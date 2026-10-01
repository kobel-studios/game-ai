"""Optional Ollama integration.
- vision model: periodic scene description from a frame
- text model: user context -> preferred/avoided action hints (JSON)
Everything degrades gracefully if the server or model is missing."""
import base64
import io
import json
import urllib.error
import urllib.request


class Ollama:
    def __init__(self, cfg):
        self.enabled = cfg.get("enabled", True)
        self.host = cfg.get("host", "http://127.0.0.1:11434").rstrip("/")
        self.text_model = cfg.get("text_model")
        self.vision_model = cfg.get("vision_model")
        self.timeout = 60
        self._models = None

    def _get(self, path):
        with urllib.request.urlopen(self.host + path, timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def models(self):
        """Cached list of pulled model names; None if server unreachable."""
        if self._models is None:
            try:
                tags = self._get("/api/tags")
                self._models = [m["name"] for m in tags.get("models", [])]
            except Exception:
                self._models = []
        return self._models

    def vision_ready(self):
        return bool(self.enabled and self.vision_model and self.vision_model in self.models())

    def text_ready(self):
        return bool(self.enabled and self.text_model and self.text_model in self.models())

    def describe_scene(self, pil_image, question):
        """Send a frame to the vision model; returns description or None."""
        if not self.vision_ready():
            return None
        try:
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=60)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            resp = self._post("/api/generate", {
                "model": self.vision_model,
                "prompt": question,
                "images": [b64],
                "stream": False,
                "options": {"num_predict": 120},
            })
            return resp.get("response", "").strip() or None
        except Exception:
            return None

    def plan(self, context_text, scene_text, action_names):
        """Map context+scene to action hints. Returns {"prefer": [...], "avoid": [...]} or None."""
        if not self.text_ready():
            return None
        prompt = (
            "You are the planner for a game-playing agent. Reply with ONLY JSON, no prose.\n"
            f"Available actions: {', '.join(action_names)}\n"
            f"Scene: {scene_text or 'unknown'}\n"
            f"User context: {context_text}\n"
            'Output: {"prefer": ["action names to favor"], "avoid": ["action names to avoid"]}'
        )
        try:
            resp = self._post("/api/generate", {
                "model": self.text_model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 80, "temperature": 0.2},
            })
            return _parse_json(resp.get("response", ""))
        except Exception:
            return None


def _parse_json(text):
    """Lenient JSON object extraction from model output."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return {
                "prefer": [str(a) for a in obj.get("prefer", [])],
                "avoid": [str(a) for a in obj.get("avoid", [])],
            }
    except (ValueError, TypeError):
        pass
    return None
