"""Episode recorder: saves frames (jpeg) + events (jsonl) so the agent
builds a memory of what it saw, did, and was told."""
import json
import time
from pathlib import Path


class Recorder:
    def __init__(self, root, profile_name):
        self.dir = Path(root) / "episodes" / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        self._log = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self._n = 0
        self.event({"type": "start", "profile": profile_name})

    def event(self, obj):
        obj = dict(obj)
        obj.setdefault("t", round(time.time(), 3))
        self._log.write(json.dumps(obj) + "\n")
        self._log.flush()

    def step(self, rec_img, mode, action_name, scores, reward):
        """Save one timestep: frame + metadata line."""
        name = f"{self._n:06d}.jpg"
        rec_img.save(self.dir / "frames" / name, format="JPEG", quality=55)
        self.event({
            "type": "step",
            "frame": name,
            "mode": mode,
            "action": action_name,
            "reward": round(reward, 4),
            "top": sorted(
                ((a, round(float(s), 3)) for a, s in scores),
                key=lambda x: -x[1],
            )[:4],
        })
        self._n += 1

    def close(self):
        self.event({"type": "end"})
        self._log.close()
