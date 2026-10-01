"""Profile loading and defaults for GameAI."""
import json
from pathlib import Path

PROFILES_DIR = Path(__file__).parent / "profiles"

DEFAULTS = {
    "name": "default",
    "capture": {
        "monitor": "primary",    # "primary" = display at (0,0), or a numeric mss index
        "region": None,          # {"left":x,"top":y,"width":w,"height":h} or null = full monitor
        "fps": 10,
        "record_width": 320,
        "record_height": 180,
        "model_grid": [64, 36],  # frame downscale fed to the reward model
    },
    "control": {
        "steps_per_sec": 6,
        "epsilon": 0.15,          # exploration rate
        "credit_window_sec": 1.2, # how far back human feedback applies
        "max_hold_sec": 3.0,
    },
    "context": {
        "ttl_sec": 90,            # a context note stays active this long
        "dim": 128,               # hashed context feature size
    },
    "humanize": {
        "enabled": True,          # jitter timing to avoid metronome-perfect input
        "step_jitter": 0.35,      # +/-35% on action rate
        "hold_jitter": 0.3,       # +/-30% on key hold durations
        "idle_chance": 0.04,      # occasional do-nothing step in control mode
    },
    "ollama": {
        "enabled": True,
        "host": "http://127.0.0.1:11434",
        "text_model": "llama3.2:3b",   # context -> action hints
        "vision_model": "moondream",   # scene descriptions; disabled if not pulled
        "scene_interval_sec": 10,
        "planner_bias": 0.25,
        "planner_bias_ttl_sec": 45,
    },
    "hotkeys": {
        "pause": "f8",
        "good": "f9",
        "bad": "f10",
        "toggle_control": "f11",
        "quit": "f12",
    },
    "keybinds": {},  # logical_name -> physical key, e.g. "interact": "e"
    "actions": [
        {"name": "idle", "type": "none"},
        {"name": "forward", "type": "key", "key": "w", "hold": 0.8},
        {"name": "back", "type": "key", "key": "s", "hold": 0.8},
        {"name": "left", "type": "key", "key": "a", "hold": 0.8},
        {"name": "right", "type": "key", "key": "d", "hold": 0.8},
        {"name": "sprint", "type": "key", "key": "shift", "hold": 1.0},
        {"name": "interact", "type": "key", "key": "e"},
        {"name": "look_left", "type": "mouse_move", "dx": -180, "dy": 0},
        {"name": "look_right", "type": "mouse_move", "dx": 180, "dy": 0},
        {"name": "look_up", "type": "mouse_move", "dx": 0, "dy": -140},
        {"name": "look_down", "type": "mouse_move", "dx": 0, "dy": 140},
        {"name": "fire", "type": "click", "button": "left"},
        {"name": "aim", "type": "click", "button": "right"},
    ],
    "pixel_rewards": [],
    "routines_dir": "routines",
    "routine_triggers": {},        # pixel hook name -> routine to auto-start
    "data_dir": "data",
}


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_profile(name_or_path=None):
    """Return profile dict. Falls back to DEFAULTS if nothing found."""
    path = None
    if name_or_path:
        cand = Path(name_or_path)
        path = cand if cand.suffix == ".json" else PROFILES_DIR / f"{name_or_path}.json"
    if path and path.exists():
        return _resolve_binds(_merge(DEFAULTS, json.loads(path.read_text(encoding="utf-8"))))
    return _resolve_binds(_merge(DEFAULTS, {}))


def _resolve_binds(profile):
    """Replace any action field value "$name" with the physical key from keybinds."""
    kb = profile.get("keybinds", {})
    for a in profile.get("actions", []):
        for field, val in list(a.items()):
            if isinstance(val, str) and val.startswith("$"):
                a[field] = kb.get(val[1:], val)
    return profile
