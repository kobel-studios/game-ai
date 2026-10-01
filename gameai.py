"""GameAI - watches your screen, takes your context, learns from your feedback.

Modes:
  watch   (default) - records screen + your inputs + context. You play; it learns passively.
  control - the agent takes actions; you steer it with good/bad hotkeys and context notes.

Usage:
  python gameai.py --profile helldivers2            # terminal UI
  python gameai.py --profile helldivers2 --gui      # Tkinter GUI
  python gameai.py --profile helldivers2 --control  # start in control mode

Hotkeys (global, work while game is focused):
  F9 good | F10 bad | F11 toggle watch/control | F12 quit
"""
import argparse
import json
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from actions import Executor
from brain import Brain
from capture import Capture
from config import load_profile
from feedback import ContextInput, Hotkeys, InputWatcher
from ollama_client import Ollama
from recorder import Recorder

SCENE_QUESTION = (
    "Describe this video game frame in one sentence: location, what the player "
    "is doing, any enemies, pickups, or UI prompts visible."
)


class GameAI:
    def __init__(self, profile, start_control=False):
        self.p = profile
        self.actions = profile["actions"]
        self.name_of = {i: a["name"] for i, a in enumerate(self.actions)}
        self.index_of = {a["name"]: i for i, a in enumerate(self.actions)}
        self.idle_idx = self.index_of.get("idle", 0)
        self.data_dir = Path(profile["data_dir"])
        if not self.data_dir.is_absolute():
            self.data_dir = Path(__file__).resolve().parent / self.data_dir
        grid = tuple(profile["capture"]["model_grid"])
        ctx_dim = profile["context"]["dim"]
        self.grid = grid
        self.ctx_dim = ctx_dim
        self.expected_dim = grid[0] * grid[1] + ctx_dim + 1
        self.task = "sc"
        self.model_path = self._model_file(self.task)
        self.brain = self._load_brain()

        humanize = profile.get("humanize", {})
        self.humanize_on = humanize.get("enabled", False)
        self.step_jitter = humanize.get("step_jitter", 0.0)
        self.idle_chance = humanize.get("idle_chance", 0.0)

        self.cap = Capture(profile["capture"])
        self.exec = Executor(self.actions, profile["control"]["max_hold_sec"],
                             hold_jitter=humanize.get("hold_jitter", 0.0))
        self.rec = Recorder(self.data_dir, profile["name"])
        self.ollama = Ollama(profile["ollama"])

        self.mode = "control" if start_control else "watch"
        self.paused = False
        self.on_ship = True  # ship = no combat keys (R opens warbonds, etc.)
        self.running = True
        self.lock = threading.Lock()
        self._started = False
        self._stopped = False

        # recent (phi, action, t) for credit assignment of human/env rewards
        self.recent = deque(maxlen=400)
        # context notes: deque of (t, text)
        self.notes = deque()
        self.ctx_ttl = profile["context"]["ttl_sec"]

        # ollama outputs
        self.scene = ""
        self._last_scene = 0.0
        self._bias = np.zeros(len(self.actions), dtype=np.float32)
        self._bias_until = 0.0

        # GUI-visible state
        self.log_lines = deque(maxlen=300)
        self.last_rec = None          # latest record-size PIL frame
        self.last_action = "-"
        self.last_reward = 0.0
        # imitation: what keys the human is holding right now (key -> time pressed)
        self.user_held = {}
        self.imitations = 0
        self._pending_imitate = None   # set by stratagem capture, consumed in step
        self._action_hist = deque(maxlen=8)  # control-mode action history (anti-stuck)
        # stratagem capture: arrows typed while the stratagem key is held
        self.strat_key = str(profile.get("keybinds", {}).get("stratagem", "ctrl")).lower()
        self._strat_active = False
        self._strat_seq = []
        self._strat_until = 0.0     # capture window ends ~3s after F1 tap or on throw click
        self.seq_to_action = {tuple(a["seq"]): i for i, a in enumerate(self.actions)
                              if a.get("type") == "stratagem" and a.get("seq")}
        # physical key -> action index (tap variants preferred over holds)
        self.key_to_action = {}
        for i, a in enumerate(self.actions):
            if a.get("type") == "key" and a.get("key"):
                k = str(a["key"]).lower()
                if k not in self.key_to_action or not a.get("hold"):
                    self.key_to_action[k] = i
            elif a.get("type") == "click" and a.get("button"):
                k = "button." + a["button"].lower()
                if k not in self.key_to_action:
                    self.key_to_action[k] = i

        self._step = 0
        self._black = 0               # consecutive all-black captures
        self._pixel_cd = {}           # hook name -> last fire time
        self._pixel_prev = {}         # hook name -> previous mean (delta mode)
        self._linger = {}             # hook name -> when region first matched (linger mode)
        self.last_auto = None         # (hook name, reward, t) most recent auto reward

        # routines: name -> {"name","steps"}; self._routine = active Routine
        self.routines = {}
        self._routine = None
        self.load_routines()
        self._services = []

    # ---------- logging / state ----------

    def log(self, msg):
        try:
            print(msg)
        except Exception:
            pass  # pythonw has no console
        self.log_lines.append(str(msg))

    def context_text(self):
        now = time.time()
        while self.notes and now - self.notes[0][0] > self.ctx_ttl:
            self.notes.popleft()
        parts = [t for _, t in self.notes]
        if self.scene:
            parts.append(self.scene)
        return " ".join(parts)

    def assign_credit(self, reward):
        """Apply reward to actions in the recent window, recency-weighted."""
        win = self.p["control"]["credit_window_sec"]
        now = time.time()
        with self.lock:
            for phi, a, t in list(self.recent):
                age = now - t
                if age <= win:
                    self.brain.update(phi, a, reward, weight=1.0 - age / win)
        self.last_reward = reward
        self.rec.event({"type": "reward", "value": reward})

    def check_pixel_hooks(self, frame):
        """Auto-reward hooks. Modes:
          range - fires while mean brightness is inside [lo, hi]
          delta - fires when mean brightness jumps by >= delta_min vs last frame
        Optional red_bias: requires R - avg(G,B) >= value (e.g. red death screens)."""
        total = 0.0
        now = time.time()
        gray = np.asarray(frame.convert("L"), dtype=np.float32)
        rgb = None
        for hook in self.p["pixel_rewards"]:
            name = hook.get("name", "hook")
            x, y, w, h = hook["region"]
            crop = gray[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            mean = float(crop.mean())

            fired = False
            mode = hook.get("mode", "range")
            if mode == "delta":
                prev = self._pixel_prev.get(name)
                self._pixel_prev[name] = mean
                if prev is not None and mean - prev >= hook.get("delta_min", 20):
                    fired = True
            elif mode == "linger":
                # fires if the region stays in range continuously for linger_sec
                lo, hi = hook.get("brightness", [0, 255])
                if lo <= mean <= hi:
                    start = self._linger.get(name)
                    if start is None:
                        self._linger[name] = now
                    elif now - start >= hook.get("linger_sec", 30):
                        fired = True
                else:
                    self._linger.pop(name, None)
            else:
                lo, hi = hook.get("brightness", [0, 255])
                fired = lo <= mean <= hi

            if fired and hook.get("red_bias") is not None:
                if rgb is None:
                    rgb = np.asarray(frame, dtype=np.float32)
                c = rgb[y : y + h, x : x + w]
                margin = float(c[..., 0].mean() - (c[..., 1].mean() + c[..., 2].mean()) / 2)
                fired = margin >= hook["red_bias"]

            cd = hook.get("cooldown_sec", 5)
            if fired and now - self._pixel_cd.get(name, 0) >= cd:
                reward = hook.get("reward", 0.0)
                total += reward
                self._pixel_cd[name] = now
                self._linger[name] = now
                self.last_auto = (name, reward, now)
                self.rec.event({"type": "pixel_hook", "name": name,
                                "mean": round(mean, 1)})
                trig = self.p.get("routine_triggers", {}).get(name)
                if trig:
                    self.start_routine(trig)
        return total

    # ---------- callbacks ----------

    def on_hotkey(self, name):
        if name == "good":
            self.log("[feedback] +1")
            self.assign_credit(1.0)
        elif name == "bad":
            self.log("[feedback] -1")
            self.assign_credit(-1.0)
        elif name == "toggle_control":
            self.toggle_mode()
        elif name == "pause":
            self.toggle_pause()
        elif name == "quit":
            self.running = False

    def reset_brain(self):
        """Wipe learned weights, pending credit, and the saved model file."""
        grid = tuple(self.p["capture"]["model_grid"])
        self.brain = Brain(len(self.actions), grid=grid,
                           ctx_dim=self.p["context"]["dim"])
        with self.lock:
            self.recent.clear()
            self._bias = np.zeros(len(self.actions), dtype=np.float32)
            self._bias_until = 0.0
        try:
            self.model_path.unlink(missing_ok=True)
        except Exception:
            pass
        self.log("[reset] learning wiped")
        self.rec.event({"type": "reset"})

    def toggle_pause(self):
        """Turn screen watching/recording off or on."""
        self.paused = not self.paused
        if self.paused:
            if self.mode == "control":
                self.exec.release_all()
                self.mode = "watch"
            self.last_rec = None
            self.log("[state] watching OFF")
        else:
            self.log("[state] watching ON")
        self.rec.event({"type": "paused", "paused": self.paused})

    def toggle_mode(self):
        if self.mode == "control":
            self.exec.release_all()
            self.mode = "watch"
        else:
            self.mode = "control"
        self.log(f"[mode] {self.mode}")
        self.rec.event({"type": "mode", "mode": self.mode})

    def set_mode(self, mode):
        if mode in ("watch", "control") and mode != self.mode:
            self.toggle_mode()

    def eval_condition(self, frame, cond):
        """Range-mode pixel check for routine 'wait' steps: brightness range
        plus optional red_bias on region [x,y,w,h]."""
        x, y, w, h = cond["region"]
        gray = np.asarray(frame.convert("L"), dtype=np.float32)
        crop = gray[y : y + h, x : x + w]
        if crop.size == 0:
            return False
        lo, hi = cond.get("brightness", [0, 255])
        if not (lo <= float(crop.mean()) <= hi):
            return False
        if cond.get("red_bias") is not None:
            rgb = np.asarray(frame, dtype=np.float32)[y : y + h, x : x + w]
            margin = float(rgb[..., 0].mean() - (rgb[..., 1].mean() + rgb[..., 2].mean()) / 2)
            if margin < cond["red_bias"]:
                return False
        return True

    # ---------- routines ----------

    def load_routines(self):
        rdir = Path(self.p.get("routines_dir", "routines"))
        if not rdir.is_absolute():
            rdir = Path(__file__).resolve().parent / rdir
        self.routines = {}
        if rdir.is_dir():
            for f in rdir.glob("*.json"):
                try:
                    spec = json.loads(f.read_text(encoding="utf-8"))
                    self.routines[spec.get("name", f.stem)] = spec
                except Exception as e:
                    self.log(f"[routine] failed to load {f.name}: {e}")

    def _model_file(self, task):
        return self.data_dir / f"model_{self.p['name']}_{task}.npz"

    def _load_brain(self):
        if self.model_path.exists():
            try:
                b = Brain.load(self.model_path, n_actions=len(self.actions))
                if b.dim == self.expected_dim:
                    return b
            except Exception:
                pass
        return Brain(len(self.actions), grid=self.grid, ctx_dim=self.ctx_dim)

    def set_task(self, task):
        """Swap brains: each task (sc / medal) learns into its own model file."""
        if task == self.task:
            return
        self.brain.save(self.model_path)
        self.task = task
        self.model_path = self._model_file(task)
        self.brain = self._load_brain()
        self.log(f"[brain] switched to task '{task}'")

    def start_routine(self, name):
        spec = self.routines.get(name)
        if not spec:
            self.log(f"[routine] unknown: {name}")
            return False
        if self._routine is not None and self._routine.name == name \
                and time.time() - getattr(self._routine, "started_at", 0) < 2.0:
            self.log(f"[routine] {name} already running - ignoring double-start")
            return False
        self.stop_routine()
        from routines import Routine
        self._routine = Routine(self, name, spec.get("steps", []))
        self._routine.started_at = time.time()
        self._routine.start()
        return True

    def stop_routine(self):
        if self._routine is not None:
            self._routine.abort = True
            self._routine = None

    def on_note(self, text):
        now = time.time()
        self.notes.append((now, text))
        self.rec.event({"type": "context", "text": text})
        self.log(f"[context] {text}")
        # ask the text model which actions this implies (background)
        threading.Thread(target=self._plan, args=(text,), daemon=True).start()

    def _plan(self, context_text):
        plan = self.ollama.plan(context_text, self.scene, list(self.index_of))
        if not plan:
            return
        bias_mag = self.p["ollama"]["planner_bias"]
        bias = np.zeros(len(self.actions), dtype=np.float32)
        for a in plan["prefer"]:
            if a in self.index_of:
                bias[self.index_of[a]] = bias_mag
        for a in plan["avoid"]:
            if a in self.index_of:
                bias[self.index_of[a]] = -bias_mag
        with self.lock:
            self._bias = bias
            self._bias_until = time.time() + self.p["ollama"]["planner_bias_ttl_sec"]
        self.rec.event({"type": "plan", **plan})
        self.log(f"[plan] prefer={plan['prefer']} avoid={plan['avoid']}")

    def _scene_loop(self):
        while self.running:
            if (not self.paused) and self.ollama.vision_ready() and (
                time.time() - self._last_scene
                > self.p["ollama"]["scene_interval_sec"]
            ):
                self._last_scene = time.time()
                try:
                    frame, _, _ = self.cap.grab()
                    desc = self.ollama.describe_scene(frame, SCENE_QUESTION)
                except Exception:
                    desc = None
                if desc:
                    self.scene = desc
                    self.rec.event({"type": "scene", "text": desc})
                    self.log(f"[scene] {desc}")
            time.sleep(1.0)

    def _watch_input(self, ev):
        if self.mode == "watch" and not self.paused:
            self.rec.event({"type": "input", **ev})
            t, key = ev.get("type"), ev.get("key") or ev.get("button")
            if not key:
                return
            key = str(key).lower()
            now = time.time()
            arrows = ("up", "down", "left", "right")
            if self._strat_active and now > self._strat_until:
                self._finish_strat()
            # F1 tap opens the stratagem menu - capture arrows until the
            # throw click (or timeout)
            if t == "key_down" and key == self.strat_key:
                self._strat_active = True
                self._strat_seq = []
                self._strat_until = now + 3.0
            elif self._strat_active and t == "key_down" and key in arrows:
                self._strat_seq.append(key)
            elif self._strat_active and t == "mouse_down":
                # throw click ends the sequence - it's part of the stratagem,
                # not a "fire" demonstration
                self._finish_strat()
            elif t in ("key_down", "mouse_down"):
                self.user_held[key] = now
            else:
                self.user_held.pop(key, None)

    def _finish_strat(self):
        if self._strat_seq:
            tgt = self.seq_to_action.get(tuple(self._strat_seq))
            if tgt is not None:
                self._pending_imitate = tgt
            else:
                self.log(f"[watch] unknown stratagem code {self._strat_seq} - "
                         "add it to profile actions to teach it")
        self._strat_active = False
        self._strat_seq = []

    def _held_to_action(self):
        """Most recently pressed still-held key -> action index, or None."""
        if not self.user_held:
            return None
        latest = max(self.user_held, key=self.user_held.get)
        return self.key_to_action.get(latest)

    # ---------- lifecycle ----------

    def start_services(self, console_input=True):
        if self._started:
            return
        self._started = True
        hotkeys = Hotkeys(self.p["hotkeys"], self.on_hotkey)
        watcher = InputWatcher(self._watch_input)
        hotkeys.start()
        watcher.start()
        self._services = [hotkeys, watcher]
        if console_input:
            ContextInput(self.on_note).start()
        threading.Thread(target=self._scene_loop, daemon=True).start()
        self.log(f"GameAI [{self.p['name']}] mode={self.mode}")
        self.log(f"  model: {self.model_path} (loaded={self.model_path.exists()})")
        self.log(f"  capture: {self.cap.backend}   "
                 f"ollama text={self.ollama.text_ready()} vision={self.ollama.vision_ready()}")
        self.log("  hotkeys: F8 watch on/off | F9 reward | F10 punish | F11 control | F12 quit")

    def step_once(self):
        """One capture/act/record iteration."""
        t0 = time.time()
        frame, rec_img, grid = self.cap.grab()
        self.last_rec = rec_img

        if grid.mean() < 3:
            self._black += 1
            if self._black == int(5 * self.p["control"]["steps_per_sec"]):
                self.log("[warn] captures are all black - if a game is running, "
                         "switch it to borderless/windowed (exclusive fullscreen can't be captured)")
        else:
            self._black = 0
        ctx = self.context_text()
        phi = self.brain.features(grid, ctx)

        scores = []
        if self.mode == "watch":
            tgt = self._pending_imitate
            self._pending_imitate = None
            if tgt is None:
                tgt = self._held_to_action()
            if tgt is not None and self.brain.imitate(phi, tgt):
                self.imitations += 1
        # on the ship only routines drive - the brain has nothing to do there
        elif self.mode == "control" and self._routine is None and not self.on_ship:
            with self.lock:
                bias = self._bias if time.time() < self._bias_until else None
            masked = set()
            if self.on_ship:
                masked = {i for i, n in self.name_of.items()
                          if n.startswith("strat_") or n in
                          ("reload", "fire", "aim", "grenade", "melee", "stim", "dive")}
                ship_bias = np.full(len(self.actions), -50.0, dtype=np.float32)
                for i in range(len(self.actions)):
                    if i not in masked:
                        ship_bias[i] = 0.0
                bias = ship_bias if bias is None else bias + ship_bias
            a, sc = self.brain.act(phi, self.p["control"]["epsilon"], bias)
            while a in masked:  # random picks ignore bias - resample
                a = int(self.brain.rng.integers(len(self.actions)))
            if self.humanize_on and random.random() < self.idle_chance:
                a = self.idle_idx
            # anti-stuck: same action 8x in a row -> force a fresh pick
            self._action_hist.append(a)
            if len(self._action_hist) >= 8 and len(set(self._action_hist)) == 1:
                allowed = [i for i in range(len(self.actions)) if i not in masked]
                a = allowed[int(self.brain.rng.integers(len(allowed)))] if allowed else a
                self._action_hist.clear()
                self.log(f"[anti-stuck] forced random action: {self.name_of[a]}")
            self.exec.run(a)
            self.last_action = self.name_of[a]
            with self.lock:
                self.recent.append((phi, a, t0))
            scores = sc.tolist()

        # auto-rewards + routine triggers only make sense while the AGENT is
        # acting - gating to control mode also stops false-positive hooks on
        # the desktop from firing routines that flip control on
        env_r = self.check_pixel_hooks(frame) if self.mode == "control" else 0.0
        if env_r:
            self.assign_credit(env_r)

        self.rec.step(rec_img, self.mode, self.last_action,
                      zip(self.name_of.values(), scores) if scores else [],
                      env_r)
        self._step += 1
        if self._step % int(30 * self.p["control"]["steps_per_sec"]) == 0:
            self.brain.save(self.model_path)

    def _step_delay(self):
        if self.mode == "watch":
            base = 1.0 / max(float(self.p["capture"].get("fps", 10)), 1.0)
        else:
            base = 1.0 / self.p["control"]["steps_per_sec"]
            if self.humanize_on:
                base *= 1.0 + random.uniform(-self.step_jitter, self.step_jitter)
        return max(0.02, base)

    def run_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.2)
                continue
            self.step_once()
            time.sleep(self._step_delay())

    def shutdown(self):
        if self._stopped:
            return
        self._stopped = True
        self.exec.release_all()
        self.brain.save(self.model_path)
        self.rec.close()
        for s in self._services:
            try:
                s.stop()
            except Exception:
                pass
        self.cap.close()
        self.log("saved model + episode. bye.")

    def run(self):
        """Blocking terminal-mode entry."""
        self.start_services(console_input=True)
        self.log("  type context lines here, e.g.:  heading to extract  |  farming POIs")
        try:
            self.run_loop()
        finally:
            self.shutdown()


def main():
    ap = argparse.ArgumentParser(description="GameAI - watch, learn, play.")
    ap.add_argument("--profile", default="default", help="profile name in profiles/ or path to .json")
    ap.add_argument("--control", action="store_true", help="start in control mode")
    ap.add_argument("--gui", action="store_true", help="launch the Tkinter GUI")
    ap.add_argument("--calibrate", metavar="X,Y,W,H", help="print mean brightness of region once, then exit")
    ap.add_argument("--keylog", action="store_true", help="summarize keys recorded in the latest episode")
    ap.add_argument("--clickfind", type=int, metavar="SEC", nargs="?", const=30,
                    help="record mouse click coords for SEC seconds (default 30)")
    args = ap.parse_args()

    if args.clickfind:
        from pynput import mouse
        print(f"click anywhere - coords print for {args.clickfind}s (Ctrl+C to stop early)")
        def _clk(x, y, button, pressed):
            if pressed:
                print(f'  {{"click_at": [{x}, {y}]}}', flush=True)
        def _scr(x, y, dx, dy):
            print(f'  {{"scroll_at": [{x}, {y}], "dy": {dy}}}', flush=True)
        listener = mouse.Listener(on_click=_clk, on_scroll=_scr)
        listener.start()
        try:
            time.sleep(args.clickfind)
        except KeyboardInterrupt:
            pass
        listener.stop()
        print("done")
        return

    if args.keylog:
        import json as _json
        eps_dir = Path(__file__).resolve().parent / "data" / "episodes"
        eps = sorted(eps_dir.glob("*/"), key=lambda p: p.stat().st_mtime)
        if not eps:
            print("no episodes yet")
            return
        counts = {}
        for line in (eps[-1] / "events.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                ev = _json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "key_down":
                counts[ev.get("key", "?")] = counts.get(ev.get("key", "?"), 0) + 1
            elif ev.get("type") == "mouse_down":
                k = "mouse:" + ev.get("button", "?")
                counts[k] = counts.get(k, 0) + 1
        print(f"keys seen in {eps[-1].name}:")
        for k, c in sorted(counts.items(), key=lambda x: -x[1])[:24]:
            print(f"  {k:>14}  x{c}")
        return

    p = load_profile(args.profile)
    if args.calibrate:
        cap = Capture(p["capture"])
        x, y, w, h = (int(v) for v in args.calibrate.split(","))
        frame, _, _ = cap.grab()
        arr = np.asarray(frame.convert("L"), dtype=np.float32)
        crop = arr[y : y + h, x : x + w]
        print(f"region ({x},{y},{w},{h}) mean brightness = {crop.mean():.1f}")
        cap.close()
        return

    if args.gui:
        from gui import run_gui
        run_gui(p, start_control=args.control)
    else:
        GameAI(p, start_control=args.control).run()


if __name__ == "__main__":
    main()
