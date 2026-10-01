"""Routines: scripted step sequences for deterministic tasks (menus, map
navigation, redeploy flow). The RL brain handles fuzzy in-mission play;
routines handle the parts that are the same every time.

Step types:
  {"sleep": seconds}
  {"act": "action_name"}                       - run a profile action
  {"key": "e", "hold": 0.5}                    - direct key press; "$name" resolves via keybinds
  {"keys": ["w", "d"], "hold": 4}              - hold multiple keys together (diagonal walk)
  {"click_at": [x, y], "button": "left"}       - absolute screen click
  {"scroll_at": [x, y], "dy": 1|-1}            - move cursor and scroll wheel
  {"wait": {"region": [x,y,w,h], "brightness": [lo,hi], "red_bias": n},
   "timeout": sec, "else": step_index}         - wait until pixel condition
  {"mode": "watch"|"control"}                  - switch agent mode
  {"routine": "other_name"}                    - chain into another routine
  {"context": "text"}                          - inject a context note
  {"log": "text"}
  {"goto": step_index}
  {"stop": true}
"""
import threading
import time


class Routine:
    def __init__(self, agent, name, steps):
        self.agent = agent
        self.name = name
        self.steps = steps
        self.abort = False
        self.thread = None
        self.current_step = -1

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        a = self.agent
        a.log(f"[routine] {self.name} started")
        a.rec.event({"type": "routine", "name": self.name, "event": "start"})
        i = 0
        while i < len(self.steps) and a.running and not self.abort:
            if a.paused:
                time.sleep(0.3)
                continue
            self.current_step = i
            try:
                nxt = self._do(self.steps[i], i)
            except Exception as e:
                a.log(f"[routine] step {i} error: {e}")
                break
            if nxt == "stop" or self.abort:
                break
            i = nxt if isinstance(nxt, int) else i + 1
        done = not self.abort and i >= len(self.steps)
        a.log(f"[routine] {self.name} {'done' if done else 'aborted'}")
        a.rec.event({"type": "routine", "name": self.name,
                     "event": "done" if done else "aborted"})
        if a._routine is self:
            a._routine = None

    def _do(self, s, i):
        a = self.agent
        if "sleep" in s:
            time.sleep(float(s["sleep"]))
        elif "act" in s:
            idx = a.index_of.get(s["act"])
            if idx is not None:
                a.exec.run(idx)
        elif "key" in s or "keys" in s:
            keys = s.get("keys", [s.get("key")])
            for key in keys:
                if isinstance(key, str) and key.startswith("$"):
                    key = a.p.get("keybinds", {}).get(key[1:], key)
                a.exec._press(key, s.get("hold", 0.12))
        elif "click_at" in s:
            x, y = s["click_at"]
            a.exec.click_at(x, y, s.get("button", "left"), s.get("hold", 0))
        elif "scroll_at" in s:
            x, y = s["scroll_at"]
            a.exec.scroll_at(x, y, s.get("dy", -1))
        elif "wait" in s:
            return self._wait(s, i)
        elif "mode" in s:
            a.set_mode(s["mode"])
        elif "routine" in s:
            a.start_routine(s["routine"])
            return "stop"
        elif "on_ship" in s:
            a.on_ship = bool(s["on_ship"])
        elif "task" in s:
            a.set_task(s["task"])
        elif "context" in s:
            a.on_note(s["context"])
        elif "log" in s:
            a.log(f"[routine] {s['log']}")
        elif "goto" in s:
            return int(s["goto"])
        elif s.get("stop"):
            return "stop"
        return None

    def _wait(self, s, i):
        cond = s["wait"]
        deadline = time.time() + float(s.get("timeout", 15))
        while time.time() < deadline and self.agent.running and not self.abort:
            frame, _, _ = self.agent.cap.grab()
            if self.agent.eval_condition(frame, cond):
                return None
            time.sleep(0.25)
        self.agent.log(f"[routine] wait timed out at step {i}")
        return int(s["else"]) if "else" in s else None
