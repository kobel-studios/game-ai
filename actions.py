"""Discrete action space + input executor (keyboard/mouse via pynput)."""
import random
import threading
import time

from pynput.keyboard import Controller as KbController, Key
from pynput.mouse import Button, Controller as MouseController

_KEY_NAMES = {k.name: k for k in Key}


def _kb_key(name):
    name = name.lower()
    return _KEY_NAMES.get(name, name)


def _mouse_button(name):
    return {"left": Button.left, "right": Button.right, "middle": Button.middle}[name.lower()]


class Executor:
    """Executes action dicts from the profile. Holds run on daemon threads."""

    def __init__(self, actions, max_hold_sec=3.0, hold_jitter=0.0):
        self.actions = actions
        self.max_hold = max_hold_sec
        self.jitter = hold_jitter
        self.kb = KbController()
        self.mouse = MouseController()
        self._held = set()
        self._lock = threading.Lock()

    def run(self, idx):
        a = self.actions[idx]
        t = a["type"]
        if t == "key":
            self._press(a["key"], a.get("hold", 0.12))
        elif t == "mouse_move":
            self.mouse.move(int(a.get("dx", 0)), int(a.get("dy", 0)))
        elif t == "click":
            b = _mouse_button(a.get("button", "left"))
            if a.get("hold", 0) > 0:
                self._hold_mouse(b, min(a["hold"], self.max_hold))
            else:
                self.mouse.click(b)
        elif t == "scroll":
            self.mouse.scroll(int(a.get("dx", 0)), int(a.get("dy", 0)))
        elif t == "stratagem":
            threading.Thread(target=self._stratagem, args=(a,), daemon=True).start()
        # "none" and unknown types are no-ops

    def _press(self, name, hold):
        key = _kb_key(name)
        if self.jitter:
            hold *= 1.0 + random.uniform(-self.jitter, self.jitter)
        hold = max(0.02, min(hold, self.max_hold))
        self.kb.press(key)
        with self._lock:
            self._held.add(key)
        threading.Thread(target=self._release_later, args=(key, hold), daemon=True).start()

    def _release_later(self, key, delay):
        time.sleep(delay)
        try:
            self.kb.release(key)
        finally:
            with self._lock:
                self._held.discard(key)

    def _hold_mouse(self, button, hold):
        self.mouse.press(button)
        threading.Thread(target=self._release_mouse_later, args=(button, hold), daemon=True).start()

    def _release_mouse_later(self, button, delay):
        time.sleep(delay)
        self.mouse.release(button)

    def _stratagem(self, a):
        """Tap the stratagem key (menu opens), type the arrow code,
        then left-click to throw the ball."""
        self.kb.tap(_kb_key(a["key"]))
        time.sleep(0.18)          # menu needs a beat to open
        for d in a.get("seq", []):
            self.kb.tap(_kb_key(d))
            time.sleep(0.07)
        throw = a.get("throw", "click")
        time.sleep(0.1)
        if throw == "click":
            self.mouse.click(Button.left)
        elif throw != "none":
            self.kb.tap(_kb_key(throw))
        # step back so the inbound pod doesn't crush the caller
        if a.get("backoff"):
            self._press(a["backoff"], a.get("backoff_hold", 0.6))

    def click_at(self, x, y, button="left", hold=0):
        self.mouse.position = (int(x), int(y))
        time.sleep(0.05)
        b = _mouse_button(button)
        if hold:
            self.mouse.press(b)
            time.sleep(min(float(hold), 5))
            self.mouse.release(b)
        else:
            self.mouse.click(b)

    def scroll_at(self, x, y, dy):
        self.mouse.position = (int(x), int(y))
        time.sleep(0.05)
        self.mouse.scroll(0, int(dy))

    def release_all(self):
        with self._lock:
            held = list(self._held)
            self._held.clear()
        for key in held:
            try:
                self.kb.release(key)
            except Exception:
                pass
