"""Global hotkeys (work while the game has focus), stdin context input,
and watch-mode input recording."""
import sys
import threading

from pynput import keyboard, mouse


class Hotkeys:
    def __init__(self, mapping, on_event):
        """mapping: {"good": "f9", ...}; on_event(name) called from listener thread."""
        self.map = {v.lower(): k for k, v in mapping.items()}
        self.on_event = on_event
        self._listener = keyboard.Listener(on_press=self._press)

    def _press(self, key):
        name = getattr(key, "char", None) or getattr(key, "name", None)
        if name and name.lower() in self.map:
            self.on_event(self.map[name.lower()])

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()


class ContextInput:
    """Reads context lines from the launching terminal; each line -> callback."""

    def __init__(self, on_note):
        self.on_note = on_note

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:  # stdin closed
                return
            line = line.strip()
            if line:
                self.on_note(line)


class InputWatcher:
    """Watch-mode recorder of the user's own keyboard/mouse events."""

    def __init__(self, on_input_event):
        self.on_input_event = on_input_event
        self._kb = keyboard.Listener(on_press=self._key, on_release=self._key_up)
        self._ms = mouse.Listener(on_click=self._click, on_scroll=self._scroll)

    def _key(self, key):
        name = getattr(key, "char", None) or getattr(key, "name", None)
        if name:
            self.on_input_event({"type": "key_down", "key": name})

    def _key_up(self, key):
        name = getattr(key, "char", None) or getattr(key, "name", None)
        if name:
            self.on_input_event({"type": "key_up", "key": name})

    def _click(self, x, y, button, pressed):
        self.on_input_event({
            "type": "mouse_down" if pressed else "mouse_up",
            "button": str(button), "x": x, "y": y,
        })

    def _scroll(self, x, y, dx, dy):
        self.on_input_event({"type": "scroll", "x": x, "y": y, "dx": dx, "dy": dy})

    def start(self):
        self._kb.start()
        self._ms.start()

    def stop(self):
        self._kb.stop()
        self._ms.stop()
