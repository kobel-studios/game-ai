"""Tkinter GUI for GameAI. Run: python gameai.py --profile helldivers2 --gui"""
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageTk

from gameai import GameAI

BG = "#14161a"
PANEL = "#1d2026"
FG = "#d8dde6"
ACCENT = "#3d8bff"
GOOD = "#3fbf5f"
BAD = "#d9534f"


def run_gui(profile, start_control=False):
    agent = GameAI(profile, start_control=start_control)
    App(agent).run()


class App:
    def __init__(self, agent):
        self.agent = agent
        self.root = tk.Tk()
        self.root.title(f"GameAI - {agent.p['name']}")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        # open centered - a dragged-off-screen window can't come back otherwise
        self.root.update_idletasks()
        w, h = 1120, 620
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ---------- widgets ----------

    def _build(self):
        a = self.agent
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=8, pady=(8, 4))

        self.mode_lbl = tk.Label(top, text="WATCHING YOU", font=("Consolas", 14, "bold"),
                                 bg=PANEL, fg=ACCENT, width=14)
        self.mode_lbl.pack(side="left")

        self.pause_btn = self._btn(top, "Stop watching", a.toggle_pause)
        self.pause_btn.pack(side="left", padx=6)
        self.toggle_btn = self._btn(top, "Give AI control", a.toggle_mode)
        self.toggle_btn.pack(side="left")
        self._btn(top, "Reward +", lambda: a.on_hotkey("good"), GOOD).pack(side="left")
        self._btn(top, "Punish -", lambda: a.on_hotkey("bad"), BAD).pack(side="left", padx=6)
        self._btn(top, "Reset learning", self._reset).pack(side="left")
        self._btn(top, "Quit", self._on_close).pack(side="right")

        mid = tk.Frame(self.root, bg=BG)
        mid.pack(fill="both", expand=True, padx=8)

        # live view
        self.view = tk.Label(mid, bg="#000")
        self.view.pack(side="left", fill="both", expand=True)

        # log panel
        self.logbox = scrolledtext.ScrolledText(
            mid, width=52, bg=PANEL, fg=FG, insertbackground=FG,
            font=("Consolas", 9), state="disabled")
        self.logbox.pack(side="right", fill="both", padx=(8, 0))

        # stats row
        self.stats = tk.Label(self.root, bg=BG, fg=FG, font=("Consolas", 9),
                              anchor="w", justify="left")
        self.stats.pack(fill="x", padx=8, pady=(4, 0))

        # routine row
        rrow = tk.Frame(self.root, bg=BG)
        rrow.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(rrow, text="routine:", bg=BG, fg=FG,
                 font=("Consolas", 10)).pack(side="left")
        self.routine_box = ttk.Combobox(rrow, state="readonly", width=20,
                                        font=("Consolas", 10))
        self.routine_box["values"] = sorted(a.routines.keys())
        if a.routines:
            self.routine_box.current(0)
        self.routine_box.pack(side="left", padx=6)
        self._btn(rrow, "Run", self._run_routine).pack(side="left")
        self._btn(rrow, "Stop", a.stop_routine).pack(side="left", padx=6)

        # context entry
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="x", padx=8, pady=8)
        tk.Label(bottom, text="context:", bg=BG, fg=FG,
                 font=("Consolas", 10)).pack(side="left")
        self.ctx_entry = tk.Entry(bottom, bg=PANEL, fg=FG, insertbackground=FG,
                                  font=("Consolas", 10))
        self.ctx_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.ctx_entry.bind("<Return>", self._send_ctx)
        self._btn(bottom, "Add", self._send_ctx).pack(side="left")

    def _btn(self, parent, text, cmd, color=None):
        return tk.Button(parent, text=text, command=cmd, bg=color or PANEL,
                         fg="#ffffff" if color else FG, relief="flat",
                         font=("Consolas", 10), padx=10, pady=4)

    def _reset(self):
        if messagebox.askyesno(
                "Reset learning",
                "Wipe everything the AI has learned? This can't be undone."):
            self.agent.reset_brain()

    def _run_routine(self):
        name = self.routine_box.get()
        if name:
            self.agent.start_routine(name)

    def _send_ctx(self, _ev=None):
        text = self.ctx_entry.get().strip()
        if text:
            self.ctx_entry.delete(0, "end")
            self.agent.on_note(text)

    # ---------- refresh ----------

    def _refresh(self):
        a = self.agent
        # mode indicator
        if a.paused:
            self.mode_lbl.config(text="OFF", fg="#666666")
        else:
            self.mode_lbl.config(
                text=(("AI PLAYING" if a.mode == "control" else "WATCHING YOU")
                      + f" [{a.task} brain]"),
                fg=BAD if a.mode == "control" else ACCENT)
        self.pause_btn.config(
            text="Start watching" if a.paused else "Stop watching")
        self.toggle_btn.config(
            text="Take back control" if a.mode == "control" else "Give AI control")

        # live view - only re-render when a new frame arrived
        if a.last_rec is not None and a._step != getattr(self, "_shown_step", -1):
            self._shown_step = a._step
            img = a.last_rec.resize((640, 360), Image.LANCZOS)
            self._tkimg = ImageTk.PhotoImage(img)
            self.view.config(image=self._tkimg)

        # log panel
        lines = "\n".join(a.log_lines)
        cur = self.logbox.get("1.0", "end-1c")
        if lines != cur:
            self.logbox.config(state="normal")
            self.logbox.delete("1.0", "end")
            self.logbox.insert("1.0", lines)
            self.logbox.see("end")
            self.logbox.config(state="disabled")

        # stats
        bias_on = time.time() < a._bias_until
        auto = ""
        if a.last_auto:
            name, rwd, t = a.last_auto
            age = time.time() - t
            if age < 60:
                auto = f"   auto: {name} {rwd:+.1f} ({age:.0f}s ago)"
        self.stats.config(text=(
            f"step {a._step}   action {a.last_action}   learned {a.imitations} from you   "
            f"last reward {a.last_reward:+.1f}{auto}   "
            f"notes {len(a.notes)}   plan-bias {'ON' if bias_on else 'off'}   "
            f"ollama text={'ok' if a.ollama.text_ready() else 'no'} "
            f"vision={'ok' if a.ollama.vision_ready() else 'no'}"
        ))

        if a.running:
            self.root.after(120, self._refresh)

    # ---------- lifecycle ----------

    def _on_close(self):
        self.agent.running = False

    def run(self):
        a = self.agent
        a.start_services(console_input=False)

        def agent_thread():
            a.run_loop()
            a.shutdown()
            try:
                self.root.after(0, self._final)
            except Exception:
                pass

        threading.Thread(target=agent_thread, daemon=True).start()
        self.root.after(200, self._refresh)
        self.root.mainloop()

    def _final(self):
        self.stats.config(text="stopped - model + episode saved")
        self.root.after(1200, self.root.destroy)
