"""Screen capture. Prefers dxcam (DXGI Desktop Duplication - can see DirectX
apps that GDI/mss renders as black); falls back to mss.
Produces: full-res PIL frame, record-size frame, and model grid."""
import numpy as np
from PIL import Image


class Capture:
    def __init__(self, cfg):
        self.rec_size = (cfg.get("record_width", 320), cfg.get("record_height", 180))
        self.grid = tuple(cfg.get("model_grid", [64, 36]))
        self.model_size = self.grid
        self.fps = cfg.get("fps", 10)
        self.region = cfg.get("region")  # {"left","top","width","height"} or None
        self.backend = "none"
        self._dx = None
        self._sct = None
        self._last_arr = None

        want = cfg.get("backend", "auto")
        if want in ("auto", "dxcam"):
            try:
                import dxcam
                self._dx = dxcam.create(output_idx=0, output_color="RGB")
                self.backend = "dxcam"
            except Exception:
                self._dx = None
        if self._dx is None:
            import mss
            self._sct = mss.mss()
            self.mon = self.region or self._resolve_monitor(cfg.get("monitor", "primary"))
            self.backend = "mss"

    def _resolve_monitor(self, monitor):
        """'primary' -> the display at desktop coords (0,0); or a numeric index."""
        if monitor == "primary" or monitor is None:
            for m in self._sct.monitors[1:]:
                if m["left"] == 0 and m["top"] == 0:
                    return m
            return self._sct.monitors[1]
        return self._sct.monitors[int(monitor)]

    def grab(self):
        """Returns (full_img, rec_img, grid_gray). grid_gray is (gh, gw) uint8."""
        if self._dx is not None:
            arr = self._dx.grab()
            if arr is not None:
                self._last_arr = arr
            elif self._last_arr is not None:
                arr = self._last_arr
            else:
                arr = np.zeros((self.rec_size[1] * 8, self.rec_size[0] * 8, 3), np.uint8)
            img = Image.fromarray(arr, "RGB")
            if self.region:
                r = self.region
                img = img.crop((r["left"], r["top"],
                                r["left"] + r["width"], r["top"] + r["height"]))
        else:
            shot = self._sct.grab(self.mon)
            img = Image.frombytes("RGB", shot.size, shot.rgb)

        rec = img.resize(self.rec_size, Image.BILINEAR)
        grid = np.asarray(rec.convert("L").resize(self.model_size, Image.BILINEAR),
                          dtype=np.uint8)
        return img, rec, grid

    def close(self):
        if self._dx is not None:
            try:
                self._dx.release()
            except Exception:
                pass
        if self._sct is not None:
            self._sct.close()
