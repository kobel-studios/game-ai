"""POI scanner: opens the map (Tab), captures the screen, counts the
'?' point-of-interest markers (the spots where Super Credits hide),
saves an annotated image so you can verify the detection.

Run while in a mission:
    python poi_scan.py
Press Tab yourself first if the map is already open.
"""
import time
from pathlib import Path

import numpy as np
import cv2
from pynput.keyboard import Controller, Key

HERE = Path(__file__).resolve().parent

# minimap/full-map region on a 2560x1440 screen (tune if detection is off)
MAP_REGION = (0, 0, 2560, 1440)
# '?' icons are small bright blobs; rough size range in pixels
BLOB_MIN, BLOB_MAX = 8, 400


def grab_frame():
    import dxcam
    cam = dxcam.create(output_idx=0, output_color="RGB")
    time.sleep(0.3)
    frame = cam.grab()
    cam.release()
    return frame


def count_pois(frame):
    """Detect small bright blob clusters = '?' POI icons."""
    x, y, w, h = MAP_REGION
    roi = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # POI icons glow bright against the dark map
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    pts = [(int(cx + x), int(cy + y)) for i in range(1, n)
           for (cx, cy) in [cent[i]]
           if BLOB_MIN < stats[i, cv2.CC_STAT_AREA] < BLOB_MAX
           and abs(stats[i, cv2.CC_STAT_WIDTH] - stats[i, cv2.CC_STAT_HEIGHT]) < 15]
    return pts, roi


def main():
    kb = Controller()
    print("opening map...")
    kb.tap(Key.tab)
    time.sleep(1.2)
    frame = grab_frame()
    kb.tap(Key.tab)  # close map
    if frame is None:
        print("no frame captured")
        return
    pts, roi = count_pois(frame)
    out = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    for (cx, cy) in pts:
        cv2.circle(out, (cx - MAP_REGION[0], cy - MAP_REGION[1]), 18, (0, 0, 255), 3)
    img_path = HERE / "poi_scan.png"
    cv2.imwrite(str(img_path), out)
    print(f"POI candidates found: {len(pts)}")
    for p in pts:
        print(f"  at {p}")
    print(f"annotated image -> {img_path}")


if __name__ == "__main__":
    main()
