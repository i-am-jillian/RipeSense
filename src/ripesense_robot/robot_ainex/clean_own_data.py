"""Drop junk crops from robot collection (shadows, feet, carpet): keep a crop only if enough of it
has the fruit's colour. Rejects are moved to own_data_rejected/<class>/ (not deleted).
usage: .venv/bin/python robot/clean_own_data.py <class> [min_fraction]"""
import cv2, glob, os, shutil, sys
import numpy as np
cls = sys.argv[1]; frac_min = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'own_data')
rej = os.path.join(os.path.dirname(root), 'own_data_rejected', cls); os.makedirs(rej, exist_ok=True)
def colour_mask(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if cls in ('apple', 'strawberry'):
        return ((h < 12) | (h > 165)) & (s > 90) & (v > 40)
    if cls == 'avocado':     # dark purple-black (ripe Hass) or dark green skin
        return (v < 80) | ((h >= 30) & (h <= 90) & (s > 60) & (v < 160))
    if cls == 'mango':       # yellow / orange / green skin
        return (h >= 5) & (h <= 75) & (s > 70) & (v > 50)
    if cls.startswith('orange'):
        return (h >= 5) & (h <= 28) & (s > 100) & (v > 60)
    if cls.startswith('banana'):
        return (h >= 15) & (h <= 60) & (s > 70) & (v > 50)
    return None
kept = moved = 0
for f in sorted(glob.glob(os.path.join(root, cls, '*.jpg'))):
    img = cv2.imread(f)
    m = colour_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    if img is None or min(img.shape[:2]) < 32 or (m is not None and m.mean() < frac_min):
        shutil.move(f, os.path.join(rej, os.path.basename(f))); moved += 1
    else:
        kept += 1
print('%s: kept %d, moved %d to %s' % (cls, kept, moved, rej))
