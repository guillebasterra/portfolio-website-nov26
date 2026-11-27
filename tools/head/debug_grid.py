"""Debug sheet: each photo with its mask outline and a labelled pixel grid."""
import sys, cv2, numpy as np
from PIL import Image, ImageOps

def load(path):
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))

def panel(name, step=100, crop=None, zoom=1.0):
    img = load(f'assets/head/{name}.jpeg')
    m = cv2.imread(f'assets/head/masks/{name}.png', 0)
    assert m.shape[:2] == img.shape[:2], (m.shape, img.shape)
    out = img.copy()
    cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (255, 0, 255), 3)
    h, w = m.shape
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(out, str(x), (x + 3, (crop[1] if crop else 0) + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(out, str(y), ((crop[0] if crop else 0) + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    if crop:
        x0, y0, x1, y1 = crop
        out = out[y0:y1, x0:x1]
    if zoom != 1.0:
        out = cv2.resize(out, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
    return out

if __name__ == '__main__':
    out_path, zoom = sys.argv[1], float(sys.argv[2])
    names = sys.argv[3:]
    ps = [panel(n, zoom=zoom) for n in names]
    H = max(p.shape[0] for p in ps)
    ps = [np.pad(p, ((0, H - p.shape[0]), (0, 12), (0, 0)), constant_values=255) for p in ps]
    Image.fromarray(np.hstack(ps)).save(out_path, quality=85)
