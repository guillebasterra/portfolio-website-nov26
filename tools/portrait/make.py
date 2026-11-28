"""Make the ASCII portrait stage from one head photo and its mask:
the masked photo as a grid of typed characters, density carrying tone.

usage: python3 tools/portrait/make.py [photo_name] [cols]
writes prototype/page/ascii.js
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

SRC = 'assets/head'
NAME = sys.argv[1] if len(sys.argv) > 1 else 'IMG_0937'
COLS = int(sys.argv[2]) if len(sys.argv) > 2 else 96
CELL = 0.6            # character cell width / height, matches the page's <pre>
RAMP = ' .,:;-=+ox*XB#%@'


def load(name):
    rgb = np.array(ImageOps.exif_transpose(Image.open(f'{SRC}/{name}.jpeg')).convert('RGB'))
    mask = np.array(Image.open(f'{SRC}/masks/{name}.png').convert('L')) > 127
    # Crop to the figure, with a little air around it.
    ys, xs = np.where(mask)
    pad = int(0.04 * (ys.max() - ys.min()))
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    return rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def ascii_grid(grey, mask):
    h, w = grey.shape
    rows = round(COLS * h / w * CELL)
    # Local contrast first, so skin doesn't flatten into one character.
    eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(grey)
    small = cv2.resize(eq, (COLS, rows), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    cover = cv2.resize(mask.astype(np.float32), (COLS, rows), interpolation=cv2.INTER_AREA)
    dark = np.clip((0.92 - small) / 0.8, 0, 1) ** 0.9
    idx = np.clip((dark * (len(RAMP) - 1)).round().astype(int), 1, len(RAMP) - 1)
    idx[cover < 0.5] = 0
    lines = [''.join(RAMP[i] for i in row).rstrip() for row in idx]
    return COLS, rows, '\n'.join(lines)


rgb, mask = load(NAME)
grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

cols, rows, text = ascii_grid(grey, mask)
os.makedirs('prototype/page', exist_ok=True)
with open('prototype/page/ascii.js', 'w') as f:
    f.write('window.PORTRAIT_ASCII = ' + json.dumps(dict(source=NAME, cols=cols, rows=rows, text=text)) + ';\n')

print(f'{NAME}: ascii {cols}x{rows}')
