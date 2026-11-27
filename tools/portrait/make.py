"""Make the two flat portrait stages from one head photo and its mask.

ascii: the masked photo as a grid of typed characters, density carrying tone.
trace: an XDoG ink trace of the same photo, a stand-in until a real drawing
       is dropped at assets/portrait/drawing.svg (or .png).

usage: python3 tools/portrait/make.py [photo_name] [cols]
writes prototype/page/ascii.js and assets/portrait/trace.png
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
INK = (0x31, 0x2F, 0x28)


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


def trace(grey, mask, width=1100):
    h, w = grey.shape
    size = (width, round(width * h / w))
    g = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6)).apply(g).astype(np.float32) / 255
    g = cv2.bilateralFilter(g, 9, 0.1, 7)
    # Extended difference of Gaussians: keeps edges as ink, lets flat areas go to paper.
    sigma, k, gamma, eps, phi = 2.6, 1.6, 0.98, 0.0, 60
    d = cv2.GaussianBlur(g, (0, 0), sigma) - gamma * cv2.GaussianBlur(g, (0, 0), sigma * k)
    out = np.where(d > eps, 1.0, 1.0 + np.tanh(phi * (d - eps)))
    ink = np.clip(1 - out, 0, 1) * cv2.GaussianBlur(m, (0, 0), 3)
    # Drop specks, then fatten what's left into one even marker line.
    solid = (ink > 0.45).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    keep = np.isin(labels, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 60])
    solid = cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ink = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 0.8)
    rgba = np.zeros((*ink.shape, 4), np.uint8)
    rgba[..., :3] = INK
    rgba[..., 3] = (ink * 255).astype(np.uint8)
    return Image.fromarray(rgba)


rgb, mask = load(NAME)
grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

cols, rows, text = ascii_grid(grey, mask)
os.makedirs('prototype/page', exist_ok=True)
with open('prototype/page/ascii.js', 'w') as f:
    f.write('window.PORTRAIT_ASCII = ' + json.dumps(dict(source=NAME, cols=cols, rows=rows, text=text)) + ';\n')

os.makedirs('assets/portrait', exist_ok=True)
trace(grey, mask).save('assets/portrait/trace.png')
print(f'{NAME}: ascii {cols}x{rows}, trace written')
