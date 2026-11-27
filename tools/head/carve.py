"""Carve a brick head from the front and side photo silhouettes.

Geometry comes from two views: the front mask gives the width of every
horizontal slice, the left-profile mask gives its depth. Each slice is a
rounded (superellipse) outline between those extents. Thin profile features
(nose, lips, chin) are added back as a bump that falls off sideways, so the
profile line is exact without becoming a full-width prow.

Colour is projected from all four photos by which way each brick faces.

usage: python3 tools/head/carve.py [step_px]
writes prototype/head/bricks.js and assets/head/preview.jpg
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

SRC = 'assets/head'
CFG = dict(
    front='IMG_0932', left='IMG_0933', back='IMG_0934', right='IMG_0935',
    # Hand-read landmarks, in each photo's own pixels.
    f_eye=1000, f_chin=1450, f_nose_x=845,
    l_chin=1365,
    r_chin=1420,
    cut=1480,            # front-photo row where the neck is cut off
    n_front=2.1, n_back=2.2,   # slice roundness, front half / back half
    widest=0.45,         # where a slice is widest, as a fraction from the back
    open_r=0.14,         # profile-opening radius, as a fraction of head height
)
# Sideways falloff (px, front units) of profile features, by front-photo row.
SIGMA_ROWS = [(700, 320), (930, 200), (1000, 42), (1115, 75), (1165, 90), (1255, 125), (1400, 160)]


def photo(name):
    return np.array(ImageOps.exif_transpose(Image.open(f'{SRC}/{name}.jpeg')).convert('RGB'))


def mask(name):
    return cv2.imread(f'{SRC}/masks/{name}.png', 0) > 127


def top_row(m):
    return int(np.argmax(m.any(1)))


def masked_blur(img, m, k):
    """Box-blur the photo using only subject pixels, so the wall never bleeds in."""
    k = max(1, int(round(k)))
    mf = m.astype(np.float32)
    num = cv2.blur(img.astype(np.float32) * mf[..., None], (k, k))
    den = cv2.blur(mf, (k, k))
    out = num / np.maximum(den, 1e-6)[..., None]
    return out, den > 0.05


def fit(ref, mov, mirror, rows, init):
    """Similarity (scale + shift, optional mirror) taking ref coords to mov coords,
    chosen to maximise silhouette overlap over the given ref rows."""
    q = 4
    r = cv2.resize(ref.astype(np.uint8), None, fx=1 / q, fy=1 / q, interpolation=cv2.INTER_AREA) > 0
    m = (mov.astype(np.uint8) * 255)
    r0, r1 = rows[0] // q, rows[1] // q
    sgn = -1.0 if mirror else 1.0

    def iou(p):
        a, bx, by = p
        # ref (small) -> mov (full): x' = sgn*a*q*x + bx
        M = np.float32([[sgn * a * q, 0, bx], [0, a * q, by]])
        w = cv2.warpAffine(m, M, (r.shape[1], r.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 127
        A, B = r[r0:r1], w[r0:r1]
        return (A & B).sum() / max(1, (A | B).sum())

    p = np.array(init, dtype=np.float64)
    best = iou(p)
    steps = np.array([0.04, 48.0, 48.0])
    while steps[1] > 0.75:
        improved = False
        for i in range(3):
            for d in (-1, 1):
                c = p.copy()
                c[i] += d * steps[i]
                v = iou(c)
                if v > best:
                    p, best, improved = c, v, True
        if not improved:
            steps /= 2
    return p, best


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    F, L, B, R = (mask(CFG[k]) for k in ('front', 'left', 'back', 'right'))
    Fp, Lp, Bp, Rp = (photo(CFG[k]) for k in ('front', 'left', 'back', 'right'))

    top_f = top_row(F)
    head_h = CFG['f_chin'] - top_f
    sL = (CFG['l_chin'] - top_row(L)) / head_h       # left px per front px

    # Left profile resampled to front units and flipped so the nose points to +z.
    def to_front_units(img, interp):
        return cv2.resize(img, None, fx=1 / sL, fy=1 / sL, interpolation=interp)[:, ::-1]
    Lz = to_front_units(L.astype(np.uint8) * 255, cv2.INTER_AREA) > 127
    Lzp = np.ascontiguousarray(to_front_units(Lp, cv2.INTER_AREA))
    row_off = top_row(Lz) - top_f                    # Lz row = front row + row_off

    # Profile with thin features (nose, lips, chin, hair wisps) opened away.
    rad = int(CFG['open_r'] * head_h)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    Lbase = cv2.morphologyEx(Lz.astype(np.uint8), cv2.MORPH_OPEN, ker) > 0
    # Front outline with stray hair flicks opened away, so slices do not step in and out.
    ker_f = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    Fg = cv2.morphologyEx(F.astype(np.uint8), cv2.MORPH_OPEN, ker_f) > 0

    # Voxel grid, in front-photo pixels: X right, Y down, Z toward the nose.
    ys = np.arange(top_f + S / 2, CFG['cut'], S)
    fx = np.flatnonzero(F[top_f:CFG['cut']].any(0))
    lz = np.flatnonzero(Lz[top_f + row_off:CFG['cut'] + row_off].any(0))
    xs = np.arange(fx.min() - S, fx.max() + S, S) + S / 2
    zs = np.arange(lz.min() - S, lz.max() + S, S) + S / 2
    occ = np.zeros((len(ys), len(xs), len(zs)), bool)
    zmid = np.zeros(len(ys))
    sig_y, sig_v = zip(*SIGMA_ROWS)
    nf, nb = CFG['n_front'], CFG['n_back']

    for j, Y in enumerate(ys):
        frow = Fg[int(Y)]
        x0 = x1 = CFG['f_nose_x']
        if not frow[x0]:
            continue
        while x0 > 0 and frow[x0 - 1]:
            x0 -= 1
        while x1 < len(frow) - 1 and frow[x1 + 1]:
            x1 += 1
        prof = np.flatnonzero(Lz[int(Y) + row_off])
        if len(prof) == 0:
            continue
        base = np.flatnonzero(Lbase[int(Y) + row_off])
        zf = prof.max()
        zfb, zbb = (base.max(), base.min()) if len(base) else (zf, prof.min())
        cx, w = (x0 + x1) / 2, max(1.0, (x1 - x0) / 2)
        zc = zmid[j] = zbb + CFG['widest'] * (zfb - zbb)
        u = np.clip(np.abs(xs - cx) / w, 0, 1)[:, None]
        g = np.exp(-((xs - CFG['f_nose_x']) / np.interp(Y, sig_y, sig_v)) ** 2)[:, None]
        front = zc + (zfb - zc) * (1 - u ** nf) ** (1 / nf) + (zf - zfb) * g
        back = zc - (zc - zbb) * (1 - u ** nb) ** (1 / nb)
        occ[j] = (u < 1) & (zs[None, :] <= front) & (zs[None, :] >= back)

    # Surface shell: filled cells with an empty neighbour.
    pad = np.pad(occ, 1)
    core = occ.copy()
    for ax in range(3):
        for d in (-1, 1):
            core &= np.roll(pad, d, ax)[1:-1, 1:-1, 1:-1]
    shell = occ & ~core

    # Normals from the smoothed occupancy gradient.
    sm = pad.astype(np.float32)
    for _ in range(2):
        for ax in range(3):
            sm = (np.roll(sm, 1, ax) + sm + np.roll(sm, -1, ax)) / 3
    gy, gx, gz = (-(np.roll(sm, -1, ax) - np.roll(sm, 1, ax))[1:-1, 1:-1, 1:-1] for ax in range(3))
    jj, ii, kk = np.nonzero(shell)
    X, Y, Z = xs[ii], ys[jj], zs[kk]
    N = np.stack([gx[jj, ii, kk], gy[jj, ii, kk], gz[jj, ii, kk]], 1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-6)

    # Register the other two photos against the views that carry the geometry.
    r_top = top_row(R)
    aR = (CFG['r_chin'] - r_top) / head_h
    nose_rows = slice(CFG['f_eye'] + row_off, CFG['f_chin'] + row_off)
    zn = max(np.flatnonzero(r).max() for r in Lz[nose_rows] if r.any())
    cn = max(np.flatnonzero(r).max() for r in R[r_top:CFG['r_chin']] if r.any())
    (aR, bxR, byR), iouR = fit(Lz, R, False, (top_f + row_off, CFG['f_chin'] + row_off + 20),
                               (aR, cn - aR * zn, r_top - aR * (top_f + row_off)))
    b_top = top_row(B)
    bcols = np.flatnonzero(B[b_top:b_top + int(head_h * 0.6)].any(0))
    fcols = np.flatnonzero(F[top_f:top_f + int(head_h * 0.6)].any(0))
    aB = (bcols.max() - bcols.min()) / (fcols.max() - fcols.min())
    (aB, bxB, byB), iouB = fit(F, B, True, (top_f, CFG['f_eye'] + 50),
                               (aB, bcols.mean() + aB * fcols.mean(), b_top - aB * top_f))
    print(f'scale left/front {sL:.3f} | right fit a={aR:.3f} iou={iouR:.3f} | back fit a={aB:.3f} iou={iouB:.3f}')

    # Colour: blend the photos by facing direction.
    fwd = Z >= zmid[jj]                               # which half a brick sits in, for ones facing up or down
    views = [
        (masked_blur(Fp, F, S), X, Y, np.maximum(N[:, 2], 0) + 0.05 * fwd),
        (masked_blur(Bp, B, S * aB), -aB * X + bxB, aB * Y + byB, np.maximum(-N[:, 2], 0) + 0.05 * ~fwd),
        (masked_blur(Lzp, Lz, S), Z, Y + row_off, np.maximum(N[:, 0], 0)),
        (masked_blur(Rp, R, S * aR), aR * Z + bxR, aR * (Y + row_off) + byR, np.maximum(-N[:, 0], 0)),
    ]
    col = np.zeros((len(X), 3))
    wsum = np.zeros(len(X))
    for (img, ok), cxs, cys, wgt in views:
        c = np.clip(np.round(cxs).astype(int), 0, img.shape[1] - 1)
        r = np.clip(np.round(cys).astype(int), 0, img.shape[0] - 1)
        w = wgt ** 2 * ok[r, c]
        col += img[r, c] * w[:, None]
        wsum += w
    good = wsum > 0
    col[good] /= wsum[good, None]
    if (~good).any():                                # orphan bricks take their nearest neighbour's colour
        P = np.stack([X, Y, Z], 1)
        for i in np.flatnonzero(~good):
            col[i] = col[good][np.argmin(((P[good] - P[i]) ** 2).sum(1))]

    # Rough region tags for later styling: 0 skin, 1 hair, 2 shirt.
    lum = col @ [0.299, 0.587, 0.114]
    dark = lum < 100
    tag = np.where(dark & (Y > CFG['f_chin'] - 60) & (col[:, 2] > col[:, 0] * 0.92), 2, np.where(dark, 1, 0))

    # Export in brick units, centred, y up.
    cxh, cyh, czh = CFG['f_nose_x'], (top_f + CFG['f_chin']) / 2, (zs[0] + zs[-1]) / 2
    pos = np.stack([(X - cxh) / S, (cyh - Y) / S, (Z - czh) / S], 1)
    nrm = N * [1, -1, 1]
    out = dict(
        step=S, count=len(X),
        pos=np.round(pos, 2).ravel().tolist(),
        nrm=np.round(nrm, 2).ravel().tolist(),
        col=np.round(col).astype(int).ravel().tolist(),
        tag=tag.astype(int).tolist(),
    )
    with open('prototype/head/bricks.js', 'w') as f:
        f.write('// Generated by tools/head/carve.py. Do not edit.\nwindow.HEAD_BRICKS = ')
        json.dump(out, f, separators=(',', ':'))
        f.write(';\n')
    print(f'{len(X)} bricks, grid {occ.shape}, tags skin/hair/shirt = {np.bincount(tag, minlength=3).tolist()}')
    preview(pos, nrm, col, f'{SRC}/preview.jpg')


def preview(pos, nrm, col, path, yaws=(0, 35, 90, 180, -90, -35), px=7):
    """Offline check render: orthographic, painter's order, simple shading."""
    span = np.abs(pos).max() * 2 + 4
    side = int(span * px)
    light = np.array([0.3, 0.5, 0.8])
    light /= np.linalg.norm(light)
    panels = []
    for yaw in yaws:
        t = np.radians(yaw)
        Rm = np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])
        p, n = pos @ Rm.T, nrm @ Rm.T
        img = np.full((side, side, 3), (245, 245, 239), np.uint8)
        shade = 0.6 + 0.4 * np.clip(n @ light, 0, 1)
        h = int(np.ceil(px * (0.5 + 0.25 * abs(np.sin(2 * t)))))   # fatter squares at oblique angles so no gaps show
        for i in np.argsort(p[:, 2]):
            if n[i, 2] < -0.15:
                continue
            x = int(side / 2 + p[i, 0] * px)
            y = int(side / 2 - p[i, 1] * px)
            c = np.clip(col[i] * shade[i], 0, 255)
            cv2.rectangle(img, (x - h, y - h), (x + h, y + h), c.tolist(), -1)
        panels.append(img)
    rows = [np.hstack(panels[:3]), np.hstack(panels[3:])]
    Image.fromarray(np.vstack(rows)).save(path, quality=88)


if __name__ == '__main__':
    main()
