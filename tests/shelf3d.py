"""Tiny ray-caster for a shelf unit, so depth counting can be tested against true distances.

Renders a colour picture and the exact per-pixel distance (metres along the camera's view axis)
of a shelf with products stacked several units deep, seen from any camera position and angle.
Taking units from the front of a column leaves the rest where they were — further back.
"""
import math

import numpy as np

W, H, F = 640, 480, 520.0        # image size, focal length (px)
ZF, SHELF_D = 1.3, 0.40          # shelf front edge distance from the camera origin, shelf depth (m)

# products: id, board top y (y points down), width, depth, height (m), facings, deep, colour (BGR), x start
PRODUCTS = [
    ("cola", 0.00, 0.080, 0.080, 0.24, 4, 4, (40, 40, 200), -0.50),
    ("soap", 0.00, 0.100, 0.060, 0.12, 3, 5, (200, 160, 60), -0.12),
    ("chips", 0.00, 0.110, 0.090, 0.20, 2, 4, (40, 190, 230), 0.22),
    ("tea", 0.38, 0.090, 0.070, 0.16, 5, 4, (60, 150, 60), -0.50),
    ("jam", 0.38, 0.070, 0.075, 0.10, 4, 5, (90, 60, 170), 0.02),
]
BOARDS = [0.00, 0.38, -0.40]     # board tops; the last one is the shelf above


def camera(yaw_deg=0.0, pitch_deg=0.0, pos=(0.0, 0.0, 0.0)):
    """Rotation (camera → world) and position. Yaw turns right, pitch tilts down (y is down)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return ry @ rx, np.array(pos, float)


def boxes(removed):
    """Axis-aligned boxes (lo, hi, colour, kind). removed[(product, facing)] = units gone from the front."""
    out = [(np.array([-0.62, -0.62, ZF + SHELF_D]), np.array([0.62, 0.62, ZF + SHELF_D + 0.02]), (170, 175, 180), "wall")]
    for b in BOARDS:
        out.append((np.array([-0.62, b, ZF - 0.01]), np.array([0.62, b + 0.025, ZF + SHELF_D]), (70, 100, 140), "board"))
    for side in (-0.62, 0.60):
        out.append((np.array([side, -0.62, ZF - 0.01]), np.array([side + 0.02, 0.62, ZF + SHELF_D]), (60, 90, 130), "board"))
    for pid, top, w, d, h, n, deep, col, x0 in PRODUCTS:
        for i in range(n):
            gone = removed.get((pid, i), 0)
            for k in range(gone, deep):
                lo = np.array([x0 + i * (w + 0.006), top - h, ZF + 0.005 + k * d])
                hi = lo + np.array([w, h, d - 0.004])
                out.append((lo, hi, col, "product"))
    return out


def slot_boxes(yaw=0.0, pitch=0.0, pos=(0.0, 0.0, 0.0)):
    """Where each product block's front face lands in the picture (normalised), for marking slots."""
    R, C = camera(yaw, pitch, pos)
    slots = []
    for pid, top, w, d, h, n, deep, col, x0 in PRODUCTS:
        x1 = x0 + n * (w + 0.006) - 0.006
        pts = []
        for x in (x0, x1):
            for y in (top - h, top):
                pc = R.T @ (np.array([x, y, ZF + 0.005]) - C)
                pts.append((F * pc[0] / pc[2] + W / 2, F * pc[1] / pc[2] + H / 2))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        slots.append({"id": pid, "name": pid, "sku": "", "x": min(xs) / W, "y": min(ys) / H,
                      "w": (max(xs) - min(xs)) / W, "h": (max(ys) - min(ys)) / H,
                      "facings": n, "deep": deep, "unit_cm": d * 100})
    return slots


def render(removed=None, yaw=0.0, pitch=0.0, pos=(0.0, 0.0, 0.0)):
    """Returns (BGR image uint8, depth in metres along the view axis)."""
    removed = removed or {}
    R, C = camera(yaw, pitch, pos)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dc = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u)], -1)
    dw = dc @ R.T
    inv = 1.0 / np.where(np.abs(dw) < 1e-9, 1e-9, dw)
    tbest = np.full((H, W), np.inf)
    img = np.zeros((H, W, 3), np.float32)
    light = np.array([0.35, -0.6, -0.7])
    light /= np.linalg.norm(light)
    for lo, hi, col, kind in boxes(removed):
        # only trace the pixels the box can cover (its projected corners' bounding rectangle)
        cs = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pc = (cs - C) @ R
        if (pc[:, 2] > 0.05).all():
            us, vs = F * pc[:, 0] / pc[:, 2] + W / 2, F * pc[:, 1] / pc[:, 2] + H / 2
            u0, u1 = max(0, int(us.min()) - 1), min(W, int(us.max()) + 2)
            v0, v1 = max(0, int(vs.min()) - 1), min(H, int(vs.max()) + 2)
            if u0 >= u1 or v0 >= v1:
                continue
        else:
            u0, u1, v0, v1 = 0, W, 0, H
        win = (slice(v0, v1), slice(u0, u1))
        t1, t2 = (lo - C) * inv[win], (hi - C) * inv[win]
        tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
        tnear, tfar = tn.max(-1), tf.min(-1)
        hit = np.zeros((H, W), bool)
        hit[win] = (tnear <= tfar) & (tfar > 0) & (tnear > 0) & (tnear < tbest[win])
        if not hit.any():
            continue
        tn_full = np.zeros((H, W, 3))
        tn_full[win] = tn
        axis = tn_full.argmax(-1)[hit]
        tnear_full = np.full((H, W), np.inf)
        tnear_full[win] = tnear
        t = tnear_full[hit]
        p = C + dw[hit] * t[:, None]
        nrm = np.zeros((axis.size, 3))
        nrm[np.arange(axis.size), axis] = -np.sign(dw[hit][np.arange(axis.size), axis])
        shade = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
        shade *= 1.0 - 0.45 * np.clip((p[:, 2] - ZF) / SHELF_D, 0, 1)   # deeper in the shelf = darker
        c = np.array(col, np.float32)[None].repeat(axis.size, 0)
        if kind == "product":        # a printed label band and a logo on the front of each unit
            front = axis == 2
            lx = (p[:, 0] - lo[0]) / (hi[0] - lo[0])
            ly = (p[:, 1] - lo[1]) / (hi[1] - lo[1])
            band = front & (ly > 0.35) & (ly < 0.62)
            c[band] = 245
            logo = band & ((lx - 0.5) ** 2 + ((ly - 0.485) * 1.6) ** 2 < 0.035)
            c[logo] = np.array(col, np.float32) * 0.4
            c[front & (ly < 0.12)] *= 0.6
        elif kind == "wall":
            c = c * (0.92 + 0.08 * np.sin(p[:, 0] * 90)[:, None])
        img[hit] = c * shade[:, None]
        tbest[hit] = t
    tbest[~np.isfinite(tbest)] = 5.0                       # nothing hit: far away
    depth = (tbest[..., None] * dw) @ (R[:, 2])            # distance along the camera's view axis
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    return img, depth.astype(np.float32)
