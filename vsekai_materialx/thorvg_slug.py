"""Connect ThorVG vector art to Slug.

ThorVG (https://github.com/thorvg/thorvg) is a vector engine that loads SVG / Lottie
/ TVG into Shape paths built from moveTo / lineTo / cubicTo / close commands. That is
exactly the input Slug wants: Bezier outlines. This module reads that path geometry
(via SVG, ThorVG's canonical interchange - and identical to ThorVG's Shape command
list if you call it through its bindings), flattens the curves to an outline, and
renders it with Slug's EXACT signed-distance coverage (Shader.Vector.slugCoverage).

Deterministic, no fitting, resolution-independent. For MaterialX deployment the same
outline is either baked to an MSDF the stock `slug_path` node samples (portable), or
fed to the analytic Slug custom node (exact).
"""
import re
import numpy as np

# ---- SVG path -> polyline contours (M/L/H/V/C/Q/Z, abs+relative) ----
def _flatten_cubic(p0, p1, p2, p3, n=24):
    t = np.linspace(0, 1, n)[1:, None]
    return ((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3)

def _flatten_quad(p0, p1, p2, n=24):
    t = np.linspace(0, 1, n)[1:, None]
    return ((1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2)

def parse_svg_path(d):
    """Return a list of contours, each an (N,2) array of points in path units."""
    toks = re.findall(r"[MmLlHhVvCcQqZz]|-?\d*\.?\d+(?:e-?\d+)?", d)
    contours, cur, pt, start, i, cmd = [], [], np.zeros(2), np.zeros(2), 0, None
    def num():
        nonlocal i; v = float(toks[i]); i += 1; return v
    while i < len(toks):
        t = toks[i]
        if re.match(r"[A-Za-z]", t): cmd = t; i += 1
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            if cur: contours.append(np.array(cur)); cur = []
            pt = (pt + [num(), num()]) if rel else np.array([num(), num()])
            start = pt.copy(); cur = [pt.copy()]
        elif c == "L":
            pt = (pt + [num(), num()]) if rel else np.array([num(), num()]); cur.append(pt.copy())
        elif c == "H":
            x = num(); pt = pt + [x, 0] if rel else np.array([x, pt[1]]); cur.append(pt.copy())
        elif c == "V":
            y = num(); pt = pt + [0, y] if rel else np.array([pt[0], y]); cur.append(pt.copy())
        elif c == "C":
            p1 = pt + [num(), num()] if rel else np.array([num(), num()])
            p2 = pt + [num(), num()] if rel else np.array([num(), num()])
            p3 = pt + [num(), num()] if rel else np.array([num(), num()])
            for q in _flatten_cubic(pt, p1, p2, p3): cur.append(q)
            pt = p3
        elif c == "Q":
            p1 = pt + [num(), num()] if rel else np.array([num(), num()])
            p2 = pt + [num(), num()] if rel else np.array([num(), num()])
            for q in _flatten_quad(pt, p1, p2): cur.append(q)
            pt = p2
        elif c == "Z":
            cur.append(start.copy()); contours.append(np.array(cur)); cur = []; pt = start.copy()
        else:
            i += 1
    if cur: contours.append(np.array(cur))
    return contours

def _normalize(contours, pad=0.08):
    allp = np.concatenate(contours)
    lo, hi = allp.min(0), allp.max(0); span = (hi - lo).max()
    return [(c - lo) / span * (1 - 2*pad) + pad for c in contours]

# ---- Slug exact coverage of the outline ----
def _inside_and_dist(gx, gy, contours):
    inside = np.zeros(gx.shape, bool); dist = np.full(gx.shape, 1e9)
    for poly in contours:
        n = len(poly); j = n - 1
        for i in range(n):
            xi, yi = poly[i]; xj, yj = poly[j]
            cond = ((yi > gy) != (yj > gy)) & (gx < (xj-xi)*(gy-yi)/(yj-yi+1e-12)+xi)
            inside ^= cond                                   # even-odd fill (holes work)
            ax, ay = poly[i]; bx, by = poly[(i+1) % n]
            abx, aby = bx-ax, by-ay
            tt = np.clip(((gx-ax)*abx + (gy-ay)*aby)/(abx*abx+aby*aby+1e-12), 0, 1)
            qx, qy = ax+tt*abx, ay+tt*aby
            dist = np.minimum(dist, np.sqrt((gx-qx)**2 + (gy-qy)**2))
            j = i
    return inside, dist

def slug_coverage(contours, res, smoothing=None):
    smoothing = smoothing or 1.0/res
    gy, gx = np.meshgrid(np.linspace(0,1,res), np.linspace(0,1,res), indexing="ij")
    inside, dist = _inside_and_dist(gx, gy, contours)
    sd = dist * np.where(inside, -1.0, 1.0)                  # signed distance
    return np.clip(0.5 - sd/(2*smoothing), 0, 1)            # Shader.Vector.slugCoverage

def load_svg(path):
    import os
    d = re.search(r'd="([^"]+)"', open(path).read()).group(1)
    return _normalize(parse_svg_path(d))
