"""EXACT raster->vector with Slug: trace a flat region's boundary, then render it
with Slug's exact signed-distance coverage (Shader.Vector.slugCoverage) - no blob
fitting, no MSDF atlas, crisp at any resolution.

For vector-like (flat) content this is exact up to the tracing tolerance; a
photograph has no finite-vector form, so that needs the approximate splat fit.
"""
import numpy as np
from PIL import Image

# --- a flat vector shape (5-point star) defined by its outline (the ground truth) ---
def star_outline(cx=0.5, cy=0.5, R=0.38, r=0.16, n=5):
    pts = []
    for k in range(2 * n):
        ang = np.pi / 2 + k * np.pi / n
        rad = R if k % 2 == 0 else r
        pts.append((cx + rad * np.cos(ang), cy - rad * np.sin(ang)))
    return np.array(pts)                      # closed polygon (exact vector outline)

outline = star_outline()

def point_in_poly(px, py, poly):
    n = len(poly); inside = np.zeros_like(px, bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond; j = i
    return inside

def dist_to_poly(px, py, poly):              # unsigned distance to the outline (segments)
    n = len(poly); d = np.full(px.shape, 1e9)
    for i in range(n):
        a = poly[i]; b = poly[(i + 1) % n]
        ab = b - a; t = ((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / (ab @ ab + 1e-12)
        t = np.clip(t, 0, 1)
        qx = a[0] + t * ab[0]; qy = a[1] + t * ab[1]
        d = np.minimum(d, np.sqrt((px - qx) ** 2 + (py - qy) ** 2))
    return d

def slug_render(poly, res, smoothing):       # EXACT: signed distance -> coverage
    gy, gx = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res), indexing="ij")
    inside = point_in_poly(gx, gy, poly)
    sd = dist_to_poly(gx, gy, poly) * np.where(inside, -1.0, 1.0)   # signed
    cov = np.clip(0.5 - sd / (2 * smoothing), 0, 1)                 # slugCoverage
    fg = np.array([0.98, 0.80, 0.22]); bg = np.array([0.10, 0.12, 0.18])
    return bg + (fg - bg) * cov[..., None]

# "raster" target the artist gave us, at a low resolution
target = slug_render(outline, 64, smoothing=1.0/64)

# --- trace the raster boundary back to a polygon (marching-squares-ish) ---
mask = (np.asarray(Image.fromarray((target*255).astype(np.uint8)).convert("L")) > 110)
# recover boundary points, order them by angle about the centroid (convex-ish star)
ys, xs = np.where(mask)
cyc = np.array([xs.mean(), ys.mean()]) / 64.0
edge = []
for yy in range(1, 63):
    for xx in range(1, 63):
        if mask[yy, xx] and not (mask[yy-1,xx] and mask[yy+1,xx] and mask[yy,xx-1] and mask[yy,xx+1]):
            edge.append((xx/64.0, yy/64.0))
edge = np.array(edge)
ang = np.arctan2(edge[:,1]-cyc[1], edge[:,0]-cyc[0])
traced = edge[np.argsort(ang)]
traced = traced[::max(1, len(traced)//40)]     # decimate to ~40 vertices

# --- render the TRACED outline with exact slug coverage at 8x the fit res ---
hi = slug_render(traced, 512, smoothing=1.0/512)
ref = slug_render(outline, 512, smoothing=1.0/512)     # ground-truth vector at 512
err = float(np.abs(hi - ref).mean())

def save(img, n): Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save("generated/"+n)
save(target, "slugvec_target_64.png")
save(hi, "slugvec_traced_512.png")
save(ref, "slugvec_groundtruth_512.png")
print(f"traced {len(traced)} boundary vertices from a 64x64 raster")
print(f"exact slug re-render at 512x512: mean abs error vs ground-truth vector = {err:.4f}")
print("wrote generated/slugvec_{target_64,traced_512,groundtruth_512}.png")
