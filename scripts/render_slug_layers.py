"""Render using ONLY slug (coverage) + splat (instancing) - no PBR, no lighting.

Evaluates the exact math MaterialX generated into splat4.slang:
  slug_path  : median(r,g,b) -> smoothstep(0.5 +- s)        (Shader.Vector.decodeMSDF)
  splat4     : max_i slug_path((uv - offset_i)/scale_i)      (Shader.Splat.splatEval)
over a UV grid, then mix(bg, fg, coverage) for color. Proof that the two nodes
alone produce a complete vector render. (GPU + differentiable variant uses SlangPy.)
"""
import numpy as np
from PIL import Image

W = H = 512
S = 0.04  # slug smoothing

# --- a degenerate-MSDF atlas for a unit disk: all 3 channels equal, so
#     median(r,g,b) == that channel (med3_eq_of_all_eq). value>0.5 inside. ---
A = 256
ax, ay = np.meshgrid(np.linspace(0, 1, A), np.linspace(0, 1, A))
sd = np.sqrt((ax - 0.5) ** 2 + (ay - 0.5) ** 2) - 0.35   # signed dist to disk
atlas = np.clip(0.5 - sd / 0.5, 0.0, 1.0)                # encode like msdfAt

def sample(uv):  # nearest-clamp sample of the (single-channel) atlas
    u = np.clip(uv[..., 0], 0, 1); v = np.clip(uv[..., 1], 0, 1)
    return atlas[(v * (A - 1)).astype(int), (u * (A - 1)).astype(int)]

def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)

def slug(uv):                       # median==channel here, so decode == smoothstep
    return smoothstep(0.5 - S, 0.5 + S, sample(uv))

# splat: 4 instances (offset, scale) -- the nodedef defaults
INSTANCES = [((0.05, 0.05), (0.45, 0.45)),
             ((0.50, 0.05), (0.45, 0.45)),
             ((0.05, 0.50), (0.45, 0.45)),
             ((0.50, 0.50), (0.45, 0.45))]

ux, uy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
uv = np.stack([ux, uy], -1)

cov = np.zeros((H, W))
for (ox, oy), (sx, sy) in INSTANCES:                    # max-fold union
    local = (uv - np.array([ox, oy])) / np.array([sx, sy])
    cov = np.maximum(cov, slug(local))

bg = np.array([0.09, 0.10, 0.13]); fg = np.array([0.98, 0.62, 0.86])
img = bg + (fg - bg) * cov[..., None]                   # mix(bg, fg, coverage)

out = "generated/render_slug_layers.png"
Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out)
print(f"rendered {W}x{H} by app-level layering of the slug node (PSO symbol-art style) -> {out}  (coverage range {cov.min():.2f}..{cov.max():.2f})")
