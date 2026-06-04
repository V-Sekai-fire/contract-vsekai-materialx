"""Differentiable toon material: fit MToon ramp params + a Slug emblem mask to a
target raster by gradient descent. Toon base + a visual (vector) map, end-to-end.

This is the CPU reference of the inverse-rendering loop; the SlangPy path runs the
SAME MaterialX-generated Slang on the GPU with hardware autodiff. The forward model
here is exactly the stock-node math (mtoon_ramp + slug_path + mix) the library emits.
"""
import numpy as np
from PIL import Image

N = 256
ys, xs = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")
mask_disk = xs**2 + ys**2 <= 1.0
zz = np.sqrt(np.clip(1 - xs**2 - ys**2, 0, 1))
Nrm = np.stack([xs, ys, zz], -1)                      # sphere normals
L = np.array([0.4, 0.5, 0.75]); L /= np.linalg.norm(L)
ndotl = np.clip((Nrm * L).sum(-1), -1, 1)             # dot(N,L)

# slug emblem: degenerate-MSDF disk atlas (med3(x,x,x)=x), bilinear-sampled
A = 128
ax, ay = np.meshgrid(np.linspace(0, 1, A), np.linspace(0, 1, A))
emblem = np.clip(0.5 - (np.sqrt((ax-.5)**2 + (ay-.5)**2) - .28)/.5, 0, 1)
def slug(smoothing):
    u = np.clip(xs*0.7+0.5, 0, 1); v = np.clip(ys*0.7+0.5, 0, 1)  # emblem placement
    s = emblem[(v*(A-1)).astype(int), (u*(A-1)).astype(int)]
    t = np.clip((s - (0.5-smoothing))/(2*smoothing), 0, 1)
    return t*t*(3-2*t)

def forward(p):
    shade = p[0:3]; lit = p[3:6]; shift = p[6]; toony = p[7]
    emcol = p[8:11]; smooth = p[11]
    lo, hi = -1+toony, 1-toony
    ramp = np.clip((ndotl + shift - lo)/(hi-lo), 0, 1)          # mtoon linearstep
    base = shade[None,None,:] + (lit-shade)[None,None,:]*ramp[...,None]   # mix
    cov = slug(smooth)[...,None]
    img = base*(1-cov) + emcol[None,None,:]*cov                 # composite emblem
    return img * mask_disk[...,None]

# target (the look we want to recover) and a wrong initial guess
p_true = np.array([.30,.18,.32, .95,.55,.62, 0.05, 0.88, 0.05,0.45,0.85, 0.05])
p_init = np.array([.50,.50,.50, .70,.70,.70, 0.00, 0.90, 0.50,0.50,0.50, 0.12])
target = forward(p_true)

def loss(p):
    d = forward(p) - target
    return float((d*d).mean())

def grad(p, eps=1e-3):                                          # finite-difference grad
    g = np.zeros_like(p); b = loss(p)
    for i in range(len(p)):
        q = p.copy(); q[i]+=eps; g[i] = (loss(q)-b)/eps
    return g

p = p_init.copy()
print(f"init  loss={loss(p):.5f}")
lr = 0.5
for it in range(400):
    p = np.clip(p - lr*grad(p), 0, 1)
    if it % 80 == 0: print(f"it{it:3d} loss={loss(p):.6f}")
print(f"final loss={loss(p):.6f}")
print("recovered vs true (shade,lit,shift,toony,emblem,smooth):")
print("  fit :", np.round(p,2)); print("  true:", np.round(p_true,2))

def save(img, name):
    Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save("generated/"+name)
save(target, "toon_target.png"); save(forward(p_init), "toon_init.png"); save(forward(p), "toon_fit.png")
print("wrote generated/toon_{target,init,fit}.png")
