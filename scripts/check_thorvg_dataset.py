"""Render ThorVG's SVG test corpus through the Slug pipeline and report correctness.
Usage: python scripts/check_thorvg_dataset.py <dir-with-svgs>
"""
import os, sys, glob
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vsekai_materialx.svg_render import render

src = sys.argv[1] if len(sys.argv) > 1 else "E:/tmp/thorvg/test/resources"
files = sorted(glob.glob(os.path.join(src, "*.svg")))
os.makedirs("generated", exist_ok=True)
tiles = []
print(f"checking {len(files)} ThorVG SVGs from {src}\n")
for f in files:
    name = os.path.basename(f)
    try:
        img, n, flags = render(f, res=256)
    except Exception as e:
        print(f"  {name:14} ERROR {type(e).__name__}: {e}"); continue
    if img is None:
        print(f"  {name:14} no fillable geometry (flags={flags})"); continue
    out = f"generated/thorvg_{os.path.splitext(name)[0]}.png"
    Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(out)
    unsup = ", ".join(f"{k}={v}" for k,v in flags.items() if v)
    print(f"  {name:14} {n:3d} filled shapes -> {out}" + (f"   [partial: {unsup}]" if unsup else "   [full]"))
    tiles.append(out)

if tiles:                                   # contact sheet
    ims = [Image.open(t).resize((192,192)) for t in tiles]
    sheet = Image.new("RGB", (192*len(ims), 192), (30,30,30))
    for i,im in enumerate(ims): sheet.paste(im, (192*i,0))
    sheet.save("generated/thorvg_contact_sheet.png")
    print("\nwrote generated/thorvg_contact_sheet.png")
print("\nNote: full = all geometry is fillable paths/shapes; partial = doc also uses\n"
      "gradients/strokes/arcs/transforms our Slug-fill path doesn't reproduce.")
