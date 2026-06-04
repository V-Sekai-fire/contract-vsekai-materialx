"""ThorVG vector art -> Slug -> render. Loads an SVG (ThorVG's canonical vector
format), feeds its Bezier outline to Slug's exact coverage, and renders crisp at any
resolution. Deterministic - no fitting. Run: python scripts/thorvg_to_slug.py [file.svg]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys
import numpy as np
from PIL import Image
from vsekai_materialx.thorvg_slug import load_svg, slug_coverage

svg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "heart.svg")
contours = load_svg(svg)
print(f"ThorVG/SVG '{os.path.basename(svg)}': {len(contours)} contour(s), "
      f"{sum(len(c) for c in contours)} flattened outline points")

fg = np.array([0.94, 0.20, 0.34]); bg = np.array([0.10, 0.11, 0.15])
os.makedirs("generated", exist_ok=True)
for res in (64, 512):
    cov = slug_coverage(contours, res)[..., None]
    img = bg + (fg - bg) * cov
    Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(f"generated/thorvg_slug_{res}.png")
    print(f"  exact Slug render {res}x{res}: coverage {cov.min():.2f}..{cov.max():.2f} "
          f"-> generated/thorvg_slug_{res}.png")
print("ThorVG -> Slug -> pipeline connected (same outline renders crisp at every res).")
