"""Generate Slang surface shaders for the stock PBR materials (materials/pbr.mtlx),
including M_showcase whose gltf_pbr base_color is driven by the NPR mtoon_ramp node.
Proves PBR + NPR + vector all compose into one stock graph that emits Slang.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import vsekai_materialx as v
from vsekai_materialx.generate import make_generator
from MaterialX import PyMaterialXGenShader as gs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
doc, sp = v.load_document(os.path.join(HERE, "materials", "pbr.mtlx"))
print("doc valid:", doc.validate()[0])
outdir = os.path.join(HERE, "generated"); os.makedirs(outdir, exist_ok=True)
for m in doc.getMaterialNodes():
    sh = v.generate_material(m, "slang")
    px = sh.getSourceCode(gs.PIXEL_STAGE)
    open(os.path.join(outdir, m.getName() + ".slang"), "w").write(px)
    print(f"  {m.getName():12} Slang surface shader: {len(px)} chars")
print("wrote PBR surface shaders ->", outdir)
