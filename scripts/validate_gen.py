"""Load stdlib + the v-sekai stock-node library and prove every node generates
a Slang shader (and GLSL) through MaterialX's own ShaderGen. No custom source."""
import os, sys
import MaterialX as mx
from MaterialX import PyMaterialXGenShader as gs
from MaterialX import PyMaterialXGenGlsl as gglsl
from MaterialX import PyMaterialXGenSlang as gslang

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR_LIB = os.path.join(HERE, "libraries")
BUNDLED = os.path.join(os.path.dirname(mx.__file__), "libraries")

def load():
    doc = mx.createDocument()
    sp = mx.FileSearchPath()
    sp.append(BUNDLED)
    sp.append(OUR_LIB)
    folders = ["targets", "stdlib", "pbrlib", "bxdf", "lights", "nprlib", "cmlib", "vsekai"]
    mx.loadLibraries(folders, sp, doc)
    return doc, sp

def gen_for(node_category, out_type, gen, label, doc):
    # build a tiny doc instantiating the node, wire an output, generate
    d = doc.copy()
    ng = d.addNodeGraph("test_ng")
    n = ng.addNode(node_category, "inst", out_type)
    o = ng.addOutput("out", out_type)
    o.setConnectedNode(n)
    ctx = gs.GenContext(gen)
    # #include paths are written as "libraries/...", so search from the PARENT
    # of each libraries/ dir (the MaterialX install root and our repo root).
    src_sp = mx.FileSearchPath()
    src_sp.append(os.path.dirname(BUNDLED))
    src_sp.append(HERE)
    ctx.registerSourceCodeSearchPath(src_sp)
    shader = gen.generate("g_" + node_category, o, ctx)
    src = shader.getSourceCode(gs.PIXEL_STAGE)
    print(f"  [{label:5}] {node_category:16} -> {len(src):6} chars  ({out_type})")
    return src

def main():
    doc, _ = load()
    # sanity: our nodedefs resolved
    cats = ["mtoon_ramp", "scss_crosstone", "slug_path", "splat4"]
    types = {"mtoon_ramp": "color3", "scss_crosstone": "color3",
             "slug_path": "float", "splat4": "float"}
    print("nodedefs present:")
    for c in cats:
        defs = [nd for nd in doc.getNodeDefs() if nd.getNodeString() == c]
        print(f"  {c:16} {'OK' if defs else 'MISSING'}  ({defs[0].getName() if defs else '-'})")
    assert all(any(nd.getNodeString()==c for nd in doc.getNodeDefs()) for c in cats), "missing nodedef"

    slang = gslang.SlangShaderGenerator.create()
    glsl  = gglsl.GlslShaderGenerator.create()
    outdir = os.path.join(HERE, "generated")
    os.makedirs(outdir, exist_ok=True)
    for c in cats:
        print(f"\n{c}:")
        s = gen_for(c, types[c], slang, "slang", doc)
        g = gen_for(c, types[c], glsl,  "glsl",  doc)
        open(os.path.join(outdir, c + ".slang"), "w").write(s)
        open(os.path.join(outdir, c + ".glsl"),  "w").write(g)
    print("\nALL NODES GENERATED Slang + GLSL. wrote ->", outdir)

if __name__ == "__main__":
    main()
