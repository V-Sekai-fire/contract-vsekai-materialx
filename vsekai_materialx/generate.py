"""Generate shader source (Slang / GLSL / MSL / OSL / MDL) from the stock nodes.

Because the v-sekai nodes are pure stock-node functional nodegraphs, MaterialX's
own ShaderGen emits every target for them - no custom per-target source.
"""
import MaterialX as mx
from MaterialX import PyMaterialXGenShader as gs
from .library import load_document, source_search_path

_GENERATORS = {}


def make_generator(target: str = "slang"):
    """A cached ShaderGenerator for a target: slang|glsl|msl|osl|mdl|essl."""
    if target in _GENERATORS:
        return _GENERATORS[target]
    if target == "slang":
        from MaterialX import PyMaterialXGenSlang as g; gen = g.SlangShaderGenerator.create()
    elif target == "glsl":
        from MaterialX import PyMaterialXGenGlsl as g; gen = g.GlslShaderGenerator.create()
    elif target == "essl":
        from MaterialX import PyMaterialXGenGlsl as g; gen = g.EsslShaderGenerator.create()
    elif target == "msl":
        from MaterialX import PyMaterialXGenMsl as g; gen = g.MslShaderGenerator.create()
    elif target == "osl":
        from MaterialX import PyMaterialXGenOsl as g; gen = g.OslShaderGenerator.create()
    elif target == "mdl":
        from MaterialX import PyMaterialXGenMdl as g; gen = g.MdlShaderGenerator.create()
    else:
        raise ValueError(f"unknown target {target!r}")
    _GENERATORS[target] = gen
    return gen


def _context(gen):
    ctx = gs.GenContext(gen)
    ctx.registerSourceCodeSearchPath(source_search_path())
    return ctx


def generate(node_category: str, out_type: str = "float", target: str = "slang",
             doc=None, sp=None) -> str:
    """Generate pixel-stage source for a single v-sekai node, by category."""
    if doc is None:
        doc, sp = load_document()
    d = doc.copy()
    ng = d.addNodeGraph("g_ng")
    n = ng.addNode(node_category, "inst", out_type)
    o = ng.addOutput("out", out_type)
    o.setConnectedNode(n)
    gen = make_generator(target)
    shader = gen.generate("g_" + node_category, o, _context(gen))
    return shader.getSourceCode(gs.PIXEL_STAGE)


def generate_material(material_node, target: str = "slang"):
    """Generate a full surface Shader (pixel+vertex) for a material node."""
    gen = make_generator(target)
    return gen.generate(material_node.getName(), material_node, _context(gen))
