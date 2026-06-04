"""v-sekai-materialx: PBR + NPR + Slug vector maps as stock MaterialX nodes.

Public API:
    load_document()            -> (doc, search_path)   stdlib + vsekai library loaded
    generate(node, out_type, target='slang') -> str    Slang/GLSL/MSL/... source
    generate_material(mat, target='slang')   -> Shader  full surface shader
"""
from .library import load_document, source_search_path, LIB_DIR, BUNDLED_LIB
from .generate import generate, generate_material, make_generator

__all__ = [
    "load_document", "source_search_path", "LIB_DIR", "BUNDLED_LIB",
    "generate", "generate_material", "make_generator",
]
