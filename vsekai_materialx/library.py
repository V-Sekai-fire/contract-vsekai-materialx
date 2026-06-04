"""Load MaterialX's stdlib plus the v-sekai stock-node library into a document.

This is the "extend Python MaterialX" entry point: after load_document() the
returned doc knows mtoon_ramp / scss_crosstone / slug_path / splat4 as first-class
node definitions, resolvable by every MaterialX ShaderGen backend (incl. Slang).
"""
import os
import MaterialX as mx

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_DIR = os.path.join(HERE, "libraries")                       # our vsekai/ library
BUNDLED_LIB = os.path.join(os.path.dirname(mx.__file__), "libraries")  # stdlib in the wheel

_FOLDERS = ["targets", "stdlib", "pbrlib", "bxdf", "lights", "nprlib", "cmlib", "vsekai"]


def search_path() -> "mx.FileSearchPath":
    sp = mx.FileSearchPath()
    sp.append(BUNDLED_LIB)
    sp.append(LIB_DIR)
    return sp


def source_search_path() -> "mx.FileSearchPath":
    """For ShaderGen #include resolution: parents of each libraries/ dir."""
    sp = mx.FileSearchPath()
    sp.append(os.path.dirname(BUNDLED_LIB))
    sp.append(HERE)
    return sp


def load_document(*material_files: str):
    """Return (doc, search_path) with stdlib + vsekai loaded, plus any extra
    material .mtlx files read on top."""
    doc = mx.createDocument()
    sp = search_path()
    mx.loadLibraries(_FOLDERS, sp, doc)
    for f in material_files:
        mx.readFromXmlFile(doc, f)
    return doc, sp
