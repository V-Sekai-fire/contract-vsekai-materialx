"""Reuse ThorVG's decomposition (don't reimplement SVG).

ThorVG parses SVG / Lottie / TVG and resolves the hard parts - arcs to cubics,
transforms baked, strokes tessellated, gradients - down to flat Shape primitives.
We bind its C API (libthorvg) via ctypes, walk the loaded Picture with an Accessor,
and read each Shape's resolved path (MoveTo / LineTo / CubicTo / Close) + fill color.
Those Bezier outlines go straight to Slug. No hand-rolled SVG parsing.

Requires libthorvg built with the capi binding:
    pip install meson && cd <thorvg> && \
    meson setup build -Dengines=sw -Dloaders=svg -Dbindings=capi -Ddefault_library=shared && \
    ninja -C build
Point THORVG_LIB at the resulting .dll/.so/.dylib (or pass lib_path=).
"""
import os, ctypes as C
import numpy as np
from .thorvg_slug import _flatten_cubic

# Tvg_Path_Command enum (thorvg_capi.h): Close=0, MoveTo=1, LineTo=2, CubicTo=3
CLOSE, MOVETO, LINETO, CUBICTO = 0, 1, 2, 3

class _Pt(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float)]

def _find_lib(lib_path=None):
    cands = [lib_path, os.environ.get("THORVG_LIB")]
    for base in (os.environ.get("THORVG_SRC", "E:/tmp/thorvg"),):
        for d in ("builddir", "build"):
            for n in ("libthorvg.dll", "thorvg.dll", "libthorvg.so", "libthorvg.dylib", "libthorvg-0.dll"):
                cands.append(os.path.join(base, d, "src", n))
                cands.append(os.path.join(base, d, n))
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("libthorvg not found; build it (see module docstring) or set THORVG_LIB")

def _bind(lib):
    lib.tvg_engine_init.argtypes = [C.c_uint]
    lib.tvg_picture_new.restype = C.c_void_p
    lib.tvg_picture_load.argtypes = [C.c_void_p, C.c_char_p]
    lib.tvg_accessor_new.restype = C.c_void_p
    lib.tvg_accessor_set.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
    lib.tvg_shape_get_path.argtypes = [C.c_void_p, C.POINTER(C.POINTER(C.c_uint8)),
        C.POINTER(C.c_uint32), C.POINTER(C.POINTER(_Pt)), C.POINTER(C.c_uint32)]
    lib.tvg_shape_get_fill_color.argtypes = [C.c_void_p] + [C.POINTER(C.c_uint8)]*4
    return lib

def _add_dll_dirs(libfile):
    if not hasattr(os, "add_dll_directory"):
        return
    dirs = [os.path.dirname(libfile)] + \
        [d for d in os.environ.get("THORVG_DLL_DIRS", "").split(os.pathsep) if d]
    for d in dirs:
        if d and os.path.isdir(d):
            try: os.add_dll_directory(d)
            except OSError: pass

def decompose(svg_path, lib_path=None):
    """Return [(contours, rgba)] using ThorVG's resolved Shape paths."""
    libfile = _find_lib(lib_path)
    _add_dll_dirs(libfile)
    lib = _bind(C.CDLL(libfile))
    lib.tvg_engine_init(0)
    pic = lib.tvg_picture_new()
    if lib.tvg_picture_load(pic, str(svg_path).encode()) != 0:
        raise RuntimeError(f"ThorVG failed to load {svg_path}")
    shapes = []

    CB = C.CFUNCTYPE(C.c_bool, C.c_void_p, C.c_void_p)
    def visit(paint, _data):
        cmds = C.POINTER(C.c_uint8)(); ncmd = C.c_uint32()
        pts = C.POINTER(_Pt)(); npt = C.c_uint32()
        if lib.tvg_shape_get_path(paint, C.byref(cmds), C.byref(ncmd),
                                  C.byref(pts), C.byref(npt)) != 0 or ncmd.value == 0:
            return True                                   # not a shape (scene/picture) - keep walking
        P = [(pts[i].x, pts[i].y) for i in range(npt.value)]
        contours, cur, k, npts = [], [], 0, npt.value
        for c in (cmds[i] for i in range(ncmd.value)):
            if c == MOVETO and k < npts:
                if cur: contours.append(np.array(cur))
                cur = [P[k]]; k += 1
            elif c == LINETO and k < npts:
                cur.append(P[k]); k += 1
            elif c == CUBICTO and k + 2 < npts:
                p0 = np.array(cur[-1]) if cur else np.array(P[k])
                seg = _flatten_cubic(p0, np.array(P[k]), np.array(P[k+1]), np.array(P[k+2]))
                cur.extend(seg.tolist()); k += 3
            elif c == CLOSE:
                if cur: contours.append(np.array(cur)); cur = []
        if cur: contours.append(np.array(cur))
        r=C.c_uint8(); g=C.c_uint8(); b=C.c_uint8(); a=C.c_uint8()
        lib.tvg_shape_get_fill_color(paint, C.byref(r), C.byref(g), C.byref(b), C.byref(a))
        if contours:
            shapes.append((contours, np.array([r.value, g.value, b.value])/255, a.value/255))
        return True

    acc = lib.tvg_accessor_new()
    lib.tvg_accessor_set(acc, pic, CB(visit), None)
    return shapes
