"""GPU path (experimental): run a MaterialX-generated Slang node via SlangPy.

    pixi run -e gpu render-gpu          # needs a D3D12/Vulkan device + slangpy

VERIFIED reference: scripts/render_slug_layers.py (CPU) renders the same math.
This module proves the *generation* half here and hands the Slang to SlangPy; the
GPU dispatch is the integration point you run on a machine with a graphics device.
"""
import os
import vsekai_materialx as v


def emit_slang(node="slug_path", out_type="float") -> str:
    src = v.generate(node, out_type, target="slang")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generated")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{node}.slang")
    open(path, "w").write(src)
    print(f"MaterialX -> Slang: {node} ({len(src)} chars) wrote {path}")
    return path


def main(node="slug_path"):
    path = emit_slang(node)
    try:
        import slangpy as spy  # noqa: F401
    except ImportError:
        print("slangpy not installed; install the gpu env:  pixi install -e gpu")
        print("Then SlangPy compiles", path, "and dispatches NG_%s over a UV grid." % node)
        return
    print("slangpy present. Load the generated NG function and dispatch over a grid:")
    print("  device = spy.Device(); module = spy.Module.load_from_file(device, <wrapper>.slang)")
    print("  see https://shader-slang.org/machine-learning/ for the call/grad API")


if __name__ == "__main__":
    main()
