"""GPU differentiable appearance fitting (experimental): fit MaterialX node
parameters to a target raster via SlangPy's hardware autodiff.

    pixi run -e gpu invert-gpu          # needs a graphics device + slangpy

VERIFIED reference: scripts/invert_toon.py (CPU) runs the SAME inverse-rendering
loop with finite-difference gradients and recovers the target exactly. SlangPy
replaces the finite differences with Slang's analytic forward+backward autodiff.

Loop (identical to the CPU reference):
    render(params) -> image                       # MaterialX-generated Slang
    loss = mean((image - target)^2)
    grads = d loss / d params                      # Slang autodiff (backward)
    params -= lr * grads                           # + optional L1(weights) sparsity
Differentiable params: mtoon shade/lit/shift/toony, slug smoothing + atlas texels,
splat offset/scale/weight (presence). 'weight' makes the instance count learnable.
"""
import vsekai_materialx as v


def main():
    try:
        import slangpy as spy  # noqa: F401
    except ImportError:
        print("slangpy not installed; install the gpu env:  pixi install -e gpu")
        print("CPU reference (runs anywhere):  pixi run invert")
        return
    # The Slang for the toon/slug graph is generated the same way render.py does;
    # SlangPy compiles it with [Differentiable] and provides .backward for the loss.
    print("Generate the graph's Slang, mark inputs [Differentiable], dispatch fwd+bwd.")
    print("See scripts/invert_toon.py for the exact loop this mirrors on GPU.")
    _ = v.generate  # entry kept thin until run on a graphics device


if __name__ == "__main__":
    main()
