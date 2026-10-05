# contract-vsekai-materialx

Toon, physically based and vector-glyph materials as standard MaterialX node graphs, generated into differentiable Slang.

## What it is for

Every material is a composition of standard nodes, so any conformant MaterialX renderer runs it and no target needs hand-written shader source. The generated Slang is differentiable, which lets a material's parameters be fitted to a target image. A Lean 4 specification under `lean` proves each node graph equal to its reference function.

## Build and run

```sh
pixi run gen
pixi run invert
```

The first generates shaders for every node; the second fits a toon ramp and an emblem to a target image. `pixi.toml` lists the other tasks.

## Licence

Apache-2.0; see LICENSE.
