# License Map & Boundary Policy

This repository contains code under two different licenses, kept strictly
separated. Read this before touching either side of the boundary.

| Path | License | Role |
|---|---|---|
| `calibrix/` | MIT | The platform: search engine, studio logic, kernels, marketplace, agents, Rust core (`calibrix/rust-core/`), bridge client |
| `comfyui_nodes/` | MIT | ComfyUI custom node (dependency-free spec parser) |
| `usecases/` | MIT | Self-contained vertical product folders |
| `heritic/heretic/` | **AGPL-3.0** | Upstream Heretic source tree, vendored **unmodified** as a reference/optional accelerator |

## Why the separation exists

Heretic is AGPL-3.0. If Calibrix imported Heretic code, linked it into the
same process, or shared data structures with it, the whole combined work
would have to be AGPL. Calibrix is MIT, and staying MIT is a product
requirement (commercial kernel marketplace, hosted calibration). The method
directional ablation + TPE co-optimization — is **not** copyrightable and is
reimplemented clean-room from published literature (Arditi et al. 2024 and
cited predecessors) in `calibrix/ablation.py` and `calibrix/optimizers.py`.

## The boundary, mechanically

`calibrix/calibrix/heretic_bridge.py` is the ONLY module that interacts with
the AGPL side, and it does so exclusively through:

```
subprocess.run(["heretic", <model>, ...])   # published CLI of heretic-llm
```

Hard rules (all machine-checked):

1. **No imports.** No `calibrix` module may `import heretic` (or
   `importlib`/`exec`-load it). Enforced by
   `calibrix/tests/test_standalone.py::TestLicenseFirewall`, which AST-parses
   every package file in CI.
2. **No shared memory.** The child process never imports calibrix; the parent
   never imports heretic. Communication is the child's stdout/stderr and
   artifact files only.
3. **No modification.** The `heritic/heretic/` tree is vendored as-is. Do not
   edit files inside it; improvements go on the MIT side
   (`calibrix/calibrix/ablation.py` implements the improved, robust
   multi-direction extraction and kernel-profile fitting natively).
4. **Optional by construction.** If the `heretic` CLI is not installed (the
   default for offline/non-commercial use), the bridge reports the AGPL side
   as unavailable and every MIT-side capability still works — the entire
   platform, tests, and solo runner run offline without it
   (`python -m calibrix solo`).

## What crossing the boundary would mean

If you install and *run* the Heretic CLI yourself (separate act, separate
machine or environment), you use Heretic under AGPL-3.0 terms. Results
produced by that run (per-layer strength numbers) are data; converting such
data into a Calibrix kernel spec via `fit_kernel_profile` happens entirely on
the MIT side and creates no derivative of Heretic's code. This mirrors
standard inter-process licensing practice and the strategy documented in
`docs/product_opportunities.md`.

## Non-commercial use

The Calibrix platform itself is usable standalone and non-commercially under
MIT: clone, run `python -m calibrix solo`, redistribute with the MIT license
text intact. The AGPL tree is not required for any MIT-side feature and can
be deleted wholesale without breaking anything (the tests prove it: they run
with the tree present but its CLI uninstalled).
