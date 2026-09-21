# Calibrix × ComfyUI

Two complementary integrations:

1. **ComfyUIAdapter** (in `calibrix/calibrix/image_adapters.py`) — Calibrix
   *drives* ComfyUI: queues workflows over the HTTP API, polls history,
   downloads images, and injects Calibrix kernel gains into the graph. Use
   this when you want Calibrix's search engine to optimize a ComfyUI-hosted
   image model.

2. **CalibrixKernelScale node** (`calibrix_node.py`) — ComfyUI *consumes*
   Calibrix: a drop-in custom node that applies an exported Calibrix kernel
   inside any ComfyUI workflow. Optimized kernels become portable ComfyUI
   workflow JSON you can share like any other workflow.

## Install (free, offline)

```bash
# 1. Get ComfyUI (free & offline) if you don't have it
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt

# 2. Install the Calibrix node
cp /path/to/this/repo/calibrix/comfyui_nodes/calibrix_node.py custom_nodes/

# 3. Start ComfyUI (default http://127.0.0.1:8188)
python main.py
```

## Use the node

In the ComfyUI graph:

```
Load Checkpoint ──> Calibrix Kernel Scale ──> CLIPTextEncode ──> KSampler ──> ...
```

Paste a kernel exported by Calibrix (`report.json` → `best.vector`, or a
kernel spec string like `attn:1.15@0.55:0.82:0.4|mlp:1.0@0.5:1.0:1e6`) into
the node's `kernel_spec` field.

## Drive ComfyUI from Calibrix

```python
from calibrix.image_adapters import ComfyUIAdapter
from calibrix.image_scorers import OfflineImageQuality, OfflineColorAlignment
from calibrix.scorers import seed_prompts
from calibrix.engine import SearchEngine, SearchConfig

adapter = ComfyUIAdapter(
    checkpoint="sd_xl_base_1.0.safetensors",
    modulation_mode="conditioning",   # inject per-block gains into the graph
)
prompts = seed_prompts(["a red boat on a blue lake", "a green forest at dawn"])

engine = SearchEngine(
    adapter,
    [OfflineImageQuality(prompts), OfflineColorAlignment(prompts)],
    prompts,
    SearchConfig(n_trials=6, popsize=6),
)
result = engine.run()
```

## Example workflow

`workflow_example.json` is a ready-to-load ComfyUI workflow containing the
Calibrix Kernel Scale node between the checkpoint loader and the sampler.
