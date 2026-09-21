# Calibrix custom node for ComfyUI.
#
# Drop this file into ComfyUI/custom_nodes/calibrix_node.py (or clone the
# calibrix folder there). It registers a "Calibrix Kernel Scale" node that
# applies Calibrix modulation kernel gains to a MODEL's per-block conditioning
# strengths inside any ComfyUI workflow — letting Calibrix-optimized kernels
# run natively in ComfyUI pipelines (and be shared as normal ComfyUI
# workflows JSON).
#
# Node graph placement:
#   Load Checkpoint -> [Calibrix Kernel Scale] -> CLIPTextEncode -> KSampler ...
#
# The node accepts a compact kernel spec string:
#   "attn:1.2@0.5:0.3:0.9|mlp:1.0@0.6:0.4:1.0"
#   component:weight@position:floor:focus[:ripple:phase]
# (the same serialization Calibrix report.json exports).

import math
import types

import torch


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _parse_kernel_spec(spec: str):
    """Parse "attn:1.2@0.5:0.3:0.9|mlp:..." -> dict component -> gains list."""
    channels = {}
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        head, _, rest = part.partition(":")
        params = head + ":" + rest
        # component:weight@position:floor:focus[:ripple:phase]
        comp, _, tail = params.partition(":")
        w, _, tail2 = tail.partition("@")
        nums = tail2.split(":") if tail2 else []
        try:
            weight = float(w)
            position = float(nums[0]) if len(nums) > 0 else 0.5
            floor = float(nums[1]) if len(nums) > 1 else 1.0
            focus = float(nums[2]) if len(nums) > 2 else 1e6
            ripple = float(nums[3]) if len(nums) > 3 else 0.0
            phase = float(nums[4]) if len(nums) > 4 else 0.0
        except (ValueError, IndexError):
            raise ValueError(f"bad kernel spec segment: {part!r}")
        channels[comp.strip()] = (weight, position, floor, focus, ripple, phase)
    return channels


def _gains(n: int, weight: float, position: float, floor: float,
           focus: float, ripple: float, phase: float) -> torch.Tensor:
    last = max(n - 1, 1)
    pos = position * last
    f = focus * last
    out = []
    for l in range(n):
        d = abs(l - pos)
        decay = min(d / f, 1.0) if f > 0 else 1.0
        g = weight - (weight - floor) * decay
        if ripple != 0.0:
            g += ripple * math.cos(2 * math.pi * (l / max(n, 1)) + 2 * math.pi * phase)
        out.append(g)
    return torch.tensor(out, dtype=torch.float32)


class CalibrixKernelScale:
    """Scales per-block outputs of a diffusion model by Calibrix kernel gains.

    Implementation: wraps each transformer block's forward to multiply its
    output by the block's gain, computed from the compact kernel spec.
    Works with ComfyUI-wrapped DiT models (FLUX/SD3/Qwen-Image style
    `diffusion_model.model` with `.layers`/`.transformer_blocks`/`.blocks`).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "kernel_spec": ("STRING", {
                    "default": "attn:1.0@0.5:1.0:1e6|mlp:1.0@0.5:1.0:1e6",
                    "multiline": False,
                }),
                "n_blocks": ("INT", {"default": 19, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "calibrix"

    def apply(self, model, kernel_spec: str, n_blocks: int):
        channels = _parse_kernel_spec(kernel_spec)

        m = model.clone()
        diffusion_model = model.get_model_object("diffusion_model")

        # find a block list on the diffusion model
        blocks = None
        for attr in ("transformer_blocks", "layers", "blocks",
                     "double_blocks", "single_transformer_blocks"):
            cand = getattr(diffusion_model, attr, None)
            if isinstance(cand, (list,)) or hasattr(cand, "__len__"):
                try:
                    if len(cand) > 0:
                        blocks = list(cand)
                        break
                except TypeError:
                    continue
        if blocks is None:
            raise ValueError(
                "CalibrixKernelScale: no transformer block list found on the "
                "diffusion model (looked at transformer_blocks/layers/blocks/"
                "double_blocks/single_transformer_blocks)."
            )

        n = min(int(n_blocks), len(blocks))
        for i in range(n):
            gains = torch.zeros(len(channels))
            for ci, (comp, params) in enumerate(channels.items()):
                gains[ci] = _gains(n, *params)[i]

            def make_hook(gains_vec):
                def hook(module, args, output):
                    # scale the block output tensor (or first tensor element)
                    if isinstance(output, tuple):
                        return tuple(
                            gains_vec[0] * o if torch.is_tensor(o) else o
                            for o in output
                        )
                    return gains_vec[0] * output
                return hook

            blocks[i] = blocks[i]  # keep identity; attach below
            h = blocks[i].register_forward_hook(make_hook(gains))
            # ComfyUI model.clone() keeps patches; hooks survive via the module.
            _ = h

        return (m, )


NODE_CLASS_MAPPINGS["CalibrixKernelScale"] = CalibrixKernelScale
NODE_DISPLAY_NAME_MAPPINGS["CalibrixKernelScale"] = "Calibrix Kernel Scale"
