# Calibrix custom node for ComfyUI.
#
# Drop this file into ComfyUI/custom_nodes/calibrix_node.py (or clone the
# calibrix folder there). It registers a "Calibrix Kernel Scale" node that
# applies Calibrix modulation kernel gains to a FLUX MODEL's per-block gate
# tuples inside a ComfyUI workflow — letting Calibrix-optimized kernels
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

import base64
import hashlib
import hmac
import json
import math
import os
import time
import types

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# ---------------------------------------------------------------------------
# Offline license verification (mirrors calibrix.marketplace.licenses).
# Kept inline so the node still works when copied standalone into
# ComfyUI/custom_nodes without the calibrix package installed.

_KEY_PREFIX = "CBX1"


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _signing_secret() -> str:
    # Set CALIBRIX_MARKETPLACE_SECRET to the marketplace's signing secret.
    return os.environ.get("CALIBRIX_MARKETPLACE_SECRET",
                          "calibrix-dev-secret-do-not-use-in-production")


def verify_license_offline(key: str, spec: str) -> tuple:
    """Verify a CBX1 license key covers `spec`. Returns (ok, reason, payload)."""
    try:
        parts = key.strip().split(".")
        if len(parts) != 3 or parts[0] != _KEY_PREFIX:
            return False, "malformed key", None
        payload_bytes = _b64url_decode(parts[1])
        sig = _b64url_decode(parts[2])
        expected = hmac.new(_signing_secret().encode("utf-8"), payload_bytes,
                            hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return False, "bad signature", None
        payload = json.loads(payload_bytes.decode("utf-8"))
        subject = hashlib.sha256(spec.encode("utf-8")).hexdigest()
        if payload.get("sub") != subject:
            return False, "license does not cover this kernel", payload
        exp = payload.get("exp")
        if exp is not None and int(time.time()) > int(exp):
            return False, "license expired", payload
        if "comfyui" not in payload.get("ent", []):
            return False, "missing entitlement: comfyui", payload
        return True, None, payload
    except Exception as e:  # noqa: BLE001
        return False, f"error: {e}", None


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
    """Scales FLUX attention/MLP gate tuples by Calibrix kernel gains.

    Implementation: hooks `norm1` (and `norm1_context`) in the cloned
    transformer blocks and multiplies tuple positions [1] and [4], matching
    the in-repo Calibri FLUX bridge. Unsupported graph shapes fail closed.
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
                "license_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional CBX1 license from the Calibrix "
                               "marketplace. Empty = free/unlicensed use.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "calibrix"

    def apply(self, model, kernel_spec: str, n_blocks: int,
              license_key: str = ""):
        if license_key.strip():
            ok, reason, payload = verify_license_offline(license_key, kernel_spec)
            if not ok:
                raise ValueError(
                    f"CalibrixKernelScale: license rejected ({reason}). "
                    "Paste the CBX1 key exactly as delivered, or leave "
                    "license_key empty for unlicensed kernels."
                )
        channels = _parse_kernel_spec(kernel_spec)
        if not {"attn", "mlp"}.issubset(channels):
            raise ValueError(
                "CalibrixKernelScale: FLUX validation requires attn and mlp channels"
            )

        # Clone first, then inspect/hook the clone. Hooking the source model
        # would make baseline and kernel runs contaminate one another.
        m = model.clone()
        diffusion_model = m.get_model_object("diffusion_model")

        # Diffusers FLUX exposes double blocks as transformer_blocks; ComfyUI
        # forks commonly call them double_blocks or blocks. We only accept a
        # real block-level hook point—never ignored workflow metadata.
        blocks = None
        for attr in ("transformer_blocks", "double_blocks", "layers", "blocks"):
            cand = getattr(diffusion_model, attr, None)
            if cand is None or not hasattr(cand, "__len__"):
                continue
            if len(cand) > 0:
                blocks = list(cand)
                break
        if blocks is None:
            raise ValueError(
                "CalibrixKernelScale: unsupported FLUX graph; no double-block "
                "list found (expected transformer_blocks/double_blocks/layers/blocks)"
            )

        n = min(int(n_blocks), len(blocks))
        attn_gains = _gains(n, *channels["attn"])
        mlp_gains = _gains(n, *channels["mlp"])
        handles = []

        def gate_hook(attn_gain, mlp_gain):
            def hook(module, args, kwargs, output):
                # This is the FLUX gate tuple contract used by the in-repo
                # Calibri bridge: [1] is attention gate, [4] is MLP gate.
                if not isinstance(output, (tuple, list)) or len(output) < 5:
                    raise RuntimeError(
                        "CalibrixKernelScale: FLUX norm hook returned an "
                        "unsupported shape; refusing to claim modulation"
                    )
                out = list(output)
                out[1] = out[1] * attn_gain
                out[4] = out[4] * mlp_gain
                return tuple(out)
            return hook

        for i, block in enumerate(blocks[:n]):
            norm = getattr(block, "norm1", None)
            if norm is None:
                raise ValueError(
                    "CalibrixKernelScale: FLUX double block lacks norm1 gate hook"
                )
            handles.append(norm.register_forward_hook(
                gate_hook(attn_gains[i], mlp_gains[i]), with_kwargs=True
            ))
            context_norm = getattr(block, "norm1_context", None)
            if context_norm is not None:
                handles.append(context_norm.register_forward_hook(
                    gate_hook(attn_gains[i], mlp_gains[i]), with_kwargs=True
                ))

        # Keep handles alive on the clone and expose an auditable marker for
        # validation tooling. The hooks are intentionally not removable during
        # the ComfyUI execution of this cloned model.
        m._calibrix_hook_handles = handles
        m._calibrix_validation = {
            "kernel_spec": kernel_spec,
            "blocks_hooked": n,
            "hook_kind": "flux_norm1_gate_tuple",
        }
        return (m, )


NODE_CLASS_MAPPINGS["CalibrixKernelScale"] = CalibrixKernelScale
NODE_DISPLAY_NAME_MAPPINGS["CalibrixKernelScale"] = "Calibrix Kernel Scale"
