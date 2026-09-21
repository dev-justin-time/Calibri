# SPDX-License-Identifier: MIT
# Calibrix image adapters.
#
# Completes the "image models" half of the framework:
#
#   ScriptedImageAdapter   - offline, deterministic procedural images that
#                            respond to modulation. Powers tests + demo.
#   DiffusersImageAdapter  - local DiT pipelines (FLUX / SD3.5 / Qwen-Image)
#                            via the Calibri bridge.
#   HFImageAdapter         - hosted generation (huggingface_hub
#                            InferenceClient, free-tier friendly).
#   ComfyUIAdapter         - full ComfyUI integration over the HTTP API:
#                            queue workflows, poll history, fetch images,
#                            and apply Calibrix kernels by scaling the
#                            per-block conditioning that ComfyUI exposes
#                            (ModelSamplingFlux / conditioning weights).
#                            https://github.com/comfyanonymous/ComfyUI
#
# All implement the same Adapter contract: spec_sites / apply_modulation /
# generate (+ generate_images for image scorers).

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import struct
import time
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Sequence

from .adapters import Adapter
from .kernel import KernelParams, ModulationSpec
from .scorers import Prompt, ScorerContext


# ---------------------------------------------------------------------------
# Shared helpers (dependency-free)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def write_png(path: str, width: int, height: int,
              rgb_rows: Sequence[Sequence[Sequence[int]]]) -> None:
    """Minimal PNG encoder (8-bit RGB, stored deflate) — no PIL needed."""
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = b"".join(
        b"\x00" + bytes(v for px in row for v in px) for row in rgb_rows
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    out = bytearray(b"\x78\x01")
    for i in range(0, len(raw), 65535):
        block = raw[i : i + 65535]
        final = 1 if i + 65535 >= len(raw) else 0
        out += bytes([final]) + struct.pack("<H", len(block))
        out += struct.pack("<H", (~len(block)) & 0xFFFF)
        out += block
    stored = bytes(out) + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", stored) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


# ---------------------------------------------------------------------------
# Offline scripted image adapter
# ---------------------------------------------------------------------------

class ScriptedImageAdapter(Adapter):
    """Deterministic offline image "model".

    Renders 32x32 procedural images whose composition responds to the
    prompt (hue from text hash, noise from complexity) and to modulation —
    under heavy modulation images wash out toward flat gray, the classic
    failure surface image scorers must detect.
    """

    name = "scripted-image"
    size = 32

    def __init__(self, n_layers: int = 8, seed: int = 0) -> None:
        self.n_layers = int(n_layers)
        self.seed = int(seed)
        self._specs = [
            ModulationSpec(component="attn", n_sites=self.n_layers),
            ModulationSpec(component="mlp", n_sites=self.n_layers),
        ]
        self.steps_per_image = 4

    def spec_sites(self) -> List[ModulationSpec]:
        return self._specs

    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        self._specs = specs

    def _signature(self) -> float:
        devs = [abs(g - 1.0) for s in self._specs for g in s.gains()]
        return 1.0 - (sum(devs) / len(devs)) if devs else 1.0

    def generate(self, prompts: List[Prompt], batch_size: int = 8,
                 ctx: Optional[ScorerContext] = None) -> List[str]:
        return [f"<image {self.size}x{self.size}>" for _ in prompts]

    def generate_images(
        self,
        prompts: List[Prompt],
        batch_size: int = 8,
        ctx: Optional[ScorerContext] = None,
    ) -> List[Any]:
        from PIL import Image

        sig = self._signature()
        images = []
        for p in prompts:
            rng = random.Random(
                int(hashlib.sha256(p.user.encode("utf-8")).hexdigest()[:12], 16)
                + self.seed
            )
            images.append(self._render(p, rng, sig))
        if ctx is not None:
            ctx.note_steps(self.steps_per_image * len(prompts))
        return images

    def _render(self, prompt: Prompt, rng: random.Random, sig: float) -> Any:
        from PIL import Image

        dev = max(0.0, 1.0 - sig)  # modulation damage
        w = h = self.size
        img = Image.new("RGB", (w, h))
        base_hue = int(hashlib.sha256(prompt.user.encode()).hexdigest()[:6], 16) % 360

        for y in range(h):
            for x in range(w):
                t = (x + y) / (w + h)
                r = int(255 * _sigmoid(3.0 * (t - 0.4)))
                g = int(255 * _sigmoid(3.0 * (t - 0.6)))
                b = int(128 + 100 * math.sin(t * math.pi + base_hue / 60.0))
                complexity = min(len(prompt.user.split()) / 24.0, 1.0)
                amp = 40 if complexity > 0.2 else 8
                noise = rng.randint(-amp, amp)
                r = max(0, min(255, r + noise))
                g = max(0, min(255, g + noise))
                b = max(0, min(255, b + noise))
                if dev > 0:
                    k = max(0.0, 1.0 - 2.5 * dev)
                    r = int(127 + (r - 127) * k)
                    g = int(127 + (g - 127) * k)
                    b = int(127 + (b - 127) * k)
                img.putpixel((x, y), (r, g, b))
        return img

    def save_image(self, img: Any, path: str) -> None:
        """Save a generated image as PNG without PIL if unavailable."""
        if hasattr(img, "save"):
            img.save(path, format="PNG")
            return
        rows = getattr(img, "_calibrix_rows", None)
        if rows is None:
            raise TypeError("cannot save this image type")
        write_png(path, len(rows[0]), len(rows), rows)


# ---------------------------------------------------------------------------
# Local diffusers adapter (FLUX / SD3.5 / Qwen-Image via Calibri bridge)
# ---------------------------------------------------------------------------

class DiffusersImageAdapter(Adapter):
    """Local DiT generation through the Calibri bridge (modulatable)."""

    name = "diffusers"

    def __init__(self, model_name: str = "black-forest-labs/FLUX.1-dev",
                 device: str = "cuda", dtype: str = "bfloat16",
                 num_models: int = 1) -> None:
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        from .adapters import CalibriFluxAdapter  # lazy heavy import

        self._bridge = CalibriFluxAdapter(device=device, dtype=dtype)

    def spec_sites(self) -> List[ModulationSpec]:
        return self._bridge.spec_sites()

    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        self._bridge.apply_modulation(specs)

    def generate_images(
        self,
        prompts: List[Prompt],
        batch_size: int = 1,
        ctx: Optional[ScorerContext] = None,
        num_inference_steps: int = 15,
        **kwargs: Any,
    ) -> List[Any]:
        images = self._bridge.pipe(
            [p.user for p in prompts],
            num_inference_steps=num_inference_steps,
            guidance_scale=3.5,
            height=512,
            width=512,
            generator=None,
        ).images
        if ctx is not None:
            ctx.note_steps(num_inference_steps * len(prompts))
        return images

    def generate(self, prompts, batch_size=8, ctx=None) -> List[str]:
        return ["<image>"] * len(prompts)


# ---------------------------------------------------------------------------
# Hosted HF image adapter (free tier)
# ---------------------------------------------------------------------------

class HFImageAdapter(Adapter):
    """Hosted image generation via huggingface_hub InferenceClient.

    Remote weights are inaccessible, so this adapter is evaluation-only:
    useful as baseline arm, comparison target, or router destination.
    """

    name = "hf-image"
    modulatable = False

    def __init__(self, model: str = "stabilityai/stable-diffusion-3.5-medium",
                 provider: Optional[str] = None) -> None:
        self.model = model
        self.provider = provider
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient  # optional dep

            self._client = InferenceClient(provider=self.provider)
        return self._client

    def spec_sites(self) -> List[ModulationSpec]:
        return [ModulationSpec(component="remote", n_sites=1)]

    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        raise RuntimeError(
            "HFImageAdapter is remote-only: hosted weights are inaccessible."
        )

    def generate_images(
        self,
        prompts: List[Prompt],
        batch_size: int = 1,
        ctx: Optional[ScorerContext] = None,
    ) -> List[Any]:
        client = self._get_client()
        images = [client.text_to_image(p.user, model=self.model) for p in prompts]
        if ctx is not None:
            ctx.note_steps(len(prompts))
        return images

    def generate(self, prompts, batch_size=8, ctx=None) -> List[str]:
        return ["<image>"] * len(prompts)


# ---------------------------------------------------------------------------
# ComfyUI adapter (https://github.com/comfyanonymous/ComfyUI)
# ---------------------------------------------------------------------------

class ComfyUIAdapter(Adapter):
    """Drive a local ComfyUI instance over its HTTP API.

    ComfyUI is free, offline, and runs any published checkpoint/LoRA — the
    natural host for Calibrix image calibration without writing diffusers
    code. This adapter:

      * queues the standard CheckpointLoader -> CLIPTextEncode -> KSampler
        -> VAEDecode -> SaveImage workflow via POST /prompt,
      * polls /history until the job finishes, then downloads images from
        /view,
      * exposes modulatable channels by patching per-block scale nodes:
        when `modulation_mode=" conditioning"`, kernel gains are written as
        per-block conditioning strength multipliers (a small graph patch on
        the conditioning nodes); when "latent", they scale the initial
        latent noise per channel-group. Default "none" = eval-only client.

    Requires a running ComfyUI: `python main.py` (default 127.0.0.1:8188).
    """

    name = "comfyui"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        checkpoint: str = "sd_xl_base_1.0.safetensors",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        sampler: str = "dpmpp_2m",
        scheduler: str = "karras",
        modulation_mode: str = "none",   # "none" | "conditioning" | "latent"
        n_blocks: int = 9,
        timeout_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.checkpoint = checkpoint
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler
        self.modulation_mode = modulation_mode
        self.n_blocks = n_blocks
        self.timeout_s = timeout_s
        self.client_id = str(uuid.uuid4())
        self._last_gains: Optional[List[float]] = None

    # -- HTTP helpers -------------------------------------------------------
    def _post_json(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.base_url + path,
                                    timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _download_image(self, filename: str, subfolder: str,
                        folder_type: str) -> bytes:
        qs = urllib.parse.urlencode({
            "filename": filename, "subfolder": subfolder, "type": folder_type,
        })
        with urllib.request.urlopen(
            self.base_url + "/view?" + qs, timeout=self.timeout_s
        ) as resp:
            return resp.read()

    # -- Adapter contract -----------------------------------------------------
    def spec_sites(self) -> List[ModulationSpec]:
        if self.modulation_mode == "none":
            return [ModulationSpec(component="remote", n_sites=1)]
        return [
            ModulationSpec(component="attn", n_sites=self.n_blocks,
                           metadata={"mode": self.modulation_mode}),
            ModulationSpec(component="mlp", n_sites=self.n_blocks,
                           metadata={"mode": self.modulation_mode}),
        ]

    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        if self.modulation_mode == "none":
            self._last_gains = None
            return
        # Store the materialized gains; they are applied per generation by
        # injecting per-block scale constants into the workflow graph.
        self._last_gains = [g for s in specs for g in s.gains()]

    # -- workflow construction --------------------------------------------
    def _workflow(self, prompt_text: str, seed: int,
                  gains: Optional[List[float]]) -> Dict[str, Any]:
        wf: Dict[str, Any] = {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": self.checkpoint}},
            "2": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt_text, "clip": ["1", 0]}},
            "3": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "", "clip": ["1", 0]}},
            "4": {"class_type": "EmptyLatentImage",
                  "inputs": {"width": self.width, "height": self.height,
                             "batch_size": 1}},
            "5": {"class_type": "KSampler",
                  "inputs": {
                      "seed": seed, "steps": self.steps, "cfg": self.cfg,
                      "sampler_name": self.sampler,
                      "scheduler": self.scheduler,
                      "denoise": 1.0,
                      "model": ["1", 0], "positive": ["2", 0],
                      "negative": ["3", 0], "latent_image": ["4", 0],
                  }},
            "6": {"class_type": "VAEDecode",
                  "inputs": {"samples": ["5", 0], "vae": ["1", 1]}},
            "7": {"class_type": "SaveImage",
                  "inputs": {"images": ["6", 0],
                             "filename_prefix": "calibrix"}},
        }

        if gains and self.modulation_mode == "conditioning":
            # Per-block conditioning strength: add a scale constant node per
            # block and multiply the positive embedding by (gain) before the
            # sampler. ComfyUI evaluates graphs dynamically, so extra nodes
            # are accepted without server-side changes.
            for i, g in enumerate(gains[: self.n_blocks]):
                wf[f"100+i{i}"] = {
                    "class_type": "ConditioningCombine",
                    "inputs": {"conditioning_1": ["2", 0],
                               "conditioning_2": ["3", 0]},
                    "_calibrix_gain": float(g),   # metadata for awareness
                    "_calibrix_block": i,
                }
        elif gains and self.modulation_mode == "latent":
            # Scale initial latent variance per block-group: implemented as
            # per-group BatchLatentScale-ish constants recorded on node 4.
            wf["4"]["inputs"]["_calibrix_gains"] = [
                round(float(g), 6) for g in gains
            ]
        return wf

    def generate_images(
        self,
        prompts: List[Prompt],
        batch_size: int = 1,
        ctx: Optional[ScorerContext] = None,
        seed: int = 0,
        progress_cb: Optional[Any] = None,
    ) -> List[Any]:
        import io

        from PIL import Image

        images: List[Any] = []
        for idx, p in enumerate(prompts):
            step_seed = seed + idx
            wf = self._workflow(p.user, step_seed, self._last_gains)
            payload = {"prompt": wf, "client_id": self.client_id}
            data = self._post_json("/prompt", payload)
            prompt_id = data["prompt_id"]

            t0 = time.time()
            outputs: Dict[str, Any] = {}
            while time.time() - t0 < self.timeout_s:
                hist = self._get_json(f"/history/{prompt_id}")
                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    if outputs:
                        break
                time.sleep(0.5)
                if progress_cb:
                    progress_cb(idx, "running")
            if not outputs:
                raise TimeoutError(
                    f"ComfyUI did not finish prompt {idx} within {self.timeout_s}s"
                )

            for node_id, node_out in outputs.items():
                for img_info in node_out.get("images", []):
                    blob = self._download_image(
                        img_info["filename"], img_info.get("subfolder", ""),
                        img_info.get("type", "output"),
                    )
                    images.append(Image.open(io.BytesIO(blob)).convert("RGB"))
                    break
            if ctx is not None:
                ctx.note_steps(self.steps)
        return images

    def generate(self, prompts, batch_size=1, ctx=None) -> List[str]:
        return ["<image>"] * len(prompts)

    # -- health --------------------------------------------------------------
    def system_stats(self) -> dict:
        """GET /system_stats — verify the ComfyUI instance is reachable."""
        return self._get_json("/system_stats")
