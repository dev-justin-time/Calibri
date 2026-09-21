# SPDX-License-Identifier: MIT
# Calibrix adapter layer.
#
# An Adapter is anything that can be modulated (kernel gains applied) and
# evaluated. This mirrors Heretic's Model + Context split and Calibri's
# BaseSGPipeline, but model-agnostic and dependency-light at the core:
#
#   ScriptedAdapter      - offline, no GPU, deterministic; powers tests,
#                          CI, and "try Calibrix in 30 seconds".
#   OpenAICompatAdapter  - any local OpenAI-compatible server (Ollama,
#                          llama.cpp server, LM Studio, vLLM) = free/offline
#                          use with real LLMs.
#   HFInferenceAdapter   - hosted generation via huggingface_hub
#                          InferenceClient (free-tier friendly; remote
#                          models can't be modulated, so they serve as
#                          baseline/eval backends and router targets).
#   CalibriFluxAdapter   - bridge into the Calibri codebase in this repo
#                          (src/models/*), applying kernel gains as gate
#                          scales. Optional: requires torch/diffusers.
#
# Modulation contract (the whole point):
#   - spec_sites() -> List[ModulationSpec]  (the modulatable channels)
#   - apply_modulation(specs)               (materialize gains into the model)
#   - generate(prompts, batch_size, ctx)    (run + report steps to ctx)

from __future__ import annotations

import json
import os
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from .kernel import ModulationSpec
from .scorers import Prompt, ScorerContext


def _stub_response(prompt: Prompt, gains_signature: float, step_cost: int) -> str:
    """Deterministic synthetic response used by the offline scripted adapter.

    The response encodes the modulation state, so scorer metrics (length,
    refusals, diversity) respond plausibly to kernel changes without a GPU.
    A signature near 1.0 == unmodulated model.
    """
    h = hashlib.sha256(prompt.user.encode("utf-8")).hexdigest()

    # Deviation from identity modulation drives "damage".
    dev = abs(gains_signature - 1.0)

    if dev > 0.55:
        return ""  # degenerate output under extreme modulation
    if dev > 0.30:
        return (
            "I'm sorry, but I can't help with that request. "
            "As an AI assistant, I must refuse."
        )

    # Mild/moderate deviation -> coherent-ish answer whose length grows with
    # deviation (mimics verbosity drift) and content varies by prompt hash.
    n_sentences = 2 + int(dev * 18)
    words = ["model", "latent", "signal", "vector", "field", "wave", "kernel",
             "gate", "layer", "flow", "index", "phase", "scale", "drift"]
    body = " ".join(
        words[int(h[i % len(h) : i % len(h) + 2], 16) % len(words)]
        for i in range(n_sentences * 7)
    )
    return f"Sure. {body}."


class Adapter(ABC):
    """Base class for all Calibrix model adapters."""

    name: str = "adapter"
    # Whether apply_modulation can actually change behavior (remote APIs can't).
    modulatable: bool = True

    @abstractmethod
    def spec_sites(self) -> List[ModulationSpec]:
        """Declare the modulatable channels of the underlying model."""

    @abstractmethod
    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        """Materialize per-site gains from the specs into the model."""

    @abstractmethod
    def generate(
        self,
        prompts: List[Prompt],
        batch_size: int = 8,
        ctx: Optional[ScorerContext] = None,
    ) -> List[str]:
        """Generate responses; report step consumption to ctx if given."""

    def logits(self, prompts: List[Prompt], ctx: Optional[ScorerContext] = None) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose logits (needed by KLDrift)."
        )

    def apply_to_vector(self, vec: Sequence[float]) -> None:
        """Convenience: unpack a flat kernel vector into the live specs."""
        specs = self.spec_sites()
        unpack_into(specs, vec)
        self.apply_modulation(specs)


def unpack_into(specs: List[ModulationSpec], vec: Sequence[float]) -> None:
    from .kernel import unpack_spec_vector

    unpack_spec_vector(specs, [float(v) for v in vec])


# ---------------------------------------------------------------------------
# Offline scripted adapter (no dependencies, deterministic)
# ---------------------------------------------------------------------------

class ScriptedAdapter(Adapter):
    """Offline, dependency-free adapter emulating a modulatable model.

    Behavior model (deterministic, seeded by prompt text):
      - modulation deviation < 0.30 : coherent output, mild length growth
      - 0.30 .. 0.55                : refusal-style output
      - > 0.55                      : empty output (degenerate)

    This gives every scorer in Calibrix a realistic, stable signal surface
    so the full search pipeline runs on a laptop with no GPU and no network.
    """

    def __init__(self, n_layers: int = 12, seed: int = 0) -> None:
        self.n_layers = int(n_layers)
        self.seed = int(seed)
        self._specs = [
            ModulationSpec(component="attn", n_sites=self.n_layers),
            ModulationSpec(component="mlp", n_sites=self.n_layers),
        ]
        self.steps_per_call = 2  # emulate 2 inference steps per generation

    def spec_sites(self) -> List[ModulationSpec]:
        return self._specs

    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        self._specs = specs

    def _signature(self) -> float:
        """Mean absolute deviation of current gains from identity (1.0)."""
        devs = []
        for spec in self._specs:
            for g in spec.gains():
                devs.append(abs(g - 1.0))
        if not devs:
            return 1.0
        return 1.0 - (sum(devs) / len(devs))

    def generate(
        self,
        prompts: List[Prompt],
        batch_size: int = 8,
        ctx: Optional[ScorerContext] = None,
    ) -> List[str]:
        sig = self._signature()
        out = [_stub_response(p, sig, self.steps_per_call) for p in prompts]
        if ctx is not None:
            # emulate step cost scaled by response length (longer = more steps)
            total_steps = self.steps_per_call * max(len(prompts), 1)
            ctx.note_steps(total_steps)
        return out

    def logits(self, prompts: List[Prompt], ctx: Optional[ScorerContext] = None) -> Any:
        import numpy as np

        sig = self._signature()
        rows = []
        for p in prompts:
            h = hashlib.sha256(p.user.encode("utf-8")).hexdigest()
            rng = np.random.default_rng(int(h[:8], 16) + self.seed)
            base = rng.normal(size=16)
            # drift logits with modulation deviation -> KL responds to damage
            rows.append(base + rng.normal(scale=4.0 * (1.0 - sig), size=16))
        if ctx is not None:
            ctx.note_steps(self.steps_per_call * len(prompts))
        return np.stack(rows)

    def hidden_states(self, prompts: List[Prompt]) -> Any:
        """Synthetic residual stream for research probes.

        Returns array shaped (n_prompts, n_layers+1, d). A contrast between
        prompt clusters is emulated by offsetting cluster "tone" so probes
        have a real (deterministic) signal to find.
        """
        import numpy as np

        rows = []
        for p in prompts:
            h = hashlib.sha256(p.user.encode("utf-8")).hexdigest()
            rng = np.random.default_rng(int(h[:8], 16))
            per_layer = []
            for l in range(self.n_layers + 1):
                vec = rng.normal(size=16)
                # layer-depth-dependent contrast signal driven by prompt tone
                tone = 1.0 if any(w in p.user.lower()
                                  for w in ("poem", "describe", "story")) else -1.0
                vec = vec + tone * math.tanh(l / 4.0) * 0.8
                per_layer.append(vec)
            rows.append(per_layer)
        return np.stack(rows)


# ---------------------------------------------------------------------------
# OpenAI-compatible local adapter (Ollama / llama.cpp / LM Studio / vLLM)
# ---------------------------------------------------------------------------

class OpenAICompatAdapter(Adapter):
    """Client for any local OpenAI-compatible chat server.

    Free/offline usage:
        ollama serve                        # then model="llama3.2"
        llama_cpp_server --model x.gguf     # then model="x"
    No API key, no cloud, runs on your machine.
    """

    name = "openai-compat"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        n_layers: int = 32,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get(
            "CALIBRIX_LOCAL_URL", "http://127.0.0.1:11434/v1"
        )
        self.api_key = api_key or os.environ.get("CALIBRIX_LOCAL_KEY", "not-needed")
        self.n_layers = n_layers
        self._specs = [
            ModulationSpec(component="attn", n_sites=n_layers),
            ModulationSpec(component="mlp", n_sites=n_layers),
        ]
        # Note: local servers cannot be weight-modulated from outside the
        # process; use CalibriFluxAdapter/LLMAdapter for true modulation.
        # This adapter is for scoring/judging/eval against real local models.
        self.modulatable = False
        self._gains_override: Optional[List[List[float]]] = None

    def spec_sites(self) -> List[ModulationSpec]:
        return self._specs

    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        # Graceful: record gains but they have no effect on a remote process.
        self._gains_override = [s.gains() for s in specs]

    def generate(
        self,
        prompts: List[Prompt],
        batch_size: int = 8,
        ctx: Optional[ScorerContext] = None,
    ) -> List[str]:
        import urllib.request

        results: List[str] = []
        for i in range(0, len(prompts), max(1, batch_size)):
            chunk = prompts[i : i + batch_size]
            chunk_out: List[Optional[str]] = [None] * len(chunk)
            # Sequential to be gentle on local servers; batch via threads if needed.
            for j, p in enumerate(chunk):
                payload = {
                    "model": self.model,
                    "messages": [
                        *([{"role": "system", "content": p.system}] if p.system else []),
                        {"role": "user", "content": p.user},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 512,
                }
                req = urllib.request.Request(
                    self.base_url.rstrip("/") + "/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                chunk_out[j] = data["choices"][0]["message"]["content"]
            results.extend(chunk_out)  # type: ignore[arg-type]
        if ctx is not None:
            ctx.note_steps(len(prompts))  # 1 "step" per request
        return results


# ---------------------------------------------------------------------------
# Hugging Face hosted inference adapter (free-tier friendly)
# ---------------------------------------------------------------------------

class HFInferenceAdapter(Adapter):
    """Hosted text_to_image / chat via huggingface_hub.InferenceClient.

    Use cases inside Calibrix:
      * baseline scorer backend for hosted models (SD3.5-medium etc.)
      * comparison arm in model routing experiments
      * LLM-judge backend when no local judge is available

    Remote models are NOT modulatable (you don't get weights), so this
    adapter raises on apply_modulation; use it for evaluation only.
    """

    name = "hf-inference"

    def __init__(self, model: str, provider: Optional[str] = None) -> None:
        self.model = model
        self.provider = provider
        self.modulatable = False
        self._client = None
        self._kind = "image" if "stable-diffusion" in model or "flux" in model.lower() else "chat"

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient  # optional dep

            self._client = InferenceClient(provider=self.provider)
        return self._client

    def spec_sites(self) -> List[ModulationSpec]:
        # Declared so router/planner can reason about it; not modulatable.
        return [ModulationSpec(component="remote", n_sites=1)]

    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        raise RuntimeError(
            "HFInferenceAdapter is remote-only: weights are not accessible, "
            "so modulation cannot be applied. Use it as an eval/baseline backend."
        )

    def generate(
        self,
        prompts: List[Prompt],
        batch_size: int = 8,
        ctx: Optional[ScorerContext] = None,
    ) -> List[str]:
        client = self._get_client()
        out: List[str] = []
        for p in prompts:
            if self._kind == "image":
                img = client.text_to_image(p.user, model=self.model)
                # Return a tiny text descriptor; image scorers should subclass
                # and override to consume the PIL image directly.
                out.append(f"<image {img.size[0]}x{img.size[1]}>")
            else:
                resp = client.chat_completion(
                    model=self.model,
                    messages=[
                        *([{"role": "system", "content": p.system}] if p.system else []),
                        {"role": "user", "content": p.user},
                    ],
                    max_tokens=512,
                )
                out.append(resp.choices[0].message.content or "")
        if ctx is not None:
            ctx.note_steps(len(prompts))
        return out


# ---------------------------------------------------------------------------
# Calibri bridge: apply kernel gains as Calibri gate scales (FLUX etc.)
# ---------------------------------------------------------------------------

class CalibriFluxAdapter(Adapter):
    """Bridges Calibrix kernels into the Calibri FLUX pipeline in this repo.

    Requires torch + diffusers (heavy deps); import lazily so Calibrix core
    stays dependency-free. Kernel gains are mapped onto the Calibri scale
    parameters: gain g on site l scales the block output, matching Calibri's
    per-block gate semantics; 'attn' channels map to gate_msa, 'mlp' to
    gate_mlp.
    """

    name = "calibri-flux"

    def __init__(self, device: str = "cuda", dtype: str = "bfloat16") -> None:
        from src.models.flux_sg import SGFluxPipeline  # lazy heavy import

        import torch

        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]
        self.pipe = SGFluxPipeline(device=device, dtype=dt)
        self.modulatable = True

    def spec_sites(self) -> List[ModulationSpec]:
        shapes = self.pipe.get_coefficient_shapes()
        n_double, n_single = shapes["n_double"], shapes["n_single"]
        return [
            ModulationSpec(component="attn", n_sites=n_double, metadata={"kind": "double_attn"}),
            ModulationSpec(component="mlp", n_sites=n_double, metadata={"kind": "double_mlp"}),
            ModulationSpec(component="single", n_sites=n_single, metadata={"kind": "single"}),
        ]

    def apply_modulation(self, specs: List[ModulationSpec]) -> None:
        gains_by_kind = {s.component: s.gains() for s in specs}
        scales_double = [[(g, g) for g in gains_by_kind["attn"]] for _ in range(1)]
        scales_single = [gains_by_kind["single"]]
        self.pipe.apply_coefficients(
            self.pipe.struct_to_flat(
                {
                    "scales_double": scales_double,
                    "scales_double_ctx": scales_double,
                    "scales_single": scales_single,
                    "models_scales": [1.0],
                }
            )
        )

    def generate(
        self,
        prompts: List[Prompt],
        batch_size: int = 8,
        ctx: Optional[ScorerContext] = None,
        num_inference_steps: int = 15,
        **kwargs: Any,
    ) -> List[str]:
        images = self.pipe(
            [p.user for p in prompts],
            num_inference_steps=num_inference_steps,
            guidance_scale=3.5,
            height=512,
            width=512,
            generator=None,
        ).images
        if ctx is not None:
            ctx.note_steps(num_inference_steps * len(prompts))
        return [f"<image {im.size[0]}x{im.size[1]}>" for im in images]
