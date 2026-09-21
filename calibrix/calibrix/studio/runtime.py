# SPDX-License-Identifier: MIT
# Calibrix Logic Studio — Domain 6: Edge / Cluster Runtime Hooks & Artifacts.
#
# The deployment layer: turn a search result into artifacts consumers can
# actually load (ComfyUI workflows, annotated safetensors, hot-swap gate
# bundles), plus runtime guardrails (quantization clamps, autoscaler math).
# Formats are written from their public specifications.

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# F066 — ComfyUI Native Node Exporter --------------------------------------------
def export_comfyui_workflow(kernel_spec: str, checkpoint: str,
                            n_blocks: int = 19,
                            workflow_name: str = "calibrix_kernel") -> Dict[str, Any]:
    """Emit a loadable ComfyUI workflow JSON embedding the kernel.

    Format per ComfyUI's published workflow schema: nodes keyed by id with
    class_type + inputs; the CalibrixKernelScale custom node (shipped in
    comfyui_nodes/) consumes kernel_spec verbatim.
    """
    return {
        "last_node_id": 3,
        "last_link_id": 2,
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "pos": [0, 0],
             "inputs": [], "outputs": [{"name": "MODEL", "type": "MODEL"}],
             "properties": {"ckpt_name": checkpoint}},
            {"id": 2, "type": "CalibrixKernelScale", "pos": [220, 0],
             "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
             "outputs": [{"name": "MODEL", "type": "MODEL", "links": [2]}],
             "widgets_values": [kernel_spec, n_blocks, ""],
             "properties": {}},
            {"id": 3, "type": "Note", "pos": [220, 200],
             "widgets_values": [f"kernel_spec={kernel_spec}"],
             "properties": {}},
        ],
        "links": [[1, 1, 0, 2, 0, "MODEL"]],
        "version": 0.4,
        "extra": {"calibrix": {"workflow_name": workflow_name,
                               "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")}},
    }


# F078 — Safetensors Metadata Injector ----------------------------------------------
def safetensors_header(tensors: Dict[str, List[int]], dtype: str = "F32",
                       metadata: Optional[Dict[str, str]] = None) -> bytes:
    """Build a safetensors header (per the published format spec):
    u64_le header_len + JSON header with dtype/shape per tensor + metadata.

    Used to *attest* artifacts: license id + kernel hash travel inside the
    file header, so provenance survives copying.
    """
    header: Dict[str, Any] = {
        name: {"dtype": dtype, "shape": shape,
               "data_offsets": [0, int(np.prod(shape) * _dtype_size(dtype))]}
        for name, shape in tensors.items()
    }
    if metadata:
        header["__metadata__"] = dict(metadata)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # pad to 8-byte alignment per spec
    pad = (8 - len(raw) % 8) % 8
    raw += b" " * pad
    return len(raw).to_bytes(8, "little") + raw


def _dtype_size(dtype: str) -> int:
    return {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1,
            "F64": 8, "I64": 8}.get(dtype, 4)


# F069 — FP8 Quantization Gate Preserver -----------------------------------------------
def quantize_gains_fp8_e4m3(gains: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate FP8 E4M3 quantization of gain vectors; returns (q, dequant).

    E4M3: 4 exponent bits, 3 mantissa bits, max 448. Gains are O(1) so the
    dynamic range is safe; the returned dequantized vector is what a runtime
    would multiply with — the scorer panel validates quality on THIS vector,
    so there are no surprises after deployment quantization.
    """
    g = np.asarray(gains, dtype=np.float64)
    max_g = float(np.max(np.abs(g))) if g.size else 0.0
    if max_g == 0.0:
        return g.copy(), g.copy()
    # encode to fp32->fp8 scale: use max=448 normalization, 3 mantissa bits
    scale = 448.0 / max_g if max_g > 448.0 else 1.0
    q = g * scale
    # quantize mantissa to 3 bits => step in binade
    sign = np.sign(q)
    a = np.abs(q)
    exp = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    frac = a / np.power(2.0, exp) - 1.0
    step = 1.0 / 8.0
    frac_q = np.round(frac / step) * step
    aq = np.power(2.0, exp) * (1.0 + np.clip(frac_q, 0.0, 7 * step))
    aq = np.where(a > 0, aq, 0.0)
    q = sign * aq
    return q, q / scale


def quantization_penalty(raw: np.ndarray, dequant: np.ndarray) -> float:
    """Max relative error introduced by quantization (gate for F069)."""
    denom = np.maximum(np.abs(raw), 1e-9)
    return float(np.max(np.abs(raw - dequant) / denom))


# F074 — Hot-Swappable Gate Server ----------------------------------------------------------
@dataclass
class GateBundle:
    gate_id: str
    gains: Dict[str, List[float]]   # component -> per-layer gains
    kernel_hash: str


class GateRegistry:
    """Versioned in-memory gate table: swap kernels without reloading
    weights. Model processes read gains per forward pass; a swap is atomic
    (dict pointer replacement) — no in-flight inconsistency."""

    def __init__(self) -> None:
        self._active: Optional[GateBundle] = None
        self._history: List[Tuple[float, str]] = []

    def activate(self, bundle: GateBundle) -> None:
        self._active = bundle
        self._history.append((time.time(), bundle.gate_id))

    def active(self) -> Optional[GateBundle]:
        return self._active

    def swap(self, bundle: GateBundle) -> Optional[str]:
        previous = self._active.gate_id if self._active else None
        self.activate(bundle)
        return previous

    def history(self) -> List[Tuple[float, str]]:
        return list(self._history)


# F075 — ONNX Runtime Serialization (metadata side; graph export needs torch) --------------
def onnx_metadata_props(kernel_spec: str, license_id: str) -> Dict[str, str]:
    """Metadata props to embed in an ONNX ModelProto.metadata_props field."""
    return {
        "calibrix.kernel_spec": kernel_spec,
        "calibrix.license_id": license_id,
        "calibrix.exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# F071 — Apple Metal / F072 — Triton: gain precompute helpers -------------------------------
def tile_gains_uniform(n_layers: int, gains: Sequence[float]) -> np.ndarray:
    """Runtime-friendly layout: (n_layers,) float32 buffer, contiguity
    guaranteed — Metal/Triton kernels consume exactly this layout."""
    arr = np.asarray(gains, dtype=np.float32)
    if arr.shape[0] != n_layers:
        raise ValueError(f"expected {n_layers} gains, got {arr.shape[0]}")
    return np.ascontiguousarray(arr)


def threadgroup_size(n_layers: int, max_size: int = 256) -> int:
    """Metal threadgroup sizing: multiple of 32 <= max_size covering layers."""
    for s in (256, 128, 64, 32):
        if n_layers <= s and s <= max_size:
            return s
    return max_size


# F076 — Edge TPU Quantization Clamp ------------------------------------------------------------
def quantize_gains_int8(gains: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Per-tensor int8 symmetric quantization with scale; clamps to [-127,127]."""
    g = np.asarray(gains, dtype=np.float64)
    scale = np.max(np.abs(g)) / 127.0 if g.size and np.max(np.abs(g)) > 0 else 1.0
    q = np.clip(np.round(g / scale), -127, 127).astype(np.int8)
    return q, float(scale)


# F070 — REST API Endpoint Dispatcher (routing plan; HTTP layer is server.py) ------------------
def build_api_plan(kernel_spec: str, max_concurrency: int = 4
                   ) -> Dict[str, Any]:
    """The contract a deployment endpoint must satisfy to serve this kernel."""
    return {
        "endpoint": "/v1/generate",
        "method": "POST",
        "middleware": ["calibrix.gate", "calibrix.meter"],
        "kernel_spec": kernel_spec,
        "concurrency": max_concurrency,
        "slo_ms": 500,
    }


# F077 — Kubernetes Cluster Auto-Scaler (HPA math) ------------------------------------------------
def desired_replicas(current_replicas: int, queue_depth: int,
                     target_per_pod: int = 4, min_pods: int = 1,
                     max_pods: int = 16) -> int:
    """Horizontal scaling rule: ceil(queue / target), clamped; uses the
    standard HPA tolerance band to avoid flapping."""
    if queue_depth <= 0:
        return min_pods
    want = math.ceil(queue_depth / max(target_per_pod, 1))
    # 10% tolerance band (as in k8s HPA default)
    if abs(want - current_replicas) / max(current_replicas, 1) < 0.10:
        return current_replicas
    return int(min(max(want, min_pods), max_pods))


# F073 — Diffusers Pipeline Adapter (monkeypatch-free hook spec) ------------------------------------
def diffusers_hook_spec(component: str = "transformer") -> Dict[str, Any]:
    """Describe where gains attach in a diffusers pipeline without patching
    source: forward hooks on block modules, gain lookup by block index."""
    return {
        "target_component": component,
        "hook_type": "forward",
        "gain_lookup": "block_index -> gain",
        "supports": ["flux", "sd3", "qwen-image"],
    }


# F067 — vLLM PagedAttention Hook (gain table contract) ----------------------------------------------
def vllm_gain_table(gains: Sequence[float], num_kv_heads: int) -> Dict[str, Any]:
    """Per-layer gain table shaped for a vLLM worker plugin: layer -> tensor."""
    return {
        "format": "per_layer_f32",
        "n_layers": len(gains),
        "num_kv_heads": num_kv_heads,
        "gains": [float(g) for g in gains],
    }


# F068 — TensorRT-LLM Engine Builder (plan spec; compile happens in TRT) ------------------------------
def trt_plan_spec(gains: Sequence[float], precision: str = "fp8") -> Dict[str, Any]:
    """Serialized plan metadata: gains baked as constant weights."""
    return {
        "precision": precision,
        "gains_f32": [float(g) for g in gains],
        "plugin": "calibrix_gate",
        "constant_weights": True,
    }


__all__ = [
    "export_comfyui_workflow", "safetensors_header", "quantize_gains_fp8_e4m3",
    "quantization_penalty", "GateBundle", "GateRegistry", "onnx_metadata_props",
    "tile_gains_uniform", "threadgroup_size", "quantize_gains_int8",
    "build_api_plan", "desired_replicas", "diffusers_hook_spec",
    "vllm_gain_table", "trt_plan_spec",
]
