# SSNF weight-only quantization for vLLM 0.23 (V1). Registers `--quantization ssnf`.
# Weights load bf16, are packed to the SSNF u3/gs16 planes (scale_exp / upper_cm / shared_cm)
# in process_weights_after_loading (bf16 freed); apply() routes M across the fused GEMM paths.
# Config: u=3, gs=16 (~5.44 b/elem). Activation stays bf16 (no W+A here).
import os
import re
from typing import Any

import torch

import ssnf_cuda  # noqa: F401  — registers torch.ops.ssnf.*

_SSNF_TRITON = os.getenv("SSNF_TRITON", "1") != "0"   # Triton fused medium-M GEMM (default ON —
# faster than the skinny kernel over the whole medium-M band in decode-heavy serving).
# SSNF_TRITON=0 disables. If the Triton import fails at load, _TRITON_OK falls back to skinny (no crash).
_TRITON_OK = True
_SSNF_RA = os.getenv("SSNF_RA", "0") != "0"           # register-aligned weight (6.5b, straddle-free RA kernel, all M)
from ssnf_vllm.pack_gpu import pack_weight_ssnf_gpu  # on-GPU packer (byte-identical to ssnf_lib.pack.pack_weight, ~86x faster)

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

SSNF_LB = 3
SSNF_G_SIZE = 16

# ---------------------------------------------------------------------------
# Encoder = greedy+H (importance-weighted shared + radius search), the deployed
# weight encoder. Free accuracy win over plain (all-u3/gs16 +4.08% -> +1.85%);
# format/kernel/decode 100% unchanged (encoder is offline packing only).
# ---------------------------------------------------------------------------
# radius: candidates searched around the H-weighted shared mean (per-candidate upper
# re-round). Deploy default = 1<<u (full range, matches the measured greedy+H).
_SSNF_RADIUS = int(os.getenv("SSNF_RADIUS", "0"))          # 0 -> use 1<<u per layer (full search)
# H (importance E[x^2]) file: torch.load -> {"<layer>.<role>": tensor[K]}, role in
# {qkv,o,gu,down}. Absent -> uniform greedy (search-only; still beats plain). Build with
# ssnf_vllm/calibrate_h.py. Keyed by _hkey so vLLM-fused prefixes map to HF calib names.
_SSNF_HFILE = os.getenv("SSNF_HFILE", "")
# Per-layer bit config. Default = uniform u3/gs16 (accuracy-per-bit champion, +1.85%,
# runs the tuned u3/gs16 RA/Triton kernels). SSNF_PROFILE=mixed = attn u4/gs4 nibble +
# gate/up u3/gs16 + down u2/gs16 (5.66b, +1.95%; for the nibble-kernel path).
_SSNF_PROFILE = os.getenv("SSNF_PROFILE", "uniform")
_PROFILES = {
    "uniform": {"qkv": (3, 16), "o": (3, 16), "gu": (3, 16), "down": (3, 16)},
    "mixed":   {"qkv": (4, 4),  "o": (4, 4),  "gu": (3, 16), "down": (2, 16)},
    # gs8: the W+KV accuracy-gate config (W u3/gs8 +H = the passing frontier; measured:
    # gs16 uniform +4.33% FAIL / gs8 uniform +3.74% FAIL / gs8+H +2.69% PASS). Needs SSNF_HFILE.
    "gs8":     {"qkv": (3, 8),  "o": (3, 8),  "gu": (3, 8),  "down": (3, 8)},
}


def _hkey(prefix: str) -> str:
    """Normalize a Linear's name to '<layer_idx>.<role>' (matches the HF-side calibrator keys)."""
    m = re.search(r"layers\.(\d+)\.", prefix)
    li = m.group(1) if m else "x"
    if "qkv_proj" in prefix or "q_proj" in prefix or "k_proj" in prefix or "v_proj" in prefix:
        role = "qkv"
    elif "o_proj" in prefix:
        role = "o"
    elif "gate_up_proj" in prefix or "gate_proj" in prefix or "up_proj" in prefix:
        role = "gu"
    elif "down_proj" in prefix:
        role = "down"
    else:
        role = "other"
    return f"{li}.{role}"


def _role(prefix: str) -> str:
    return _hkey(prefix).split(".", 1)[1]


_H_CACHE = None


def _load_H() -> dict[str, torch.Tensor]:
    global _H_CACHE
    if _H_CACHE is None:
        _H_CACHE = torch.load(_SSNF_HFILE, map_location="cpu") if _SSNF_HFILE and os.path.exists(_SSNF_HFILE) else {}
    return _H_CACHE


@register_quantization_config("ssnf")
class SSNFConfig(QuantizationConfig):
    """SSNF mantissa-shared weight-only quant (u3/gs16)."""

    def __init__(self, l_b: int = SSNF_LB, g_size: int = SSNF_G_SIZE) -> None:
        super().__init__()
        self.l_b = l_b
        self.g_size = g_size

    @classmethod
    def get_name(cls) -> str:
        return "ssnf"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120  # sm_120 (Blackwell) — kernels are gencode'd for it

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SSNFConfig":
        return cls(l_b=config.get("l_b", SSNF_LB), g_size=config.get("g_size", SSNF_G_SIZE))

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase | None:
        # Quantize ordinary Linear layers only; embeddings / lm_head (not a
        # LinearBase) fall through to vLLM's default unquantized path.
        if isinstance(layer, LinearBase):
            return SSNFLinearMethod(self, prefix)
        return None


class SSNFLinearMethod(LinearMethodBase):
    def __init__(self, cfg: SSNFConfig, prefix: str = "") -> None:
        self.cfg = cfg
        self.prefix = prefix

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Load as a plain bf16 weight (mirrors UnquantizedLinearMethod); we pack
        # it to SSNF after the checkpoint has streamed in.
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.ssnf_K = input_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        W = layer.weight.data  # [OUT, K] bf16, cuda
        OUT, K = int(W.shape[0]), int(W.shape[1])
        # Per-layer bit config (role from prefix); default profile = uniform u3/gs16.
        role = _role(self.prefix)
        l_b, g_size = _PROFILES.get(_SSNF_PROFILE, _PROFILES["uniform"]).get(role, (self.cfg.l_b, self.cfg.g_size))
        layer.ssnf_l_b, layer.ssnf_g_size = l_b, g_size
        # Encoder = greedy+H: importance H (E[x^2], per-input-channel) from the
        # calib file, keyed so vLLM's fused qkv/gate_up map to their shared input. Absent -> None
        # = uniform greedy (search-only). radius = full range (1<<u) unless SSNF_RADIUS overrides.
        Hd = _load_H().get(_hkey(self.prefix))
        H = Hd.to(W.device).float() if Hd is not None else None
        radius = _SSNF_RADIUS if _SSNF_RADIUS > 0 else (1 << l_b)
        # Pack on-GPU (greedy+H planes are format-identical to plain; ~86x faster than numpy).
        p = pack_weight_ssnf_gpu(W, l_b, g_size, H=H, radius=radius)
        layer.ssnf_s = p["scale_exp"]                                  # int8 [nb, OUT] (cuda)
        layer.ssnf_hc = p["upper_cm"]                                 # uint8 (cuda)
        layer.ssnf_lgc = p["shared_cm"]                                # uint8 (cuda)
        layer.ssnf_OUT, layer.ssnf_K, layer.ssnf_nb = OUT, K, int(p["nb"])
        # free the bf16 master copy
        layer.weight.data = torch.empty(0, device=W.device, dtype=W.dtype)
        # #2: warm the Triton autotune HERE (at load) so serving has no first-call stall.
        # autotune keys on (OUT,K) -> one tune per weight shape, cached for all later M.
        _br = None
        if _SSNF_RA and l_b == 3 and g_size in (8, 16):
            # register-aligned: repack upper (6 codes/word, straddle-free, 5.44->6.5b), FREE dense upper.
            # Upper plane is group-size-independent; the shared field (gs8 -> 2 bytes/col) is handled
            # by the SB-aware kernel, so gs8 rides the same RA path.
            from ssnf_vllm.ssnf_triton import pack_upper_ra, ssnf_gemm_ra
            layer.ssnf_ra = pack_upper_ra(layer.ssnf_hc)                  # int32 [nb, OUT, 6]
            layer.ssnf_hc = torch.empty(0, device=W.device, dtype=torch.uint8)   # free dense (RA path only)
            try:
                for mm in (128, 256, 512):
                    xw = torch.zeros(mm, K, dtype=torch.bfloat16, device=W.device)
                    ssnf_gemm_ra(xw, layer.ssnf_s, layer.ssnf_ra, layer.ssnf_lgc, mm, OUT, K, layer.ssnf_nb, g_size=g_size)
            except Exception:
                pass
        elif _SSNF_TRITON and l_b == 3 and g_size in (8, 16):
            # Warm the SAME entry apply() uses (ssnf_gemm_best: rw + split-K kernels) at decode-like
            # Ms. Previously only ssnf_gemm(rw) was warmed -> the split-K kernel autotuned (30+
            # configs) DURING the first serving decode step. M=64 added: covers the low decode band.
            global _TRITON_OK
            try:
                from ssnf_vllm.ssnf_triton import ssnf_gemm_best
                for mm in (16, 32, 64, 128, 256, 512):   # one per _mbucket band : the autotune
                    # key now includes the M bucket, so every decode band must be warmed at load time
                    # or it would autotune during the first serving step in that band.
                    xw = torch.zeros(mm, K, dtype=torch.bfloat16, device=W.device)
                    ssnf_gemm_best(xw, layer.ssnf_s, layer.ssnf_hc, layer.ssnf_lgc, mm, OUT, K, layer.ssnf_nb, g_size=g_size)
            except Exception:
                _TRITON_OK = False   # fall back to skinny/dequant in apply (no crash)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        x2d = x.reshape(-1, layer.ssnf_K)
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        x2d = x2d.contiguous()
        M = int(x2d.shape[0])
        # Route M to whichever path is faster (both bit-exact; wonly_dequant_bf16 rel<=2e-3).
        # dequant+cuBLAS overtakes the skinny quant GEMM at a shape-dependent M crossover:
        # 192 for the large FFN matrices, 256 for the square attention projections.
        thr = 192 if max(layer.ssnf_OUT, layer.ssnf_K) >= 8192 else 256
        l_b, g_size = layer.ssnf_l_b, layer.ssnf_g_size
        # SSNF_TRITON: fused Triton mixed-input GEMM (no bf16 materialize) for the medium-M valley
        # (M∈[64,1536]): faster there than both the skinny kernel and dequant->cuBLAS.
        # u3/gs16 only; keeps dequant->cuBLAS for large M (re-unpack loses there), skinny for tiny M.
        _br = None
        if _SSNF_RA and l_b == 3 and g_size in (8, 16):
            # register-aligned RA kernel for ALL M (dense upper was freed). straddle-free.
            from ssnf_vllm.ssnf_triton import ssnf_gemm_ra
            _br = "ra"
            y = ssnf_gemm_ra(x2d, layer.ssnf_s, layer.ssnf_ra, layer.ssnf_lgc,
                             M, layer.ssnf_OUT, layer.ssnf_K, layer.ssnf_nb, g_size=g_size)
        elif _SSNF_TRITON and _TRITON_OK and l_b == 3 and g_size in (8, 16) and (1 if g_size == 8 else 64) <= M < 1536:
            # ssnf_gemm_best: shape-aware split-K (fills SMs on block-starved small-OUT/large-K
            # shapes: down_proj/o_proj/qkv). Large-OUT (gate_up) -> plain. SSNF_NO_SPLITK=1 disables.
            # gs8: the gs8 skinny kernel is slower at small M, so gs8 routes Triton for all
            # M<1536 and low-batch long-context decode stays fast.
            from ssnf_vllm.ssnf_triton import ssnf_gemm, ssnf_gemm_best
            fn = ssnf_gemm if os.getenv("SSNF_NO_SPLITK", "0") != "0" else ssnf_gemm_best
            _br = "triton"
            y = fn(x2d, layer.ssnf_s, layer.ssnf_hc, layer.ssnf_lgc,
                   M, layer.ssnf_OUT, layer.ssnf_K, layer.ssnf_nb, g_size=g_size)
        elif M >= thr:
            _br = "dequant"
            W = torch.ops.ssnf.wonly_dequant_bf16(
                layer.ssnf_s, layer.ssnf_hc, layer.ssnf_lgc,
                layer.ssnf_OUT, layer.ssnf_K, layer.ssnf_nb, l_b, g_size,
            )
            y = torch.matmul(x2d, W.t())
        else:
            _br = "skinny"
            y = torch.ops.ssnf.wonly_gemm_tc(
                x2d, layer.ssnf_s, layer.ssnf_hc, layer.ssnf_lgc,
                M, layer.ssnf_OUT, layer.ssnf_K, layer.ssnf_nb, l_b, g_size,
            )
        y = y.reshape(*x.shape[:-1], layer.ssnf_OUT)
        if bias is not None:
            y = y + bias
        return y
