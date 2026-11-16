# Plain MXINT8 weight-only quant for vLLM — the FAIR baseline twin of SSNF (ssnf_vllm/quant.py).
# Registers `--quantization mxint8`. Identical integration/vehicle to SSNF (packing, M-adaptive
# routing); only the format differs: straight per-block E8M0 int8 (no upper/shared split, no
# u/gs), making SSNF-vs-MXINT8 an "element-handling only" comparison (for_fair_comparison.md).
import os
from typing import Any

import torch

import ssnf_cuda  # noqa: F401 — registers torch.ops.ssnf.*
from ssnf_vllm.pack_gpu import pack_weight_mxint8_gpu

_SSNF_TRITON = os.getenv("SSNF_TRITON", "1") != "0"   # Triton fused medium-M GEMM (fair SSNF twin).
# Default ON so MXINT8 runs its own fastest weight path, matching quant.py (SSNF); a paired E2E
# probe measured ON faster than OFF at every context length.

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs


@register_quantization_config("mxint8")
class MXINT8Config(QuantizationConfig):
    """Plain block-scaled MXINT8 weight-only quant (8.25 b/elem) — SSNF's fair baseline."""

    @classmethod
    def get_name(cls) -> str:
        return "mxint8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 120

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MXINT8Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> "MXINT8LinearMethod | None":
        if isinstance(layer, LinearBase):
            return MXINT8LinearMethod(self)
        return None


class MXINT8LinearMethod(LinearMethodBase):
    def __init__(self, cfg: MXINT8Config) -> None:
        self.cfg = cfg

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs) -> None:
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype),
            input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.mxi_K = input_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        W = layer.weight.data  # [OUT, K] bf16 cuda
        OUT, K = int(W.shape[0]), int(W.shape[1])
        p = pack_weight_mxint8_gpu(W)                          # on-GPU, byte-identical to numpy twin
        layer.mxi_s = p["scale_exp"]                           # int8 [nb, OUT]
        layer.mxi_qw = p["qweight_cm"]                         # int8 [nb, OUT, 32]
        layer.mxi_OUT, layer.mxi_K, layer.mxi_nb = OUT, K, int(p["nb"])
        layer.weight.data = torch.empty(0, device=W.device, dtype=W.dtype)
        # #2a fair twin: warm the Triton autotune at load (no serving first-call stall).
        if _SSNF_TRITON:
            try:
                from ssnf_vllm.ssnf_triton import mxint8_gemm
                for mm in (16, 32, 64, 128, 256, 512):   # one per _mbucket band 
                    xw = torch.zeros(mm, K, dtype=torch.bfloat16, device=W.device)
                    mxint8_gemm(xw, layer.mxi_s, layer.mxi_qw, mm, OUT, K, layer.mxi_nb)
            except Exception:
                pass

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        x2d = x.reshape(-1, layer.mxi_K)
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        x2d = x2d.contiguous()
        M = int(x2d.shape[0])
        # Same M-adaptive routing as SSNF (quant.py): large-M -> dequant+cuBLAS, small-M -> skinny.
        # Shape-aware threshold mirrors SSNF (crossover re-measurable per format for a fair sweep).
        thr = 192 if max(layer.mxi_OUT, layer.mxi_K) >= 8192 else 256
        # Fair twin: Triton fused int8 mixed-input (no materialize) — same band as SSNF (quant.py)
        # for an apples-to-apples W+KV comparison. The fused Triton path is faster than the skinny
        # kernel below M=64 and than dequant+cuBLAS up to M=8192, so it takes the whole band.
        if _SSNF_TRITON and 1 <= M < 8192:
            from ssnf_vllm.ssnf_triton import mxint8_gemm
            y = mxint8_gemm(x2d, layer.mxi_s, layer.mxi_qw, M, layer.mxi_OUT, layer.mxi_K, layer.mxi_nb)
        elif M >= thr:
            # mxint8_dequant_bf16 returns [K, OUT] (already transposed / matmul-ready,
            # unlike SSNF's wonly_dequant_bf16 which is [OUT, K]) -> no .t().
            Wt = torch.ops.ssnf.mxint8_dequant_bf16(layer.mxi_s, layer.mxi_qw,
                                                    layer.mxi_OUT, layer.mxi_K, layer.mxi_nb)
            y = torch.matmul(x2d, Wt)
        else:
            y = torch.ops.ssnf.mxint8_gemm_fused_skinny(x2d, layer.mxi_s, layer.mxi_qw,
                                                        M, layer.mxi_OUT, layer.mxi_K, layer.mxi_nb)
        y = y.reshape(*x.shape[:-1], layer.mxi_OUT)
        if bias is not None:
            y = y + bias
        return y
