# Plain MXFP8-E4M3 weight-only quant for vLLM — the native-FP twin of MXINT8 (quant_mxint8.py).
# Registers `--quantization mxfp8w`. IDENTICAL integration/vehicle to MXINT8 (packing layout,
# M-adaptive routing); only the element codec differs: per-block E8M0 e4m3 (8.25 b/elem).
# Activation stays bf16; GEMM is bf16 tensor-core (fp8/W8A8 paths are out-of-category peers).
from typing import Any

import torch

import ssnf_cuda  # noqa: F401 — registers torch.ops.ssnf.*
from ssnf_vllm.pack_gpu import pack_weight_mxfp8_gpu

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs


# NOTE: registered as "mxfp8w" (NOT "mxfp8") on purpose — vLLM 0.23 ships a NATIVE "mxfp8"
# quant method (W8A8: vllm::mxfp8_quantize on the activation + vllm::mm_mxfp8 fp8 tensor-core
# GEMM) that hijacks the "mxfp8" name. That native path is the fp8-datapath (activation-quantized),
# the same out-of-category W8A8 we exclude from the block-wise W-only comparison. This config is the
# FAIR twin of MXINT8/SSNF: W-only (activation bf16), bf16-dequant GEMM — same vehicle, codec-only diff.
@register_quantization_config("mxfp8w")
class MXFP8Config(QuantizationConfig):
    """Plain block-scaled MXFP8-E4M3 weight-only quant (8.25 b/elem) — native-FP twin of MXINT8."""

    @classmethod
    def get_name(cls) -> str:
        return "mxfp8w"

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
    def from_config(cls, config: dict[str, Any]) -> "MXFP8Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> "MXFP8LinearMethod | None":
        if isinstance(layer, LinearBase):
            return MXFP8LinearMethod(self)
        return None


class MXFP8LinearMethod(LinearMethodBase):
    def __init__(self, cfg: MXFP8Config) -> None:
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
        layer.mxf_K = input_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        W = layer.weight.data  # [OUT, K] bf16 cuda
        OUT, K = int(W.shape[0]), int(W.shape[1])
        p = pack_weight_mxfp8_gpu(W)
        layer.mxf_s = p["scale_exp"]                           # int8 [nb, OUT]
        layer.mxf_qw = p["qweight_cm"]                         # int8 [nb, OUT, 32] (e4m3 bytes)
        layer.mxf_OUT, layer.mxf_K, layer.mxf_nb = OUT, K, int(p["nb"])
        layer.weight.data = torch.empty(0, device=W.device, dtype=W.dtype)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        x2d = x.reshape(-1, layer.mxf_K)
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        x2d = x2d.contiguous()
        M = int(x2d.shape[0])
        # Same M-adaptive routing as MXINT8 (default path, no Triton): large-M -> dequant+cuBLAS,
        # small-M -> skinny fused bf16-dequant TC-GEMM.
        # measured per shape instead of inheriting
        # the SSNF thresholds (192/256), which were never measured for this format. The skinny->dequant
        # crossover is qkv 128, o_proj 80, gate_up 96, down 80, so the old rule kept the FFN shapes on
        # the slower skinny path for the whole M in [96,192) band.
        thr = 96 if max(layer.mxf_OUT, layer.mxf_K) >= 8192 else 128
        if M >= thr:
            Wt = torch.ops.ssnf.mxfp8_dequant_bf16(layer.mxf_s, layer.mxf_qw,
                                                   layer.mxf_OUT, layer.mxf_K, layer.mxf_nb)
            y = torch.matmul(x2d, Wt)
        else:
            y = torch.ops.ssnf.mxfp8_gemm_fused_skinny(x2d, layer.mxf_s, layer.mxf_qw,
                                                       M, layer.mxf_OUT, layer.mxf_K, layer.mxf_nb)
        y = y.reshape(*x.shape[:-1], layer.mxf_OUT)
        if bias is not None:
            y = y + bias
        return y
