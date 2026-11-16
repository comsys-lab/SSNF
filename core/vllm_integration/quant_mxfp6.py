# Plain MXFP6-E2M3 weight-only quant for vLLM — native-FP twin of MXINT8/MXFP8 (quant_mxfp8.py).
# Registers `--quantization mxfp6w`. Same integration/vehicle as MXFP8w (W-only, bf16-dequant GEMM,
# M-adaptive routing); only the element codec differs: per-block E8M0 e2m3 6-bit codes
# (6.25 b/elem, 24 B/32-block). Activation stays bf16; GEMM is bf16 tensor-core.
from typing import Any

import torch

import ssnf_cuda  # noqa: F401 — registers torch.ops.ssnf.*
from ssnf_vllm.pack_gpu import pack_weight_mxfp6_gpu

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs


@register_quantization_config("mxfp6w")
class MXFP6Config(QuantizationConfig):
    """Plain block-scaled MXFP6-E2M3 weight-only quant (6.25 b/elem) — native-FP twin of MXINT8."""

    @classmethod
    def get_name(cls) -> str:
        return "mxfp6w"

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
    def from_config(cls, config: dict[str, Any]) -> "MXFP6Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> "MXFP6LinearMethod | None":
        if isinstance(layer, LinearBase):
            return MXFP6LinearMethod(self)
        return None


class MXFP6LinearMethod(LinearMethodBase):
    def __init__(self, cfg: MXFP6Config) -> None:
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
        layer.mx6_K = input_size_per_partition

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        W = layer.weight.data  # [OUT, K] bf16 cuda
        OUT, K = int(W.shape[0]), int(W.shape[1])
        p = pack_weight_mxfp6_gpu(W)
        layer.mx6_s = p["scale_exp"]                           # int8 [nb, OUT]
        layer.mx6_qw = p["qweight_cm"]                         # int8 [nb, OUT, 24] (packed 6-bit)
        layer.mx6_OUT, layer.mx6_K, layer.mx6_nb = OUT, K, int(p["nb"])
        layer.weight.data = torch.empty(0, device=W.device, dtype=W.dtype)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        x2d = x.reshape(-1, layer.mx6_K)
        if x2d.dtype != torch.bfloat16:
            x2d = x2d.to(torch.bfloat16)
        x2d = x2d.contiguous()
        M = int(x2d.shape[0])
        # measured per shape, same crossovers as
        # MXFP8 (qkv 128, o_proj 80, gate_up 96, down 80). The inherited 192/256 rule left MXFP6 on the
        # slower skinny path at its own 2k decode batch (M=106), costing ~4.6% of its weight-GEMM time.
        thr = 96 if max(layer.mx6_OUT, layer.mx6_K) >= 8192 else 128
        if M >= thr:
            Wt = torch.ops.ssnf.mxfp6_dequant_bf16(layer.mx6_s, layer.mx6_qw,
                                                   layer.mx6_OUT, layer.mx6_K, layer.mx6_nb)
            y = torch.matmul(x2d, Wt)
        else:
            y = torch.ops.ssnf.mxfp6_gemm_fused_skinny(x2d, layer.mx6_s, layer.mx6_qw,
                                                       M, layer.mx6_OUT, layer.mx6_K, layer.mx6_nb)
        y = y.reshape(*x.shape[:-1], layer.mx6_OUT)
        if bias is not None:
            y = y + bias
        return y
