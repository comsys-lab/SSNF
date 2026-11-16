# MXFP4 / NVFP4 weight-only FAKE-quant for vLLM (bf16 store, bf16 GEMM) — accuracy foils, no CUDA.
# Registers `--quantization mxfp4` and `--quantization nvfp4`. process_weights_after_loading rounds each
# Linear weight to the format and keeps it bf16, so the GEMM is a plain bf16 matmul on rounded weights.
# KV is fake-quantized separately by a forward hook (kv_fakequant.py). Pair: SSNF_KV_FAKEQUANT=mxfp4|nvfp4.
from typing import Any

import torch

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from ssnf_vllm.fp4_quant import QUANTIZERS


class _FP4LinearMethod(LinearMethodBase):
    def __init__(self, fmt: str) -> None:
        self.q = QUANTIZERS[fmt]

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int, output_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs) -> None:
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype),
            input_dim=1, output_dim=0, weight_loader=weight_loader)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = self.q(layer.weight.data)      # fake-quant in place, stays bf16

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        return torch.nn.functional.linear(x, layer.weight, bias)


def _mk(fmt: str) -> type[QuantizationConfig]:
    class _Cfg(QuantizationConfig):
        @classmethod
        def get_name(cls) -> str:
            return fmt
        @classmethod
        def get_supported_act_dtypes(cls) -> list[torch.dtype]:
            return [torch.bfloat16, torch.float16]
        @classmethod
        def get_min_capability(cls) -> int:
            return 70
        @staticmethod
        def get_config_filenames() -> list[str]:
            return []
        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "_Cfg":
            return cls()
        def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> LinearMethodBase | None:
            if isinstance(layer, LinearBase):
                return _FP4LinearMethod(fmt)
            return None
    _Cfg.__name__ = f"{fmt.upper()}Config"
    return register_quantization_config(fmt)(_Cfg)


MXFP4Config = _mk("mxfp4")
NVFP4Config = _mk("nvfp4")
