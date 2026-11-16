"""Only stage uncompressed checkpoint weights on CPU; preserve GPU packing/apply."""
import json
from pathlib import Path
import torch


def install(directory):
    from ssnf_vllm.quant import SSNFLinearMethod
    from ssnf_vllm.quant_mxfp6 import MXFP6LinearMethod
    audit = []
    output = Path(directory) / 'layer_loading_audit.json'

    def patch(cls):
        original_create = cls.create_weights
        original_process = cls.process_weights_after_loading

        def create(self, *args, **kwargs):
            with torch.device('cpu'):
                return original_create(self, *args, **kwargs)

        def process(self, layer):
            # vLLM device_loading_context has moved this single layer to CUDA.
            assert layer.weight.device.type == 'cuda'
            shape = list(layer.weight.shape)
            # Original code warms M=16,32,64,128,256,512 for every layer.
            # Keep the packer unchanged; warm only actual run sizes in run_trial.
            from ssnf_vllm import quant
            triton_before = quant._SSNF_TRITON
            if cls is SSNFLinearMethod:
                quant._SSNF_TRITON = False
            try:
                result = original_process(self, layer)
            finally:
                quant._SSNF_TRITON = triton_before
            fields = ['ssnf_s', 'ssnf_hc', 'ssnf_lgc'] if cls is SSNFLinearMethod else ['mx6_s', 'mx6_qw']
            assert all(getattr(layer, x).device.type == 'cuda' for x in fields)
            assert layer.weight.numel() == 0
            audit.append(dict(method=cls.__name__, prefix=getattr(self, 'prefix', ''), shape=shape,
                              packed_fields={x:dict(shape=list(getattr(layer,x).shape), bytes=getattr(layer,x).numel()*getattr(layer,x).element_size()) for x in fields}))
            output.write_text(json.dumps(dict(policy='CPU checkpoint staging; original GPU packers and inference apply methods unchanged; defer load-time autotuning to actual-size run warmups; no inference CPU offload', layers=audit), indent=2)+'\n')
            return result

        cls.create_weights = create
        cls.process_weights_after_loading = process

    patch(SSNFLinearMethod)
    patch(MXFP6LinearMethod)
