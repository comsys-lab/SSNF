"""Official KIVI with whole decoder layers placed on two identical GPUs.

Only exact tensor transfers and device selection are added. KIVI's CUDA GEMV
launches on the current device, so attention hooks explicitly select its device.
Embedding and LM head stay on logical device 0; each layer's cache stays local.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch

from . import ppl_external as external
from .ppl import atomic_json, load_torch, sha_file

DEVICE_MAP = {'model.embed_tokens': 0, 'model.norm': 1, 'lm_head': 0,
              **{f'model.layers.{i}': 0 if i < 16 else 1 for i in range(32)}}
original_sources = external.source_hashes
original_environment = external.environment


def sources(root, method):
    result = original_sources(root, method)
    result[str(Path(__file__).resolve())] = sha_file(Path(__file__))
    return result


def environment():
    result = original_environment()
    result['layer_device_map'] = DEVICE_MAP
    result['gpu_models'] = [torch.cuda.get_device_name(i) for i in range(2)]
    result['placement'] = '16 whole layers per logical GPU; embedding and LM head on logical GPU 0'
    return result


class ShardedAdapter(external.Adapter):
    def __init__(self, root, method):
        from accelerate import dispatch_model
        from transformers import LlamaConfig
        assert method in ('kivi2', 'kivi4') and torch.cuda.device_count() == 2
        assert torch.cuda.get_device_name(0) == torch.cuda.get_device_name(1)
        self.root, self.method, self.dtype = root, method, torch.float16
        self.calls, self.handles = Counter(), []
        self.real_sdpa = external.F.scaled_dot_product_attention
        self.inputs = json.loads((root/'inputs/manifest.json').read_text())
        model_path = self.inputs['model']['snapshot']
        sys.path.insert(0, str(root/'sources/KIVI'))
        sys.path.insert(0, str(root/'sources/KIVI/quant'))
        from models.llama_kivi import LlamaForCausalLM_KIVI
        config = LlamaConfig.from_pretrained(model_path, local_files_only=True)
        config.k_bits = config.v_bits = int(method[-1])
        config.group_size, config.residual_length, config.use_flash = 32, 128, False
        model = LlamaForCausalLM_KIVI.from_pretrained(model_path, config=config,
            torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True,
            attn_implementation='eager').eval()
        self.model = dispatch_model(model, device_map=DEVICE_MAP, main_device=0,
            skip_keys=['past_key_values'])
        self.layers = list(model.model.layers)
        assert len(self.layers) == 32 and model.hf_device_map == DEVICE_MAP
        for i, layer in enumerate(self.layers):
            expected = 0 if i < 16 else 1
            assert layer.self_attn.q_proj.weight.device.index == expected
            def select_device(module, inp):
                torch.cuda.set_device(module.q_proj.weight.device)
            self.handles.append(layer.self_attn.register_forward_pre_hook(select_device))
        def head_device(module, inp):
            torch.cuda.set_device(0)
        self.handles.append(model.lm_head.register_forward_pre_hook(head_device))
        def verify_cache_devices(module, inp, output):
            assert output.logits.device.index == 0
            for i, cache in enumerate(output.past_key_values):
                assert all(t.device.index == (0 if i < 16 else 1) for t in cache if isinstance(t, torch.Tensor))
            torch.cuda.set_device(0)
        self.handles.append(model.register_forward_hook(verify_cache_devices))

    def score(self, inputs, target_starts, heartbeat=None):
        totals, audit = super().score(inputs, target_starts, heartbeat)
        for i in range(2):
            torch.cuda.synchronize(i)
        audit.update(cache_placement_verified_all_layers=True,
                     peak_cuda_bytes_per_logical_device={str(i):torch.cuda.max_memory_allocated(i) for i in range(2)})
        return totals, audit


def validate(root):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    adapter = ShardedAdapter(root, 'kivi2')
    ids = load_torch(root/'inputs/test.pt')['ids'][:256][None].cuda(0)
    totals, audit = adapter.score(ids, [1])
    old = json.loads((root/'smoke/kivi2/summary.json').read_text())
    difference = abs(totals[0]-old['nll'])/255
    assert difference < 1e-6, difference
    assert audit['compressed_cache_reads_all_layers'] > 0
    atomic_json(root/'kivi_sharding_validation.json', dict(passed=True, compared_with='original single-GPU KIVI-2 256-token run',
        target_count=255, mean_nll_difference=difference, tolerance=1e-6, device_map=DEVICE_MAP,
        source_sha256=sha_file(Path(__file__)), audit=audit))
    adapter.close()
    print(json.dumps(dict(passed=True,mean_nll_difference=difference)), flush=True)


external.Adapter = ShardedAdapter
external.source_hashes = sources
external.environment = environment

if __name__ == '__main__':
    if '--validate-sharding' in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument('--root', required=True)
        p.add_argument('--validate-sharding', action='store_true')
        validate(Path(p.parse_args().root).resolve())
    else:
        external.main()
