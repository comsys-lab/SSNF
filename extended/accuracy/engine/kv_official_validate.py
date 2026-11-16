"""Real-GPU equivalence checks for KIVI's compatibility and packed kernels."""
import argparse
import gc
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from .ppl import atomic_json, load_torch, sha_file
from .ppl_external import Adapter


@torch.inference_mode()
def validate(root):
    sys.path.insert(0, str(root/'sources/KIVI'))
    sys.path.insert(0, str(root/'sources/KIVI/quant'))
    from quant.new_pack import triton_quantize_and_pack_along_last_dim, unpack_and_dequant_vcache
    from quant.matmul import cuda_bmm_fA_qB_outer
    torch.manual_seed(32)
    checks = []
    for bits in (2, 4):
        for length in (1, 7, 31, 127, 128, 383, 1919):
            value = torch.randn(2, 32, length, 128, device='cuda', dtype=torch.float16)
            a = torch.randn(2, 32, 1, length, device='cuda', dtype=torch.float16)/max(length, 1)
            packed, scale, zero = triton_quantize_and_pack_along_last_dim(value, 32, bits)
            reconstructed = unpack_and_dequant_vcache(packed, scale.unsqueeze(-1), zero.unsqueeze(-1), 32, bits)
            reference = a @ reconstructed
            actual = cuda_bmm_fA_qB_outer(32, a, packed, scale, zero, bits)
            # CUDA accumulates in FP32; the reference reconstructs in FP16 first.
            error = (actual-reference).abs().max().item()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, rtol=.005, atol=.003)
            checks.append(dict(bits=bits, value_cache_length=length, max_abs_error=error, rtol=.005, atol=.003))
    del value, a, packed, scale, zero, reconstructed, reference, actual
    gc.collect(); torch.cuda.empty_cache()
    ids = load_torch(root/'inputs/test.pt')['ids'][:128][None].cuda()
    adapter = Adapter(root, 'kivi2')
    losses = []
    cache = None
    for t in range(127):
        out = adapter.model(ids[:, t:t+1], past_key_values=cache, use_cache=True,
            position_ids=torch.tensor([[t]], device='cuda'))
        cache = out.past_key_values
        assert all(c[0] is None and c[4] is None for c in cache)
        losses.append(F.cross_entropy(out.logits[:, -1].float(), ids[:, t+1], reduction='none'))
    kivi = torch.cat(losses).cpu()
    adapter.close()
    del adapter, out, cache, losses
    gc.collect(); torch.cuda.empty_cache()
    reference_adapter = Adapter(root, 'fp16')
    logits = reference_adapter.model(ids, use_cache=False).logits[:, :-1].float()
    reference = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction='none').cpu().reshape(-1)
    mean_difference = abs((kivi-reference).double().mean().item())
    assert mean_difference < .002, mean_difference
    atomic_json(root/'kivi_validation.json', dict(passed=True, packed_value_checks=checks,
        unquantized_prefix=dict(tokens=128, targets=127, mean_nll_kivi=kivi.double().mean().item(),
            mean_nll_fp16=reference.double().mean().item(), absolute_mean_nll_difference=mean_difference,
            mean_nll_tolerance=.002, same_frozen_tokens=True, compressed_cache_absent_all_layers=True),
        validation_source_sha256=sha_file(Path(__file__))))
    print(json.dumps(dict(passed=True, packed_kernel_checks=len(checks), prefix_mean_nll_difference=mean_difference)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    torch.set_num_threads(4)
    validate(Path(p.parse_args().root).resolve())
