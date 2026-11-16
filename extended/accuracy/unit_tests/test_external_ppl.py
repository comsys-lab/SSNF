"""Independent causal-target and Fisher-equivalence checks for external adapters."""
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from engine.ppl import canonical_windows
from engine.ppl_external import teacher_forced


class PrefixModel:
    """A deterministic autoregressive model whose logits depend on every past token."""
    def __init__(self):
        self.model = SimpleNamespace(layers=[None, None])
        self.prefix = None
        self.resets = 0

    def __call__(self, inputs, past_key_values=None, use_cache=True, position_ids=None):
        if past_key_values is None:
            self.prefix = inputs
            self.resets += 1
        else:
            self.prefix = torch.cat((self.prefix, inputs), dim=1)
        length = self.prefix.shape[1]
        assert position_ids[0, 0] == length-1
        signal = self.prefix.sum(1) % 11
        logits = -(torch.arange(11)[None]-signal[:, None]).float().square()
        marker = torch.ones(1) if length > 128 else None
        cache = tuple((marker, None, None, None, marker, None, None, None, length) for _ in self.model.layers)
        return SimpleNamespace(logits=logits[:, None], past_key_values=cache)


class ExternalPPLTest(unittest.TestCase):
    def test_corpus_targets_exactly_once_including_short_tail(self):
        for length in (2, 256, 2048, 2049, 3073, 10019):
            windows = canonical_windows(length)
            targets = [i for w in windows for i in range(w['target_start'], w['end'])]
            self.assertEqual(targets, list(range(1, length)))

    def test_teacher_forcing_mask_cache_boundary_and_resets(self):
        ids = torch.arange(2*257).reshape(2, 257) % 11
        starts = [1, 129]
        model = PrefixModel()
        actual, audit = teacher_forced(model, ids, starts)
        expected = []
        for i, start in enumerate(starts):
            total = 0.
            for target in range(start, ids.shape[1]):
                signal = ids[i, :target].sum() % 11
                logits = -(torch.arange(11)-signal).float().square()
                total += F.cross_entropy(logits[None], ids[i, target:target+1]).item()
            expected.append(total)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=1e-6, atol=1e-5)
        self.assertGreater(audit['compressed_cache_reads_all_layers'], 0)
        repeated, _ = teacher_forced(model, ids, starts)
        self.assertEqual(actual, repeated)
        self.assertEqual(model.resets, 2)

    def test_frozen_weight_hook_matches_official_retained_activation_fisher(self):
        torch.manual_seed(17)
        layers = torch.nn.Sequential(torch.nn.Linear(7, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
        inputs = torch.randn(2, 7, requires_grad=True)
        activation = layers[0](inputs)
        activation.retain_grad()
        loss = layers[2](layers[1](activation)).square().mean()
        loss.backward()
        reference = activation.grad.square()
        layers.requires_grad_(False)
        result = []
        def capture(module, inp, output):
            output.register_hook(lambda grad: result.append(grad.square()))
        h = layers[0].register_forward_hook(capture)
        layers(inputs.detach().requires_grad_()).square().mean().backward()
        h.remove()
        torch.testing.assert_close(reference, result[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
