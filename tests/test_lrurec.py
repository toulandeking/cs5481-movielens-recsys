"""LRURec regression checks: causality, scan, ID mapping and alignment."""

import unittest
from types import SimpleNamespace

import pandas as pd
import torch

from src.adapters.lrurec import histories, inference_tokens, training_pairs
from src.adapters.lrurec_model import LRU, LRULayer


class LRURecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_no_holdout_in_histories(self):
        frame = pd.DataFrame(
            dict(
                uid=[0, 0, 0, 0],
                iid=[7, 4, 8, 9],
                ts=[2, 1, 3, 4],
                split=["train", "train", "val", "test"],
            )
        )
        self.assertEqual(histories(frame)[0], [5, 8])
        self.assertEqual(histories(frame, include_val=True)[0], [5, 8, 9])
        self.assertEqual(inference_tokens([[5, 8]], 4).tolist(), [[0, 0, 5, 8]])

    def test_windows_cover_all_transitions(self):
        sequence = list(range(1, 12))
        x, y = training_pairs([sequence], 4)
        self.assertEqual(y[y > 0].tolist(), sequence[1:])
        self.assertEqual(x[y > 0].tolist(), sequence[:-1])
        self.assertTrue(torch.all(x[y == 0] == 0))

    def test_parallel_scan_matches_sequential(self):
        layer = LRULayer(4, dropout=0).eval()
        x = torch.randn(2, 8, 4)
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [1] * 8]).bool()
        nu, theta, gamma = torch.exp(layer.params_log).split((1, 1, 1))
        lam = torch.exp(torch.complex(-nu, theta))
        b = layer.in_proj(x.to(torch.cfloat)) * gamma
        state = torch.zeros_like(b[:, 0])
        outs = []
        for t in range(8):
            if t > 0:
                state = state * mask[:, t - 1, None]
            state = lam * state + b[:, t]
            outs.append(state)
        expected = layer.layer_norm(layer.out_proj(torch.stack(outs, 1)).real + x)
        torch.testing.assert_close(layer(x, mask), expected, atol=2e-5, rtol=2e-5)

    def test_causality_and_projection(self):
        args = SimpleNamespace(
            num_items=20,
            bert_hidden_units=8,
            bert_num_blocks=2,
            bert_dropout=0.0,
            bert_attn_dropout=0.0,
            dataset_code="ml-1m",
        )
        model = LRU(args).eval()
        x = torch.tensor([[0, 1, 2, 3, 4]])
        changed = x.clone()
        changed[0, -1] = 10
        full = model(x)[0]
        torch.testing.assert_close(full[:, :-1], model(changed)[0][:, :-1])
        torch.testing.assert_close(full[:, -1], model(x, only_last=True)[0])
        labels = torch.tensor([[0, 2, 3, 4, 5]])
        torch.testing.assert_close(full[labels > 0], model(x, labels=labels)[0])
        loss = torch.nn.functional.cross_entropy(
            model(x, labels=labels)[0], labels[labels > 0]
        )
        loss.backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )


if __name__ == "__main__":
    unittest.main()
