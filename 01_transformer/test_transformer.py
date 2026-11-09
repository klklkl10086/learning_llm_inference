"""验证因果性、next-token 数据对齐、梯度和保存加载。"""

import tempfile
import unittest
from pathlib import Path

import torch

from train import get_batch
from transformer import LiteGPT, get_config


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.config = get_config(17)
        self.config.update(d_model=32, n_layers=2, n_heads=4, d_ff=128, max_seq_len=8)
        self.model = LiteGPT(self.config)

    def test_future_tokens_cannot_change_prefix_logits(self):
        self.model.eval()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 8, 9]])
        with torch.no_grad():
            logits_a, _ = self.model(first)
            logits_b, _ = self.model(second)
        torch.testing.assert_close(logits_a[:, :3], logits_b[:, :3])

    def test_training_windows_are_shifted_once(self):
        x, y = get_batch(torch.arange(30), 6, 4, torch.Generator().manual_seed(5), "cpu")
        torch.testing.assert_close(y[:, :-1], x[:, 1:])
        torch.testing.assert_close(y[:, -1], x[:, -1] + 1)

    def test_loss_backpropagates_into_attention_and_embeddings(self):
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        y = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        _, loss = self.model(x, y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in [self.model.token_embedding.weight, self.model.trans[0].atte.Wq.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIs(self.model.lm_head.weight, self.model.token_embedding.weight)

    def test_saved_model_reproduces_logits_and_generation(self):
        self.model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"config": self.config, "model_state_dict": self.model.state_dict()}, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            restored = LiteGPT(checkpoint["config"])
            restored.load_state_dict(checkpoint["model_state_dict"])
            restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(self.model(prompt)[0], restored(prompt)[0])
        generated = self.model.generate(prompt, max_new_tokens=10)
        torch.testing.assert_close(generated, restored.generate(prompt, max_new_tokens=10))
        torch.testing.assert_close(generated[:, :3], prompt)
        self.assertEqual(generated.shape, (1, 13))
        self.assertIs(restored.lm_head.weight, restored.token_embedding.weight)

    def test_context_length_is_checked(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 9, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
