"""验证不同长度请求合批 decode 的数值结果和请求状态边界。"""

import unittest

import torch

from decode_batch import LiteGPT, RequestState, get_config


class DecodeBatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = get_config(23)
        self.config.update(d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=12)
        self.model = LiteGPT(self.config).eval()

    def make_request(self, prompt):
        input_ids = torch.tensor([prompt], dtype=torch.long)
        with torch.no_grad():
            logits, _, past_key_values = self.model(input_ids, use_cache=True)
        request = RequestState(prompt, past_key_values)
        request.append_token(logits[0, -1].argmax().item())
        return request

    def test_different_lengths_match_full_recomputation_across_steps(self):
        requests = [self.make_request([1, 2, 3]), self.make_request([4, 5, 6, 7, 8, 9, 10])]

        with torch.inference_mode():
            for step in range(4):
                # 换顺序后仍须按请求自身的位置和缓存长度计算。
                batch = requests if step % 2 == 0 else requests[::-1]
                old_lengths = [request.cached_len for request in batch]
                expected = torch.cat(
                    [
                        self.model(torch.tensor([request.token_ids]))[0][:, -1:, :]
                        for request in batch
                    ],
                    dim=0,
                )
                actual = self.model.decode_batch(batch)

                self.assertEqual(actual.shape, (2, 1, self.config["vocab_size"]))
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                for request, old_length in zip(batch, old_lengths):
                    self.assertEqual(request.cached_len, old_length + 1)
                    self.assertTrue(
                        all(cache.length == request.cached_len for cache in request.past_key_values)
                    )

                if step < 3:
                    for request, row in zip(batch, actual):
                        request.append_token(row[-1].argmax().item())

    def test_single_request_matches_full_recomputation(self):
        request = self.make_request([1, 2, 3, 4])
        with torch.inference_mode():
            expected = self.model(torch.tensor([request.token_ids]))[0][:, -1:, :]
            actual = self.model.decode_batch([request])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(request.cached_len, 5)

    def test_batch_matches_independent_cached_decode(self):
        prompts = ([1, 2, 3], [4, 5, 6, 7, 8, 9, 10])
        batched = [self.make_request(prompt) for prompt in prompts]
        independent = [self.make_request(prompt) for prompt in prompts]

        with torch.inference_mode():
            actual = self.model.decode_batch(batched)
            expected = torch.cat(
                [
                    self.model(
                        torch.tensor([[request.pending_token]]),
                        past_key_values=request.past_key_values,
                        use_cache=True,
                    )[0]
                    for request in independent
                ],
                dim=0,
            )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual([request.cached_len for request in batched], [4, 8])

    def test_prefill_and_generation_run_inside_inference_mode(self):
        prompts = ([1, 2, 3], [4, 5, 6, 7, 8, 9, 10])
        expected = [list(prompt) for prompt in prompts]

        with torch.inference_mode():
            for _ in range(3):
                for token_ids in expected:
                    logits = self.model(torch.tensor([token_ids]))[0]
                    token_ids.append(logits[0, -1].argmax().item())

            requests = [self.make_request(prompt) for prompt in prompts]
            for _ in range(2):
                logits = self.model.decode_batch(requests)
                for request, row in zip(requests, logits):
                    request.append_token(row[-1].argmax().item())

        self.assertEqual([request.token_ids for request in requests], expected)
        self.assertEqual(
            [request.cached_len for request in requests],
            [len(token_ids) - 1 for token_ids in expected],
        )

    def test_duplicate_request_is_rejected_before_cache_write(self):
        request = self.make_request([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "同一个请求不能重复"):
            self.model.decode_batch([request, request])
        self.assertEqual(request.cached_len, 3)

    def test_shared_layer_cache_is_rejected_before_cache_write(self):
        first = self.make_request([1, 2, 3])
        second = RequestState([1, 2, 3], first.past_key_values)
        second.append_token(first.pending_token)

        with self.assertRaisesRegex(ValueError, "KV cache 被多个请求共用"):
            self.model.decode_batch([first, second])
        self.assertEqual(first.cached_len, 3)

    def test_pending_token_and_capacity_are_checked(self):
        prompt = [1, 2, 3]
        with torch.no_grad():
            _, _, past_key_values = self.model(torch.tensor([prompt]), use_cache=True)
        request = RequestState(prompt, past_key_values)

        with self.assertRaisesRegex(ValueError, "恰有一个尚未计算的token"):
            self.model.decode_batch([request])
        self.assertEqual(request.cached_len, 3)

        full_prompt = list(range(self.config["max_seq_len"]))
        full_request = self.make_request(full_prompt)
        with self.assertRaisesRegex(ValueError, "超过 max_seq_len"):
            self.model.decode_batch([full_request])
        self.assertEqual(full_request.cached_len, self.config["max_seq_len"])

    def test_decode_batch_does_not_build_autograd_graph(self):
        request = self.make_request([1, 2, 3])
        self.assertTrue(torch.is_grad_enabled())

        logits = self.model.decode_batch([request])

        self.assertFalse(logits.requires_grad)
        for cache in request.past_key_values:
            self.assertFalse(cache.k.requires_grad)
            self.assertFalse(cache.v.requires_grad)


if __name__ == "__main__":
    unittest.main()
