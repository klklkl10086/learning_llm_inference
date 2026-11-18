"""Measure variable-length batched decode against independent cached decode."""

import argparse
import statistics
import time

import torch

from decode_batch import LiteGPT, RequestState, get_config


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_requests(model, prompts, prompt_ids, decode_ids):
    requests = []
    for prompt, ids, future in zip(prompts, prompt_ids, decode_ids):
        _, _, past_key_values = model(prompt, use_cache=True)
        request = RequestState(ids, past_key_values)
        request.append_token(future[0])
        requests.append(request)
    return requests


def decode_step(model, requests, mode, device):
    if mode == "batched":
        return model.decode_batch(requests)
    return torch.cat(
        [
            model(
                torch.tensor([[request.pending_token]], dtype=torch.long, device=device),
                past_key_values=request.past_key_values,
                use_cache=True,
            )[0]
            for request in requests
        ],
        dim=0,
    )


@torch.inference_mode()
def check_correctness(model, prompts, prompt_ids, decode_ids, steps, device):
    batched = make_requests(model, prompts, prompt_ids, decode_ids)
    independent = make_requests(model, prompts, prompt_ids, decode_ids)
    max_diff = 0.0
    for step in range(steps):
        actual = decode_step(model, batched, "batched", device)
        expected = decode_step(model, independent, "independent", device)
        max_diff = max(max_diff, (actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        if step + 1 < steps:
            for row, request in enumerate(batched):
                request.append_token(decode_ids[row][step + 1])
                independent[row].append_token(decode_ids[row][step + 1])
    return max_diff


@torch.inference_mode()
def measure_once(model, prompts, prompt_ids, decode_ids, steps, mode, device):
    sync(device)
    start = time.perf_counter()
    requests = make_requests(model, prompts, prompt_ids, decode_ids)
    sync(device)
    prefill_ms = (time.perf_counter() - start) * 1000

    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for step in range(steps):
        decode_step(model, requests, mode, device)
        if step + 1 < steps:
            for row, request in enumerate(requests):
                request.append_token(decode_ids[row][step + 1])
    sync(device)
    decode_ms = (time.perf_counter() - start) * 1000

    layers = [layer for request in requests for layer in request.past_key_values]
    allocated_bytes = sum(
        tensor.numel() * tensor.element_size()
        for layer in layers for tensor in (layer.k, layer.v)
    )
    valid_bytes = sum(
        tensor[:, :, :layer.length].numel() * tensor.element_size()
        for layer in layers for tensor in (layer.k, layer.v)
    )
    peak_extra_mib = (
        (torch.cuda.max_memory_allocated(device) - baseline) / 1024**2
        if device.type == "cuda" else None
    )
    return prefill_ms, decode_ms, allocated_bytes / 1024**2, valid_bytes / 1024**2, peak_extra_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prompt-lens", default="32,64,96,128")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    try:
        prompt_lens = [int(value) for value in args.prompt_lens.split(",")]
    except ValueError:
        parser.error("prompt-lens 应为逗号分隔的正整数，例如 32,64,96")
    if not prompt_lens or min(prompt_lens) < 1:
        parser.error("prompt-lens 中的每个长度必须为正数")
    if args.decode_steps < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("decode-steps、repeats 必须为正数；warmup 不能为负数")

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("当前没有可用的 CUDA 设备")

    torch.set_num_threads(4)
    torch.manual_seed(2026)
    config = get_config(vocab_size=330)
    if max(prompt_lens) + args.decode_steps > config["max_seq_len"]:
        parser.error(f"最长 prompt + decode-steps 不能超过 {config['max_seq_len']}")
    model = LiteGPT(config).to(device).eval()

    prompt_ids = [
        torch.randint(config["vocab_size"], (length,)).tolist()
        for length in prompt_lens
    ]
    prompts = [torch.tensor([ids], dtype=torch.long, device=device) for ids in prompt_ids]
    decode_ids = [
        torch.randint(config["vocab_size"], (args.decode_steps,)).tolist()
        for _ in prompt_lens
    ]

    max_diff = check_correctness(model, prompts, prompt_ids, decode_ids, args.decode_steps, device)
    print(
        f"device={device}, dtype=float32, prompts={prompt_lens}, steps={args.decode_steps}, "
        f"warmup={args.warmup}, repeats={args.repeats}",
        flush=True,
    )
    print(f"correctness: PASS; max_abs_logit_diff={max_diff:.3g}", flush=True)

    for _ in range(args.warmup):
        for mode in ("independent", "batched"):
            measure_once(model, prompts, prompt_ids, decode_ids, args.decode_steps, mode, device)

    samples = {"independent": [], "batched": []}
    for repeat in range(args.repeats):
        order = ("independent", "batched") if repeat % 2 == 0 else ("batched", "independent")
        for mode in order:
            samples[mode].append(
                measure_once(model, prompts, prompt_ids, decode_ids, args.decode_steps, mode, device)
            )

    for mode in ("independent", "batched"):
        prefill, decode, allocated, valid = (
            statistics.median(sample[index] for sample in samples[mode])
            for index in range(4)
        )
        peak = (
            statistics.median(sample[4] for sample in samples[mode])
            if device.type == "cuda" else None
        )
        throughput = len(prompt_lens) * args.decode_steps / (decode / 1000)
        print(
            f"{mode}: prefill={prefill:.3f} ms, decode={decode:.3f} ms "
            f"({decode / args.decode_steps:.3f} ms/step, {throughput:.1f} token/s), "
            f"kv_allocated={allocated:.3f} MiB, kv_valid={valid:.3f} MiB, "
            f"decode_peak_extra={f'{peak:.3f} MiB' if peak is not None else 'N/A'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
