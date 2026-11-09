"""Compare dynamic (torch.cat) and preallocated KV caches with identical inputs."""

import argparse
import gc
import importlib.util
import statistics
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = HERE.parent / "01_transformer" / "runs" / "demo" / "model.pt"


def load_model(path, checkpoint, device):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.LiteGPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def check_correctness(dynamic, static, tokens, prompt_len):
    prompt = tokens[:, :prompt_len]
    old_logits, _, old_cache = dynamic(prompt, use_cache=True)
    new_logits, _, new_cache = static(prompt, use_cache=True)
    max_diff = (old_logits - new_logits).abs().max().item()
    torch.testing.assert_close(new_logits, old_logits, rtol=1e-4, atol=1e-5)

    pointers = tuple((layer.k.data_ptr(), layer.v.data_ptr()) for layer in new_cache)
    for position in range(prompt_len, tokens.size(1)):
        new_token = tokens[:, position:position + 1]
        old_logits, _, old_cache = dynamic(
            new_token, past_key_values=old_cache, use_cache=True
        )
        new_logits, _, new_cache = static(
            new_token, past_key_values=new_cache, use_cache=True
        )
        max_diff = max(max_diff, (old_logits - new_logits).abs().max().item())
        torch.testing.assert_close(new_logits, old_logits, rtol=1e-4, atol=1e-5)
        assert pointers == tuple(
            (layer.k.data_ptr(), layer.v.data_ptr()) for layer in new_cache
        ), "预分配缓存的底层地址发生了变化"
    assert all(layer.length == tokens.size(1) for layer in new_cache)
    return max_diff


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_once(model, tokens, prompt_len, device):
    prompt = tokens[:, :prompt_len]
    sync(device)
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    _, _, cache = model(prompt, use_cache=True)
    sync(device)
    prefill_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    for position in range(prompt_len, tokens.size(1)):
        _, _, cache = model(
            tokens[:, position:position + 1],
            past_key_values=cache,
            use_cache=True,
        )
    sync(device)
    decode_ms = (time.perf_counter() - start) * 1000

    if hasattr(cache[0], "k"):
        tensors = [tensor for layer in cache for tensor in (layer.k, layer.v)]
    else:
        tensors = [tensor for layer in cache for tensor in layer]
    cache_mib = sum(t.numel() * t.element_size() for t in tensors) / 1024**2
    peak_mib = (
        (torch.cuda.max_memory_allocated(device) - baseline) / 1024**2
        if device.type == "cuda" else None
    )
    return prefill_ms, decode_ms, cache_mib, peak_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if min(args.batch_size, args.prompt_len, args.decode_steps, args.repeats) < 1 or args.warmup < 0:
        parser.error("batch-size、prompt-len、decode-steps、repeats 必须为正数；warmup 不能为负数")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("当前没有可用的 CUDA 设备")
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    max_len = checkpoint["config"]["max_seq_len"]
    if args.prompt_len + args.decode_steps > max_len:
        parser.error(f"prompt-len + decode-steps 不能超过 {max_len}")
    tokens = torch.randint(
        checkpoint["config"]["vocab_size"],
        (args.batch_size, args.prompt_len + args.decode_steps),
    )
    paths = (("torch.cat", HERE / "kv_cache.py"), ("preallocated", HERE / "kv_cache_pro.py"))

    tokens = tokens.to(device)
    dynamic = load_model(paths[0][1], checkpoint, device)
    static = load_model(paths[1][1], checkpoint, device)
    max_diff = check_correctness(dynamic, static, tokens, args.prompt_len)
    print(f"correctness ({device}): PASS; max_abs_logit_diff={max_diff:.3g}", flush=True)
    del dynamic, static
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for name, path in paths:
        model = load_model(path, checkpoint, device)
        for _ in range(args.warmup):
            measure_once(model, tokens, args.prompt_len, device)
        samples = [
            measure_once(model, tokens, args.prompt_len, device)
            for _ in range(args.repeats)
        ]
        prefill, decode, cache = (
            statistics.median(sample[i] for sample in samples) for i in range(3)
        )
        peak = (
            statistics.median(sample[3] for sample in samples)
            if device.type == "cuda" else None
        )
        print(
            f"{name}: prefill={prefill:.3f} ms, "
            f"decode={decode:.3f} ms ({decode / args.decode_steps:.3f} ms/token), "
            f"cache={cache:.3f} MiB, "
            f"peak_extra={f'{peak:.3f} MiB' if peak is not None else 'N/A'}",
            flush=True,
        )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
