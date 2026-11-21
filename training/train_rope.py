"""训练与主线推理模型兼容的 RoPE 权重；保留既有模型和训练代码。"""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_transformer"))

from train import CharacterTokenizer, get_batch
from llm_inference.model import LiteGPT as InferenceModel


def reference_module():
    path = ROOT / "04_batched_decode" / "batch_prefill_decode.py"
    spec = importlib.util.spec_from_file_location("rope_training_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_text(path):
    with path.open(encoding="utf-8") as handle:
        return "\n\n".join(json.loads(line)["text"] for line in handle if line.strip())


def autocast_context(device, precision):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(model, splits, args, device, precision):
    was_training = model.training
    model.eval()
    metrics = {}
    for offset, (name, tokens) in enumerate(splits.items()):
        generator = torch.Generator().manual_seed(args.seed + 100 + offset)
        losses = []
        for _ in range(args.eval_batches):
            x, y = get_batch(tokens, args.seq_len, args.batch_size, generator, device)
            with autocast_context(device, precision):
                _, loss, _ = model(x, y)
            losses.append(loss.item())
        metrics[name] = sum(losses) / len(losses)
    model.train(was_training)
    return metrics


@torch.inference_mode()
def inference_samples(model, tokenizer, prompts, count):
    samples = {}
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        logits, caches = model.prefill_batch([ids], max_new_tokens=count)
        output = list(ids)
        for step in range(count):
            token = logits.reshape(-1, logits.size(-1))[-1].argmax().item()
            output.append(token)
            if step + 1 < count:
                logits = model.decode_batch([token], caches)
        samples[prompt] = tokenizer.decode(output)
    return samples


@torch.inference_mode()
def verify_checkpoint(checkpoint, ref_module, prompts):
    config = checkpoint["config"]
    reference = ref_module.LiteGPT(config).eval()
    reference.load_state_dict(checkpoint["model_state_dict"], strict=True)
    inference = InferenceModel(config).eval()
    inference.load_state_dict(checkpoint["model_state_dict"], strict=True)

    ref_logits, ref_states = reference.prefill_batch(prompts)
    logits, caches = inference.prefill_batch(prompts, max_new_tokens=6)
    errors = [(ref_logits - logits).abs().max().item()]
    torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-5)

    for _ in range(5):
        reference_tokens = ref_logits.reshape(len(prompts), -1, config["vocab_size"])[:, -1].argmax(-1)
        inferred_tokens = logits.reshape(len(prompts), -1, config["vocab_size"])[:, -1].argmax(-1)
        assert reference_tokens.tolist() == inferred_tokens.tolist()
        for state, token in zip(ref_states, reference_tokens.tolist()):
            state.append_token(token)
        ref_logits = reference.decode_batch(ref_states)
        logits = inference.decode_batch(inferred_tokens.tolist(), caches)
        errors.append((ref_logits - logits).abs().max().item())
        torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-5)

    # 每个请求独立执行，与合批执行对照。
    batch_logits, batched_caches = inference.prefill_batch(prompts, max_new_tokens=6)
    batch_outputs = [[] for _ in prompts]
    for step in range(6):
        tokens = batch_logits.reshape(len(prompts), -1, config["vocab_size"])[:, -1].argmax(-1).tolist()
        for row, token in enumerate(tokens):
            batch_outputs[row].append(token)
        if step < 5:
            batch_logits = inference.decode_batch(tokens, batched_caches)

    for row, prompt in enumerate(prompts):
        single_logits, single_caches = inference.prefill_batch([prompt], max_new_tokens=6)
        single_output = []
        for step in range(6):
            token = single_logits.reshape(-1, config["vocab_size"])[-1].argmax().item()
            single_output.append(token)
            if step < 5:
                single_logits = inference.decode_batch([token], single_caches)
        assert single_output == batch_outputs[row]

    return {
        "strict_weight_loading": True,
        "prefill_and_five_decode_max_abs_errors": errors,
        "batched_vs_independent_greedy_tokens_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.seq_len, args.eval_interval, args.eval_batches) < 1:
        parser.error("步数、序列长度、batch 和评估参数必须为正数")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    precision = args.precision if device.type == "cuda" else "fp32"

    texts = {name: read_text(args.data_dir / f"{name}.jsonl") for name in ("train", "val")}
    tokenizer = CharacterTokenizer(texts["train"] + texts["val"])
    splits = {name: torch.tensor(tokenizer.encode(text), dtype=torch.long) for name, text in texts.items()}
    if any(len(tokens) <= args.seq_len for tokens in splits.values()):
        parser.error("训练集和验证集必须各包含至少 seq-len + 1 个字符")

    ref_module = reference_module()
    config = ref_module.get_config(len(tokenizer))
    if args.seq_len > config["max_seq_len"]:
        parser.error("seq-len 超过模型上下文上限")
    if args.max_new_tokens + 8 > config["max_seq_len"]:
        parser.error("生成样例超过上下文上限")
    model = ref_module.LiteGPT(config).to(device)

    # 在花费训练算力前检查主线参数结构，禁止忽略不匹配的键。
    check_model = InferenceModel(config)
    check_model.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    del check_model

    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": 0.1},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))
    generator = torch.Generator().manual_seed(args.seed)
    data_metadata = json.loads((args.data_dir / "metadata.json").read_text())
    prompts = [p for p in ("人工智能", "今天的天气", "学习", "从前") if all(c in tokenizer.stoi for c in p)]
    if not prompts:
        prompts = [texts["train"][:4]]
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "04_batched_decode/batch_prefill_decode.py", ROOT / "llm_inference/model.py")
    }
    history = []
    best_val = float("inf")
    best_step = 0
    training_start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"device={device}, precision={precision}, params={sum(p.numel() for p in parameters):,}, vocab={len(tokenizer)}, train_tokens={len(splits['train']):,}, val_tokens={len(splits['val']):,}", flush=True)
    print("Strict state_dict compatibility with llm_inference/model.py: PASS", flush=True)

    def checkpoint(step):
        return {
            "architecture": "LiteGPT-RoPE",
            "config": config,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "tokenizer": {"kind": "char", "chars": tokenizer.chars},
            "step": step,
            "seed": args.seed,
            "history": list(history),
            "dataset": data_metadata,
            "train_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "source_sha256": source_hashes,
        }

    def record(step):
        nonlocal best_val, best_step
        metrics = {"step": step, **evaluate(model, splits, args, device, precision)}
        history.append(metrics)
        print(f"step={step:4d} train_loss={metrics['train']:.4f} val_loss={metrics['val']:.4f} elapsed={time.monotonic() - training_start:.1f}s", flush=True)
        if metrics["val"] < best_val:
            best_val = metrics["val"]
            best_step = step
            torch.save(checkpoint(step), args.output_dir / "model.pt")
        (args.output_dir / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n")

    record(0)
    model.train()
    for step in range(1, args.steps + 1):
        if step <= args.warmup_steps:
            lr = args.lr * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(splits["train"], args.seq_len, args.batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            _, loss, _ = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps:
            record(step)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - training_start
    peak_memory = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    torch.save(checkpoint(args.steps), args.output_dir / "last.pt")
    best = torch.load(args.output_dir / "model.pt", map_location="cpu", weights_only=True)
    best["history"] = history
    best["completed_steps"] = args.steps
    torch.save(best, args.output_dir / "model.pt")
    print(f"Training complete: best_step={best_step}, best_val_loss={best_val:.4f}, elapsed={elapsed:.1f}s", flush=True)

    inference = InferenceModel(config).to(device).eval()
    inference.load_state_dict(best["model_state_dict"], strict=True)
    samples = inference_samples(inference, tokenizer, prompts, args.max_new_tokens)
    del inference, model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_prompts = [splits["train"][offset:offset + length].tolist() for offset, length in ((0, 3), (10, 5), (30, 7))]
    verification = verify_checkpoint(best, ref_module, test_prompts)
    summary = {
        "architecture": best["architecture"],
        "best_step": best_step,
        "completed_steps": args.steps,
        "initial_val_loss": history[0]["val"],
        "best_val_loss": best_val,
        "final_val_loss": history[-1]["val"],
        "parameters": sum(p.numel() for p in parameters),
        "vocab_size": len(tokenizer),
        "train_characters": len(splits["train"]),
        "val_characters": len(splits["val"]),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "pytorch": str(torch.__version__),
        "precision": precision,
        "training_seconds_including_evaluation": elapsed,
        "training_tokens": args.steps * args.batch_size * args.seq_len,
        "peak_allocated_gpu_bytes": peak_memory,
        "verification": verification,
        "samples": samples,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "samples.txt").write_text("\n\n".join(f"提示：{p}\n生成：{s}" for p, s in samples.items()) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
