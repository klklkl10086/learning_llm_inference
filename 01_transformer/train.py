"""小语料训练入口：构造 next-token 标签、训练、验证、保存模型。"""

import argparse
import json
from pathlib import Path

import torch

from transformer import LiteGPT, get_config


class CharacterTokenizer:
    """示例按字符编码，无需下载；chars 也会随 checkpoint 保存。"""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {char: index for index, char in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        unknown = set(text) - self.stoi.keys()
        if unknown:
            raise ValueError(f"字符不在示例词表中：{sorted(unknown)!r}")
        return [self.stoi[char] for char in text]

    def decode(self, ids):
        return "".join(self.chars[index] for index in ids)


def get_batch(tokens, seq_len, batch_size, generator, device):
    # 同一段连续文本错开一位，避免在模型内部再次移动标签。
    starts = torch.randint(len(tokens) - seq_len, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i + seq_len] for i in starts.tolist()])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, splits, args, device):
    was_training = model.training
    model.eval()
    losses = {}
    try:
        for offset, (name, tokens) in enumerate(splits.items()):
            # 每次评估抽取同一组窗口，让不同训练阶段的数值可比较。
            generator = torch.Generator().manual_seed(args.seed + 100 + offset)
            total = 0.0
            for _ in range(args.eval_batches):
                x, y = get_batch(tokens, args.seq_len, args.batch_size, generator, device)
                _, loss = model(x, y)
                total += loss.item()
            losses[name] = total / args.eval_batches
    finally:
        model.train(was_training)
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("demo_chinese.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runs" / "demo")
    parser.add_argument("--tokenizer", choices=["char", "bert"], default="char")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt", default="我正在学习")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.seq_len, args.eval_interval, args.eval_batches) < 1:
        parser.error("步数、batch 大小、序列长度和评估参数必须为正数")
    if args.lr <= 0 or args.max_new_tokens < 0:
        parser.error("学习率必须为正数，生成数量不能为负数")

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    text = args.data.read_text(encoding="utf-8")
    if args.tokenizer == "char":
        tokenizer = CharacterTokenizer(text)
        token_ids = tokenizer.encode(text)
        prompt_ids = tokenizer.encode(args.prompt)
        tokenizer_info = {"kind": "char", "chars": tokenizer.chars}
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
        tokenizer_info = {"kind": "bert", "name": "google-bert/bert-base-chinese"}
    if not prompt_ids:
        parser.error("生成提示不能编码为空序列")

    config = get_config(len(tokenizer))
    if args.seq_len > config["max_seq_len"]:
        parser.error("seq-len 不能超过模型的 max_seq_len")
    tokens = torch.tensor(token_ids, dtype=torch.long)
    boundary = int(len(tokens) * 0.9)
    splits = {"train": tokens[:boundary], "val": tokens[boundary:]}
    if any(len(part) <= args.seq_len for part in splits.values()):
        parser.error("语料过短：训练集和验证集都至少需要 seq-len + 1 个 token")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.tokenizer == "bert":
        tokenizer.save_pretrained(args.output_dir / "tokenizer")
    model = LiteGPT(config).to(device)
    # 一个共享参数仅加入优化器一次；矩阵使用 weight decay，bias/LN 不使用。
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": 0.1},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))
    generator = torch.Generator().manual_seed(args.seed)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    def generate_text():
        model.eval()
        ids = model.generate(prompt, max_new_tokens=args.max_new_tokens)
        values = ids[0].tolist()
        if args.tokenizer == "char":
            return tokenizer.decode(values)
        return tokenizer.decode(values, skip_special_tokens=True)

    print(f"device={device}, parameters={sum(p.numel() for p in parameters):,}, "
          f"vocab={len(tokenizer)}, train_tokens={len(splits['train'])}, "
          f"val_tokens={len(splits['val'])}", flush=True)
    before_text = generate_text()
    print(f"训练前生成：{before_text}", flush=True)
    history = []

    def record(step):
        metrics = {"step": step, **evaluate(model, splits, args, device)}
        history.append(metrics)
        print(f"step={step:4d} train_loss={metrics['train']:.4f} "
              f"val_loss={metrics['val']:.4f}", flush=True)

    record(0)
    model.train()
    for step in range(1, args.steps + 1):
        x, y = get_batch(splits["train"], args.seq_len, args.batch_size, generator, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"第 {step} 步出现非有限 loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps:
            record(step)

    after_text = generate_text()
    checkpoint = {
        "config": config,
        "model_state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
        "tokenizer": tokenizer_info,
        "step": args.steps,
        "seed": args.seed,
        "history": history,
        "train_args": {name: str(value) if isinstance(value, Path) else value
                       for name, value in vars(args).items()},
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    (args.output_dir / "loss_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "samples.txt").write_text(
        f"训练前：\n{before_text}\n\n训练后：\n{after_text}\n", encoding="utf-8")
    print(f"训练后生成：{after_text}", flush=True)
    print(f"模型已保存：{args.output_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
