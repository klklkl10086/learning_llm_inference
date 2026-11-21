# 主线 RoPE 模型训练

这个目录提供独立的数据准备和训练入口，复用已有的模型计算代码：

- [prepare_minimind.py](prepare_minimind.py)：读取固定版本 MiniMind，筛选中文文本、去重，并按完整样本划分训练与验证集。
- [train_rope.py](train_rope.py)：通过 [04 的通用前向模型](../04_batched_decode/batch_prefill_decode.py) 训练，保存可严格加载到 [主线模型](../llm_inference/model.py) 的 RoPE 权重。

01 的训练脚本和旧 checkpoint 保留原样。新的模型从随机初始化开始训练，不使用绝对位置编码的旧权重。

## 本次数据

数据来源是 MiniMind 的 pretrain_t2t_mini.jsonl，固定版本为 312afb4f76391145c6902f765bb51691c09a12f5。

顺序读取满足长度 80–6000、中文字符比例至少 25% 的样本，累计约 20 MiB UTF-8 文本。按规范化文本的 SHA256 去除完全重复样本，并用哈希确定约 95% / 5% 的文档级划分。这是源文件前部的中文子集，不代表整个数据集的随机抽样。

本次共 27,504 条样本：训练集 26,058 条、验证集 1,446 条。实际数据与版本信息保存在 [metadata.json](../runs/datasets/minimind_20m_20261120/metadata.json)。

## 重新执行

在项目根目录、kvcache 环境中执行。再次运行时改用新的输出目录，脚本会拒绝覆盖已有目录。

```bash
python -B training/prepare_minimind.py \
  --revision 312afb4f76391145c6902f765bb51691c09a12f5 \
  --output-dir runs/datasets/minimind_20m_20261120 \
  --target-mib 20

srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=00:15:00 \
  python -B -u training/train_rope.py \
  --data-dir runs/datasets/minimind_20m_20261120 \
  --output-dir runs/rope_minimind_20261120_001 \
  --steps 4000 --batch-size 16 --seq-len 128 \
  --eval-interval 200 --eval-batches 10 \
  --device cuda --precision bf16
```

本次沿用 6 层、256 维、4 个 attention head、FFN 1024 维，最大上下文为 256。词表由本次子集的字符集确定，共 5573 个字符；参数约 617 万。使用 AdamW、学习率 warmup 与余弦衰减，训练激活采用 BF16，保存的参数保持 FP32。

验证使用固定窗口，每 200 步评估一次。model.pt 保存验证损失最低的权重；last.pt 保存最后一步权重。训练脚本没有提供优化器断点续训功能。

## 训练产物

本次输出位于 [runs/rope_minimind_20261120_001](../runs/rope_minimind_20261120_001/)：

| 文件 | 内容 |
| --- | --- |
| model.pt | 最佳验证权重、config、字符词表、训练与数据来源信息 |
| last.pt | 最后一步权重 |
| loss_history.json | 初始及每次评估的训练、验证损失 |
| summary.json | 最佳步数、训练环境、耗时和推理数值验证结果 |
| samples.txt | 最佳权重生成的文本样例 |

数据和 checkpoint 位于已有忽略规则覆盖的 runs/ 目录，不会被当作源码提交。

## 加载到主线模型

在项目根目录运行：

```python
import torch
from llm_inference.model import LiteGPT

checkpoint = torch.load(
    "runs/rope_minimind_20261120_001/model.pt",
    map_location="cpu",
    weights_only=True,
)

model = LiteGPT(checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

chars = checkpoint["tokenizer"]["chars"]
stoi = {char: index for index, char in enumerate(chars)}

prompt_ids = [stoi[char] for char in "人工智能"]
logits, caches = model.prefill_batch([prompt_ids])
next_token_id = logits[0].argmax().item()
print(chars[next_token_id])

# 消费刚生成的 token，继续生成下一个。
logits = model.decode_batch([next_token_id], caches)
```

之后把这个已加载权重的 model 传给 ModelRunner。仅调用 LiteGPT(config) 仍是随机初始化，不会自动寻找 checkpoint。字符不在本次词表中时，需要换提示或后续扩展 tokenizer。

训练完成后脚本会检查严格权重加载、普通版与主线的 prefill 和五轮 decode logits，以及异长批量与独立执行的贪心 token 是否一致。当前训练没有加入 EOS 特殊 token，推理结束条件由生成上限和上下文上限控制。

## 本次训练结果

4000 步已完成，最佳验证权重位于第 4000 步。验证 loss 从 8.6662 降至 3.2776，训练与评估耗时 189.3 秒。主线严格加载和批量推理数值对照均通过。中文生成仍有重复，完整记录见 [summary.json](../runs/rope_minimind_20261120_001/summary.json) 与 [samples.txt](../runs/rope_minimind_20261120_001/samples.txt)。
