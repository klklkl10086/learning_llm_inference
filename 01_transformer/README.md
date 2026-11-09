# 01 Transformer

要实现：

- Token embedding 和位置编码。
- 多头 Attention、causal mask。
- 残差、LayerNorm、MLP，组成可配置层数的 Decoder（当前默认 6 层）。
- 输出 logits，逐个生成 token。

完成标准：能解释张量形状，验证未来 token 不影响之前的输出。

参考资料（按顺序看）：

- [动手学深度学习：Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html)：先看多头注意力、位置编码和残差连接；本阶段只需实现 Decoder。
- [minGPT](https://github.com/karpathy/minGPT)：重点阅读 `mingpt/model.py` 中的 `CausalSelfAttention`、`Block` 和 `generate`，对应自己的实现。

## 先跑通训练

使用已有的 `kvcache` 环境。在本目录运行：

```bash
conda activate kvcache
python -m unittest -v test_transformer
# 集群 GPU 通过 Slurm 分配，300 步为一次小语料实验。
srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=00:10:00 python -u train.py
```

无 GPU 时也可以使用 `python train.py --device cpu --steps 100`。

默认使用 `demo_chinese.txt`，按字符建立小词表，无需下载 tokenizer。模型仍然是
6 层、4 个头、256 维、FFN 1024 维；每个训练窗口长度为 64，batch 为 8。
语料按 token 顺序分成前 90% 训练、后 10% 验证，不会跨分界取窗口。
它只用于检查训练流程和准备 KV Cache 基线，小语料的损失下降不代表通用语言能力。

也可以指定自己的 UTF-8 文本，或使用原来的 BERT 中文分词器（首次使用需要下载）：

```bash
python train.py --data /path/to/corpus.txt --prompt '语料中出现过的提示'
python train.py --tokenizer bert
```

训练输入与标签只错开一次：

```text
同一窗口：我 正 在 学 习
input_ids：我 正 在 学
targets：  正 在 学 习
```

`LiteGPT.forward(input_ids, targets=None)` 始终返回 `(logits, loss)`。
传入标签时，对原始 logits 计算交叉熵；不传标签时 loss 为 `None`。
训练循环执行 `zero_grad → backward → 梯度裁剪 → optimizer.step`。
每隔 50 步切换到 eval 模式，在固定的训练/验证窗口上记录平均损失。

默认结果保存在 `runs/demo/`：

- `model.pt`：模型配置、参数、词表、训练参数和损失记录。
- `loss_history.json`：训练前及各个评估阶段的损失。
- `samples.txt`：同一提示下训练前后的贪心生成结果。

加载默认字符级模型，作为后续 KV Cache 实验基线：

```python
import torch
from train import CharacterTokenizer
from transformer import LiteGPT

checkpoint = torch.load('runs/demo/model.pt', map_location='cpu', weights_only=True)
model = LiteGPT(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tokenizer = CharacterTokenizer(checkpoint['tokenizer']['chars'])
input_ids = torch.tensor([tokenizer.encode('我正在学习')], dtype=torch.long)
output_ids = model.generate(input_ids, max_new_tokens=40)
print(tokenizer.decode(output_ids[0].tolist()))
```

字符级词表只包含语料里出现过的字符；提示中有未见字符时会明确报错。
BERT 选项会另存 `runs/demo/tokenizer/`，加载时应使用对应的 `AutoTokenizer`。
