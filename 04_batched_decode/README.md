# 04 不同长度请求的批量执行

本阶段在 [03 RoPE](../03_rope/README.md) 基础上，为每个请求保存独立 KV cache，支持异长 prompt 的批量 prefill，以及不同历史长度请求的批量 decode。

## 文件与系统主线

| 文件 | 作用 |
| --- | --- |
| [batch_prefill_decode.py](batch_prefill_decode.py) | 保留通用 forward、生成循环和批量入口，作为数值对照版本 |
| [decode_batch.py](decode_batch.py) | 指向 batch_prefill_decode.py 的符号链接，兼容已有测试与 benchmark 的旧导入 |
| [test_decode_batch.py](test_decode_batch.py) | 对照完整重算与逐请求缓存 decode，检查状态和边界 |
| [benchmark_decode_batch.py](benchmark_decode_batch.py) | 同一随机 RoPE 模型的逐请求与合批 decode 比较 |
| [主线 model.py](../llm_inference/model.py) | 原 batch_prefill_decode_optimized.py，后续系统使用的模型 |

后续系统功能写入 [llm_inference](../llm_inference/README.md)。本目录保留已有代码作为对照；符号链接仅提供旧模块名，不额外维护一份模型实现。

主线模型在 prefill 时直接建立各请求最终使用的缓存，并仅为每个 prompt 最后一个有效位置计算 logits。普通版保留完整前向，便于逐位置、逐请求比较。

## 请求与模型状态

`RequestState.token_ids` 包含 prompt 与已追加的生成 token，`cached_len` 是已经经过模型并写入 KV 的 token 数。prefill 最后一个有效位置的 logits 预测第一个新 token；调用 `append_token` 后，这个 token 成为 `pending_token`，到下一轮前向才写入 KV。

每轮 `decode_batch(states)` 为每个请求计算一个 token，以其自己的 `cached_len` 作为 RoPE 位置。注意力按本批最长历史临时补齐 KV，屏蔽较短请求的无效位置。返回 logits 形状为 `[B, 1, vocab_size]`，顺序与输入状态相同。

调用方先执行 `model.eval()`。相同请求不能在同一个 batch 重复出现，不同请求不能共享同一层 KV cache。当前两个版本都采用连续缓存，分块 prefill 与分页缓存尚未实现。

## 已有验证入口

在本目录、已安装 PyTorch 的环境中运行：

```bash
python -B -m unittest -v test_decode_batch
python -B benchmark_decode_batch.py --device cuda --prompt-lens 32,64,96,128 --decode-steps 32 --warmup 2 --repeats 7
```

测试使用固定随机权重比较合批、逐请求缓存 decode 和完整重算；它当前验证本目录的普通版，不自动验证主线优化版。

现有 benchmark 逐请求 prefill，再使用预设后续 token 比较 decode；不包括采样、调度、排队或端到端首 token 延迟。它报告 logits 对照、耗时、KV 已分配与有效空间、decode 额外峰值显存。性能结论的输入与测量口径见[实验记录](../docs/EXPERIMENT_LOG.md)。

下一步把主线模型接入 [05 调度](../05_scheduler/README.md) 和 [07 最小引擎](../07_inference_engine/README.md)。
