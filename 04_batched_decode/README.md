# 04 不同长度请求的批量 decode

本阶段在 [03 RoPE](../03_rope/README.md) 的基础上，为每个请求分别保存 KV cache，再将不同历史长度的请求放进同一次 decode 前向。

| 文件 | 作用 |
| --- | --- |
| [`decode_batch.py`](decode_batch.py) | `RequestState`、每请求独立缓存、异长请求的合批 decode |
| [`test_decode_batch.py`](test_decode_batch.py) | 对照完整重算与逐请求缓存 decode，检查状态和边界 |
| [`benchmark_decode_batch.py`](benchmark_decode_batch.py) | 同一随机 RoPE 模型的逐请求与合批 decode 对比 |

当前实现先分别 prefill。`RequestState.token_ids` 记录全部已生成 token，`cached_len` 是已经进入模型并写入 KV 的 token 数。prefill 最后一个 logits 预测第一个新 token；调用 `append_token` 后，这个 token 成为 `pending_token`，到下一轮前向才写入 KV。

每轮 `decode_batch(requests)` 为每个请求各处理 1 个 token，以其自己的 `cached_len` 作为 RoPE 位置。注意力按本批最长历史临时补齐 KV，并屏蔽较短请求的无效位置。返回 logits 形状为 `[请求数, 1, vocab_size]`，顺序与输入请求相同；贪心采样可使用 `logits[:, -1, :].argmax(dim=-1)`，然后分别调用 `append_token`。不同长度请求的批量 prefill 和 PagedAttention 仍属后续工作。

调用方先执行 `model.eval()`，并将 prefill 和生成循环放在 `with torch.inference_mode():` 内。相同请求不能在同一个 batch 中重复出现，不同请求也不能共享同一层 KV cache。

在本目录、已安装 PyTorch 的环境中运行：

```bash
python -m unittest -v test_decode_batch
python benchmark_decode_batch.py --device cuda --prompt-lens 32,64,96,128 --decode-steps 32 --warmup 2 --repeats 7
```

测试以固定随机权重比较合批、逐请求缓存 decode 和完整重算；覆盖 3/7 长度、连续多轮、请求换序、状态边界及容量限制。benchmark 分别 prefill，每轮使用预设后续 token；不包括采样、调度、排队或端到端首 token 延迟。它先检查 logits，再报告 prefill/decode 中位数、KV 已分配/有效空间和 decode 期间的额外峰值显存。当前实现每轮会复制有效 KV 到临时补齐张量，因此合批未必更快。已执行的测试和性能数据见[实验记录](../docs/EXPERIMENT_LOG.md)。

下一阶段是[05 调度](../05_scheduler/README.md)：决定不同时间到达的请求每轮如何 prefill、decode 和结束。
