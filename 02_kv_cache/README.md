# 02 KV cache

在上一阶段模型上实现：

- Prefill 保存每层 K/V，decode 每次只处理新 token。
- 正确处理位置偏移和 attention mask。
- 先用拼接缓存，再改成预分配缓存。
- 加入 RoPE，比较缓存大小和生成耗时。

完成标准：相同输入和权重下，缓存与完整重算的 logits 在合理误差内一致。

## 文件地图与当前任务

这些文件保留了逐步演进的版本，便于比较，不需要互相覆盖：

| 文件 | 用途 |
| --- | --- |
| `kv_cache.py` | 使用 `torch.cat` 扩展 KV 的第一版 |
| `kv_cache_pro.py` | 预分配 KV 空间，用长度标记已写入区域 |
| `RoPE.py` | 在预分配 KV 上为 Q/K 加入 RoPE |
| `decode_batch.py` | 从 RoPE 版出发，练习不同长度请求的批量 decode；目前是未验收的草稿 |
| `benchmark_kv_cache.py` | 对比前两版缓存的正确性、prefill/decode 耗时与占用；目前不测 RoPE 或多请求 |

下一步先确保 `decode_batch.py` 可导入，再让每个请求分别完成 prefill 并持有自己的缓存，把每个请求的一个待计算 token 组成 decode batch。为每个请求分别生成位置、写入并读取 KV，按有效长度屏蔽补齐部分。用两个长度分别为 3 和 7 的请求验证：合批 decode 与分别 decode 的 logits 一致，且缓存长度独立增长。

`benchmark_kv_cache.py` 使用 01 阶段的绝对位置编码 checkpoint，仅对比 `kv_cache.py` 与 `kv_cache_pro.py`。RoPE 版改变了位置编码，不能直接用这个 checkpoint 证明 RoPE 生成质量；RoPE 数值对比要使用同一组 RoPE 模型权重。

阶段计划和验收顺序见[学习计划](../docs/LEARNING_PLAN.md)。

参考资料（按顺序看）：

- [Transformers KV cache 文档](https://huggingface.co/docs/transformers/kv_cache)：了解动态缓存、静态缓存和缓存的使用方式。
- [RoFormer / RoPE 论文](https://arxiv.org/abs/2104.09864)：理解对 Q/K 施加旋转，以及位置差如何进入注意力计算。
