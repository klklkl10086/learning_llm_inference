# 02 基础 KV cache

本阶段沿用 [01 Transformer](../01_transformer/README.md) 的可训练绝对位置 embedding，先实现单请求缓存，再比较动态拼接和预分配两种存储方式。

| 文件 | 作用 |
| --- | --- |
| [`kv_cache.py`](kv_cache.py) | 每轮用 `torch.cat` 扩展 K/V |
| [`kv_cache_append.py`](kv_cache_append.py) | 预分配 K/V，用 `length` 标记有效前缀 |
| [`benchmark_kv_cache.py`](benchmark_kv_cache.py) | 读取 01 阶段 checkpoint，验证输出并比较两种缓存 |

prefill 将每层 K/V 写入缓存；decode 每轮只输入新增 token，并使用缓存长度确定绝对位置和因果 mask。`kv_cache.py` 的单层缓存是 `(K, V)`；预分配版用 `LayerKVCache.k/.v` 保存固定容量，`.length` 表示已写入长度。完成标准是在同一权重和输入下，缓存版与完整重算的 logits 一致，并能说明有效 KV 大小和已分配空间的区别。

在本目录、已安装 PyTorch 的环境中运行：

```bash
python benchmark_kv_cache.py --device cuda --batch-size 1 --prompt-len 64 --decode-steps 32 --warmup 2 --repeats 7
```

脚本先比较拼接与预分配版本的 logits，再报告 prefill/decode 耗时、KV 已分配/有效空间和峰值额外显存。GPU 计时在边界同步，结果取重复测量的中位数；decode 使用预设 token，不包含采样、排队或端到端首 token 延迟。`peak_extra` 从 prefill 前的模型常驻显存计起。实际命令和数据见[实验记录](../docs/EXPERIMENT_LOG.md)。

后续按 [03 RoPE](../03_rope/README.md) → [04 异长批量 decode](../04_batched_decode/README.md) 推进。RoPE 版改变了位置编码，不能直接使用本阶段的绝对位置编码 checkpoint 评价其生成质量。

参考：[Transformers KV cache 文档](https://huggingface.co/docs/transformers/kv_cache)。
