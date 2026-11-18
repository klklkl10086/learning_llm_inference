# 学习计划与项目地图

更新于 2026-11-16。先完成 `01`～`05` 的单卡推理主线，再围绕测量结果开展量化、CUDA 和多卡通信实验。本页的“已有”表示文件或实验产物存在；尚未重新运行的检查不标记为已通过。每周约 4 小时的情况下，后续单卡主线可按约 10～14 周安排；科研和课程繁忙时按验收结果顺延。

## 现在有哪些文件

- [01 Transformer](../01_transformer/README.md)：`transformer.py` 是无缓存基线；`train.py` 使用 `demo_chinese.txt` 跑训练；`test_transformer.py` 检查因果性、梯度、保存加载等。`runs/demo/` 已有本地训练产物，属于生成结果。
- [02 KV cache](../02_kv_cache/README.md)：`kv_cache.py` 用 `torch.cat`；`kv_cache_pro.py` 用预分配存储；`RoPE.py` 叠加旋转位置编码；`decode_batch.py` 是正在推进的不同长度请求练习，目前仍是草稿。`benchmark_kv_cache.py` 仅比较 `kv_cache.py` 与 `kv_cache_pro.py`。
- [03 调度](../03_scheduler/README.md)、[04 分页缓存](../04_paged_cache/README.md)、[05 推理引擎](../05_inference_engine/README.md)：目前是阶段说明，代码待写。
- [06 NCCL](../06_distributed_nccl/README.md)：多 GPU 通信和张量并行的后续阶段。
- [推理优化专题](OPTIMIZATION.md)：单卡引擎完成后，按实验需求学习 SDPA、量化和 CUDA。

## 当前关卡：不同长度请求的批量 decode

目标：在一次模型前向中处理多个历史长度不同、各自本轮输入 1 个 token 的请求。先确保 `decode_batch.py` 可导入并分别 prefill；待 decode 正确后，再研究不同长度的批量 prefill。

1. **请求状态**：每个请求保存 token 列表和各层的 KV cache。区分“已生成/已追加到请求的 token 数”和“已经经过模型、写入 KV 的 token 数”。刚采样出来的 token 在下一轮 decode 才进入 KV。
2. **位置**：若 A、B 分别已缓存 3、7 个 token，下一轮位置应为 `[[3], [7]]`，不能从一个全 batch 的 `past_len` 推出。RoPE 的 cos/sin 索引要支持每个 batch 行使用不同位置。
3. **KV 写入与读取**：每请求独立写入自己对应层的缓存。初版可以先按请求读取有效 KV，再临时补齐到本批最大长度；注意力必须屏蔽补齐区域。`T=1` 时，本轮 query 可看见当前及以前的有效 KV。
4. **正确性**：相同权重、相同输入 token、`eval()` 模式下，将合批 decode 的每个请求 logits 与单请求分别 decode 对比。覆盖提示长度 `3/7`、连续多轮、达到块边界前后（以后接分页缓存时）、缓存容量上限。

建议为这个实验保留现有单请求 `forward`，单独设计批量 decode 入口，便于从已工作的路径逐步比较。现在的连续缓存尚无 block table；待 [04 分页缓存](../04_paged_cache/README.md) 再引入物理块、`-1` 补齐和块复用。

`RoPE.py` 和 `decode_batch.py` 改变了位置编码，不能直接把 `01_transformer` 的绝对位置 embedding checkpoint 当成已训练的 RoPE 模型。数值正确性实验应让被比较的 RoPE 路径使用同一组权重；要评价生成质量，则需要与 RoPE 架构匹配的训练权重。

## 后续关卡

| 顺序 | 要交付的行为 | 过关检查 |
| --- | --- | --- |
| [03 调度](../03_scheduler/README.md) | 等待/运行/完成/取消状态，固定批再到每轮重新组批 | 不同到达时间和完成时间的请求都能推进；完成后释放状态 |
| [04 分页缓存](../04_paged_cache/README.md) | 固定大小物理块、空闲列表、每请求 block table；再做前缀共享 | 扩容/回收与引用计数正确；与连续缓存输出一致 |
| [05 推理引擎](../05_inference_engine/README.md) | 调度 → 模型执行 → 采样 → 更新状态 → 回收 | 记录首 token 延迟、decode token/s、峰值显存，并比较改造前后 |
| [06 NCCL](../06_distributed_nccl/README.md) | 先测集合通信，再把简化张量并行接入模型 | 单 GPU 与多 GPU 数值接近；记录通信耗时和总耗时 |

单卡主线完成后，在 [05 推理引擎](../05_inference_engine/README.md) 固定测试负载、建立性能基线，再从 [推理优化专题](OPTIMIZATION.md)选择 SDPA、KV cache 量化或 CUDA 单算子实验。NCCL 是随后围绕多卡推理开展的专题。

## 验证和实验记录

先固定输入、模型权重、设备与精度，核对 logits；性能比较则分开测 prefill 与 decode，记录 batch/并发数、提示长度、生成步数、预热次数、重复次数、显存和运行命令。GPU 计时要在测量边界同步；区分 KV 有效数据量与预分配空间。小模型上预分配缓存不一定比 `torch.cat` 快，不以单次耗时作为正确性依据。

结果统一记入[实验记录](EXPERIMENT_LOG.md)，避免把不同硬件、输入长度或热身次数下的数字直接比较。

现有入口：

```bash
# 01：在 01_transformer/ 目录执行
python -m unittest -v test_transformer

# 02：在 02_kv_cache/ 目录执行；默认读取 01_transformer/runs/demo/model.pt
python benchmark_kv_cache.py --device cpu --prompt-len 32 --decode-steps 16 --warmup 1 --repeats 3
```

`benchmark_kv_cache.py` 当前只比较动态与预分配缓存；RoPE 和不同长度批量 decode 需要单独的正确性测试，不能把这个基准的 PASS 当作它们的结果。

阅读源码时可对照本地 nano-vLLM：`engine/model_runner.py` 的 `prepare_prefill`、`prepare_decode`，`engine/sequence.py` 的请求状态，`engine/block_manager.py` 的块管理和 `layers/attention.py` 的 KV 写入/读取。先追踪数据形状和每个请求的长度，再看 CUDA Graph 与张量并行。
