# 推理引擎学习

项目按“模型 → 缓存 → 多请求 → 分页缓存 → 推理引擎 → 多 GPU 通信”逐步推进。每阶段的具体目标在对应目录的 README；[学习计划与当前进度](docs/LEARNING_PLAN.md)记录文件地图、下一步和验收标准，[实验记录](docs/EXPERIMENT_LOG.md)用于保存可复现的对比结果。

| 阶段 | 内容 | 当前状态 |
| --- | --- | --- |
| [01_transformer](01_transformer/README.md) | 小型 Decoder、训练与生成 | 已有实现、测试和小语料训练产物 |
| [02_kv_cache](02_kv_cache/README.md) | 动态/预分配 KV cache、RoPE、不同长度请求 | 前三项已有实现；不同长度批量 decode 待实现 |
| [03_scheduler](03_scheduler/README.md) | 请求状态与 continuous batching | 待实现 |
| [04_paged_cache](04_paged_cache/README.md) | 物理块、block table、共享与回收 | 待实现 |
| [05_inference_engine](05_inference_engine/README.md) | 组装执行循环并测量性能 | 待实现 |
| [06_distributed_nccl](06_distributed_nccl/README.md) | 多 GPU 集合通信与张量并行 | 待实现 |

当前从 `02_kv_cache/request_len.py` 继续：先分别 prefill 不同长度的请求，再实现单步批量 decode，并用独立执行作为正确性基准。完成这一关后再进入调度和分页缓存。

参考资料：[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)、[vLLM 架构](https://docs.vllm.ai/en/latest/design/arch_overview/)。阅读源码时固定一个提交版本；具体章节见各阶段 README。
