# 推理优化专题

在 [llm_inference 主线](../llm_inference/README.md) 跑通最小引擎并建立性能基线后，围绕同一组请求负载选择实验；每次只改一个变量，把结果记入[实验记录](EXPERIMENT_LOG.md)。先看正确性，再比较首 token 延迟、后续 token 延迟、吞吐和峰值显存。小模型可用来验证实现，但其性能结论不能直接推广到较大模型。

| 主题 | 最小实验 | 要回答的问题 |
| --- | --- | --- |
| SDPA | 保持模型、输入与精度一致，将手写 attention 与 PyTorch SDPA 比较 | 数值是否接近？prefill/decode 延迟和显存如何变化？不同形状下结果是否一致？ |
| KV cache 量化 | 在已有缓存上做一个 8 位存储、读取时还原的原型，再对照未量化缓存 | 存储节省多少？logits 误差多少？还原开销是否抵消了收益？ |
| CUDA 单算子 | 通过测量选一个算子，例如 LayerNorm，写最小 CUDA 实现并与 PyTorch 比较 | 哪些输入形状受益？优化的是计算、访存还是调用开销？ |

量化实验先从已实现的 KV cache 入手；权重量化和激活量化属于后续扩展。8 位原型的目标是理解尺度、误差和内存布局，不能仅凭存储位数推断端到端加速。实际部署可对照 [vLLM 量化功能](https://docs.vllm.ai/en/latest/features/quantization/)和所用 GPU、后端的支持情况。

CUDA 实验放在引擎能测量性能之后。先用 PyTorch profiler 或分阶段计时确认耗时位置，再实现一个算子；验收是数值对齐、重复测量以及对收益边界的解释。参考 [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/)与 [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)。

多卡专题见[08 NCCL](../08_distributed_nccl/README.md)：从两卡 collective 到简化张量并行推理，重点测通信时间及端到端收益。
