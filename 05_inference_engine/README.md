# 05 推理引擎

要实现：

- 连接请求、调度器、缓存管理器和模型执行器。
- 跑通“调度 → 执行 → 更新状态 → 回收”的循环。
- 测量首 token 延迟、吞吐和显存，用 SDPA 等方式尝试优化。
- 对照 nano-vLLM，再阅读 vLLM 的调度、KV 管理和 ModelRunner。

完成标准：能追踪一条请求的完整生命周期，并解释一次优化的测量结果。

参考资料（按顺序看）：

- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)：先追踪请求如何经过调度器、块管理器和模型执行器。
- [vLLM 架构说明](https://docs.vllm.ai/en/latest/design/arch_overview/)：对照 Engine Core、Worker 和 ModelRunner 的职责，追踪一次请求。
- [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)：将手写 Attention 替换为 SDPA，比较正确性、耗时和显存。
