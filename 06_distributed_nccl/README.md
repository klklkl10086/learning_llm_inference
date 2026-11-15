# 06 多 GPU 通信与 NCCL

这一阶段接在单 GPU 推理引擎之后。先通过 `torch.distributed` 学习 NCCL 集合通信，再对照 nano-vLLM 的张量并行实现。

建议按以下顺序完成：

1. 两个 GPU、两个进程跑通 `broadcast`、`all_reduce`、`all_gather`，理解 `rank`、`world_size`、进程组和 collective 的调用顺序。
2. 分别测消息大小变化时的通信耗时；正确记录 GPU 同步和运行环境，区分通信与模型计算。
3. 把一层线性层按输出维或输入维切分，确认何时需要 `all_gather` 或 `all_reduce`；与单 GPU 输出比较。
4. 再研究注意力头、MLP 分片和 KV cache 在各 rank 上的布局，对照 nano-vLLM 的 ModelRunner 和并行层。

验收：两 rank 结果与单 GPU 基线在合理数值误差内一致；每个 rank 参与相同次序的 collective；能够解释通信开销和端到端收益。需要至少两块可用 GPU 才能实际验证 NCCL 部分。

参考：[NVIDIA NCCL 概览](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)、[NCCL 集合操作](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)、[PyTorch 分布式教程](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html)。
