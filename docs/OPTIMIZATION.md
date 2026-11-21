# 推理优化实验怎么选

分成独立基础实验与系统集成实验。单张量量化、单算子 CUDA 现在即可学习；集成时使用 [llm_inference](../llm_inference/README.md) 的相同模型、请求和基线，每次只改变一个变量。

| 专题 | 起步实验 | 系统接入前要回答的问题 |
| --- | --- | --- |
| Attention / SDPA | 相同输入、dtype 和 mask 对照手写 attention | prefill/decode 的 mask 与布局是否正确？实际耗时和显存是否改善？ |
| [量化](../09_quantization/README.md) | 张量 INT8 → Linear 权重 → 分组/INT4 → KV cache | 误差如何累积？元数据占多少？反量化后浮点计算是否仍有收益？ |
| [CUDA / Triton](../10_cuda_kernels/README.md) | vector add → 归约 → softmax/LayerNorm → RoPE | 是否正确？哪些形状受益？是否减少访存或 launch？ |
| [多 GPU](../08_distributed_nccl/README.md) | 两卡 collective → 并行 Linear | 分片与单卡结果一致吗？通信是否抵消计算收益？ |

## 测量顺序

1. 固定架构、权重、输入与推理设置，先确认正确性；量化时明确允许的误差及质量下降范围。
2. 用分阶段计时或 [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html) 定位实际瓶颈，再选择集成点。
3. 测单算子、模型阶段、引擎端到端三个层级；收益分别记录。
4. GPU 采用 CUDA events 或计时边界同步；JIT/编译、首次初始化和正常执行分别计时。预热后重复测量并记录中位数及波动。
5. 固定形状之外，再扫 batch、prompt/decode 长度，说明何时受益、何时变慢。

## 记录口径

- 正确性：局部输出和 logits 的最大绝对/相对误差；量化增加 MSE、重构误差，必要时在固定验证集测困惑度或任务指标。
- 存储：权重/KV 实际张量与 scale、zero point、padding 等元数据字节；另记进程/分配器峰值显存。
- 性能：prefill 时间、decode 时间、token/s；Engine 可运行后再测包含排队与采样的 TTFT/每 token 延迟。
- 条件：GPU、软件版本、代码版本、shape、dtype、输入/生成策略、预热与重复次数、计时范围。

INT4 保存在 int8 容器里且没有打包时，不会自然得到 4 位存储。先反量化成浮点再 matmul 的原型用于学习误差，是否节省峰值显存或加速都需要实测。单算子加速不能直接推断系统加速。小模型性能结论也不能直接推广到大模型。

具体模板见[实验记录](EXPERIMENT_LOG.md)。官方资料与论文按专题收在 [09](../09_quantization/README.md) 和 [10](../10_cuda_kernels/README.md)。
