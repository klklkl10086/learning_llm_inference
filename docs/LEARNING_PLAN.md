# 推理系统、量化与 GPU 算子的学习路线

以可运行里程碑组织学习。目录编号用于检索，不等于先修顺序；08 多 GPU 不挡住 09 量化或 10 CUDA 基础。当前状态与阻塞问题见[检查记录](REVIEW_2026-11-20.md)。

## 主线：把已有代码接成系统

| 顺序 | 要做的内容 | 完成标准 |
| --- | --- | --- |
| 1 模型契约 | 读主线 RequestState、prefill_batch、decode_batch | 能解释采样 token 与已计算 token 的区别；同权重对照 04 |
| 2 ModelRunner | 完成已有骨架的执行、argmax、状态映射和 release | 单请求完整生成、多个异长请求换序执行正常 |
| 3 Scheduler + Engine | 修正 update 问题，再实现提交请求和 step | schedule → execute → update → release 完整闭环 |
| 4 动态请求 | 运行中接纳新请求、结束旧请求 | 日志能解释每轮选择，长期活跃请求能持续推进 |
| 5 分块 prefill | 计划与模型支持历史 KV 上的分段计算 | 长 prompt 跨轮执行，结果与完整 prefill 对照 |
| 6 分页缓存 | 物理块、block table、分配与回收，最后再加共享 | 与连续缓存对照；覆盖块边界、资源不足和回收 |
| 7 按瓶颈集成优化 | 接入已验证的量化/算子/attention 路径 | 正确性、延迟、吞吐、显存有可复现对照 |
| 8 多 GPU | collective → 并行 Linear → 并行模型 | 对照单卡结果，分清通信与计算收益 |

最小引擎先于分页缓存。初版只做完整 prefill 或单 token decode，避免同时引入调度、分页和算子三类变量。

## 支线一：量化，可以现在开始

按 [09 量化](../09_quantization/README.md) 完成：

1. 单张量量化/还原：scale、zero point、clipping、对称/非对称、不同粒度。
2. 单 Linear 的 W8A16 原型：量化权重，还原后浮点计算，理解误差与真实存储。
3. 分组量化与 INT4 打包，区分 4 位数值范围和实际 4 位存储。
4. 结合已有 KV cache 做 INT8 存储实验，比较长序列的误差、显存与还原开销。
5. 需要时扩展 W8A8、校准与 GPTQ/AWQ/SmoothQuant，再选择一个后端对照。

基础实验不依赖 Engine。模型质量评估需要训练权重和固定评估集；随机小模型只适合验证数值路径。

## 支线二：CUDA / Triton，可以现在开始

按 [10 算子](../10_cuda_kernels/README.md) 推进：

1. CUDA C++ vector add：grid/block/thread、边界判断和访存。
2. reduction：warp、shared memory、同步与归约。
3. softmax、LayerNorm：数值稳定性、行归约和访存；对应当前模型的实际算子。
4. RoPE：位置索引、layout、融合访存；可与现有 RoPE 对照。
5. 小型 matmul 学习 tile，然后依据 profiler 选择模型瓶颈，最后再研究 attention 融合。

CUDA C++ 是理解线程与存储的主线；可为同一算子增加 Triton 对照，初期不必每个算子都维护两套实现。RMSNorm 可以独立学，但当前模型用 LayerNorm，不能直接替换后要求输出一致。

## 下一次动手的具体安排

建议先完成一个可运行的主线步骤，再做一个支线小实验：

- 主线：继续 model_runner.py，明确 plan → 请求状态 → logits → 采样结果的映射；自查调度 update，再接 Engine。
- 量化第一个实验：随机张量做对称 INT8 量化/还原，覆盖全零和离群值，输出误差与实际存储字节。
- CUDA 第一个实验：vector add，对比 PyTorch，覆盖长度不能被 block size 整除的情况。

每次完成后写简短实验记录与 Git 提交。不要为了新阶段复制完整模型；不要先堆积空代码文件。

## 验证与记录

已有测试命令见[项目首页](../README.md)。新增实现后围绕实际行为验证；04 测试不自动覆盖主线优化模型、Scheduler 或未来 Engine。

数值对照固定架构、权重、输入、eval 模式和 dtype，先核对 logits。量化实验再增加局部重构误差与质量指标。性能比较固定负载，记录环境、预热、重复次数和计时边界；区分 kernel、模型与包含排队的端到端耗时。

已有 benchmark 仍在对应目录运行：

```bash
# 在 02_kv_cache/，读取 01 的 checkpoint
python -B benchmark_kv_cache.py --device cuda --batch-size 1 --prompt-len 64 --decode-steps 32 --warmup 2 --repeats 7

# 在 04_batched_decode/，使用随机初始化的同一个 RoPE 模型
python -B benchmark_decode_batch.py --device cuda --prompt-lens 32,64,96,128 --decode-steps 32 --warmup 2 --repeats 7
```

01–02 的绝对位置编码与 03 之后的 RoPE 不同，不能直接共用训练 checkpoint 评价质量。实验结果见[实验记录](EXPERIMENT_LOG.md)，选题与测量见[优化专题](OPTIMIZATION.md)。
