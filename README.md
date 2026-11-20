# LLM 推理优化学习项目

目标是把自己写的 Transformer、KV cache、批量执行和调度代码逐步接成一个推理系统，并通过数值对照和性能实验理解每次优化。

**后续系统代码统一放在 [`llm_inference/`](llm_inference/README.md)。** `01–08` 保留学习过程、对照实现、测试和专题说明；编号可以保留，实际学习顺序按功能依赖调整。

## 当前系统主线

```text
llm_inference/
├── model.py          # 原 batch_prefill_decode_optimized.py，推理模型
├── scheduler.py      # 原 05_scheduler/scheduler.py，仍在编写
└── README.md         # 当前接口与后续接入位置
```

目前已有批量 prefill/decode 模型和未完成的调度器。ModelRunner、Engine 和缓存管理器尚未实现，因此系统还不能运行端到端多请求生成。

建议现在沿着这条顺序继续手写：

```text
模型 → ModelRunner → 补完 Scheduler → Engine.step()
     → 多请求连续组批 → 分块 prefill → 分页缓存
     → 性能优化 → 多 GPU
```

组件边界见[系统设计](docs/ARCHITECTURE.md)，具体任务见[学习路线](docs/LEARNING_PLAN.md)，文件迁移位置见[整理记录](docs/FILE_LAYOUT.md)。

## 学习目录与系统的关系

| 目录 | 保留的学习内容 | 在系统中的作用 |
| --- | --- | --- |
| [01 Transformer](01_transformer/README.md) | 无缓存模型、训练、测试和训练产物 | 理解模型计算，提供绝对位置编码基线 |
| [02 KV cache](02_kv_cache/README.md) | 拼接缓存、预分配缓存和 benchmark | 理解缓存写入与存储开销 |
| [03 RoPE](03_rope/README.md) | RoPE 与单请求缓存版本 | 理解每个 token 的位置 |
| [04 批量执行](04_batched_decode/README.md) | 通用前向对照、测试和 benchmark | 对照主模型的数值与缓存行为 |
| [05 调度](05_scheduler/README.md) | 调度说明与回调草稿 | 继续编写主线中的 scheduler.py |
| [06 分页缓存](06_paged_cache/README.md) | 分页缓存接入计划 | 最小引擎跑通后再改造缓存布局 |
| [07 推理引擎](07_inference_engine/README.md) | ModelRunner、Engine 与测量计划 | 提前与 05 一起完成最小执行循环 |
| [08 多 GPU](08_distributed_nccl/README.md) | NCCL 与张量并行计划 | 在已有单卡系统上扩展执行方式 |

01–04 是模型的演进与实验版本。后续功能集中加入 `llm_inference/`，沿用同一个系统入口。每完成一个可运行阶段，用 Git 提交记录版本，避免再复制一份完整模型来承接下一阶段。

## 验证与实验

在项目根目录、已安装 PyTorch 的环境中执行：

```bash
conda activate kvcache
python -B -m unittest discover -s 01_transformer -p 'test_transformer.py' -v
python -B -m unittest discover -s 04_batched_decode -p 'test_decode_batch.py' -v
```

上述测试分别检查 01 与 04 的对照实现，尚未覆盖整个系统。04 的旧导入名 `decode_batch.py` 通过符号链接指向 `batch_prefill_decode.py`，测试源代码保持原样。

性能数据保存在[实验记录](docs/EXPERIMENT_LOG.md)；已有引擎测量基线后，再按[优化专题](docs/OPTIMIZATION.md)选择 SDPA、量化或 CUDA 实验。03 及后续使用 RoPE，与 01 的绝对位置编码模型结构不同；数值对照应使用结构匹配的相同权重。
