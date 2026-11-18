# 推理实验记录

这里只记录实际执行过的实验；未运行或未测到的项目写“待测”，不要填写估计值。每次比较使用同一模型、权重、输入和生成策略，并保存可复现的命令。

目录整理后，基础 KV cache 实验仍从 `02_kv_cache/` 运行；RoPE 与异长批量 decode 文件分别移至 `03_rope/`、`04_batched_decode/`。下文保留当时的测量结果；重新执行异长测试和 benchmark 时，请在 `04_batched_decode/` 运行同一命令。

## 2026-11-18：不同长度请求批量 decode 正确性

- 环境：CPU；`kvcache` Conda 环境，PyTorch `2.9.1+cu128`。模型使用固定随机种子 7、2 层、4 个 head、`d_model=32`、`max_seq_len=12`，未加载训练 checkpoint。
- 输入与基准：提示长度 3/7，比较合批 decode、逐请求缓存 decode 和完整重算；覆盖连续多轮、请求换序、单请求、完整 `inference_mode` 生成流程及状态边界。
- 运行命令：当时在 `02_kv_cache/` 目录执行 `/workplace/home/lizhiying/.conda/envs/kvcache/bin/python -m unittest -v test_decode_batch`；目录整理后从 `04_batched_decode/` 运行。
- 结果：8 个测试全部通过；logits 对照使用 `rtol=1e-5, atol=1e-6`。本次没有测量吞吐、延迟或显存。

## 2026-11-18：KV cache 单卡性能基线

- 环境：`cuda:0`，NVIDIA A30 24 GiB，驱动 `580.178.04`；PyTorch `2.9.1+cu128`，CUDA runtime `12.8`，FP32。代码基于提交 `1c0c003` 和本次工作区内的未提交 benchmark 修改。
- 计时：`torch.inference_mode()`，预热 2 次，重复 7 次取中位数；CUDA 计时边界调用 `torch.cuda.synchronize()`。每个对照内保持模型权重、输入 token 与形状相同；decode 使用预设 token，不含采样、请求调度和排队。下表的 `token/s` 只按 decode 阶段的 `batch × steps / 时间` 计算。TTFT 未测。

### 绝对位置编码：`torch.cat` 对比预分配

读取 `01_transformer/runs/demo/model.pt`；模型为 6 层、4 head、`d_model=256`、`d_ff=1024`、`max_seq_len=256`、词表 330。两个实现的 logits 完全一致（记录到的最大绝对差为 0）。在 `02_kv_cache/` 执行：

```bash
/workplace/home/lizhiying/.conda/envs/kvcache/bin/python benchmark_kv_cache.py --device cuda --batch-size 1 --prompt-len 64 --decode-steps 32 --warmup 2 --repeats 7
/workplace/home/lizhiying/.conda/envs/kvcache/bin/python benchmark_kv_cache.py --device cuda --batch-size 4 --prompt-len 128 --decode-steps 64 --warmup 2 --repeats 7
```

| 负载 | 版本 | Prefill (ms) | Decode (ms) | ms/step | token/s | KV 分配/有效 (MiB) | 峰值额外分配 (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B=1，P=64，S=32 | `torch.cat` | 2.964 | 90.353 | 2.824 | 354.2 | 1.125 / 1.125 | 2.250 |
| B=1，P=64，S=32 | 预分配 | 3.286 | 92.467 | 2.890 | 346.1 | 3.000 / 1.125 | 3.750 |
| B=4，P=128，S=64 | `torch.cat` | 3.180 | 180.034 | 2.813 | 1422.0 | 9.000 / 9.000 | 18.000 |
| B=4，P=128，S=64 | 预分配 | 3.400 | 183.581 | 2.868 | 1394.5 | 12.000 / 9.000 | 18.001 |

在这两组小模型负载下，预分配版 decode 没有更快。这里的峰值额外分配以 prefill 前的模型常驻显存为基线，包含缓存、临时张量与计算中间量；`KV 分配/有效` 只统计 K/V 张量，不等于进程总显存。

### RoPE：不同长度请求逐个 decode 对比合批 decode

两条路径使用同一个随机初始化的 `decode_batch.LiteGPT`，配置与上面相同，随机种子 2026；各请求分别 prefill。逐请求路径调用该模型的缓存 `forward`，合批路径调用 `decode_batch`。提示长度为 32/64/96/128，每个请求处理 32 个预设 token；各步 logits 最大绝对差 `5.07e-07`。原在 `02_kv_cache/` 执行；目录整理后从 `04_batched_decode/` 重跑：

```bash
/workplace/home/lizhiying/.conda/envs/kvcache/bin/python benchmark_decode_batch.py --device cuda --prompt-lens 32,64,96,128 --decode-steps 32 --warmup 2 --repeats 7
```

| 路径 | 分别 Prefill (ms) | Decode (ms) | ms/step | token/s | KV 分配/有效 (MiB) | Decode 峰值额外分配 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 逐请求 | 20.333 | 608.794 | 19.025 | 210.3 | 12.000 / 5.250 | 0.017 |
| 合批 | 20.359 | 213.249 | 6.664 | 600.2 | 12.000 / 5.250 | 1.299 |

在这组负载下，合批 decode 阶段约快 `2.85×`，同时需要更多临时张量。此处的 `Decode 峰值额外分配` 从分别 prefill 结束后的常驻显存计起，不能与上表的峰值列直接比较。预分配 KV 的容量在两条路径中相同；合批额外开销主要来自按最长请求建立的临时 K/V。性能结果只对应当前小模型、固定输入和本机环境，未包含真正服务系统的请求到达、采样或调度开销。

## 实验模板

### 问题与结论

- 日期、阶段：
- 本次只改变的变量：
- 预期现象：
- 实际结论与适用范围：

### 环境与输入

- 代码版本或提交号：
- 设备、GPU 型号、PyTorch/CUDA 版本：
- 模型与 checkpoint（RoPE/绝对位置编码要区分）：
- dtype、batch/并发数、prompt 长度、生成步数：
- 预热次数、重复次数、GPU 同步方式：
- 运行命令：

### 正确性

- 比较对象与基准：
- 最大/平均 logits 误差，或生成结果差异：
- 覆盖的边界情况：

### 性能

| 版本 | prefill 耗时 | 首 token 延迟（TTFT） | decode 每 token 延迟 | 吞吐（token/s） | KV 占用 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 改动后 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

- 测量口径、重复测量的统计量（仅测模型前向时，TTFT 填“未测”）：
- 异常现象和下一步验证：
