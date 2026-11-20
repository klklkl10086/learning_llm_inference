# 从已有代码接成推理系统

目标是沿着同一套请求、模型执行和缓存约定持续增加功能，最终形成可以解释每轮行为、验证正确性并测量性能的推理系统。目录编号是学习专题，实际顺序按功能依赖调整。

## 当前起点

- [主线模型](../llm_inference/model.py)：由 batch_prefill_decode_optimized.py 原样迁入，已有批量 prefill/decode。
- [主线调度器](../llm_inference/scheduler.py)：由 05 原样迁入，结果处理未完成。
- [回调草稿](../05_scheduler/scheduler_callbacks_draft.py)：原 tmp.py，尚未接入 Scheduler。
- [04 对照实现与测试](../04_batched_decode/README.md)：保留完整前向、逐请求执行、已有测试和 benchmark。
- 01–03 保留模型、缓存和位置编码的学习过程；现有训练产物位置保持不变。

目录入口见[项目首页](../README.md)，职责与数据流见[系统设计](ARCHITECTURE.md)。

## 按可运行里程碑推进

| 顺序 | 要写的内容 | 完成标准 |
| --- | --- | --- |
| 1 模型契约 | 读懂主线 prefill/decode 的输入、输出和 KV 变化 | 能解释 pending_token 与 cached_len，使用同权重与 04 对照 |
| 2 ModelRunner | 包装模型执行、状态对应和 argmax 采样 | 独立运行完整 prefill 和多轮 decode，请求与结果顺序正确 |
| 3 Scheduler 与最小 Engine | 补完调度回调、结束处理、提交请求和 step | 先跑一个请求，再跑多个异长请求，结束后释放模型状态 |
| 4 连续组批 | 执行过程中接纳新请求，移除完成或取消的请求 | 日志能解释每轮选了谁，已有请求不会被持续插入的短请求无限推迟 |
| 5 分块 prefill | 计划表达每请求计算进度，模型支持历史 KV 上续算 | 长 prompt 可跨轮推进，结果与整段 prefill 对照一致 |
| 6 分页缓存 | 物理块、block table、分配和回收，再考虑共享 | 与连续缓存输出接近，块边界和回收正确 |
| 7 性能实验 | 固定负载、记录基线，逐个改动计算与存储路径 | 数值对照通过，能解释耗时、吞吐和显存变化 |
| 8 多 GPU | collective、小型并行层、接入现有执行器 | 与单卡输出接近，能区分通信开销和计算收益 |

最小引擎与调度一起完成，提前于分页缓存。首个引擎不依赖 block table，也不要求先完成分块 prefill。

## 下一次手写的具体顺序

1. 先读主线 model.py 中的 RequestState、prefill_batch 和 decode_batch，记录调用前后 token 数与缓存长度。
2. 新增 model_runner.py，先包装当前模型接口，不改模型计算，使用 argmax。
3. 参考回调草稿，继续完成 scheduler.py；确认每个 token 只追加一次。
4. 新增 engine.py，接通 schedule → execute → update → release。
5. 写一个多请求例子，逐轮输出请求 ID、阶段、本轮输入 token 数、缓存长度、生成数量和结束原因。
6. 对照逐请求执行检查 token 和 logits，再考虑增加新的调度策略。

ModelRunner、Engine 与上述例子当前均未创建；这里保留给后续自己实现。

## 验证入口与口径

在项目根目录、已安装 PyTorch 的环境中运行：

```bash
python -B -m unittest discover -s 01_transformer -p 'test_transformer.py' -v
python -B -m unittest discover -s 04_batched_decode -p 'test_decode_batch.py' -v
```

在对应实验目录运行已有 benchmark：

```bash
# 02_kv_cache/；默认读取 01_transformer/runs/demo/model.pt
python -B benchmark_kv_cache.py --device cuda --batch-size 1 --prompt-len 64 --decode-steps 32 --warmup 2 --repeats 7

# 04_batched_decode/；同一个随机 RoPE 模型的逐请求与合批比较
python -B benchmark_decode_batch.py --device cuda --prompt-lens 32,64,96,128 --decode-steps 32 --warmup 2 --repeats 7
```

04 的测试和 benchmark 当前针对普通版，通过旧名 decode_batch.py 的符号链接导入。它们不自动覆盖主线优化模型、调度器或未来引擎。新增系统功能时，围绕实际行为增加验证，保留普通版作对照。

数值比较固定权重、输入 token、eval 模式和精度，先核对 logits。性能比较记录设备、模型、提示长度、生成步数、并发数、预热、重复次数与计时边界；区分仅模型执行与包含排队、采样、调度的端到端耗时。GPU 计时在边界同步。

01–02 的绝对位置编码与 03 之后的 RoPE 是不同模型结构。历史 benchmark 的输入和测量口径也各不相同，不能跨实验直接计算加速比。已执行的数据保留在[实验记录](EXPERIMENT_LOG.md)，后续选题见[优化专题](OPTIMIZATION.md)。
