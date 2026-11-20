# 推理系统主线

这个目录集中保存后续要接成系统的代码。现有 Python 文件只移动和重命名，内容保留原样。

## 已有文件

| 文件 | 来源 | 当前状态 |
| --- | --- | --- |
| [model.py](model.py) | `04_batched_decode/batch_prefill_decode_optimized.py` | 支持完整 prompt 的批量 prefill，以及每请求一个 token 的批量 decode |
| [scheduler.py](scheduler.py) | `05_scheduler/scheduler.py` | 请求、计划和批次选择已有草稿；结果处理尚未接完 |

模型提供 `LiteGPT`、`RequestState` 和 `LayerKVCache`。当前模型没有实现 `forward()` 或 `generate()`；执行入口是 `prefill_batch()` 和 `decode_batch()`，使用前调用 `model.eval()`。

- `prefill_batch(prompts, pad_token_id=0, max_new_tokens=None)` 返回 `[B, vocab_size]` 的 logits 和逐请求状态。
- `decode_batch(states)` 返回 `[B, 1, vocab_size]` 的 logits，并原地推进各层 KV cache 的长度。
- 采样和 `append_token()` 由调用方负责。刚生成的 token 到下一次 decode 才会写入 KV。
- prefill 当前只接受完整 prompt，不支持已有历史 KV 上的分块续算。
- prefill 的 `max_new_tokens` 参数当前用于规划连续缓存容量；分配职责以后再移交缓存管理器。

调度器的 `on_prefill_done()` 尚未完成；它引用的 `_require_plan()` 等方法还在[回调草稿](../05_scheduler/scheduler_callbacks_draft.py)中，没有接入 Scheduler 类。这个目录当前可以导入，但尚无可运行的 Engine。

## 接下来增加什么

以下文件尚未创建，按实际实现进度再增加：

| 未来文件 | 职责 |
| --- | --- |
| `model_runner.py` | 包装现有模型入口，准备输入、执行和采样，维护请求与模型状态的对应关系 |
| `engine.py` | 实现请求提交和 `step()`，串联调度、执行、状态更新与回收 |
| `request.py` | 当跨模块共享类型出现时，集中定义请求、计划、执行结果，避免重复定义 |
| `kv_cache.py` | 接入独立缓存管理后，承接缓存分配、扩容与回收 |

当前用 Python 的 namespace package 方式导入，从项目根目录可使用 `from llm_inference.model import LiteGPT`，无需添加空的 Python 文件。外部脚本以后通过 Engine 使用系统。

设计边界见[系统设计](../docs/ARCHITECTURE.md)，下一步任务见[学习路线](../docs/LEARNING_PLAN.md)。
