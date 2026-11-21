# 05 调度与请求生命周期

当前有两个版本：[本专题 schedule.py](schedule.py) 和 [系统主线 scheduler.py](../llm_inference/scheduler.py)。检查时（2026-11-20）除文件头外的实现一致；系统后续使用主线，专题版本保留作学习参照。本次未移动或修改源码。

当前已有 `Request`、`ScheduledRequest`、`BatchPlan`、请求注册、队列、批次选择和统一 `update()`。旧文档提到的 `scheduler_callbacks_draft.py` 当前不存在，`on_prefill_done()` 也不是现在的接口。

## 当前需要完成的行为

1. 自行修正 `update()` 中 `sample_tokens` 与形参 `sampled_tokens` 不一致的问题；错误发生前已推进计算进度，修正后需用新请求重新验证。
2. 与 ModelRunner 约定计划顺序、状态映射和采样结果；让逻辑进度与物理 KV 长度一致。
3. 先验证已有的生成上限结束处理，再补充 EOS、取消和模型状态释放。
4. 与 [07 最小引擎](../07_inference_engine/README.md) 一起跑通执行循环。
5. 用不同到达时间和生成长度观察优先级与公平性，并验证错误批次/错误结果不会破坏状态。

当前策略：上一完成批次若为 prefill 且有 decode，则先 decode；否则尝试接纳 prefill，再选择 decode。一个批次不混合两个阶段。decode 被预算截断时，已执行的活跃请求会在 update 中轮转到队尾。

prefill 按补齐后的 `batch_size × 最长 prompt 长度` 计算预算，decode 每请求输入一个 token。活跃请求上限、输入 token 预算和 KV 显存预算分别约束不同资源。

## 后续扩展

第一版先保留完整 prompt prefill，验证请求执行中能接纳新请求、退出已结束请求。然后再做分块 prefill；它需要模型支持在历史 KV 上继续处理多个 token，不能只改调度器预算。

`Request.request_id` 和结果映射声明为 `str`，`add_request()` 参数目前标为 `int`，后续自己统一。构造器预算的有效性、EOS/取消、上下文上限和失败后的状态处理仍需要完善。详见[检查记录](../docs/REVIEW_2026-11-20.md)。

规划见[系统设计](../docs/ARCHITECTURE.md)和[学习路线](../docs/LEARNING_PLAN.md)。参考：[nano-vLLM 调度器源码](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py)。对照时记录 commit，映射到自己的最小实现。
