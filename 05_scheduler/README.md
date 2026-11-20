# 05 调度与请求生命周期

调度器代码已移到系统主线的 [llm_inference/scheduler.py](../llm_inference/scheduler.py)。原 `tmp.py` 改名为 [scheduler_callbacks_draft.py](scheduler_callbacks_draft.py)，保留其全部内容，继续作为手写时的参考草稿。

当前 Scheduler 有请求注册、队列、批次计划和选择逻辑，但 prefill 回调尚未写完。草稿中的回调与辅助方法仍是文件顶层定义，尚未接入类，不能视为已经完成的功能。

## 当前要接通的行为

1. 与 ModelRunner 约定：调度计划中的请求顺序如何对应模型返回的状态和 token。
2. 补完 prefill/decode 的结果处理；明确 KV 长度由模型推进，生成 token 由结果处理统一追加。
3. 处理生成上限、EOS、完成和取消，让请求退出活跃队列并释放模型状态。
4. 与 [07 最小引擎](../07_inference_engine/README.md) 一起跑通多请求执行循环。
5. 用不同到达时间和生成长度的请求观察接纳顺序、prefill/decode 优先级与公平性。

当前按 padding 后的 `batch_size × 最长 prompt 长度` 计算 prefill 输入预算，与主线模型的补齐执行方式一致。decode 每请求输入一个 token。输入 token 预算与活跃请求数量是不同限制，也都不能直接等同于总显存预算。

## 后续扩展

最小执行循环可用后，再引入逐请求计算进度和分块 prefill。分块 prefill 同时需要扩展模型在历史 KV 上继续处理多个 token 的能力，不能只修改调度器的预算。

规划中的组件关系见[系统设计](../docs/ARCHITECTURE.md)，实际顺序见[学习路线](../docs/LEARNING_PLAN.md)。

参考资料：[nano-vLLM 调度器源码](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py)、[vLLM 优化说明](https://docs.vllm.ai/en/latest/configuration/optimization/)。对照源码时记录具体版本，再映射到自己的最小实现。
