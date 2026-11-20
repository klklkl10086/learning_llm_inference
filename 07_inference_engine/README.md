# 07 模型执行器与推理引擎

这个专题提前与 [05 调度](../05_scheduler/README.md) 一起完成。先用连续 KV cache 跑通最小系统，再加入 [06 分页缓存](../06_paged_cache/README.md)。后续实现集中写入 [llm_inference](../llm_inference/README.md)。

## 第一步 ModelRunner

未来文件为 `llm_inference/model_runner.py`，当前尚未创建。先包装已有 `model.py` 的两个入口：

- prefill：按计划顺序收集 prompts，执行 prefill_batch，采样首个 token，返回请求对应的模型状态和结果。
- decode：按计划顺序收集状态，执行 decode_batch，采样下一个 token，返回结果。
- 请求结束时释放执行器持有的状态引用；将来再调用独立缓存管理器回收物理块。

先使用 argmax。明确谁追加生成 token，避免执行器与调度器各追加一次。

## 第二步 Engine

未来文件为 `llm_inference/engine.py`，当前尚未创建。对外接收请求，使用一轮 step 串联：

```text
Scheduler.schedule
  → ModelRunner.execute
  → Scheduler 更新请求与队列
  → 回收已结束请求的模型状态
  → 返回本轮生成结果
```

先跑一个请求，再跑多个异长请求，最后在执行过程中加入新请求。每轮记录选中了谁、本轮输入 token 数、已缓存长度、生成数量和结束原因。

当前调度器的回调尚未完成。执行失败时如何清理正在执行的计划，以及模型已部分写入 KV 时如何处理，也应在接通执行循环时明确。

## 第三步 测量与优化

先用逐请求执行核对多请求的 token 和 logits，再固定负载，测首 token 延迟、后续 token 延迟、吞吐和显存。根据测量结果选择[优化专题](../docs/OPTIMIZATION.md)，将实际数据写入[实验记录](../docs/EXPERIMENT_LOG.md)。

完整组件边界见[系统设计](../docs/ARCHITECTURE.md)。参考：[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)、[vLLM 架构说明](https://docs.vllm.ai/en/latest/design/arch_overview/)、[PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/sdpa_tutorial.html)。
