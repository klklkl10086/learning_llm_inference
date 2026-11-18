# 05 调度

承接 [04 异长批量 decode](../04_batched_decode/README.md)，决定不同时间进入的请求何时 prefill、何时加入 decode batch。`scheduler.py` 已有草稿，后续按下列目标完善。

要实现：

- 请求的等待、运行、完成和取消状态。
- 固定批处理，再实现 continuous batching。
- 每轮 token 预算和长输入的分块 prefill。
- 请求结束后释放缓存。

完成标准：多个不同长度的请求能正确推进，能解释每轮调度了谁。

参考资料（按顺序看）：

- [nano-vLLM 调度器源码](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py)：重点看 `waiting`、`running`、`schedule`、`preempt` 和 `postprocess`。
- [vLLM 优化说明](https://docs.vllm.ai/en/latest/configuration/optimization/)：重点阅读 chunked prefill 和批处理 token 预算，理解延迟与吞吐的取舍。
