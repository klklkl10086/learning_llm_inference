# 06 分页缓存

要实现：

- 把 KV 存储分成固定大小的物理块。
- 维护每个请求的 block table 和空闲块列表。
- 分配、扩容、回收，处理容量不足。
- 相同前缀共享缓存块，通过引用计数控制释放。

完成标准：共享块不会被提前释放，全部请求结束后所有块都能回收。

参考资料（按顺序看）：

- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：先理解逻辑块、物理块、碎片与缓存共享。
- [nano-vLLM 块管理源码](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/block_manager.py)：对照实现空闲块、引用计数、分配与回收。
- [vLLM 前缀缓存设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)：理解前缀缓存键和共享块的生命周期。
