# 06 分页缓存

先完成 [05 调度](../05_scheduler/README.md) 与 [07 最小引擎](../07_inference_engine/README.md) 的执行循环，再在同一个 [llm_inference](../llm_inference/README.md) 系统中替换缓存布局。编号保留为学习专题标识。

## 接入顺序

1. 从主线模型现有的连续 LayerKVCache 出发，明确谁申请、谁写入、谁回收。
2. 实现固定大小的物理块、空闲块列表和每请求 block table，先处理独占块。
3. 让 ModelRunner 把物理块映射传给模型，按逻辑 token 位置读写 KV；可以先用 PyTorch 索引还原有效序列验证结果。
4. 处理容量不足与请求结束后的回收，再讨论是否需要抢占和重算。
5. 正确性稳定后再增加前缀共享、引用计数及共享块写入时的处理。
6. 再研究直接访问分页布局的 attention 实现，并比较临时还原连续 KV 的开销。

未来缓存管理器放入主线，负责块分配、映射和释放。Scheduler 根据可用资源做选择，ModelRunner 准备映射，模型执行张量读写。具体边界见[系统设计](../docs/ARCHITECTURE.md)。

完成标准：与连续缓存使用相同权重和输入时输出接近；块边界上的写入正确；结束和取消后能回收资源；共享块不会提前释放。

参考资料：[PagedAttention 论文](https://arxiv.org/abs/2309.06180)、[nano-vLLM 块管理源码](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/block_manager.py)、[vLLM 前缀缓存设计](https://docs.vllm.ai/en/latest/design/prefix_caching/)。
