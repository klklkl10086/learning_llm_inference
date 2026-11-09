# 推理引擎学习

按顺序学习，每个目录里自行编写代码。

1. `01_transformer`：实现一个小型 Transformer 和文本生成。
2. `02_kv_cache`：加入 KV cache，验证与完整重算一致。
3. `03_scheduler`：实现多请求批处理和调度。
4. `04_paged_cache`：实现缓存块的分配、共享和回收。
5. `05_inference_engine`：组装小推理器，对照 nano-vLLM / vLLM。

参考资料（具体章节和源码见各目录 README）：

- [动手学深度学习：Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html)：中文原理与代码。
- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)：从小模型实现过渡到推理系统。
- [vLLM 架构说明](https://docs.vllm.ai/en/latest/design/arch_overview/)：理解调度、缓存管理与模型执行的分工。

按阶段选读即可；阅读源码时固定一个版本，并使用对应版本的文档。
