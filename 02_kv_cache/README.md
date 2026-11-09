# 02 KV cache

在上一阶段模型上实现：

- Prefill 保存每层 K/V，decode 每次只处理新 token。
- 正确处理位置偏移和 attention mask。
- 先用拼接缓存，再改成预分配缓存。
- 加入 RoPE，比较缓存大小和生成耗时。

完成标准：相同输入和权重下，缓存与完整重算的 logits 在合理误差内一致。

参考资料（按顺序看）：

- [Transformers KV cache 文档](https://huggingface.co/docs/transformers/kv_cache)：了解动态缓存、静态缓存和缓存的使用方式。
- [RoFormer / RoPE 论文](https://arxiv.org/abs/2104.09864)：理解对 Q/K 施加旋转，以及位置差如何进入注意力计算。
