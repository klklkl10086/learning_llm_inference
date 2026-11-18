# 03 RoPE 与单请求缓存

本阶段在预分配 KV cache 上引入 RoPE，代码是 [`RoPE.py`](RoPE.py)。Q/K 按各自 token 的位置旋转，prefill 保存旋转后的 K；后续 decode 使用缓存长度确定新 token 的位置。当前实现适用于单请求或 batch 内历史长度相同的请求。

完成标准：在**同一个 RoPE 模型、相同权重和输入**下，缓存 decode 与完整重算的 logits 在合理误差内一致；能解释位置偏移、Q/K 旋转和 K/V 的存储时机。

这里是一个独立的模型演进版本：它不再使用 [01 Transformer](../01_transformer/README.md) 的可训练绝对位置 embedding。01 阶段的 checkpoint 不能直接当作已经训练好的 RoPE 模型。检查缓存数值时可以固定随机权重；若要评价生成质量，则需要与 RoPE 架构匹配的训练权重。

下一阶段是[04 异长批量 decode](../04_batched_decode/README.md)：为每个请求分别保存位置和 KV 长度。

参考：[RoFormer / RoPE 论文](https://arxiv.org/abs/2104.09864)。
