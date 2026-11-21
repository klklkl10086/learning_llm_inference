# 推理系统的组件与数据流

系统主线位于 [llm_inference](../llm_inference/README.md)。当前已有模型、调度器和 ModelRunner，可以通过调用循环接通；Engine 尚待实现。

## 组件边界

| 组件 | 负责 | 当前情况 |
| --- | --- | --- |
| Request | 请求 ID、输入和输出 token、生成限制、生命周期及逻辑计算进度 | 当前定义为 scheduler.py 中的 Request，记录 token、prompt 长度和计算进度 |
| Scheduler | 接纳请求，选择本轮执行内容，根据结果推进请求与队列 | 已有 schedule/update，计划记录起始位置和本轮 token 数 |
| ModelRunner | 准备模型输入，执行前向和采样，将结果对应回请求 | execute 读取 Request 的 token，保存逐请求 KVCache；release 删除结束请求的缓存 |
| Model | embedding、RoPE、attention、MLP、logits 和 KV 读写 | model.py 已支持完整批量 prefill 与单 token decode |
| KVCacheManager | 分配、扩容、释放缓存，后续管理块映射和共享 | 待实现；当前缓存由模型内部直接创建 |
| Engine | 接收请求并组织一轮 step，返回生成结果，协调回收 | 待实现 |

共享类型在需要跨模块使用时再从 scheduler.py 移到 request.py。这里列出的职责是后续手写时的目标，不要求现在一次性拆出所有类。

## 当前模型契约

prefill 输入多个非空 prompt，按右侧补齐后执行。它返回每个 prompt 最后有效位置的 logits 和逐请求 KVCache。一个 KVCache 保存所有层的 K/V、容量和统一的 length，不保存 token 列表。decode 显式接收输入 token ID 和对应缓存。

```text
prompt [a, b, c]
  prefill 完成          token_ids = [a, b, c]       KVCache.length = 3
  采样 x 并追加        token_ids = [a, b, c, x]    KVCache.length = 3
  decode 计算 x        token_ids = [a, b, c, x]    KVCache.length = 4
  采样 y 并追加        token_ids = [a, b, c, x, y] KVCache.length = 4
```

生成 token 与计算 token 是两件事。模型前向推进物理 KV 的长度；结果处理更新逻辑进度与生成 token。当前只实现了生成上限判断，EOS 处理尚待增加。decode 前必须恰好有一个尚未计算的 token。

Request 是 token 历史的唯一持有者，由 Scheduler.update 追加采样结果。ModelRunner 按 ScheduledRequest 的起始位置和数量读取输入，并维护请求 ID 到 KVCache 的映射。各层只写 KV，完整模型前向成功后统一推进一次缓存 length；Scheduler 接收结果后推进 num_computed_tokens，两者在一次完整 step 结束后应一致。

## 下一步的执行循环

```text
提交请求
  → Scheduler 登记到等待队列
  → Engine 发起 step
  → Scheduler 选择批次并保留本轮计划
  → ModelRunner 按计划准备输入、调用 Model、采样
  → Scheduler 接收结果、追加 token、更新请求
  → Engine 协调释放结束请求的模型状态
  → 下一轮 step
```

第一版继续沿用整批 PREFILL 或整批 DECODE，先形成闭环。当前策略是：上轮完成 prefill 且有 decode 时先 decode，否则尝试接纳 prefill，再选择 decode。update 会将本批仍活跃的请求移到队尾；用持续到达和预算截断的负载验证公平性。

第一版同时只允许一个未完成计划。模型执行或调度更新抛出异常后，要明确计划与状态如何收尾；若 KV 或逻辑进度已部分更新，不能只清空计划后直接重试。

## 后续如何接入优化

| 功能 | 主要改动位置 | 需要核对的约定 |
| --- | --- | --- |
| 连续组批 | Scheduler、Engine | 每轮可加入新请求并移除结束请求，保持请求与结果对应 |
| 分块 prefill | 计划、Scheduler、ModelRunner、Model | 逐请求记录起始位置与本轮 token 数，支持历史 KV 上继续 prefill；prompt 未算完时不提前追加生成 token |
| 分页缓存 | 缓存管理器、ModelRunner、attention | 逻辑位置映射到物理块，结束时回收，数值对照连续缓存 |
| attention 等计算优化 | Model 中对应的计算路径 | 相同权重与输入下核对 logits，再测耗时和显存 |
| 权重 / KV 量化 | 稳定后按需增加量化模块，并修改对应投影或缓存读写 | 明确存储格式、尺度元数据、计算 dtype 与误差 |
| CUDA / Triton 算子 | Model 中对应算子的稳定封装 | 保持模型计算定义和接口，分别测算子与端到端收益 |
| 张量并行 | 并行层、ModelRunner 与多进程执行 | 请求顺序、通信顺序、分片布局与单卡结果对应 |

计划已经记录“请求 ID、起始位置、本轮计算 token 数”。缓存容量、输入 token 预算、活跃请求数量分别描述不同资源，命名和计量口径需要分开说明。

## 对照版本

[04 的普通版](../04_batched_decode/batch_prefill_decode.py) 保留通用 forward，可用于完整重算和逐请求缓存对照。系统运行使用主线 model.py，后续功能围绕这一份实现演进。

[01](../01_transformer/README.md) 和 [02](../02_kv_cache/README.md) 使用绝对位置编码，[03](../03_rope/README.md) 及后续使用 RoPE。优化数值对照要使用结构匹配的相同权重，不能直接把 01 checkpoint 当作已训练的主线 RoPE 模型。
