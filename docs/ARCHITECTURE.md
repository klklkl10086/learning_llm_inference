# 推理系统的组件与数据流

系统主线位于 [llm_inference](../llm_inference/README.md)。当前只有模型和未完成的调度器；下面区分现有行为与下一步的设计，避免把目录结构当作已经实现的功能。

## 组件边界

| 组件 | 负责 | 当前情况 |
| --- | --- | --- |
| Request | 请求 ID、输入和输出 token、生成限制、生命周期及逻辑计算进度 | 暂时定义在 scheduler.py，字段与命名仍待继续完善 |
| Scheduler | 接纳请求，选择本轮执行内容，根据结果推进请求与队列 | 草稿已迁入主线，回调尚未接完 |
| ModelRunner | 准备模型输入，执行前向和采样，将结果对应回请求 | 待实现 |
| Model | embedding、RoPE、attention、MLP、logits 和 KV 读写 | model.py 已支持完整批量 prefill 与单 token decode |
| KVCacheManager | 分配、扩容、释放缓存，后续管理块映射和共享 | 待实现；当前缓存由模型内部直接创建 |
| Engine | 接收请求并组织一轮 step，返回生成结果，协调回收 | 待实现 |

共享类型在需要跨模块使用时再从 scheduler.py 移到 request.py。这里列出的职责是后续手写时的目标，不要求现在一次性拆出所有类。

## 当前模型契约

prefill 输入多个非空 prompt，按右侧补齐后执行。它返回每个 prompt 最后有效位置的 logits 和逐请求 RequestState，后者持有 token 列表及各层连续 KV cache。

```text
prompt [a, b, c]
  prefill 完成          token_ids = [a, b, c]       cached_len = 3
  采样 x 并追加        token_ids = [a, b, c, x]    cached_len = 3
  decode 计算 x        token_ids = [a, b, c, x]    cached_len = 4
  采样 y 并追加        token_ids = [a, b, c, x, y] cached_len = 4
```

生成 token 与计算 token 是两件事。模型前向推进物理 KV 的长度；结果处理统一追加生成 token，判断 EOS 和生成上限。decode 前必须恰好有一个尚未计算的 token。

现有 InferenceRequest 和模型 RequestState 是两个不同层次：前者描述生命周期与输出，后者描述模型执行状态。当前代码暂时同时保存了部分 token 数据；接通 ModelRunner 时明确对应关系，之后再统一 token 的所有权。任何后续拆分都应避免让逻辑进度和各层 KV 长度各自独立增长。

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

第一版继续沿用整批 PREFILL 或整批 DECODE，先形成闭环。当前 schedule 会优先接纳可执行的 prefill，再选择 decode；观察持续到达的短请求是否延迟已有请求，并把优先级明确写成策略。

第一版同时只允许一个未完成计划。模型执行抛出异常后，要明确计划与状态如何收尾；若 KV 已被部分写入，不能只清空计划后直接重试。

## 后续如何接入优化

| 功能 | 主要改动位置 | 需要核对的约定 |
| --- | --- | --- |
| 连续组批 | Scheduler、Engine | 每轮可加入新请求并移除结束请求，保持请求与结果对应 |
| 分块 prefill | 计划、Scheduler、ModelRunner、Model | 逐请求记录起始位置与本轮 token 数，支持历史 KV 上继续 prefill；prompt 未算完时不提前追加生成 token |
| 分页缓存 | 缓存管理器、ModelRunner、attention | 逻辑位置映射到物理块，结束时回收，数值对照连续缓存 |
| attention 等计算优化 | Model 中对应的计算路径 | 相同权重与输入下核对 logits，再测耗时和显存 |
| 张量并行 | 并行层、ModelRunner 与多进程执行 | 请求顺序、通信顺序、分片布局与单卡结果对应 |

后续计划可以逐步扩展为“请求 ID、起始位置、本轮计算 token 数”。缓存容量、输入 token 预算、活跃请求数量分别描述不同资源，命名和计量口径需要分开说明。

## 对照版本

[04 的普通版](../04_batched_decode/batch_prefill_decode.py) 保留通用 forward，可用于完整重算和逐请求缓存对照。系统运行使用主线 model.py，后续功能围绕这一份实现演进。

[01](../01_transformer/README.md) 和 [02](../02_kv_cache/README.md) 使用绝对位置编码，[03](../03_rope/README.md) 及后续使用 RoPE。优化数值对照要使用结构匹配的相同权重，不能直接把 01 checkpoint 当作已训练的主线 RoPE 模型。
