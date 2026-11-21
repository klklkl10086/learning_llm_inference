# 类的设计
| 对象 | 保存内容 | 谁修改 |
|---|---|---|
| `Request` | 完整 token 序列、生成限制、逻辑进度 | Scheduler |
| `ScheduledRequest` | 请求 ID、本轮起始位置、本轮数量 | 创建后固定 |
| `KVCache` | 一个请求所有层的 K/V、缓存容量和有效长度 | 模型前向 |
| `ModelRunner` | 请求 ID 到缓存状态的映射 | ModelRunner |

`KVCache.keys[layer_idx]` 和 `values[layer_idx]` 的形状均为
`[1, n_heads, capacity, head_dim]`。各层通过 `write()` 写入同一个逻辑位置；
完整模型前向结束后，统一调用 `advance()` 推进一次 `length`。
缓存不保存 token 序列，decode 的 token 显式传入 `model.decode_batch(token_ids, caches)`。

调度器与执行器可以这样接通：

```python
while (plan := scheduler.schedule()) is not None:
    requests = {
        item.request_id: scheduler.get_request(item.request_id)
        for item in plan.items
    }
    sampled_tokens = runner.execute(plan, requests)
    finished_ids = scheduler.update(plan.batch_id, sampled_tokens)
    runner.release(finished_ids)
```

当前采用逐请求的连续缓存。后续分页时，可将 `KVCache` 内部改成请求的 block table，
实际 K/V 张量由全局缓存池持有，一个请求关联多个物理 block。
