import torch

from .model import LiteGPT, KVCache
from .scheduler import BatchKind, BatchPlan, Request


class ModelRunner:
    def __init__(
        self,
        model: LiteGPT,
        pad_token_id: int = 0,
    ):
        self.model = model.eval()
        self.pad_token_id = pad_token_id

        #一个请求对应一个KVCache
        self._caches: dict[str, KVCache] = {}
        

    @torch.inference_mode()
    def execute(
        self,
        plan: BatchPlan,
        requests: dict[str, Request],
    ) -> dict[str, int]:
        """读取plan的Request类的token"""
        if plan.kind is BatchKind.PREFILL:
            logits = self._execute_prefill(plan, requests)

        elif plan.kind is BatchKind.DECODE:
            logits = self._execute_decode(plan, requests)

        else:
            raise RuntimeError("未知批次类型")

        # 两条执行路径都统一返回 [B, vocab_size]。
        next_token_ids = logits.argmax(dim=-1).tolist()

        return {
            item.request_id: token_id
            for item, token_id in zip(
                plan.items,
                next_token_ids,
            )
        }

    def _execute_prefill(
        self,
        plan: BatchPlan,
        requests: dict[str, Request],
    ) -> torch.Tensor:
        # 本次需要计算的token
        prompts = [
            requests[item.request_id].token_ids[
                item.start_pos : item.start_pos + item.num_tokens
            ]
            for item in plan.items
        ]

        logits, caches = self.model.prefill_batch(
            prompts,
            pad_token_id=self.pad_token_id,
        )

        for item, cache in zip(plan.items, caches):
            self._caches[item.request_id] = cache #按照请求编号保存KVcache

        # [B, vocab_size]
        return logits

    def _execute_decode(
        self,
        plan: BatchPlan,
        requests: dict[str, Request],
    ) -> torch.Tensor:
        caches: list[KVCache] = [] #每一个请求的KVcache
        input_token_ids: list[int] = []

        for item in plan.items:
            cache = self._caches[item.request_id] #取出历史记录保存的KVCache

            # 模型缓存的位置必须与计划的输入位置对应。
            if cache.length != item.start_pos:
                raise RuntimeError(
                    f"请求 {item.request_id} 的缓存进度" f"与调度计划不一致"
                )

            # 当前计划每请求只有一个 decode token。
            if item.num_tokens != 1:
                raise ValueError("decode 每请求应恰有一个输入 token")
            input_token_ids.append(
                requests[item.request_id].token_ids[item.start_pos]
            )
            caches.append(cache)

        # 运行过程中修改caches
        logits = self.model.decode_batch(input_token_ids, caches)

        # [B, 1, vocab_size] -> [B, vocab_size]
        return logits[:, -1, :]

    def release(
        self,
        finished_ids: tuple[str, ...],
    ) -> None:
        for request_id in finished_ids:
            self._caches.pop(request_id, None)
