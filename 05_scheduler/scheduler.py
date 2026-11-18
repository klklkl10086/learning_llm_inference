"""

由于尚未实现批量prefill,因此一次只会有一个prompt进行prefill
先约定整个prompt一次prefill完 然后才放入running
调度器 请求状态管理 :
请求在不同时间到达时，决定
1. 每轮让哪些请求执行prefill
2. 哪些请求组成 decode batch
3. 何时结束并释放缓存
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, Sequence


class RequestStateLike(Protocol):
    """现有decode_batch.py中RequestState所提供的接口。"""

    token_ids: list[int]

    @property
    def cached_len(self) -> int: ...

    @property
    def pending_token(self) -> int: ...

    def append_token(self, token_id: int) -> None: ...


class RequestStatus(Enum):
    """
    请求处于哪个阶段,定义WAITING、RUNNING、FINISHED、CANCELLED
    """

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    CANCELED = auto()


@dataclass(eq=False)
class InferenceRequest:
    """
    保存请求 ID、prompt、最多生成 token 数、状态、生成结果，以及可选的 RequestState
    """

    request_id: str
    prompt_ids: tuple[int, ...]  # 零个或多个元素的元组
    max_new_tokens: int
    status: RequestStatus = RequestStatus.WAITING

    # prefill前为None()之后是现有的 RequestState
    state: RequestStateLike | None = None  # 记录每个请求的token 缓存等信息

    # 单独保存输出，因为结束时会释放state及其KV cache
    output_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None


@dataclass(frozen=True)
class SchedulePlan:
    """
    表示本轮选出的 prefill 请求和 decode 请求
    """

    # 当前实现逐请求 prefill，所以每轮最多选一个,因为尚未实现批量prefill
    prefill: tuple[InferenceRequest, ...]
    # 已完成 prefill 的请求，可以一起执行 decode_batch
    decode: tuple[InferenceRequest, ...]  # 零个或多个元素的元组


class Scheduler:
    """
    管理等待/运行队列，选请求，并在执行结果返回后更新状态
    """

    def __init__(
        self,
        max_num_seqs: int,
        max_seq_len: int,
        eos_token_id: int | None = None,
    ):
        if max_num_seqs <= 0 or max_seq_len <= 0:
            raise ValueError("max_num_seqs 和 max_seq_len 必须为正数")

        self.max_num_seqs = max_num_seqs
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id

        self.waiting: deque[InferenceRequest] = deque()  #双端队列
        self.running: deque[InferenceRequest] = deque() 
        self.requests: dict[str,InferenceRequest] = {}

        self._active_plan:SchedulePlan | None = None
        self._prefill_done=False
        self._decode_done = False


    def add_request(
        self,
        request_id:str,
        prompt_ids:Sequence[int],
        max_new_tokens: int,
    )-> InferenceRequest:
        if request_id in self.requests:
            raise ValueError(f"重复的 request_id: {request_id}")
        if not prompt_ids:
            raise ValueError("prompt 不能为空")
        if len(prompt_ids) > self.max_seq_len:
            raise ValueError("prompt 超过 max_seq_len")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能为负数")
        
        req = InferenceRequest(
            request_id = request_id,
            prompt_ids = tuple(int(x) for x in prompt_ids),
            max_new_tokens =  max_new_tokens
        )

        self.requests[request_id] = req
        
        if max_new_tokens == 0:
            req.status = RequestStatus.FINISHED
            req.finish_reason = "max_new_tokens"
        else:
            self.waiting.append(req)
        
        return req
    

    def schedule(self) -> SechedulePlan |None:
        if self._active_plan is not None:
            raise RuntimeError("上一轮尚未处理完，不能再次调度")
    
        prefill = None
        if self.waiting and len(self.running) < self.max_num_seqs:
            prefill = self.waiting.popleft()

        decode = tuple(self.running)#tuple固定本轮的请求名单和顺序

        if prefill is None and not decode:#无请求可以调度
            return None
         
        plan = SchedulePlan(prefill=prefill,decode=decode)
        self._active_plan = plan
        self._prefill_done = prefill is None
        self._decode_done = not decode
        return plan

    def on_prefill_done(
        self,
        request: InferenceRequest,
        state: RequestStateLike,
        first_token_id: int,
    ) -> None:
        plan = self._require_active_plan()
        
        