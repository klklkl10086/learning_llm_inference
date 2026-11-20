"""
来源 ./05_scheduler/schedule.py
整批 prefill 或整批 decode尚未实现分块prefill
调度策略:
1. 上一批是 prefill并且有可执行的 decode
       执行 decode
2. 否则有空闲槽位且能接纳 prefill
       执行 prefill
3. 否则有可执行的decode
       执行 decode
4. 否则
       没有任务
同一个批次中prefill 和 decode 不同时出现。
"""

from dataclasses import dataclass
from enum import Enum, auto
from collections import deque


class RequestStatus(Enum):
    WAITING = auto()  # 等待
    ACTIVE = auto()  # 占用并发位置,已经接纳,本轮是否执行由调度计划决定。
    FINISHED = auto()  # 结束


@dataclass
class Request:
    request_id: str

    token_ids: list[int]  # prompt + 已生成的token
    prompt_len: int  # prompt=token_ids[:prompt_len]
    max_new_tokens: int

    num_computed_tokens: int = 0

    status: RequestStatus = RequestStatus.WAITING
    finish_reason: str | None = None

    @property
    def num_generated_tokens(self) -> int:
        return len(self.token_ids) - self.prompt_len


class BatchKind(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    start_pos: int
    input_token_ids: tuple[int, ...]

    @property
    def num_tokens(self) -> int:
        return len(self.input_token_ids)


@dataclass(frozen=True)
class BatchPlan:
    batch_id: int
    kind: BatchKind
    items: tuple[ScheduledRequest, ...]


class Scheduler:

    def __init__(
        self,
        max_active_requests,
        max_batch_input_tokens,
    ):
        self.max_active_requests = max_active_requests
        self.max_batch_input_tokens = max_batch_input_tokens

        self._requests: dict[str, Request] = {}

        self._waiting: deque[str] = deque()
        self._active: deque[str] = deque()  # 表示请求已经占用并发槽位

        self._inflight_plan: BatchPlan | None = None  # 本轮模型执行批次

        self._last_completed_kind: BatchKind | None = None

        self._next_batch_id = 0

    def _select_prefill_requests(self) -> tuple[str, ...]:
        free_slots = self.max_active_requests - len(self._active)

        if free_slots <= 0:
            return ()
        selected: list[str] = []
        longest_prompt = 0

        for request_id in self._waiting:
            if len(selected) >= free_slots:
                break
            req = self._requests[request_id]
            if req.status is not RequestStatus.WAITING:
                raise RuntimeError(f"等待队列中的请求 {request_id} 状态错误")
            if req.num_computed_tokens != 0 or len(req.token_ids) != req.prompt_len:
                raise RuntimeError(f"等待请求 {request_id} 已有计算或生成进度")
            next_batch_size = len(selected) + 1
            next_longest_prompt = max(longest_prompt, req.prompt_len)

            padded_input_tokens = next_batch_size * next_longest_prompt

            if (
                padded_input_tokens > self.max_batch_input_tokens
            ):  # 使用padded后的token数目进行比较而非真实需要计算的token数目
                break
            selected.append(request_id)
            longest_prompt = next_longest_prompt
        return tuple(selected)

    def _select_decode_requests(self) -> tuple[str, ...]:
        """
        对于已经完成 prefill、尚未结束的请求:
        len(req.token_ids) == req.num_computed_tokens + 1
        """
        selected: list[str] = []

        for request_id in self._active:  # 检查已经完成 prefill、尚未结束的请求

            if len(selected) >= self.max_batch_input_tokens:
                # 每个decode请求只有一个token需要计算
                break

            req = self._requests[request_id]

            if req.status is not RequestStatus.ACTIVE:
                raise RuntimeError(f"等待队列中的请求 {request_id} 状态错误")
            # 当前不支持分块prefill,上一批次完成后,active的prompt应该全部完成了prefill
            if req.num_computed_tokens < req.prompt_len:
                raise RuntimeError(f"活跃请求 {request_id} 的 prefill 尚未完成")

            pending_tokens = len(req.token_ids) - req.num_computed_tokens
            if pending_tokens != 1:
                raise RuntimeError(
                    f"请求 {request_id} 应恰有一个待计算 token，"
                    f"实际为 {pending_tokens}"
                )

            selected.append(request_id)

        return tuple(selected)

    def _build_plan(
        self,
        kind: BatchKind,
        request_ids: tuple[str, ...],
    ) -> BatchPlan:
        if not request_ids:
            raise RuntimeError("不能创建空批次")
        items: list[ScheduledRequest] = []

        for request_id in request_ids:
            req = self._requests[request_id]

            if kind is BatchKind.PREFILL:
                start_pos = 0
                input_token_ids = tuple(req.token_ids[: req.prompt_len])
            elif kind is BatchKind.DECODE:
                start_pos = req.num_computed_tokens

                input_token_ids = (req.token_ids[start_pos],)
            else:
                raise RuntimeError("未知批次类型")
            items.append(
                ScheduledRequest(
                    request_id=request_id,
                    start_pos=start_pos,
                    input_token_ids=input_token_ids,
                )
            )

        plan = BatchPlan(batch_id=self._next_batch_id, kind=kind, items=tuple(items))

        # 如果是prefill证明其第一次进入推理,需要初始化状态status
        if kind is BatchKind.PREFILL:
            for request_id in request_ids:
                queued_id = self._waiting.popleft()

                if queued_id != request_id:
                    raise RuntimeError("等待队列与调度名单不一致")
                req = self._requests[request_id]
                req.status = RequestStatus.ACTIVE
                self._active.append(request_id)
        self._next_batch_id += 1
        self._inflight_plan = plan
        return plan

    def schedule(self) -> BatchPlan | None:
        if self._inflight_plan is not None:
            raise RuntimeError("上一批尚未完成，不能再次调度")

        decode_ids = self._select_decode_requests()  # 计算候选名单

        if self._last_completed_kind is BatchKind.PREFILL and decode_ids:
            return self._build_plan(
                BatchKind.DECODE,
                decode_ids,
            )

        # 如果没有上面的那种情况,接纳新的请求
        prefill_ids = self._select_prefill_requests()
        if prefill_ids:
            return self._build_plan(
                BatchKind.PREFILL,
                prefill_ids,
            )
        # 没有可以接纳的新请求，就推进已有请求。
        if decode_ids:
            return self._build_plan(
                BatchKind.DECODE,
                decode_ids,
            )

        if self._waiting or self._active:
            raise RuntimeError("存在未结束请求，但没有可执行批次，请检查请求状态或预算")

        return None

    def add_request(
        self,
        request_id: int,
        prompt_token_ids: list[int],
        max_new_tokens: int,
    ) -> Request:
        if request_id in self._requests:
            raise ValueError(f"重复的 request_id: {request_id}")
        if not prompt_token_ids:
            raise ValueError("prompt 不能为空")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能为负数")
        if max_new_tokens > 0 and len(prompt_token_ids) > self.max_batch_input_tokens:
            raise ValueError("prompt 超过本轮输入预算，当前不支持分块 prefill")
        req = Request(
            request_id=request_id,
            token_ids=list(prompt_token_ids),
            prompt_len=len(prompt_token_ids),
            max_new_tokens=max_new_tokens,
        )
        self._requests[request_id] = req
        if max_new_tokens == 0:
            req.status = RequestStatus.FINISHED
            req.finish_reason = "max_new_tokens"
        else:
            self._waiting.append(request_id)
        return req

    def update(
        self,
        batch_id: int,
        sampled_tokens: dict[str, int],
        # 表示每个请求本轮生成的token
        # 例如sampled_tokens = {"requestA": 42, "requestB": 17}
        # 请求A本轮采样得到 token 42
    ) -> tuple[str, ...]:
        """接收执行结果，推进请求并结束本轮计划"""
        plan = self._inflight_plan

        if plan is None:
            raise RuntimeError("当前没有等待更新的批次")
        if batch_id != plan.batch_id:
            raise ValueError(
                f"批次编号不匹配：" f"当前为 {plan.batch_id}，收到 {batch_id}"
            )
        expected_ids = {item.request_id for item in plan.items}
        if (
            set(sampled_tokens) != expected_ids
        ):  # set(sampled_tokens) 会取出它的所有键(请求编号)
            raise ValueError("采样结果中的请求与调度计划不一致")
        for item in plan.items:
            req = self._requests[item.request_id]
            if req.num_computed_tokens != item.start_pos:
                raise RuntimeError(f"请求 {item.request_id} 的计算进度与计划不一致")
        finished_ids: list[str] = []
        for item in plan.items:
            req = self._requests[item.request_id]
            req.num_computed_tokens += item.num_tokens

            next_token = sample_tokens[item.request_id]
            req.token_ids.append(next_token)

            if req.num_generated_tokens >= req.max_new_tokens:
                req.status = RequestStatus.FINISHED
                req.finish_reason = "max_new_tokens"
                self._active.remove(req.request_id)
                finished_ids.append(req.request_id)
        for item in plan.items:
            req = self._requests[item.request_id]
            if req.status is RequestStatus.ACTIVE:
                # 将仍然活跃的本批请求移到队尾
                self._active.remove(req.request_id)
                self._active.append(req.request_id)
        self._last_completed_kind = plan.kind
        self._inflight_plan = None

        return tuple(finished_ids)
