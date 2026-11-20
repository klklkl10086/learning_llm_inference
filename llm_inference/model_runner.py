

class ModelRunner:
    def __init__(self, model):
        self.model = model.eval()

        # request_id -> 模型的 RequestState
        self._states = {}

    def execute(self, plan: BatchPlan) -> dict[str, int]:
        """执行计划，返回 请求ID -> 本轮采样token。"""
        ...

    def release(self, finished_ids: tuple[str, ...]) -> None:
        """移除已结束请求的模型状态。"""
        ...