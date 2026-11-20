# 文件整理与代码保留记录

本次整理只移动 Python 文件、增加一个旧导入名的符号链接，并调整项目说明。整理前已有的 13 个 Python 源文件逐一按 SHA256 校验，迁移后内容保持一致；已有测试代码、训练代码和模型计算均未改写。

## 文件迁移

| 原位置 | 当前位置 | 用途 |
| --- | --- | --- |
| `04_batched_decode/batch_prefill_decode_optimized.py` | [llm_inference/model.py](../llm_inference/model.py) | 后续推理系统使用的模型 |
| `05_scheduler/scheduler.py` | [llm_inference/scheduler.py](../llm_inference/scheduler.py) | 继续手写的调度器 |
| `05_scheduler/tmp.py` | [05_scheduler/scheduler_callbacks_draft.py](../05_scheduler/scheduler_callbacks_draft.py) | 未接入类的回调草稿 |

01、02、03 的代码、数据与训练产物留在原位置。04 的普通版、测试和 benchmark 留在原位置。调度草稿没有自动拼接到 Scheduler；未完成的实现继续保留当前状态。

## 旧导入名

现有测试和 benchmark 使用 `from decode_batch import ...`。为保留这些源文件的全部内容，新增相对符号链接：

```text
04_batched_decode/decode_batch.py → batch_prefill_decode.py
```

这个路径仅提供普通版的旧模块名，后续系统使用 `llm_inference/model.py`。符号链接不维护另一份模型源码；复制项目时需要保留符号链接，使用 Git 获取项目可以保留该关系。

## 当前目录与后续代码

系统实现集中到 [llm_inference](../llm_inference/README.md)，编号目录保留学习内容和对照实验。ModelRunner、Engine、独立缓存管理器尚未创建，也没有新增空的 Python 占位文件。

[系统设计](ARCHITECTURE.md) 描述组件如何连接，[学习路线](LEARNING_PLAN.md) 将最小引擎提前到分页缓存之前。原有[历史实验记录](EXPERIMENT_LOG.md)保留原样，已有结果不代表新系统的端到端性能。

## 内容校验

[CODE_PRESERVATION.sha256](CODE_PRESERVATION.sha256) 记录整理前的 13 个 Python 源文件校验值，并使用整理后的路径。在项目根目录执行：

```bash
sha256sum -c docs/CODE_PRESERVATION.sha256
```

这份清单是本次文件整理的快照。以后自己继续修改代码时，校验值自然会变化；它不限制后续学习和开发，也不代替功能测试。
