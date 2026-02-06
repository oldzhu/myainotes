# vLLM 自定义算子映射表：采样 + logits 阶段

[English](vllm-custom-ops-mapping-sampling.md) | [简体中文](vllm-custom-ops-mapping-sampling.zh-CN.md)

范围：logits 后处理与采样相关自定义算子。

## 映射表：采样 + logits

| 算子 | Python 调用点 + GPT 推理用途 | 原生实现 + 说明/伪代码 |
|---|---|---|
| `apply_repetition_penalties_` | 封装见 [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L399)；logits 工具调用 [vllm/vllm/model_executor/layers/utils.py](vllm/vllm/model_executor/layers/utils.py#L104)。用途：在采样前对 logits 应用重复惩罚。 | CUDA 内核在 [vllm/csrc/sampler.cu](vllm/csrc/sampler.cu#L605)。对每个 token 根据正负值缩放 logits。伪代码：`logit = logit > 0 ? logit / penalty : logit * penalty`。
