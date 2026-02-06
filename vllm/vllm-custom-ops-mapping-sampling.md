# vLLM custom ops mapping table: sampling + logits stage

[English](vllm-custom-ops-mapping-sampling.md) | [Chinese (ZH-CN)](vllm-custom-ops-mapping-sampling.zh-CN.md)

Scope: logits post-processing and sampling-related custom ops.

## Mapping table: sampling + logits

| Op | Python call site(s) + GPT inference role | Native implementation + notes/pseudo code |
|---|---|---|
| `apply_repetition_penalties_` | Wrapper in [vllm/vllm/_custom_ops.py](vllm/vllm/_custom_ops.py#L399); called in logits utilities [vllm/vllm/model_executor/layers/utils.py](vllm/vllm/model_executor/layers/utils.py#L104). Role: apply repetition penalties to logits before sampling. | CUDA kernel in [vllm/csrc/sampler.cu](vllm/csrc/sampler.cu#L605). For each token, scales positive/negative logits by repetition factors. Pseudocode: `logit = logit > 0 ? logit / penalty : logit * penalty`.
