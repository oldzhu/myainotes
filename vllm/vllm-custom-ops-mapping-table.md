# vLLM custom ops mapping table (overview)

[English](vllm-custom-ops-mapping-table.md) | [Chinese (ZH-CN)](vllm-custom-ops-mapping-table.zh-CN.md)

This overview links to pipeline-stage mapping tables. Each table maps Python call sites
to native kernels and the role in GPT inference.

## Pipeline stages

- KV cache stage: [vllm-custom-ops-mapping-kv-cache.md](vllm-custom-ops-mapping-kv-cache.md)
- Attention stage: [vllm-custom-ops-mapping-attention.md](vllm-custom-ops-mapping-attention.md)
- Normalization + RoPE + activation stage: [vllm-custom-ops-mapping-norm-activation.md](vllm-custom-ops-mapping-norm-activation.md)
- Sampling + logits stage: [vllm-custom-ops-mapping-sampling.md](vllm-custom-ops-mapping-sampling.md)
- Quantization + GEMM stage: [vllm-custom-ops-mapping-quant-gemm.md](vllm-custom-ops-mapping-quant-gemm.md)
- Distributed + all-reduce stage: [vllm-custom-ops-mapping-distributed-ar.md](vllm-custom-ops-mapping-distributed-ar.md)
- CPU backends stage: [vllm-custom-ops-mapping-cpu-backends.md](vllm-custom-ops-mapping-cpu-backends.md)

## Notes

- Stage files are intentionally scoped to keep tables focused; cross-stage ops are linked from their primary call sites.
