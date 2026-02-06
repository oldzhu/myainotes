# vLLM custom ops mapping table (overview)

[English](vllm-custom-ops-mapping-table.md) | [Chinese (ZH-CN)](vllm-custom-ops-mapping-table.zh-CN.md)

This overview links to pipeline-stage mapping tables. Each table maps Python call sites
to native kernels and the role in GPT inference.

## Pipeline stages

- KV cache stage: [vllm-custom-ops-mapping-kv-cache.md](vllm-custom-ops-mapping-kv-cache.md)
- Attention stage: [vllm-custom-ops-mapping-attention.md](vllm-custom-ops-mapping-attention.md)
- Normalization + RoPE + activation stage: [vllm-custom-ops-mapping-norm-activation.md](vllm-custom-ops-mapping-norm-activation.md)

## Notes

- New stages (quant/GEMM, distributed/AR, CPU backends) will be added as separate files.
