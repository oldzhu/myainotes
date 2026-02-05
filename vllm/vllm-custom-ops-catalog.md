# vLLM custom ops catalog (Python-exposed)

Repo: `/home/oldzhu/vllm`
Generated: `2026-02-05 16:24`

This document lists custom operators exposed to Python via `torch.ops.*` that are registered by vLLM native extensions, plus a few Python glue files that call into them.

## 1) Native-registered operators (authoritative)

Scanned `csrc/**` for `TORCH_LIBRARY*` blocks and extracted their `.def("...")` registrations.

Native source files with registrations: **5**

- `csrc/cpu/torch_bindings.cpp`
- `csrc/moe/torch_bindings.cpp`
- `csrc/quantization/machete/machete_pytorch.cu`
- `csrc/rocm/torch_bindings.cpp`
- `csrc/torch_bindings.cpp`

### `torch.ops._C`

Sources: `csrc/cpu/torch_bindings.cpp`, `csrc/torch_bindings.cpp`

Count: **109**

| Op |
|---|
| `allspark_w8a16_gemm` |
| `apply_repetition_penalties_` |
| `awq_dequantize` |
| `awq_gemm` |
| `awq_marlin_repack` |
| `convert_vertical_slash_indexes` |
| `convert_vertical_slash_indexes_mergehead` |
| `convert_weight_packed` |
| `cpu_attention_with_kv_cache` |
| `cpu_attn_reshape_and_cache` |
| `cpu_fused_moe` |
| `cpu_gemm_wna16` |
| `create_onednn_mm_handler` |
| `create_onednn_scaled_mm_handler` |
| `cutlass_encode_and_reorder_int4b` |
| `cutlass_encode_and_reorder_int4b_grouped` |
| `cutlass_fp4_group_mm` |
| `cutlass_group_gemm_supported` |
| `cutlass_moe_mm` |
| `cutlass_pack_scale_fp8` |
| `cutlass_scaled_fp4_mm` |
| `cutlass_scaled_mm` |
| `cutlass_scaled_mm_azp` |
| `cutlass_scaled_mm_supports_block_fp8` |
| `cutlass_scaled_mm_supports_fp4` |
| `cutlass_scaled_mm_supports_fp8` |
| `cutlass_scaled_sparse_mm` |
| `cutlass_sparse_compress` |
| `cutlass_sparse_scaled_mm_supported` |
| `cutlass_w4a8_mm` |
| `cutlass_w4a8_moe_mm` |
| `dynamic_4bit_int_moe` |
| `dynamic_per_token_scaled_fp8_quant` |
| `dynamic_scaled_fp8_quant` |
| `dynamic_scaled_int8_quant` |
| `fatrelu_and_mul` |
| `fused_add_rms_norm` |
| `fused_add_rms_norm_static_fp8_quant` |
| `fused_experts_cpu` |
| `fused_qk_norm_rope` |
| `gelu_and_mul` |
| `gelu_fast` |
| `gelu_new` |
| `gelu_quick` |
| `gelu_tanh_and_mul` |
| `get_cuda_view_from_cpu_tensor` |
| `get_cutlass_moe_mm_data` |
| `get_cutlass_moe_mm_problem_sizes_from_expert_offsets` |
| `get_cutlass_pplx_moe_mm_data` |
| `get_scheduler_metadata` |
| `ggml_dequantize` |
| `ggml_moe_a8` |
| `ggml_moe_a8_vec` |
| `ggml_moe_get_block_size` |
| `ggml_mul_mat_a8` |
| `ggml_mul_mat_vec_a8` |
| `gptq_gemm` |
| `gptq_marlin_repack` |
| `gptq_shuffle` |
| `hadacore_transform` |
| `init_shm_manager` |
| `int8_scaled_mm_with_quant` |
| `is_onednn_acl_supported` |
| `join_shm_manager` |
| `machete_mm` |
| `machete_prepack_B` |
| `machete_supported_schedules` |
| `marlin_gemm` |
| `marlin_int4_fp8_preprocess` |
| `merge_attn_states` |
| `mul_and_silu` |
| `onednn_mm` |
| `onednn_scaled_mm` |
| `paged_attention_v1` |
| `paged_attention_v2` |
| `per_token_group_fp8_quant` |
| `per_token_group_fp8_quant_packed` |
| `per_token_group_quant_int8` |
| `permute_cols` |
| `persistent_masked_m_silu_mul_quant` |
| `prepack_moe_weight` |
| `rearrange_kn_weight_as_n32k16_order` |
| `release_dnnl_matmul_handler` |
| `rms_norm` |
| `rms_norm_dynamic_per_token_quant` |
| `rms_norm_per_block_quant` |
| `rms_norm_static_fp8_quant` |
| `rotary_embedding` |
| `scaled_fp4_experts_quant` |
| `scaled_fp4_quant` |
| `selective_scan_fwd` |
| `shm_all_gather` |
| `shm_allreduce` |
| `shm_gather` |
| `shm_recv_tensor_list` |
| `shm_send_tensor_list` |
| `silu_and_mul` |
| `silu_and_mul_nvfp4_quant` |
| `silu_and_mul_quant` |
| `silu_and_mul_scaled_fp4_experts_quant` |
| `sm100_cutlass_mla_decode` |
| `sm100_cutlass_mla_get_workspace_size` |
| `static_scaled_fp8_quant` |
| `static_scaled_int8_quant` |
| `swigluoai_and_mul` |
| `top_k_per_row_decode` |
| `top_k_per_row_prefill` |
| `weak_ref_tensor` |
| `weight_packed_linear` |

### `torch.ops._C_cache_ops`

Sources: `csrc/torch_bindings.cpp`

Count: **11**

| Op |
|---|
| `concat_and_cache_mla` |
| `concat_and_cache_mla_rope_fused` |
| `convert_fp8` |
| `cp_gather_and_upconvert_fp8_kv_cache` |
| `cp_gather_cache` |
| `cp_gather_indexer_k_quant_cache` |
| `gather_and_maybe_dequant_cache` |
| `indexer_k_quant_and_cache` |
| `reshape_and_cache` |
| `reshape_and_cache_flash` |
| `swap_blocks` |

### `torch.ops._C_cpu`

Sources: `csrc/cpu/torch_bindings.cpp`

Count: **1**

| Op |
|---|
| `mla_decode_kvcache` |

### `torch.ops._C_cuda_utils`

Sources: `csrc/torch_bindings.cpp`

Count: **2**

| Op |
|---|
| `get_device_attribute` |
| `get_max_shared_memory_per_block_device_attribute` |

### `torch.ops._C_custom_ar`

Sources: `csrc/torch_bindings.cpp`

Count: **16**

| Op |
|---|
| `all_reduce` |
| `allocate_shared_buffer_and_handle` |
| `dispose` |
| `free_shared_buffer` |
| `get_graph_buffer_ipc_meta` |
| `init_custom_ar` |
| `init_custom_qr` |
| `meta_size` |
| `open_mem_handle` |
| `qr_all_reduce` |
| `qr_destroy` |
| `qr_get_handle` |
| `qr_max_size` |
| `qr_open_handles` |
| `register_buffer` |
| `register_graph_buffers` |

### `torch.ops._C_utils`

Sources: `csrc/cpu/torch_bindings.cpp`

Count: **1**

| Op |
|---|
| `init_cpu_threads_env` |

### `torch.ops._moe_C`

Sources: `csrc/moe/torch_bindings.cpp`

Count: **14**

| Op |
|---|
| `batched_moe_align_block_size` |
| `grouped_topk` |
| `marlin_gemm_moe` |
| `moe_align_block_size` |
| `moe_lora_align_block_size` |
| `moe_permute` |
| `moe_permute_unpermute_supported` |
| `moe_sum` |
| `moe_unpermute` |
| `moe_wna16_gemm` |
| `moe_wna16_marlin_gemm` |
| `shuffle_rows` |
| `topk_sigmoid` |
| `topk_softmax` |

### `torch.ops._rocm_C`

Sources: `csrc/rocm/torch_bindings.cpp`

Count: **5**

| Op |
|---|
| `LLMM1` |
| `paged_attention` |
| `wvSplitK` |
| `wvSplitKQ` |
| `wvSplitKrc` |

## 2) Python `torch.ops.*` usage

Scanned `vllm/**/*.py` for `torch.ops.<namespace>.<op>` references (this includes third-party namespaces too, e.g. `torch.ops.aten`).

### Aggregated by namespace

- `torch.ops._C` (111 ops): `allspark_w8a16_gemm`, `apply_repetition_penalties_`, `awq_dequantize`, `awq_gemm`, `awq_marlin_repack`, `convert_vertical_slash_indexes`, `convert_vertical_slash_indexes_mergehead`, `convert_weight_packed`, `cpu_attention_with_kv_cache`, `cpu_attn_reshape_and_cache`, `cpu_fused_moe`, `cpu_gemm_wna16`, `create_onednn_mm_handler`, `create_onednn_scaled_mm_handler`, `cutlass_encode_and_reorder_int4b`, `cutlass_encode_and_reorder_int4b_grouped`, `cutlass_fp4_group_mm`, `cutlass_group_gemm_supported`, `cutlass_moe_mm`, `cutlass_pack_scale_fp8`, `cutlass_scaled_fp4_mm`, `cutlass_scaled_mm`, `cutlass_scaled_mm_azp`, `cutlass_scaled_mm_supports_block_fp8`, `cutlass_scaled_mm_supports_fp4`, `cutlass_scaled_mm_supports_fp8`, `cutlass_scaled_sparse_mm`, `cutlass_sparse_compress`, `cutlass_sparse_scaled_mm_supported`, `cutlass_w4a8_mm`, `cutlass_w4a8_moe_mm`, `dynamic_4bit_int_moe`, `dynamic_per_token_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`, `dynamic_scaled_int8_quant`, `fatrelu_and_mul`, `flash_mla_fwd_kvcache`, `fused_add_rms_norm`, `fused_add_rms_norm_static_fp8_quant`, `fused_experts_cpu`, `fused_qk_norm_rope`, `gelu_and_mul`, `gelu_fast`, `gelu_new`, `gelu_quick`, `gelu_tanh_and_mul`, `get_cuda_view_from_cpu_tensor`, `get_cutlass_moe_mm_data`, `get_cutlass_moe_mm_problem_sizes_from_expert_offsets`, `get_cutlass_pplx_moe_mm_data`, `get_flash_mla_metadata`, `get_scheduler_metadata`, `ggml_dequantize`, `ggml_moe_a8`, `ggml_moe_a8_vec`, `ggml_moe_get_block_size`, `ggml_mul_mat_a8`, `ggml_mul_mat_vec_a8`, `gptq_gemm`, `gptq_marlin_repack`, `gptq_shuffle`, `hadacore_transform`, `init_shm_manager`, `int8_scaled_mm_with_quant`, `is_onednn_acl_supported`, `join_shm_manager`, `machete_mm`, `machete_prepack_B`, `machete_supported_schedules`, `marlin_gemm`, `marlin_int4_fp8_preprocess`, `merge_attn_states`, `mul_and_silu`, `onednn_mm`, `onednn_scaled_mm`, `paged_attention_v1`, `paged_attention_v2`, `per_token_group_fp8_quant`, `per_token_group_fp8_quant_packed`, `per_token_group_quant_int8`, `permute_cols`, `persistent_masked_m_silu_mul_quant`, `prepack_moe_weight`, `rearrange_kn_weight_as_n32k16_order`, `release_dnnl_matmul_handler`, `rms_norm`, `rms_norm_dynamic_per_token_quant`, `rms_norm_per_block_quant`, `rms_norm_static_fp8_quant`, `rotary_embedding`, `scaled_fp4_experts_quant`, `scaled_fp4_quant`, `selective_scan_fwd`, `shm_all_gather`, `shm_allreduce`, `shm_gather`, `shm_recv_tensor_list`, `shm_send_tensor_list`, `silu_and_mul`, `silu_and_mul_nvfp4_quant`, `silu_and_mul_quant`, `silu_and_mul_scaled_fp4_experts_quant`, `sm100_cutlass_mla_decode`, `sm100_cutlass_mla_get_workspace_size`, `static_scaled_fp8_quant`, `static_scaled_int8_quant`, `swigluoai_and_mul`, `top_k_per_row_decode`, `top_k_per_row_prefill`, `weak_ref_tensor`, `weight_packed_linear`
- `torch.ops._C_cache_ops` (11 ops): `concat_and_cache_mla`, `concat_and_cache_mla_rope_fused`, `convert_fp8`, `cp_gather_and_upconvert_fp8_kv_cache`, `cp_gather_cache`, `cp_gather_indexer_k_quant_cache`, `gather_and_maybe_dequant_cache`, `indexer_k_quant_and_cache`, `reshape_and_cache`, `reshape_and_cache_flash`, `swap_blocks`
- `torch.ops._C_cpu` (1 ops): `mla_decode_kvcache`
- `torch.ops._C_cuda_utils` (2 ops): `get_device_attribute`, `get_max_shared_memory_per_block_device_attribute`
- `torch.ops._C_custom_ar` (16 ops): `all_reduce`, `allocate_shared_buffer_and_handle`, `dispose`, `free_shared_buffer`, `get_graph_buffer_ipc_meta`, `init_custom_ar`, `init_custom_qr`, `meta_size`, `open_mem_handle`, `qr_all_reduce`, `qr_destroy`, `qr_get_handle`, `qr_max_size`, `qr_open_handles`, `register_buffer`, `register_graph_buffers`
- `torch.ops._C_utils` (1 ops): `init_cpu_threads_env`
- `torch.ops._flashmla_extension_C` (2 ops): `fwd_kvcache_mla_fp8`, `get_mla_decoding_metadata_dense_fp8`
- `torch.ops._moe_C` (13 ops): `batched_moe_align_block_size`, `grouped_topk`, `moe_align_block_size`, `moe_lora_align_block_size`, `moe_permute`, `moe_permute_unpermute_supported`, `moe_sum`, `moe_unpermute`, `moe_wna16_gemm`, `moe_wna16_marlin_gemm`, `shuffle_rows`, `topk_sigmoid`, `topk_softmax`
- `torch.ops._qutlass_C` (5 ops): `fusedQuantizeMxAbsMax`, `fusedQuantizeMxQuest`, `fusedQuantizeNv`, `matmul_ada_mxf4_bf16_tn`, `matmul_mxf4_bf16_tn`
- `torch.ops._rocm_C` (5 ops): `LLMM1`, `paged_attention`, `wvSplitK`, `wvSplitKQ`, `wvSplitKrc`
- `torch.ops.aiter` (1 ops): `paged_attention_v1`
- `torch.ops.aten` (12 ops): `_dyn_quant_matmul_4bit`, `_dyn_quant_pack_4bit_weight`, `_scaled_mm`, `copy_`, `full`, `mm`, `permute`, `reshape`, `slice`, `slice_scatter`, `split_with_sizes`, `view`
- `torch.ops.fbgemm` (1 ops): `f4f4bf16`
- `torch.ops.higher_order` (1 ops): `auto_functionalized`
- `torch.ops.symm_mem` (6 ops): `fused_all_gather_matmul`, `fused_all_gather_scaled_matmul`, `fused_matmul_reduce_scatter`, `fused_scaled_matmul_reduce_scatter`, `multimem_all_reduce_`, `two_shot_all_reduce_`
- `torch.ops.vllm` (81 ops): `_apply_gguf_embedding`, `_fused_moe_gguf`, `_fused_mul_mat_gguf`, `all_gather`, `all_reduce`, `all_reduce_symmetric_with_copy`, `apply_bnb_4bit`, `cpu_fused_moe_torch`, `dequant_mxfp4`, `dequant_mxfp6`, `fi_trtllm_fp8_per_tensor_moe`, `flash_attn_maxseqlen_wrapper`, `flashinfer_fp8_blockscale_gemm`, `flashinfer_fused_moe_bf16`, `flashinfer_fused_moe_blockscale_fp8`, `flashinfer_rotary_embedding`, `flashinfer_trtllm_fused_allreduce_norm`, `fp8_gemm_nt_op`, `fused_moe_lora`, `fused_moe_lora_expand`, `fused_moe_lora_shrink`, `fused_quantize_mx`, `fused_quantize_nv`, `gdn_attention_core`, `gemm_with_dynamic_quant`, `inplace_fused_experts`, `kda_attention`, `linear_attention`, `lora_expand`, `lora_shrink`, `mamba_mixer`, `mamba_mixer2`, `matmul_mxf4_bf16`, `matmul_nvf4_bf16`, `maybe_calc_kv_scales`, `maybe_populate_sink`, `moe_forward`, `moe_forward_shared`, `outplace_fused_experts`, `padded_cutlass`, `patched_fused_scaled_matmul_reduce_scatter`, `plamo2_mamba_mixer`, `quant_dequant_mxfp4`, `quant_dequant_mxfp6`, `reduce_scatter`, `rocm_aiter_act_mul_and_fp8_group_quant`, `rocm_aiter_asm_moe_tkw1`, `rocm_aiter_biased_grouped_topk`, `rocm_aiter_fused_moe`, `rocm_aiter_gemm_a8w8`, `rocm_aiter_gemm_a8w8_blockscale`, `rocm_aiter_group_fp8_quant`, `rocm_aiter_grouped_topk`, `rocm_aiter_mla_decode_fwd`, `rocm_aiter_per_tensor_quant`, `rocm_aiter_per_token_quant`, `rocm_aiter_rms_norm`, `rocm_aiter_rmsnorm2d_fwd_with_add`, `rocm_aiter_rmsnorm_fp8_group_quant`, `rocm_aiter_rmsnorm_fused_add_dynamic_quant`, `rocm_aiter_rmsnorm_fused_dynamic_quant`, `rocm_aiter_rmsnorm_with_add_fp8_group_quant`, `rocm_aiter_sparse_attn_indexer`, `rocm_aiter_topk_sigmoid`, `rocm_aiter_topk_softmax`, `rocm_aiter_triton_add_rmsnorm_pad`, `rocm_aiter_triton_gemm_a8w8_blockscale`, `rocm_per_tensor_float_w8a8_scaled_mm_impl`, `rocm_unquantized_gemm`, `sequence_parallel_chunk_impl`, `short_conv`, `sparse_attn_indexer`, `torch_sdpa_wrapper`, `transformers_moe_forward`, `triton_per_token_group_quant_fp8`, `unified_attention`, `unified_attention_with_output`, `unified_kv_cache_update`, `unified_mla_attention`, `unified_mla_attention_with_output`, `w8a8_triton_block_scaled_mm_func`

### Selected call sites (top files)

- `vllm/_custom_ops.py` (128 refs)
  - `torch.ops._C` (78 ops): `allspark_w8a16_gemm`, `apply_repetition_penalties_`, `awq_dequantize`, `awq_gemm`, `awq_marlin_repack`, `convert_vertical_slash_indexes`, `convert_vertical_slash_indexes_mergehead`, `cpu_attention_with_kv_cache`, `cpu_attn_reshape_and_cache`, `cpu_fused_moe`, `cpu_gemm_wna16`, `create_onednn_mm_handler`, `create_onednn_scaled_mm_handler`, `cutlass_encode_and_reorder_int4b`, `cutlass_encode_and_reorder_int4b_grouped`, `cutlass_fp4_group_mm`, `cutlass_group_gemm_supported`, `cutlass_moe_mm`, `cutlass_pack_scale_fp8`, `cutlass_scaled_fp4_mm`, `cutlass_scaled_mm`, `cutlass_scaled_mm_azp`, `cutlass_scaled_mm_supports_block_fp8`, `cutlass_scaled_mm_supports_fp4`, `cutlass_scaled_mm_supports_fp8`, `cutlass_scaled_sparse_mm`, `cutlass_sparse_compress`, `cutlass_sparse_scaled_mm_supported`, `cutlass_w4a8_mm`, `cutlass_w4a8_moe_mm`, `dynamic_per_token_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`, `dynamic_scaled_int8_quant`, `flash_mla_fwd_kvcache`, `fused_add_rms_norm`, `fused_qk_norm_rope`, `get_cutlass_moe_mm_data`, `get_cutlass_moe_mm_problem_sizes_from_expert_offsets`, `get_cutlass_pplx_moe_mm_data`, `get_flash_mla_metadata`, `get_scheduler_metadata`, `ggml_dequantize`, `ggml_moe_a8`, `ggml_moe_a8_vec`, `ggml_moe_get_block_size`, `ggml_mul_mat_a8`, `ggml_mul_mat_vec_a8`, `gptq_gemm`, `gptq_marlin_repack`, `gptq_shuffle`, `hadacore_transform`, `is_onednn_acl_supported`, `machete_mm`, `machete_prepack_B`, `machete_supported_schedules`, `marlin_gemm`, `marlin_int4_fp8_preprocess`, `merge_attn_states`, `onednn_mm`, `onednn_scaled_mm`, `paged_attention_v1`, `paged_attention_v2`, `permute_cols`, `prepack_moe_weight`, `rearrange_kn_weight_as_n32k16_order`, `release_dnnl_matmul_handler`, `rms_norm`, `rms_norm_dynamic_per_token_quant`, `rms_norm_per_block_quant`, `rotary_embedding`, `scaled_fp4_experts_quant`, `scaled_fp4_quant`, `selective_scan_fwd`, `silu_and_mul_scaled_fp4_experts_quant`, `sm100_cutlass_mla_decode`, `sm100_cutlass_mla_get_workspace_size`, `static_scaled_fp8_quant`, `static_scaled_int8_quant`
  - `torch.ops._C_cache_ops` (11 ops): `concat_and_cache_mla`, `concat_and_cache_mla_rope_fused`, `convert_fp8`, `cp_gather_and_upconvert_fp8_kv_cache`, `cp_gather_cache`, `cp_gather_indexer_k_quant_cache`, `gather_and_maybe_dequant_cache`, `indexer_k_quant_and_cache`, `reshape_and_cache`, `reshape_and_cache_flash`, `swap_blocks`
  - `torch.ops._C_cpu` (1 ops): `mla_decode_kvcache`
  - `torch.ops._C_cuda_utils` (2 ops): `get_device_attribute`, `get_max_shared_memory_per_block_device_attribute`
  - `torch.ops._C_custom_ar` (16 ops): `all_reduce`, `allocate_shared_buffer_and_handle`, `dispose`, `free_shared_buffer`, `get_graph_buffer_ipc_meta`, `init_custom_ar`, `init_custom_qr`, `meta_size`, `open_mem_handle`, `qr_all_reduce`, `qr_destroy`, `qr_get_handle`, `qr_max_size`, `qr_open_handles`, `register_buffer`, `register_graph_buffers`
  - `torch.ops._moe_C` (10 ops): `batched_moe_align_block_size`, `grouped_topk`, `moe_align_block_size`, `moe_lora_align_block_size`, `moe_sum`, `moe_wna16_gemm`, `moe_wna16_marlin_gemm`, `shuffle_rows`, `topk_sigmoid`, `topk_softmax`
  - `torch.ops._qutlass_C` (5 ops): `fusedQuantizeMxAbsMax`, `fusedQuantizeMxQuest`, `fusedQuantizeNv`, `matmul_ada_mxf4_bf16_tn`, `matmul_mxf4_bf16_tn`
  - `torch.ops._rocm_C` (5 ops): `LLMM1`, `paged_attention`, `wvSplitK`, `wvSplitKQ`, `wvSplitKrc`
- `vllm/_aiter_ops.py` (21 refs)
  - `torch.ops.vllm` (21 ops): `rocm_aiter_act_mul_and_fp8_group_quant`, `rocm_aiter_asm_moe_tkw1`, `rocm_aiter_biased_grouped_topk`, `rocm_aiter_fused_moe`, `rocm_aiter_gemm_a8w8`, `rocm_aiter_gemm_a8w8_blockscale`, `rocm_aiter_group_fp8_quant`, `rocm_aiter_grouped_topk`, `rocm_aiter_mla_decode_fwd`, `rocm_aiter_per_tensor_quant`, `rocm_aiter_per_token_quant`, `rocm_aiter_rms_norm`, `rocm_aiter_rmsnorm2d_fwd_with_add`, `rocm_aiter_rmsnorm_fp8_group_quant`, `rocm_aiter_rmsnorm_fused_add_dynamic_quant`, `rocm_aiter_rmsnorm_fused_dynamic_quant`, `rocm_aiter_rmsnorm_with_add_fp8_group_quant`, `rocm_aiter_topk_sigmoid`, `rocm_aiter_topk_softmax`, `rocm_aiter_triton_add_rmsnorm_pad`, `rocm_aiter_triton_gemm_a8w8_blockscale`
- `vllm/compilation/fix_functionalization.py` (13 refs)
  - `torch.ops._C` (10 ops): `fused_add_rms_norm`, `fused_add_rms_norm_static_fp8_quant`, `fused_qk_norm_rope`, `rms_norm`, `rms_norm_dynamic_per_token_quant`, `rms_norm_static_fp8_quant`, `rotary_embedding`, `silu_and_mul`, `silu_and_mul_nvfp4_quant`, `silu_and_mul_quant`
  - `torch.ops.aten` (2 ops): `slice_scatter`, `split_with_sizes`
  - `torch.ops.vllm` (1 ops): `flashinfer_trtllm_fused_allreduce_norm`
- `vllm/compilation/collective_fusion.py` (12 refs)
  - `torch.ops._C` (2 ops): `cutlass_scaled_mm`, `scaled_fp4_quant`
  - `torch.ops.aten` (2 ops): `_scaled_mm`, `mm`
  - `torch.ops.higher_order` (1 ops): `auto_functionalized`
  - `torch.ops.symm_mem` (3 ops): `fused_all_gather_matmul`, `fused_all_gather_scaled_matmul`, `fused_matmul_reduce_scatter`
  - `torch.ops.vllm` (4 ops): `all_gather`, `flashinfer_trtllm_fused_allreduce_norm`, `patched_fused_scaled_matmul_reduce_scatter`, `reduce_scatter`
- `vllm/compilation/matcher_utils.py` (11 refs)
  - `torch.ops._C` (9 ops): `dynamic_per_token_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`, `fused_add_rms_norm`, `per_token_group_fp8_quant`, `rms_norm`, `rotary_embedding`, `scaled_fp4_quant`, `silu_and_mul`, `static_scaled_fp8_quant`
  - `torch.ops.vllm` (2 ops): `flashinfer_rotary_embedding`, `triton_per_token_group_quant_fp8`
- `vllm/compilation/fusion.py` (11 refs)
  - `torch.ops._C` (11 ops): `dynamic_per_token_scaled_fp8_quant`, `dynamic_scaled_fp8_quant`, `fused_add_rms_norm`, `fused_add_rms_norm_static_fp8_quant`, `per_token_group_fp8_quant`, `rms_norm`, `rms_norm_dynamic_per_token_quant`, `rms_norm_per_block_quant`, `rms_norm_static_fp8_quant`, `scaled_fp4_quant`, `static_scaled_fp8_quant`
- `vllm/model_executor/layers/activation.py` (9 refs)
  - `torch.ops._C` (9 ops): `fatrelu_and_mul`, `gelu_and_mul`, `gelu_fast`, `gelu_new`, `gelu_quick`, `gelu_tanh_and_mul`, `mul_and_silu`, `silu_and_mul`, `swigluoai_and_mul`
- `vllm/distributed/device_communicators/cpu_communicator.py` (7 refs)
  - `torch.ops._C` (7 ops): `init_shm_manager`, `join_shm_manager`, `shm_all_gather`, `shm_allreduce`, `shm_gather`, `shm_recv_tensor_list`, `shm_send_tensor_list`
- `vllm/model_executor/layers/quantization/utils/fp8_utils.py` (6 refs)
  - `torch.ops._C` (2 ops): `per_token_group_fp8_quant`, `per_token_group_fp8_quant_packed`
  - `torch.ops.vllm` (4 ops): `flashinfer_fp8_blockscale_gemm`, `fp8_gemm_nt_op`, `padded_cutlass`, `w8a8_triton_block_scaled_mm_func`
- `vllm/model_executor/layers/quantization/gguf.py` (5 refs)
  - `torch.ops._C` (2 ops): `gelu_and_mul`, `silu_and_mul`
  - `torch.ops.vllm` (3 ops): `_apply_gguf_embedding`, `_fused_moe_gguf`, `_fused_mul_mat_gguf`

## 3) Reconciliation

### Namespaces used in Python but not found in vLLM native bindings scan

- `torch.ops._flashmla_extension_C`
- `torch.ops._qutlass_C`
- `torch.ops.aiter`
- `torch.ops.aten`
- `torch.ops.fbgemm`
- `torch.ops.higher_order`
- `torch.ops.symm_mem`
- `torch.ops.vllm`

### Namespaces registered natively but not referenced by Python scans

- (none)

## 4) Notes / caveats

- Some operators are conditionally compiled (CUDA vs ROCm vs CPU ISA). Presence depends on your build/platform.
- This catalog lists operator names/namespaces. Exact schemas live in the `.def("...")` strings in the binding files.
- Third-party namespaces (e.g. `torch.ops.aiter.*` or `torch.ops.vllm.*`) may appear in Python glue but are not necessarily registered by vLLM’s `_C` extensions.
