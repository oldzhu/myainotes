# vLLM (repo @ ~/vllm) — Architecture, Source Tree, and Contributor Onboarding Notes

> Goal: a practical, code-pointer-heavy guide to understand vLLM’s runtime pipeline (v1), how the repo is laid out, and how to start contributing.

---

## 0) TL;DR Architecture

vLLM is an inference/serving engine optimized for **high-throughput, low-latency generation**. The core design splits responsibilities:

- **Frontend / Entry points**: parse user inputs (offline API or HTTP server), validate params, tokenize, manage streaming.
- **Engine (v1)**: manages request lifecycle and output streaming while delegating heavy work to the core loop.
- **EngineCore loop**: repeats **schedule → execute → postprocess**.
- **Scheduler**: chooses which requests/tokens to run each step.
- **Executor**: abstracts local vs multiprocess vs Ray execution and handles RPC/fanout.
- **Worker / ModelRunner**: owns device-specific model execution; GPU path is the hot path.
- **KV cache manager**: allocates/frees KV blocks; enables prefix caching and memory-aware scheduling.
- **Attention backends + custom ops**: most speed comes from specialized attention kernels and fused ops (C++/CUDA/HIP + Triton).

### Canonical implementation: “v1”
In this checkout, “legacy” engine modules are thin aliases to v1. Treat v1 as the source of truth:

- Sync engine: `~/vllm/vllm/v1/engine/llm_engine.py`
- Async engine: `~/vllm/vllm/v1/engine/async_llm.py`
- Core loop: `~/vllm/vllm/v1/engine/core.py`

---

## 1) Request Flow (end-to-end)

### 1.1 Offline inference API
Start here if you want to understand the user-facing Python API:

- `~/vllm/vllm/entrypoints/llm.py`

Typical flow:

1. `LLM(...)` / engine args are built.
2. Prompts + sampling params validated.
3. Requests are added.
4. A loop calls `engine.step()` until outputs are complete.

### 1.2 OpenAI-compatible server
Start here if you want serving, streaming, request parsing, and FastAPI lifecycle:

- `~/vllm/vllm/entrypoints/openai/api_server.py`

Typical flow:

1. CLI args → async engine client.
2. Requests come in via HTTP; each request becomes an internal v1 `Request`.
3. Streaming responses are produced while the core loop runs.

---

## 1.3 Choose a Starting Area (A/B/C)

When you’re starting to contribute, it helps to pick a “lane” first. Here are three common entry paths:

### Quick decision guide

| Pick this if you want to… | Choose | Usually need a GPU locally? | Typical PR size | Notes |
|---|---|---:|---:|---|
| Improve API behavior, request parsing, streaming, compatibility | A | No (often) | Small–Medium | Very testable; lots of correctness/UX wins |
| Own batching/scheduling behavior and throughput/latency tradeoffs | B | Helpful | Medium | High impact; requires understanding request state + KV cache |
| Work on raw performance (kernels/attention/quant/graphs) | C | Yes (usually) | Medium–Large | Highest perf impact; higher setup + debugging complexity |

Suggested first PR types:
- **A**: one endpoint behavior fix + focused tests.
- **B**: one scheduling invariant/metric improvement + unit test.
- **C**: one kernel/backend bugfix + correctness test (and ideally a micro-benchmark).

### A) OpenAI server / streaming (frontend & serving)

What it is:
- Everything from HTTP request parsing → engine request creation → streaming tokens back.
- OpenAI API compatibility, request/response schemas, cancellation, timeouts, metrics, logging.

Where it lives:
- `~/vllm/vllm/entrypoints/openai/` (main OpenAI-compatible server implementation)
- `~/vllm/vllm/entrypoints/` (shared entrypoint utilities)

Typical starter tasks:
- Fix streaming edge cases and output formatting.
- Improve validation/error messages and add tests.
- Add/adjust metrics and request lifecycle handling.

### B) Scheduler / core throughput (engine core)

What it is:
- The “heart” of vLLM correctness + throughput: continuous batching, preemption, prefill/decode mixing, KV-cache-aware scheduling.

Where it lives:
- `~/vllm/vllm/v1/engine/core.py` (core loop)
- `~/vllm/vllm/v1/core/sched/` (scheduler)
- `~/vllm/vllm/v1/core/kv_cache_manager.py` (KV cache management)
- `~/vllm/vllm/v1/executor/` (execution strategy: uni/mp/ray)
- `~/vllm/vllm/v1/request.py` (request state machine)

Typical starter tasks:
- Add/adjust scheduler policies (fairness/latency) with targeted tests.
- Add instrumentation for scheduling/KV usage and validate invariants.
- Fix subtle request lifecycle issues (abort/preempt/finish transitions).

### C) Attention / kernels (low-level performance)

What it is:
- The compute-heavy part: attention backends, fused kernels, custom ops, and their integration.
- Usually involves CUDA/ROCm/Triton and/or compiler/CUDA-graphs work.

Where it lives:
- `~/vllm/vllm/v1/attention/` (backend interface + selector + implementations)
- `~/vllm/vllm/_custom_ops.py` (Python ↔ native op calls)
- `~/vllm/csrc/` (C++/CUDA/HIP kernels and registration)
- `~/vllm/vllm/model_executor/` (optimized layers, quantization, model code)
- `~/vllm/vllm/compilation/` and `~/vllm/vllm/v1/cudagraph_dispatcher.py` (compile/cudagraph control)

Typical starter tasks:
- Add or optimize a kernel (and add correctness tests).
- Fix a backend-specific bug (layout, dtype, shapes, cudagraph support).
- Improve dispatch/compatibility between attention backends and compilation modes.

---

## 2) The v1 “Spine” (Core Classes + Methods)

This is the minimum set of modules to read to understand how tokens get generated.

### 2.1 Configuration: the control plane
- `~/vllm/vllm/config/vllm.py` — `VllmConfig`

Why it matters:
- It aggregates *everything*: model config, cache config, parallelism, scheduler policy, attention backend, compilation/cudagraphs settings, LoRA, speculative decode, structured outputs, observability, KV transfer, etc.
- It is passed around widely; understanding how a setting flows is mostly “follow VllmConfig”.

Key thing to look for:
- `VllmConfig.compute_hash()` — identifies graph-affecting settings for caching/compilation.

### 2.2 Request state machine
- `~/vllm/vllm/v1/request.py` — `Request`

What it does:
- Tracks per-request lifecycle: waiting/running/preempted/finished.
- Holds prompt tokens, generated token ids, sampling params, counters.
- Implements ordering / priority comparisons.

### 2.3 Input/Output processing
- `~/vllm/vllm/v1/engine/input_processor.py` — request validation + normalization, tokenization, multimodal parsing.
- `~/vllm/vllm/v1/engine/output_processor.py` — assembles deltas, detokenizes, logprobs, streaming collector.

If you want to contribute quickly, this area is usually approachable and testable.

### 2.4 Engine (frontend)
- `~/vllm/vllm/v1/engine/llm_engine.py` — sync engine
  - Important methods: `from_engine_args()`, `add_request()`, `step()`, `abort_request()`
- `~/vllm/vllm/v1/engine/async_llm.py` — async engine
  - Important methods: `from_engine_args()`, `add_request()`, `generate()` (async generator), `shutdown()`

### 2.5 EngineCore (core loop)
- `~/vllm/vllm/v1/engine/core.py` — `EngineCore`

Responsibilities:
- Picks executor type, initializes worker(s).
- Initializes KV cache and scheduler.
- Runs the step loop: schedule tokens → execute model → return outputs.

### 2.6 Scheduler
- `~/vllm/vllm/v1/core/sched/scheduler.py` — `Scheduler.schedule()`

Responsibilities:
- Continuous batching: pick which requests advance.
- Handle prefill/decode mix, chunked prefill, preemption.
- Coordinate with KV cache budgets and block allocation.

### 2.7 KV cache manager
- `~/vllm/vllm/v1/core/kv_cache_manager.py` — `KVCacheManager`

Responsibilities:
- Tracks available KV blocks.
- Computes prefix-cache hits.
- Allocates “slots” for scheduled tokens.

### 2.8 Executor abstraction (local, multiprocess, Ray)
- `~/vllm/vllm/v1/executor/abstract.py` — base `Executor`, `get_class()` selection.
- `~/vllm/vllm/v1/executor/uniproc_executor.py` — simplest implementation.
- `~/vllm/vllm/v1/executor/multiproc_executor.py` — multi-process worker orchestration.
- `~/vllm/vllm/v1/executor/ray_executor.py` — Ray orchestration.

Key methods:
- `execute_model(...)`
- `sample_tokens(...)`
- `collective_rpc(...)` (multi-worker coordination)

### 2.9 Worker + GPU model runner
- `~/vllm/vllm/v1/worker/worker_base.py` — `WorkerBase` interface.
- `~/vllm/vllm/v1/worker/gpu_worker.py` — GPU worker wiring/init.
- `~/vllm/vllm/v1/worker/gpu_model_runner.py` — `GPUModelRunner.execute_model()` hot path.

If you want to understand “where time goes”, read `GPUModelRunner.execute_model()` carefully.

---

## 3) Source Tree Tour (what lives where)

### 3.1 Top-level directories
- `~/vllm/vllm/` — main Python package.
- `~/vllm/csrc/` — native kernels + torch extension registration.
- `~/vllm/docs/` — design docs, user docs, contributor docs.
- `~/vllm/tests/` — pytest suite (lots of targeted subfolders).
- `~/vllm/benchmarks/` — benchmark scripts and harnesses.

### 3.2 Python package highlights

- `~/vllm/vllm/v1/`
  - The canonical “v1 engine”: engine/core/scheduler/worker/executor/attention.

- `~/vllm/vllm/entrypoints/`
  - User-facing CLIs and servers.
  - Includes OpenAI-compatible server implementation.

- `~/vllm/vllm/model_executor/`
  - Model loading and model implementations.
  - Quantization and custom layers.
  - Model registry for mapping HF configs/classes → vLLM model implementations.

  Key subfolders:
  - `model_executor/models/` — model implementations and the registry.
  - `model_executor/model_loader/` — weight loading strategies.
  - `model_executor/layers/` — optimized layers (attention, linear, fused MoE, quant kernels).
  - `model_executor/custom_op.py` — the `CustomOp` abstraction (dispatch to CUDA/HIP/CPU/XPU).

- `~/vllm/vllm/compilation/`
  - torch.compile / inductor integration, graph transforms, fusion passes.
  - CUDA Graph wrappers, caching, and dispatch support.

- `~/vllm/vllm/distributed/`
  - Device communicators, parallel state, KV transfer/connector infrastructure.
  - Integration points for multi-node and special deployment modes.

- `~/vllm/vllm/platforms/`
  - Platform detection + hooks for backend-specific behavior.
  - Important for how attention backends and kernels are chosen.

- `~/vllm/vllm/plugins/`
  - Plugin system entry point loader.

---

## 4) Native Kernels & Custom Ops (Python ↔ C++/CUDA/HIP)

### 4.1 Where kernels live
- Native sources: `~/vllm/csrc/`
  - Notable subdirs: `csrc/attention/`, `csrc/moe/`, `csrc/mamba/`, `csrc/quantization/`, `csrc/rocm/`, etc.

### 4.2 How kernels are built
- Build system: `~/vllm/CMakeLists.txt`
  - Builds torch extensions (e.g., `_C`) for CUDA/ROCm/CPU depending on target.

### 4.3 How Python calls custom ops
There are two common patterns:

1) **Direct torch extension symbols** (common for legacy core ops)
- `~/vllm/vllm/_custom_ops.py` calls ops like:
  - `torch.ops._C.paged_attention_v1(...)`
  - `torch.ops._rocm_C.paged_attention(...)`
  - `torch.ops._C_cpu.*` for some CPU ops

2) **CustomOp abstraction** (model/layer level)
- `~/vllm/vllm/model_executor/custom_op.py` defines `CustomOp(nn.Module)` which dispatches to:
  - `forward_cuda`, `forward_hip`, `forward_cpu`, `forward_xpu`, etc.

This is a key extensibility mechanism: new optimized kernels can be integrated behind a stable Python layer API.

---

## 5) Attention Backend Selection (v1)

Attention is pluggable: multiple backends can exist and are chosen based on config + platform support.

- Selector entry: `~/vllm/vllm/v1/attention/selector.py` — `get_attn_backend(...)`
- Backend interface: `~/vllm/vllm/v1/attention/backend.py`
- Backend implementations registry: `~/vllm/vllm/v1/attention/backends/`

Key idea:
- The platform decides what attention backend class to use.
- Backends can require specific KV cache layouts; the selector can adjust layout accordingly.

---

## 6) CUDA Graphs & Compilation (v1)

If you care about latency and “graph mode” behavior:

- Design doc: `~/vllm/docs/design/cuda_graphs.md`
- Dispatcher implementation: `~/vllm/vllm/v1/cudagraph_dispatcher.py` — `CudagraphDispatcher`
- Wrapper code: `~/vllm/vllm/compilation/cuda_graph.py` — `CUDAGraphWrapper`

Takeaway:
- v1 separates “compilation” and “cudagraph capture” more cleanly.
- Runtime dispatch chooses FULL vs PIECEWISE vs NONE based on batch shape (and backend support).

---

## 7) Model Registry (how models map into vLLM)

When adding a model or debugging why a model loads the “wrong” implementation:

- Registry: `~/vllm/vllm/model_executor/models/registry.py`
  - `ModelRegistry.resolve_model_cls(...)`
  - `ModelRegistry.register_model(...)`

This registry maps HF model/config identifiers to vLLM model implementations.

---

## 8) Tests, Lint, and Dev Workflow

### 8.1 Lint/format
Upstream workflow uses `pre-commit` hooks.

- Install: `uv pip install pre-commit` then `pre-commit install`
- Run: `pre-commit run -a`

### 8.2 Tests
- Test suite root: `~/vllm/tests/`
- Quick single file: `pytest -s -v tests/test_logger.py`
- Full suite: `pytest tests/`

Dependencies are split by files under:
- `~/vllm/requirements/` (e.g., `dev.txt`, `test.txt`, `cuda.txt`, `docs.txt`)

### 8.3 Fast iteration tips
- If you’re only changing Python code, prefer the precompiled build path if available:
  - `VLLM_USE_PRECOMPILED=1 uv pip install -e .`
- If you’re changing C++/CUDA, you’ll want the source build path and incremental build workflow.

---

## 9) A Practical Learning Roadmap

### Phase 1: Understand the control plane (1–2 days)
- Read `VllmConfig` and identify which config controls:
  - scheduler policy, attention backend, compilation/cudagraphs, cache behavior.

### Phase 2: Trace one request all the way (2–4 days)
Read in order:

1) `~/vllm/vllm/entrypoints/llm.py`
2) `~/vllm/vllm/v1/engine/llm_engine.py`
3) `~/vllm/vllm/v1/engine/core.py`
4) `~/vllm/vllm/v1/core/sched/scheduler.py`
5) `~/vllm/vllm/v1/executor/abstract.py` → `uniproc_executor.py`
6) `~/vllm/vllm/v1/worker/gpu_model_runner.py`

### Phase 3: Become productive in one subsystem (1–2 weeks)
Pick one:

- **Serving/frontend**: OpenAI server request parsing, streaming correctness.
- **Core scheduling**: fairness, preemption behavior, KV cache pressure, metrics.
- **Attention/backend/kernels**: attention backend selection, kernel dispatch, custom ops.

---

## 10) Good First Contribution Ideas (low-risk)

- Improve error messages / validation around sampling params or structured outputs.
- Add targeted tests for corner cases in output streaming.
- Scheduler instrumentation/metrics (ensure tests cover invariants).
- Documentation fixes: design docs, troubleshooting, examples.

Higher-risk (do later):
- Kernels (csrc), attention backend changes, compilation graph passes.

---

## 11) Extending vLLM without forking: Plugin System

If you want to add out-of-tree models/platforms/loggers:

- Design doc: `~/vllm/docs/design/plugin_system.md`

Supported plugin types include:
- general plugins (register models)
- platform plugins (custom device/platform)
- IO processor plugins
- stat logger plugins

---

## 12) Cheatsheet

- Start reading architecture: `~/vllm/docs/design/arch_overview.md`
- Trace runtime loop: `~/vllm/vllm/v1/engine/core.py`
- Scheduling decisions: `~/vllm/vllm/v1/core/sched/scheduler.py`
- GPU hot path: `~/vllm/vllm/v1/worker/gpu_model_runner.py`
- Attention backend selection: `~/vllm/vllm/v1/attention/selector.py`
- Python↔native ops glue: `~/vllm/vllm/_custom_ops.py` and `~/vllm/csrc/`

