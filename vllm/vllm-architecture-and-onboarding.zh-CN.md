# vLLM（仓库 @ ~/vllm）— 架构、源码结构与贡献者入门笔记

[English](vllm-architecture-and-onboarding.md) | [简体中文](vllm-architecture-and-onboarding.zh-CN.md)

> 目标：一份实用、以代码指针为主的指南，帮助理解 vLLM 的运行时流水线（v1）、仓库结构，以及如何开始贡献。

---

## 0) TL;DR 架构概览

vLLM 是一个面向**高吞吐、低延迟生成**优化的推理/服务引擎。核心设计将职责拆分为：

- **前端 / 入口**：解析用户输入（离线 API 或 HTTP 服务）、校验参数、分词、管理流式输出。
- **Engine（v1）**：管理请求生命周期与输出流，把重活交给核心循环。
- **EngineCore 循环**：反复执行 **schedule → execute → postprocess**。
- **Scheduler**：决定每一步运行哪些请求/哪些 token。
- **Executor**：抽象本地/多进程/Ray 执行并处理 RPC/fanout。
- **Worker / ModelRunner**：设备相关的模型执行；GPU 路径是热路径。
- **KV cache manager**：分配/释放 KV block；支持前缀缓存与内存感知调度。
- **Attention backend + custom ops**：主要性能来自专用注意力内核和融合算子（C++/CUDA/HIP + Triton）。

### 典型实现：“v1”
在此代码库中，“legacy” 引擎模块是 v1 的薄封装。把 v1 当作权威实现：

- 同步引擎：`~/vllm/vllm/v1/engine/llm_engine.py`
- 异步引擎：`~/vllm/vllm/v1/engine/async_llm.py`
- 核心循环：`~/vllm/vllm/v1/engine/core.py`

---

## 1) 请求流（端到端）

### 1.1 离线推理 API
想理解用户侧 Python API，请从这里开始：

- `~/vllm/vllm/entrypoints/llm.py`

典型流程：

1. 构建 `LLM(...)` / engine args。
2. 校验 prompts 与 sampling 参数。
3. 添加请求。
4. 循环调用 `engine.step()` 直到输出完成。

### 1.2 OpenAI 兼容服务
想理解服务端、流式输出、请求解析与 FastAPI 生命周期，请从这里开始：

- `~/vllm/vllm/entrypoints/openai/api_server.py`

典型流程：

1. CLI args → async engine client。
2. HTTP 请求进入，每个请求变成内部 v1 `Request`。
3. 核心循环执行时，产生流式响应。

---

## 1.3 选择一个起步方向（A/B/C）

开始贡献时，先选一条“跑道”更容易。这里是三条常见入口：

### 快速决策指南

| 如果你想… | 选择 | 本地通常需要 GPU 吗？ | 典型 PR 规模 | 说明 |
|---|---|---:|---:|---|
| 改进 API 行为、请求解析、流式输出、兼容性 | A | 否（通常） | 小–中 | 易测试，正确性/UX 价值高 |
| 负责 batching/scheduling 行为与吞吐/延迟权衡 | B | 有帮助 | 中 | 影响大，需要理解请求状态 + KV cache |
| 关注性能（内核/注意力/量化/图优化） | C | 通常需要 | 中–大 | 性能影响最大，但设置与调试更复杂 |

建议的首个 PR 类型：
- **A**：修一个端点行为问题 + 定向测试。
- **B**：改一个调度不变量/指标 + 单元测试。
- **C**：修一个内核/后端 bug + 正确性测试（最好带微基准）。

### A) OpenAI 服务 / 流式输出（前端 & 服务）

内容：
- HTTP 请求解析 → 引擎请求创建 → 流式输出 token。
- OpenAI API 兼容性、请求/响应 schema、取消、超时、指标、日志。

位置：
- `~/vllm/vllm/entrypoints/openai/`（主服务实现）
- `~/vllm/vllm/entrypoints/`（共享入口工具）

典型入门任务：
- 修复流式输出边界情况与格式问题。
- 改进参数校验/错误信息并补测试。
- 调整指标与请求生命周期处理。

### B) 调度器 / 核心吞吐（engine core）

内容：
- vLLM 正确性与吞吐的核心：连续 batching、抢占、prefill/decode 混合、KV cache 感知调度。

位置：
- `~/vllm/vllm/v1/engine/core.py`（核心循环）
- `~/vllm/vllm/v1/core/sched/`（调度器）
- `~/vllm/vllm/v1/core/kv_cache_manager.py`（KV cache 管理）
- `~/vllm/vllm/v1/executor/`（执行策略：单进程/多进程/Ray）
- `~/vllm/vllm/v1/request.py`（请求状态机）

典型入门任务：
- 调整调度策略（公平性/延迟）并补测试。
- 增加调度/KV 使用指标并验证不变量。
- 修复请求生命周期细节问题（abort/preempt/finish）。

### C) 注意力 / 内核（底层性能）

内容：
- 计算最重的部分：注意力后端、融合内核、自定义算子及其集成。
- 常涉及 CUDA/ROCm/Triton 与编译/图捕获相关工作。

位置：
- `~/vllm/vllm/v1/attention/`（后端接口 + 选择 + 实现）
- `~/vllm/vllm/_custom_ops.py`（Python ↔ 原生算子调用）
- `~/vllm/csrc/`（C++/CUDA/HIP 内核与注册）
- `~/vllm/vllm/model_executor/`（优化层、量化、模型代码）
- `~/vllm/vllm/compilation/` 与 `~/vllm/vllm/v1/cudagraph_dispatcher.py`（编译/图捕获控制）

典型入门任务：
- 新增或优化一个内核（并加入正确性测试）。
- 修复后端特定 bug（layout、dtype、shape、cudagraph 支持）。
- 改进注意力后端与编译模式的兼容性/分发。

---

## 2) v1 的“主干”（核心类与方法）

这是理解 token 生成流程的最小阅读集。

### 2.1 配置：控制平面
- `~/vllm/vllm/config/vllm.py` — `VllmConfig`

重要原因：
- 它聚合了**所有**配置：模型配置、cache 配置、并行、调度策略、注意力后端、编译/cudagraph 设置、LoRA、speculative decode、structured outputs、可观测性、KV 传输等。
- 它被广泛传递；理解配置流向几乎就是“沿着 VllmConfig 走”。

关键点：
- `VllmConfig.compute_hash()` — 标识影响图/编译的设置，用于缓存。

### 2.2 请求状态机
- `~/vllm/vllm/v1/request.py` — `Request`

作用：
- 跟踪请求生命周期：waiting/running/preempted/finished。
- 持有 prompt tokens、已生成 token ids、采样参数与计数器。
- 实现排序/优先级比较。

### 2.3 输入/输出处理
- `~/vllm/vllm/v1/engine/input_processor.py` — 请求校验/标准化、分词、多模态解析。
- `~/vllm/vllm/v1/engine/output_processor.py` — 组装增量输出、反分词、logprobs、流式收集。

如果想快速贡献，这块通常比较易上手、也容易测试。

### 2.4 Engine（前端）
- `~/vllm/vllm/v1/engine/llm_engine.py` — 同步引擎
  - 关键方法：`from_engine_args()`, `add_request()`, `step()`, `abort_request()`
- `~/vllm/vllm/v1/engine/async_llm.py` — 异步引擎
  - 关键方法：`from_engine_args()`, `add_request()`, `generate()`（异步生成器）, `shutdown()`

### 2.5 EngineCore（核心循环）
- `~/vllm/vllm/v1/engine/core.py` — `EngineCore`

职责：
- 选择 executor 类型，初始化 worker。
- 初始化 KV cache 与 scheduler。
- 运行 step 循环：调度 tokens → 执行模型 → 返回输出。

### 2.6 调度器
- `~/vllm/vllm/v1/core/sched/scheduler.py` — `Scheduler.schedule()`

职责：
- 连续 batching：挑选哪些请求推进。
- 处理 prefill/decode 混合、分块 prefill、抢占。
- 与 KV cache 预算与 block 分配协作。

### 2.7 KV cache manager
- `~/vllm/vllm/v1/core/kv_cache_manager.py` — `KVCacheManager`

职责：
- 跟踪可用 KV blocks。
- 计算 prefix-cache 命中。
- 为调度 tokens 分配“槽位”。

### 2.8 Executor 抽象（本地、多进程、Ray）
- `~/vllm/vllm/v1/executor/abstract.py` — base `Executor`, `get_class()` 选择。
- `~/vllm/vllm/v1/executor/uniproc_executor.py` — 最简单实现。
- `~/vllm/vllm/v1/executor/multiproc_executor.py` — 多进程 worker 编排。
- `~/vllm/vllm/v1/executor/ray_executor.py` — Ray 编排。

关键方法：
- `execute_model(...)`
- `sample_tokens(...)`
- `collective_rpc(...)`（多 worker 协调）

### 2.9 Worker + GPU model runner
- `~/vllm/vllm/v1/worker/worker_base.py` — `WorkerBase` 接口。
- `~/vllm/vllm/v1/worker/gpu_worker.py` — GPU worker 初始化与接线。
- `~/vllm/vllm/v1/worker/gpu_model_runner.py` — `GPUModelRunner.execute_model()` 热路径。

若要理解“时间都花在哪里”，请仔细阅读 `GPUModelRunner.execute_model()`。

---

## 3) 源码结构导览（哪里有什么）

### 3.1 顶层目录
- `~/vllm/vllm/` — 主 Python 包。
- `~/vllm/csrc/` — 原生内核 + torch 扩展注册。
- `~/vllm/docs/` — 设计文档、用户文档、贡献者文档。
- `~/vllm/tests/` — pytest 测试套件（大量子目录）。
- `~/vllm/benchmarks/` — 基准脚本与工具。

### 3.2 Python 包重点

- `~/vllm/vllm/v1/`
  - 权威 “v1 engine”：engine/core/scheduler/worker/executor/attention。

- `~/vllm/vllm/entrypoints/`
  - 面向用户的 CLI 与服务入口。
  - 包含 OpenAI 兼容服务实现。

- `~/vllm/vllm/model_executor/`
  - 模型加载与模型实现。
  - 量化与自定义层。
  - 模型注册表：HF 配置/类 → vLLM 模型实现。

  关键子目录：
  - `model_executor/models/` — 模型实现与注册表。
  - `model_executor/model_loader/` — 权重加载策略。
  - `model_executor/layers/` — 优化层（attention、linear、fused MoE、量化内核）。
  - `model_executor/custom_op.py` — `CustomOp` 抽象（分发到 CUDA/HIP/CPU/XPU）。

- `~/vllm/vllm/compilation/`
  - torch.compile / inductor 集成、图变换、融合 pass。
  - CUDA Graph 封装、缓存与分发支持。

- `~/vllm/vllm/distributed/`
  - 设备通信、并行状态、KV 传输/连接器基础设施。
  - 多机与特殊部署模式的集成点。

- `~/vllm/vllm/platforms/`
  - 平台检测 + 后端特定行为钩子。
  - 决定注意力后端与内核选择。

- `~/vllm/vllm/plugins/`
  - 插件系统入口加载器。

---

## 4) 原生内核与自定义算子（Python ↔ C++/CUDA/HIP）

### 4.1 内核在哪里
- 原生源码：`~/vllm/csrc/`
  - 典型子目录：`csrc/attention/`, `csrc/moe/`, `csrc/mamba/`, `csrc/quantization/`, `csrc/rocm/` 等。

### 4.2 内核如何构建
- 构建系统：`~/vllm/CMakeLists.txt`
  - 构建 torch 扩展（如 `_C`），根据目标支持 CUDA/ROCm/CPU。

### 4.3 Python 如何调用自定义算子
常见两种模式：

1) **直接调用 torch 扩展符号**（老的核心算子常用）
- `~/vllm/vllm/_custom_ops.py` 调用：
  - `torch.ops._C.paged_attention_v1(...)`
  - `torch.ops._rocm_C.paged_attention(...)`
  - `torch.ops._C_cpu.*`（部分 CPU 算子）

2) **CustomOp 抽象**（模型/层级调用）
- `~/vllm/vllm/model_executor/custom_op.py` 定义 `CustomOp(nn.Module)`，分发到：
  - `forward_cuda`, `forward_hip`, `forward_cpu`, `forward_xpu` 等。

这是一个关键的扩展机制：新的优化内核可以隐藏在稳定的 Python 层 API 之后。

---

## 5) 注意力后端选择（v1）

注意力是可插拔的：多个后端并存，基于配置与平台能力选择。

- 选择入口：`~/vllm/vllm/v1/attention/selector.py` — `get_attn_backend(...)`
- 后端接口：`~/vllm/vllm/v1/attention/backend.py`
- 后端实现注册表：`~/vllm/vllm/v1/attention/backends/`

关键点：
- 平台决定使用哪个注意力后端类。
- 后端可能要求特定 KV cache 布局；selector 可据此调整布局。

---

## 6) CUDA Graph 与编译（v1）

关注延迟与“图模式”行为：

- 设计文档：`~/vllm/docs/design/cuda_graphs.md`
- 分发实现：`~/vllm/vllm/v1/cudagraph_dispatcher.py` — `CudagraphDispatcher`
- 封装代码：`~/vllm/vllm/compilation/cuda_graph.py` — `CUDAGraphWrapper`

要点：
- v1 更清晰地分离了“编译”和“cudagraph capture”。
- 运行时基于 batch 形状（与后端支持）在 FULL/PIECEWISE/NONE 之间选择。

---

## 7) 模型注册表（模型如何映射到 vLLM）

当你添加模型或调试“为何加载了错误实现”时：

- 注册表：`~/vllm/vllm/model_executor/models/registry.py`
  - `ModelRegistry.resolve_model_cls(...)`
  - `ModelRegistry.register_model(...)`

该注册表将 HF 模型/配置标识符映射到 vLLM 模型实现。

---

## 8) 测试、Lint 与开发流程

### 8.1 Lint/format
上游工作流使用 `pre-commit` 钩子。

- 安装：`uv pip install pre-commit` 然后 `pre-commit install`
- 运行：`pre-commit run -a`

### 8.2 Tests
- 测试根目录：`~/vllm/tests/`
- 快速单文件：`pytest -s -v tests/test_logger.py`
- 全量：`pytest tests/`

依赖分散在：
- `~/vllm/requirements/`（如 `dev.txt`, `test.txt`, `cuda.txt`, `docs.txt`）

### 8.3 快速迭代建议
- 若只改 Python，优先使用预编译路径（如果可用）：
  - `VLLM_USE_PRECOMPILED=1 uv pip install -e .`
- 若改 C++/CUDA，建议使用源码构建与增量构建流程。

---

## 9) 实用学习路线

### Phase 1：理解控制平面（1–2 天）
- 阅读 `VllmConfig`，识别哪些配置控制：
  - 调度策略、注意力后端、编译/cudagraph、缓存行为。

### Phase 2：端到端跟踪一次请求（2–4 天）
按顺序阅读：

1) `~/vllm/vllm/entrypoints/llm.py`
2) `~/vllm/vllm/v1/engine/llm_engine.py`
3) `~/vllm/vllm/v1/engine/core.py`
4) `~/vllm/vllm/v1/core/sched/scheduler.py`
5) `~/vllm/vllm/v1/executor/abstract.py` → `uniproc_executor.py`
6) `~/vllm/vllm/v1/worker/gpu_model_runner.py`

### Phase 3：在一个子系统内产出（1–2 周）
任选一个：

- **Serving/frontend**：OpenAI 服务请求解析、流式正确性。
- **Core scheduling**：公平性、抢占行为、KV cache 压力、指标。
- **Attention/backend/kernels**：注意力后端选择、内核分发、自定义算子。

---

## 10) 低风险的首个贡献

- 改进采样参数/structured outputs 的错误提示与校验。
- 增加流式输出边界情况测试。
- 调度器指标与不变量测试。
- 文档修复：设计文档、故障排查、示例。

高风险（后续再做）：
- 内核（csrc）、注意力后端、编译图 pass。

---

## 11) 无需 fork 的扩展方式：插件系统

如果你想添加外部模型/平台/日志：

- 设计文档：`~/vllm/docs/design/plugin_system.md`

支持的插件类型：
- 通用插件（注册模型）
- 平台插件（自定义设备/平台）
- IO processor 插件
- 统计日志插件

---

## 12) 速查表

- 架构起点：`~/vllm/docs/design/arch_overview.md`
- 运行时循环：`~/vllm/vllm/v1/engine/core.py`
- 调度决策：`~/vllm/vllm/v1/core/sched/scheduler.py`
- GPU 热路径：`~/vllm/vllm/v1/worker/gpu_model_runner.py`
- 注意力后端选择：`~/vllm/vllm/v1/attention/selector.py`
- Python↔原生算子：`~/vllm/vllm/_custom_ops.py` 与 `~/vllm/csrc/`
- 深入指南：`vllm-custom-ops-and-native-kernels-deep-dive.md`

---
