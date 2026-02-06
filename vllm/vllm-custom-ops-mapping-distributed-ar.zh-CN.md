# vLLM 自定义算子映射表：分布式 + All-Reduce 阶段

[English](vllm-custom-ops-mapping-distributed-ar.md) | [简体中文](vllm-custom-ops-mapping-distributed-ar.zh-CN.md)

范围：GPU 自定义 all-reduce 与 CPU 共享内存集体通信相关算子。

## 映射表：分布式 + all-reduce

| 算子 | Python 调用点 + GPT 推理用途 | 原生实现 + 说明/伪代码 |
|---|---|---|
| `init_custom_ar` | 自定义 all-reduce 初始化调用 [vllm/vllm/distributed/device_communicators/custom_all_reduce.py](vllm/vllm/distributed/device_communicators/custom_all_reduce.py#L194)。用途：创建自定义 all-reduce 上下文并初始化 IPC 缓冲区。 | CUDA/C++ 入口在 [vllm/csrc/custom_all_reduce.cu](vllm/csrc/custom_all_reduce.cu#L13)。分配通信句柄并初始化拓扑。伪代码：`ctx = new CustomAR(handles)`。
| `register_buffer` | 初始化后调用 [vllm/vllm/distributed/device_communicators/custom_all_reduce.py](vllm/vllm/distributed/device_communicators/custom_all_reduce.py#L197)。用途：注册每个 rank 的 IPC buffer。 | CUDA/C++ 入口在 [vllm/csrc/custom_all_reduce.cu](vllm/csrc/custom_all_reduce.cu#L113)。保存 IPC 指针。伪代码：`ctx.register_buffer(ipc_ptrs)`。
| `get_graph_buffer_ipc_meta` | 图捕获前调用 [vllm/vllm/distributed/device_communicators/custom_all_reduce.py](vllm/vllm/distributed/device_communicators/custom_all_reduce.py#L215)。用途：导出 graph buffer 的 IPC 句柄与偏移。 | CUDA/C++ 入口在 [vllm/csrc/custom_all_reduce.cu](vllm/csrc/custom_all_reduce.cu#L125)。返回 handle 与 offsets。伪代码：`handle, offsets = ctx.get_graph_buffer_ipc_meta()`。
| `register_graph_buffers` | 图模式设置调用 [vllm/vllm/distributed/device_communicators/custom_all_reduce.py](vllm/vllm/distributed/device_communicators/custom_all_reduce.py#L231)。用途：注册 graph buffer 元数据。 | CUDA/C++ 入口在 [vllm/csrc/custom_all_reduce.cu](vllm/csrc/custom_all_reduce.cu#L133)。记录 graph buffer 信息。伪代码：`ctx.register_graph_buffers(handles, offsets)`。
| `all_reduce` | collectives 调用 [vllm/vllm/distributed/device_communicators/custom_all_reduce.py](vllm/vllm/distributed/device_communicators/custom_all_reduce.py#L260)。用途：执行自定义 all-reduce。 | CUDA/C++ 入口在 [vllm/csrc/custom_all_reduce.cu](vllm/csrc/custom_all_reduce.cu#L62)。启动自定义 all-reduce kernel。伪代码：`custom_all_reduce(inp, out, world_size)`。
| `allocate_shared_buffer_and_handle` | 图模式准备调用 [vllm/vllm/distributed/device_communicators/custom_all_reduce.py](vllm/vllm/distributed/device_communicators/custom_all_reduce.py#L302)。用途：分配共享 IPC buffer 并导出 handle。 | CUDA/C++ 入口在 [vllm/csrc/custom_all_reduce.cu](vllm/csrc/custom_all_reduce.cu#L146)。返回指针与 handle。伪代码：`ptr, handle = alloc_shared(size)`。
| `open_mem_handle` | 打开 IPC handle 调用 [vllm/vllm/distributed/device_communicators/custom_all_reduce.py](vllm/vllm/distributed/device_communicators/custom_all_reduce.py#L314)。用途：将 IPC handle 映射为本地指针。 | CUDA/C++ 入口在 [vllm/csrc/custom_all_reduce.cu](vllm/csrc/custom_all_reduce.cu#L179)。伪代码：`ptr = open_mem_handle(handle)`。
| `init_shm_manager` | CPU communicator 初始化调用 [vllm/vllm/distributed/device_communicators/cpu_communicator.py](vllm/vllm/distributed/device_communicators/cpu_communicator.py#L219)。用途：创建共享内存管理器。 | CPU 入口在 [vllm/csrc/cpu/shm.cpp](vllm/csrc/cpu/shm.cpp#L856)。创建共享内存控制结构。伪代码：`handle = shm_init(name, group_size)`。
| `join_shm_manager` | 非 leader rank 调用 [vllm/vllm/distributed/device_communicators/cpu_communicator.py](vllm/vllm/distributed/device_communicators/cpu_communicator.py#L226)。用途：加入已有共享内存管理器。 | CPU 入口在 [vllm/csrc/cpu/shm.cpp](vllm/csrc/cpu/shm.cpp#L862)。伪代码：`join(handle, name)`。
| `shm_allreduce` | CPU communicator 调用 [vllm/vllm/distributed/device_communicators/cpu_communicator.py](vllm/vllm/distributed/device_communicators/cpu_communicator.py#L237)。用途：共享内存 all-reduce。 | CPU 入口在 [vllm/csrc/cpu/shm.cpp](vllm/csrc/cpu/shm.cpp#L828)。在共享内存中执行求和归约。伪代码：`sum_reduce(ctx, tensor)`。
| `shm_gather` | CPU communicator 调用 [vllm/vllm/distributed/device_communicators/cpu_communicator.py](vllm/vllm/distributed/device_communicators/cpu_communicator.py#L247)。用途：收集各 rank 的数据到输出缓冲。 | CPU 入口在 [vllm/csrc/cpu/shm.cpp](vllm/csrc/cpu/shm.cpp#L777)。伪代码：`gather(ctx, data, dst)`。
| `shm_all_gather` | CPU communicator 调用 [vllm/vllm/distributed/device_communicators/cpu_communicator.py](vllm/vllm/distributed/device_communicators/cpu_communicator.py#L260)。用途：全量收集 all-gather。 | CPU 入口在 [vllm/csrc/cpu/shm.cpp](vllm/csrc/cpu/shm.cpp#L803)。伪代码：`all_gather(ctx, data, output)`。
| `shm_send_tensor_list` | CPU communicator 调用 [vllm/vllm/distributed/device_communicators/cpu_communicator.py](vllm/vllm/distributed/device_communicators/cpu_communicator.py#L279)。用途：通过共享内存发送 tensor 列表。 | CPU 入口在 [vllm/csrc/cpu/shm.cpp](vllm/csrc/cpu/shm.cpp#L838)。伪代码：`send_list(ctx, tensors, dst)`。
| `shm_recv_tensor_list` | CPU communicator 调用 [vllm/vllm/distributed/device_communicators/cpu_communicator.py](vllm/vllm/distributed/device_communicators/cpu_communicator.py#L287)。用途：通过共享内存接收 tensor 列表。 | CPU 入口在 [vllm/csrc/cpu/shm.cpp](vllm/csrc/cpu/shm.cpp#L848)。伪代码：`tensors = recv_list(ctx, src)`。
