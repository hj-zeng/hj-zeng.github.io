---
title: "slime 主流程"
date: 2026-10-29
draft: false
weight: 81
tags: ["slime", "RL", "Ray", "SGLang", "Megatron"]
math: false
toc: true
summary: "slime 训练主流程梳理：placement/rollout/training 初始化、训练循环与权重同步、offload 流程，以及权重更新时的 megatron→hf→IPC/SGLang 通信链路。"
---


大体流程（省略 offload 和 critic 流程）：

```mermaid
flowchart TB
    A["create_placement_groups<br/>创建通信、Ray 管理组"]
    B["create_rollout_manager<br/>初始化 rollout 管理组"]
    C["create_training_models<br/>创建 actor / critic 模型"]
    D["actor_model.update_weights<br/>更新权重到 SGLang"]
    E["训练循环<br/>for rollout_id in range(...)"]

    A --> B --> C --> D --> E

    classDef node fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1a1a1a
    class A,B,C,D,E node
```

## 初始化流程

### create_placement_groups（通信、Ray 管理组）

1. 根据 GPU 数量和训练策略（训推一体 / 训推分离）确定资源分配；
2. 创建 Ray placement group 对象。

### create_rollout_manager（rollout 管理组）

初始化 rollout 管理组：统计 GPU 数据，用 Ray 管理资源。

```mermaid
flowchart LR
    D{"资源分配模式"}
    M1["train-only"]
    M2["rollout-only"]
    M3["colocate<br/>训推共卡"]
    M4["默认模式<br/>actor 和 rollout 分别分配 GPU"]

    D --> M1
    D --> M2
    D --> M3
    D --> M4

    classDef node fill:#FFF8E1,stroke:#F9AB00,stroke-width:2px,color:#1a1a1a
    class D,M1,M2,M3,M4 node
```

返回对象：`ray_placement_group`。

步骤：

1. 用 Ray 创建一个远程 actor（分布式对象）实例；
2. 调用 remote 的 `get_num_rollout_per_epoch` 获取 rollout 数量（`num_rollout`）；
3. 若开启 `check_weight_update_equal`，会调用 `check_weight.remote(action="snapshot")` 和 `check_weight.remote(action="reset_tensor")`。

### create_training_models（actor/critic 模型）

创建 actor/critic 模型。`create_training_models` 初始化时会实例化 `RayTrainGroup`，它实际上就是 `MegatronTrainRayActor`。

步骤：

1. 调用 `allocate_train_group`，返回 `actor_model`（一个 Ray `RayTrainGroup` 对象）；
2. 调用 `actor_model.async_init`，返回 rollout id，用于后续采样流程；
3. `actor_model.set_rollout_manager`：把 rollout_manager 传入 actor_model。

init 调用链（最终调用 `MegatronTrainRayActor.init`）：

```mermaid
flowchart LR
    I1["create_training_models"]
    I2["allocate_train_group"]
    I3["RayTrainGroup"]
    I4["_allocate_group_for_actor"]
    I5["trainRayActor"]
    I6["MegatronTrainRayActor"]

    I1 --> I2 --> I3 --> I4 --> I5 --> I6

    classDef node fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1a1a1a
    class I1,I2,I3,I4,I5,I6 node
```

## 训练循环

```mermaid
flowchart LR
    S["for rollout_id in range(start_rollout_id, num_rollout)"]
    E{"满足 eval 条件?<br/>rollout_manager.eval"}
    G["数据生成<br/>rollout_manager.generate"]
    T["训练<br/>actor_model.async_train"]
    W["存权重<br/>save(rollout_id)"]
    U["权重更新<br/>actor_model.update_weights"]

    S --> E
    E -- 是 --> G
    E -- 否 --> G
    G --> T --> W --> U
    U -. "下一个 rollout_id" .-> S

    classDef node fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1a1a1a
    class S,E,G,T,W,U node
```

## offload 流程

```mermaid
flowchart TB
    subgraph PRE["train loop 前"]
        direction TB
        P1["rollout_manager.onload_weights<br/>加载推理权重"]
        P2["actor_model.update_weights<br/>更新模型参数"]
        P3["rollout_manager.onload_kv<br/>加载 KV cache"]
        P1 --> P2 --> P3
    end

    subgraph LOOP["train loop 时"]
        direction TB
        L1["rollout_manager.offload<br/>卸载推理权重"]
        L2["xxx.async_train 训练"]
        L3["critic_model.offload / actor_model.offload<br/>卸载训练权重"]
        L4["rollout_manager.onload_weights<br/>加载推理权重"]
        L5["actor_model.update_weights<br/>更新模型参数"]
        L6["rollout_manager.onload_kv<br/>加载 KV cache"]
        L1 --> L2 --> L3 --> L4 --> L5 --> L6
    end

    PRE --> LOOP

    classDef pre fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1a1a1a
    classDef loop fill:#FFF8E1,stroke:#F9AB00,stroke-width:2px,color:#1a1a1a
    class P1,P2,P3 pre
    class L1,L2,L3,L4,L5,L6 loop
```

## 权重更新与通信

权重更新实际调用的是 `weight_updater.update_weights` 方法。

### 权重更新（update_weights）

**actor.update_weights 步骤**

1. 获取需要更新的 engines：`get_updateable_engines_and_lock`；
2. 若有新 engine 加入，调用 `connect_rollout_engines` 重新建立拓扑；
3. 更新参数：`megatron → hf → flatten/序列化 → ipc/nccl → rollout`（调用 `UpdateWeightFromTensor.update_weights` / `UpdateWeightFromDistributed.update_weights`）。

**colocate 模式**下调用 `UpdateWeightFromTensor.update_weights`，大体逻辑：

1. 获取当前权重；
2. 调用 `get_hf_weight_chunks` 转为 HF 权重；
3. 调用 IPC（`__send_hf_params`）发送给 SGLang：
   - 拍平、序列化；
   - CPU 通信 metadata；
   - IPC 共享内存 tensor。

```mermaid
flowchart LR
    S1["权重版本 + 1"]
    S2["暂停 rollout"]
    S3["weights_getter 获取当前 Megatron 权重"]
    S4["get_hf_weight_chunk 拿到 HF chunk"]
    S5["_send_hf_params 发送参数给 SGLang"]
    S6["torch.cuda.ipc_collect<br/>回收 IPC 共享内存"]

    S1 --> S2 --> S3 --> S4 --> S5 --> S6

    classDef node fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1a1a1a
    class S1,S2,S3,S4,S5,S6 node
```

### 初始化与更新相关对象（create_rollout_manager / create_training_models）

`create_training_models` 初始化时会实例化 `RayTrainGroup`（即 `MegatronTrainRayActor`），详见上文"初始化流程"。权重更新依赖这些对象：`actor_model`、`rollout_manager` 与 `weight_updater` 协作完成权重从 Megatron 到 rollout（SGLang）的同步。

### IPC 通信链路

`_send_hf_params` 大体逻辑（colocate 调用 `_send_to_colocated_engine`）：

```mermaid
flowchart LR
    H1["遍历 HF 参数"]
    H2["FlattendTensorBucket 拍平"]
    H3["MultiprocessingSerialize 序列化"]
    H4["dist.gather_object"]
    H5["ipc_engine.update_weights_from_tensor"]

    H1 --> H2 --> H3 --> H4 --> H5

    classDef node fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1a1a1a
    class H1,H2,H3,H4,H5 node
```

整体链路：`GPU → CPU serialize → gather_object（Gloo CPU）→ Ray IPC to engine`。

### IPC 通信开销分析

colocate 是训练推理共卡模式，不能通过 tensor 指针传递数据（因为跨进程），需要通过 IPC 通信：

```mermaid
flowchart LR
    T1["各 rank 的 GPU tensor"]
    T2["_share_cuda_<br/>将 CUDA tensor 显存导出为<br/>可跨进程共享的 IPC 句柄"]
    T3["dist.gather_object 收集所有句柄<br/>（走 CPU，数据量小）"]
    T4["Ray remote call"]
    T5["SGLang engine 进程"]
    T6["cuIpcOpenMemHandle()<br/>映射同一块 GPU 物理内存（零拷贝）"]
    T7["SGLang 直接读到 GPU tensor"]

    T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7

    classDef node fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#1a1a1a
    class T1,T2,T3,T4,T5,T6,T7 node
```
