---
title: "slime 权重转换逻辑"
date: 2026-10-13
draft: false
weight: 1
tags: ["RL", "slime", "权重更新"]
math: false
toc: true
summary: "slime 权重更新两类 UpdateWeightFromDistributed / UpdateWeightFromTensor 的实现逻辑。"
---

# 一、权重更新函数调用

slime 权重更新有两个类：UpdateWeightFromDistributed / UpdateWeightFromTensor ，分别对应 distributed 异步模式和 colocate 同步模式。

* 权重更新类的初始化

![](/images/slime_update_weight/image1.png)


* 权重更新类的调用

![](/images/slime_update_weight/image2.png)

# 二、UpdateWeightFromDistributed 的具体实现

该类主要有两个重要的成员函数，一个是 connect_rollout_engines，一个是 update_weights，前者的作用主要是建立通信组用于通信，后者的作用是进行权重更新。

## 2.1 connect_rollout_engines

该函数在 pp_rank=0的设备上，选取一个端口和 ip 作为通信的 master 地址，将推理引擎和训练侧都加入通信组。

![](/images/slime_update_weight/image3.png)

假设用两机16卡跑强化学习训练，资源分配为：推理（8张卡）+ 训练（4张卡）+奖励（4张卡），这里是 8 * rollout + 4 * actor + 4 * prm 的异步配置，8个rollout：4个推理引擎，每个推理引擎 2个GPU；由于在权重通信前，会先进行all_gather，此时 rank=0 拥有全部的权重，所以 actor 只需要一个rank 加入通信组，而推理侧接收到所有权重之后再进行划分，所有推理的所有rank都在通信组内。

## 2.2 update_weights

该函数首先会对 rollout engines 进行处理：暂停生成、kv cache、量化参数；

其次根据是否有专家分 EP 进行处理，最后调用的是 NCCL 进行通信。

![](/images/slime_update_weight/image4.png)


## 2.3 _update_weight_from_distributed / _update_bucket_weights_from_distributed

这两个函数作为类内的函数提供给 update_weights 调用，功能是进行实际的权重传输；

函数首先会判断当前参数大小是否超过更新buffer的阈值，大于的话就触发一次 NCCL 通信，之后再进行转换 buffer 的填充。

![](/images/slime_update_weight/image5.png)

![](/images/slime_update_weight/image6.png)

## 2.4 _update_expert_weight_from_distributed / _update_expert_bucket_weights_from_distributed

expert 种类的参数和非 expert 的种类参数一样，先要进行 TP 层面的 all_gather，然后判断当前转换的大小是否超过缓存区，超过则进行发送。不同点在于，在发送前还要进行一次 EP 的 all_gather，聚合之后再进行参数的转换，最后进行参数的发送。

和非 expert 对比，expert 是先 gather TP -> 再 gather EP -> 转换 -> 发送，而非 expert 是 gather TP -> 发送 -> 继续填充缓冲区。

![](/images/slime_update_weight/image7.png)

![](/images/slime_update_weight/image8.png)

# 三、UpdateWeightFromTensor 的具体实现

和 UpdateWeightFromDistributed 类似，同步模式的权重更新类 UpdateWeightFromTensor 也提供 connect_rollout_engines 和 update_weights 接口让外部调用。

除了通信方式的不同外，对比 UpdateWeightFromDistributed 模式，同步模式还会区分 bridge 和 raw 模式：" Bridge 模式存在的根本原因是 megatron-bridge 是外部维护的模型适配层，对新模型的支持更好，不需要在 slime 内部手写映射规则。但它依赖 HF checkpoint 来构建 bridge 对象，而 distributed 模式在设计时只用了 slime 内置的 `convert_to_hf`，没有接入 bridge。"

## 3.1 __init__

除了 bridge 或 raw 类的初始化之外，init 函数还会进行通信组的创建，不过这里创建的 backend 是 gloo，走的是CPU 通信，只传输一些 metadata，不传输实际的 tensor。

![](/images/slime_update_weight/image9.png)

## 3.2 connect_rollout_engines

connect_rollout_engines 函数会判断是否是同节点，不同节点走的NCCL；相同节点则走的 ipc，这个函数里面只是简单进行了 _ipc_engine 的映射。

![](/images/slime_update_weight/image10.png)

## 3.3 update_weights

update_weights 函数的前后处理和 UpdateWeightFromDistributed 的 update_weights 函数处理类似；中间处理的逻辑有些差别，UpdateWeightFromDistributed 中是判断是否超过 buffer区域，而这里是通过获取 chunk 的数据块来防止显存峰值占用太高；

get_hf_weight_chunks 是 raw / bridge 类中的方法，功能是获取chunk数据，然后进行megatron -> hf 的转换。

![](/images/slime_update_weight/image11.png)

## 3.4 权重发送

对于同节点主要走的是 ipc 零拷贝的模式，会将数据依赖进行拍平 -> 序列化 -> ipc 通信的操作，不会拷贝实际的GPU Tensor，没有Tensor的拷贝开销，MultiprocessingSerializer.serialize 接口识别到 GPU Tensor的时候，会获取 cuda ipc 的句柄，不会拷贝实际的 Tensor。

![](/images/slime_update_weight/image12.png)

![](/images/slime_update_weight/image13.png)

## 3.5 权重接收

这部分是 sglang 的接收逻辑，主要是进行发送逻辑的反操作

![](/images/slime_update_weight/image14.png)
