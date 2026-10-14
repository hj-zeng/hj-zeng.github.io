---
title: "slime megatron 权重转为 hf 权重的转换方式"
date: 2026-06-06
draft: false
weight: 88
tags: ["RL", "slime", "权重转换"]
math: false
toc: true
summary: "slime 参数 megatron_to_hf_mode 的 bridge 与 raw 两种模式在模型构建和权重迭代导出阶段的区别与细节梳理。"
---

# 一、slime 权重转换方式

slime 参数`megatron_to_hf_mode` 的 `bridge` 和 `raw` 两种模式影响的是 **Megatron 模型权重导出为 HuggingFace 格式**的方式，涉及两个阶段：**模型构建 (model provider)** 和 **权重迭代导出 (weight iterator)**。

## 1.1 bridge 模式

bridge 模式通过借助 megatron-bridge 实现权重转换。

**模型构建** 

使用 `megatron.bridge.AutoBridge` 从 HF checkpoint 的 config 自动推导 Megatron 模型结构。特点是：

* `AutoBridge.from_hf_pretrained()` 读取 `args.hf_checkpoint`，调用 `provider = bridge.to_megatron_provider(load_weights=False)` 生成 model provider。
* 自动适配 TP/PP/EP/SP 等并行配置，**无需手写模型参数映射。**
* 支持与 `args.spec` 联合：在 bridge 构建模型后，再按 spec 替换特定层（如 Qwen3.5 GDN 层的 self_attention → `HuggingfaceAttention` 子类）

**权重导出**

利用 bridge 的 `get_conversion_tasks(model)` + `export_hf_weights(model, ...)` 做转换：

* Bridge 持有 HF ↔ Megatron 参数名/形状的映射关系，**自动处理 TP all-gather、PP/EP 通信**，直接导出 HF 格式权重
* 手动补充一些后处理：RMSNorm `+1.0` 还原、conv1d 维度 squeeze、量化等

## 1.2 raw 模式

raw 模式主要在slime 侧自主实现；并且raw 模式初始时需要指定 megatron的权重。

**模型构建** 

手动用 `core_transformer_config_from_args(args)` 构造 `TransformerConfig`，再 `GPTModel(**kwargs)` 创建模型：

* 需要手动指定 `vocab_size`、`transformer_layer_spec`、`position_embedding_type` 等全部参数
* `args.spec` 决定每层结构（本地 spec / TE spec / 自定义 spec 函数）

**权重导出**

不依赖 bridge，**手动收集** Megatron 模型的 `named_parameters`，构建 `ParamInfo`（含 TP/PP/EP 分片元信息）

* 手动做 PP broadcast → EP broadcast → TP all-gather 拼回完整参数
* 然后调用 `convert_to_hf()` 做 Megatron → HF 的参数名/形状转换

## 1.3 核心对比

|****|**bridge**|**raw**|
|:---|:---|:---|
|**模型构建**|AutoBridge 从 HF config 自动推导|手动传参构造 GPTModel|
|**HF↔Megatron 映射**|Bridge 自动维护|`convert_to_hf()` 手写映射逻辑|
|**TP/PP/EP 通信**|Bridge 内部处理|手动 broadcast + all-gather|
|**新模型适配**|只需 Bridge 支持，改动少|需要手写 `convert_to_hf` 映射|
|**灵活度**|受 Bridge 支持的模型限制，但可通过 spec 扩展|完全自由，但维护成本高|

简单说：**bridge 是"自动挡"**，让 `megatron-bridge` 库帮你搞定模型结构和权重映射；**raw 是"手动挡"**，所有映射和通信逻辑都要自己写。对于已支持 bridge 的模型（如 Qwen 系列），bridge 模式更省心；对于 bridge 尚未覆盖的模型或需要深度定制的场景，raw 模式更灵活。

# 二、细节梳理

## 2.1 两种模式 get_model_provider_func 的区别

`get_model_provider_func` 的作用是**根据启动参数，返回一个 Megatron 框架所需的 `model_provider` 闭包函数**。Megatron 训练框架在初始化时会调用这个 provider 来构建模型实例。slime 有三种方式返回 provider：

```bash
get_model_provider_func(args, role)
  ├─ ① custom_model_provider_path → 自定义 provider
  ├─ ② megatron_to_hf_mode == "bridge" → bridge provider
  └─ ③ 默认 → raw GPTModel provider
```
### 2.1.1 bridge 模式产生 provider 

大体分为两步，首先声明一个bridge类，然后再利用bridge 类生成 provider。

1. 声明 bridge 类：**AutoBridge.from_hf_pretrained**
    1. 根据 hf 中的config配置，多进程/多线程读取
    2. 读取完hf的配置之后，会检查当前的模型结构是否有AutoBridge 支持，以及有没有对应的转换关系
        1. 判断是否是自回归模型（目前只支持自回归模型）
        2. 获取架构名：arch_key
        3. 根据 arch_key 判断是否有转换实现


2. 获取 provider 模型类，hf->megatron 类转换：**bridge.to_megatron_provider**
    1. ModelProviderMixin（provider）类声明，将HuggingFace 预训练模型配置转换为 Megatron 模型 Provider 实例。
    2. 判断是否加载权重

3. 设置 provider 的 TP/PP/EP/SP/CP/variable_seq_lengths



示例 hf_name <-> megatron name 映射表：

|**HF 字段**|**Megatron 字段**|**说明**|
|:---|:---|:---|
|`num_hidden_layers`|`num_layers`|层数|
|`hidden_size`|`hidden_size`|隐藏维度|
|`intermediate_size`|`ffn_hidden_size`|FFN中间维度|
|`num_attention_heads`|`num_attention_heads`|注意力头数|
|`num_key_value_heads`|`num_query_groups`|GQA组数|
|`head_dim`|`kv_channels`|头维度|
|`vocab_size`|`vocab_size`|词表大小|
|`max_position_embeddings`|`seq_length`|最大序列长度|
|`rms_norm_eps`|`layernorm_epsilon`|LayerNorm epsilon|
|...|...|还有MoE、MLA、MTP等字段|

### 2.1.2 raw 模式产生 provider 

1. 构建模型时调用 model_provider 函数，返回的是 GPTModel 对象
2. 判断是否指定了 args.spec
    1. 如果指定了，则导入spec 指定的 attention（调用 transformer_layer_spec = import_module(args.spec)）
    2. 如果没有指定 spec 则
        1. num_experts > 0：get_gpt_decoder_block_spec
        2. num_experts < 0：get_gpt_layer_with_transformer_engine_spec(使用 transformer_engine 的情况) / get_gpt_layer_local_spec

    3. kwargs 初始化，如 attention space、mtp等
    4. kwargs 传入 GPTModel 进行实例化后返回


## **2.2 两种方式权重转换的区别**

同步模式的权重更新函数：UpdateWeightFromTensor 的init 函数，create 返回不同实例化的类，bridge 模式实例化HfWeightIteratorBridge类，raw 模式实例化HfWeightIteratorDirect类。

raw/bridge 的区分实际上只对 colocate 路径生效，非 colocate / NCCL transport：使用 `UpdateWeightFromDistributed`，直接调用 `convert_to_hf()`，始终走 raw 风格转换。

```python
self._hf_weight_iterator = HfWeightIteratorBase.create(
    args=args, model=model, model_name=model_name, quantization_config=quantization_config
)
```
HfWeightIteratorBridge 和HfWeightIteratorDirect 都继承HfWeightIteratorBase 类，各自实现 **get_hf_weight_chunks** 方法

get_hf_weight_chunks 的调用栈：

```shell
UpdateWeightFromTensor.update_weights -> _hf_weight_iterator.get_hf_weight_chunks
```
两种模式对应两个类的执行流程区别：

### bridge 模式

HfWeightIteratorBridge

![](/images/slime_megatron_to_hf_mode/image1.png)
### raw 模式

HfWeightIteratorDirect

![](/images/slime_megatron_to_hf_mode/image2.png)