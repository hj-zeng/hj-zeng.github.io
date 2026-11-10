---
title: "deepspec 框架解读"
date: 2026-09-07
draft: false
weight: 5500
tags: ["投机解码"]
math: true
toc: true
summary: "DeepSpec 框架解读：训练部分的代码结构、关键概念（block_size/anchor/loss_mask）与 Qwen3DSparkModel 实现、block 并行预测与 flex_attention、验证部分的 eval 结构与回调、损失计算（CE/L1/Confidence 分布式归一化），以及 DFlash/Eagle3/MTP 算法对比。子标签：DeepSpec、Dspark。"
---

# 训练部分

## 代码结构

```python
DeepSpec/
├── train.py
├── eval.py
├── requirements.txt
├── README.md
├── LICENSE / NOTICE
│
├── config/
│   ├── dspark/
│   ├── dflash/
│   └── eagle3/
├── deepspec/
│   ├── __init__.py
│   ├── data/
│   ├── modeling/
│   │   ├── dspark/
│   │   └── eagle3/
│   ├── trainer/
│   ├── eval/
│   │   ├── base_evaluator.py
│   │   ├── dspark/
│   │   └── eagle3/
│   └── utils/
├── scripts/
│   ├── data/
│   ├── train/
│   └── eval/
├── eval_datasets/
├── train_datasets/
├── docs/
```


## 训练结构



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image1.png" alt="训练结构图" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">训练结构图</p>
</div>


对于每一类模型，都有一个对应的 Trainer，它主要实现两件事：1）补全基类没有实现的 `run_batch`，实现模型训练；2）构建草稿模型：

_build_draft_model

```python
def _build_draft_model(self, *, target_config, model_args):
    draft_config = build_qwen3_draft_config(
        target_config=target_config,
        model_args=model_args,
    )
    return Qwen3DSparkModel(draft_config)
```
model_args 定义在 DeepSpec/config/dspark/dspark_qwen3_4b.py。

target config 来自目标模型的模型文件：

```python
target_config = AutoConfig.from_pretrained(
    model_args.target_model_name_or_path,
)
```
Qwen3DSparkModel 定义在 DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py，继承自 transformers.models.qwen3.modeling_qwen3.Qwen3PreTrainedModel。

## 关键概念

### block_size

`block_size` = **一个 anchor 后面一次并行猜多少个 token**，也就是投机解码里的草稿长度

### assistant 回答

训练数据是多轮对话，不是纯文本。每条样本形如：

```python
{"conversations": [  {"role": "user",      "content": "帮我写个快排"},  {"role": "assistant", "content": "def quicksort(arr): ..."},  {"role": "user",      "content": "改成原地的"},  {"role": "assistant", "content": "def quicksort_inplace(arr, lo, hi): ..."}]}
```
`role` 有三种：`system`（系统设定）、`user`（人类说的话）、`assistant`（模型说的话）。「assistant 回复」就是 `role == "assistant"` 的那些 `content`，即模型自己生成出来的文本。

### loss_mask：哪些 token 值得学

形状 `[bsz, seq_len]`，0/1（uint8，代码里都用 `> 0.5` 判断）。它标记序列里**属于 assistant 回复内容的 token**，先全零，再用正则匹配对话文本里每一段 **assistant 回复**，把该段对应的 token 区间置 1。所以 system prompt、user 提问、chat template 的特殊标记都是 0。

它和 `attention_mask` 不是一回事：`attention_mask` 管 padding（哪些位置是真实 token），`loss_mask` 管监督（哪些真实 token 要算 loss）。草稿模型只需要学会预测模型自己会生成的内容，学用户输入没意义。

在 `forward` 里 `loss_mask` 有两处用途：

* 决定哪些位置可以当 anchor（`sample_anchor_positions`）
* 决定哪些草稿位算 loss（`build_eval_mask`）

### anchor：草稿块的起点

`anchor_positions` 形状 `[bsz, num_anchors]`，存的是**序列下标**。每个 anchor 代表「假设推理时已经生成到位置 `p`，从这里开始并行猜后面 `block_size` 个 token」。这就是在训练里模拟投机解码的现场。

具体机制：

* 候选条件：`loss_mask[p] == 1` 且 `loss_mask[p+1] == 1`。即 anchor 本身在监督区，且它的第一个预测目标也在监督区，否则这个块一个有效标签都没有。
* 采样：给候选位随机打分排序，取前 `num_anchors` 个再按下标升序排。无效位赋 `2.0` 保证排在最后。

> `anchors` 排成升序不是正确性必需的——各块之间在注意力上完全隔离，换顺序不影响结果。它的作用是给出一个规范化的布局：`create_position_ids` 生成的扁平 position_ids 随之单调递增，调试和看 mask 图时更直观。
* block_keep_mask：候选数不足 `num_anchors` 时会补 dummy（下标填 0），这个 mask 标记哪些块是真的。后续注意力 mask、eval mask 都会把 dummy 块整块屏蔽掉。

另外 anchor 在 forward 里串起了四件事：

1. `create_noise_embed`：块首位放 `input_ids[anchor]`（真实 token），其余 `block_size - 1` 位填 `mask_token_id` — 模型看到「我在 anchor，后面全是待填的洞」
2. `create_position_ids`：块内位置 id 为 `anchor + 0..block_size-1`，与真实序列位置对齐，RoPE 才正确
3. 注意力 mask：草稿 query 只能看 `kv_idx < anchor_pos` 的上下文 + 自己所在块，模拟推理时只有 anchor 之前的 KV 可用
4. 标签：`target_ids = input_ids[anchor+1 .. anchor+block_size]`

一个 4 大小的块，`anchor=10`，`seq_len=100` 时大致长这样：

```python
输入块:  [tok10, MASK, MASK, MASK]      position_ids = [10, 11, 12, 13]
可见KV:  context[0..9] + 本块自身
标签:    [tok11, tok12, tok13, tok14]
```
块内输入不含 `tok11..tok13`，全是 MASK，所以这是一次并行的多 token 预测，而不是逐位 teacher forcing。真实的前一个 token 只在 `prev_token_ids`里给 markov head / confidence head 用作修正特征。

### num_anchors 

num_anchors 是一次前向里从这条序列上采样多少个 anchor，也就是同时模拟多少个投机解码现场。

1. 与 block_size 的区别

两者是正交的两个维度，不是相关参数：

|**参数**|**含义**|**类比**|
|:---|:---|:---|
|`block_size=7`|每个块往后猜几个 token|块的「深度」|
|`num_anchors=512`|采多少个块|块的「数量」|


草稿模型这一路的 query 长度 = num_anchors * block_size = 512 * 7 = 3584。num_anchors 的上限受这个乘积约束——所以如果把 block_size 调大，通常得同步调小 num_anchors 才能维持同样的显存占用。

> 注意 KV 长度是 seq_len + num_anchors * block_size（上下文 + 草稿块），即 4096 + 3584 ≈ 7680。attention 矩阵名义上 3584 × 7680，但 mask 极度稀疏（每个块只看 anchor 之前的上下文 + 自己那 7 个位置），这就是为什么用 flex_attention 的 `create_block_mask` 而不是普通 dense mask——稀疏块能被整块跳过。
2. num_anchors=512 这个值的含义

候选 anchor 是所有满足 loss_mask[p] == 1 且 loss_mask[p+1] == 1 的位置。在 max_length=4096 的对话样本里，assistant 内容通常占几百到两千个 token。所以 512 大致是「把这条样本里能用的监督位置尽量吃满」的量级：

* 有效候选 > 512：随机抽 512 个，每个 epoch 抽到不同子集，起到数据增强的作用
* 有效候选 < 512：全部用上，剩下的槽位补 dummy 块（block_keep_mask 为 False，无梯度）

3. 为什么需要采这么多 anchor

一个 anchor 只给 block_size 个（且经 eval_mask 前缀截断后往往更少）监督信号。如果只采少量 anchor，一步梯度的信噪比很低。采 512 个则是在一次前向里就把这条序列的大部分位置都当成过「推理起点」练了一遍，**等效于极大的 batch**。这也是为什么 local_batch_size=1 就够用——真正的并行度藏在 num_anchors 里，而不是 batch 维度。

## Qwen3DSparkModel 草稿模型类的实现

### init

* 初始化 embed_tokens、layers（Qwen3DSparkDecoderLayer）、rotary_emb、lm_head、norm、block_size
*  markov_head = build_markov_head

> 给 block 内相邻 token 加一阶依赖修正（`markov_rank > 0` 时启用）
* confidence_head

```python
input_dim += config.markov_rank
self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
```
> 预测每个草稿位的接受率；若 `confidence_head_with_markov` 则额外拼接 markov 的 prev-token embedding，输入维度 `hidden_size + markov_rank`
### initialize_embeddings_and_head

输入 `embed_tokens` 和 `lm_head`（nn.Module）；初始化 token embedding 和 lm head。

### compute_logits

返回 lm head 计算的 logits

### predict_confidence_step

推理时单步版接受率预测；无 head 时返回 `None`，带 markov 时拼接 `prev_token_ids` 的 embedding。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image2.png" alt="predict_confidence_step 流程" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">predict_confidence_step 流程</p>
</div>


* 指定了 confidence_head_with_markov，拼接 prev_embeddings 和 hidden state，然后输入到 self.confidence_head 的前向
* 没有指定 confidence_head_with_markov，直接收入 hidden state 返回 self.confidence_head 的前向

self.confidence_head  的声明：

```python
from deepspec.modeling.dspark.common import AcceptRatePredictor
self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
```
### sample_draft_tokens

整块采样。`proposal_len == 0` 返回空张量；无 markov head 直接 `sample_tokens`；有则走 `markov_head.sample_block_tokens`，块内顺序展开、逐位用上一个采样结果修正 logits。返回 `(tokens, logits)`。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image3.png" alt="sample_draft_tokens 流程" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">sample_draft_tokens 流程</p>
</div>


sample_tokens 的实现

```python
#根据 logits 采样出每个位置的 token id
def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)

    bsz, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(-1, vocab_size) / temperature
    probs = torch.softmax(flat_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).reshape(bsz, seq_len)
```
* 当 `temperature < 1e-5`（近似 0）时，视为温度为 0，直接对最后一维取 `argmax`，返回每个位置概率最高的 token。结果形状 `(bsz, seq_len)`。这是确定性输出。
* 其他分支：把 `logits` 展平成二维 `(bsz*seq_len, vocab_size)`，并除以 `temperature`。温度越高分数被压缩得越平缓，采样越随机；温度越低越接近贪心。
* 然后做 `softmax`，得到词表上的概率分布 `probs`。
* 最后用 `torch.multinomial` 按该分布对每一行抽 1 个 token（随机采样），再 reshape 回 `(bsz, seq_len)`。

### sample_draft_token_step

sample_draft_tokens 的单步版本，要求 `base_logits` 为 `[batch, vocab]`，用 `apply_step_logits` 施加 markov 修正后采样一个 token。给逐步式解码循环用。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image4.png" alt="sample_draft_token_step 流程" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">sample_draft_token_step 流程</p>
</div>




### _forward_backbone

核心前向：`hidden_norm(fc(target_hidden_states))` 得到上下文 KV 源，`rotary_emb` 按 `position_ids` 生成 RoPE，逐层跑 decoder，最后 `norm`。噪声 embedding 作为 query 流。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image5.png" alt="_forward_backbone 前向" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">_forward_backbone 前向</p>
</div>


### forward

在一次 forward 里，从序列上随机挑 N 个位置，让 draft 模型基于 target 模型的隐状态特征，并行猜出每个位置之后的 B 个 token，并输出算 loss 所需的全部素材。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image6.png" alt="forward 流程" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">forward 流程</p>
</div>


1. **采样 anchor：**`sample_anchor_positions` 采 `num_anchors` 个 anchor，得到 `block_keep_mask`（不足时补 dummy anchor 并置 False）

> `num_anchors` 在训练时设置为 512，即 512 个 block 同时进行预测（相当于 batch size 的概念）。`num_anchors` 的设置最好配合输入序列长度，一般 `block_size` × `num_anchors` ≈ seq_len 效率更高
2. **创建噪声块（num_anchors × block_size）**：`create_noise_embed`：每块首位放真实 anchor token，其余填 `mask_token_id`
3. **确认位置编码 id：**位置 id = 上下文 `[0..seq_len)` 拼草稿块的 `anchor+offset`
4. **确认attention mask：**`create_dspark_attention_mask`：flex-attention block mask，草稿 query 只能看 anchor 之前的上下文 + 自己所在块
5. **进行模型前向：**`_forward_backbone` 得 `output_hidden`，reshape 成 `[bsz, num_blocks, block_size, H]`
6. **确认标签：**用 `anchor+1..anchor+block_size` gather 出 `target_ids`（越界 clamp、无效块归零）
7. **teacher logits 确认：**若传入 `target_last_hidden_states`，在偏移 -1 处 gather 出 target hidden 并送入 `compute_logits`，产出 `aligned_target_logits` 供蒸馏用
8. **创建eval mask：**`build_eval_mask`：越界 / `loss_mask` 关闭 / 无效块的位置去掉，并 `cumprod` 保证每块只监督连续有效前缀
9. **prev_token_ids 确认**：`prev_token_ids` = anchor token 拼 `target_ids[:, :, :-1]`（teacher forcing 的前一个 token）

> 主干输入里除块首外全是 `[M]`，所以主干看不到真实前驱。这一步单独把真值前驱拼出来
10. **计算草稿模型logit输出**：`compute_logits` → 可选 `markov_head.apply_block_logits` 修正 → `draft_logits`
11. `log_sampler_stats` 记录采样指标
12. 预测接收率：可选算 `confidence_pred`
13. 打包成 `DSparkForwardOutput(draft_logits, target_ids, eval_mask, block_keep_mask, confidence_pred, aligned_target_logits)`



前向传播的一个例子：

设定 `bsz=1`、`seq_len=12`、`block_size=3`、`num_anchors=5`。

输入：

```python
idx:          0      1      2      3      4      5      6      7      8      9     10     11
input_ids:  [sys]  [usr]  "写"   "首"   "诗"  [end]  [ast]  "床"   "前"   "明"   "月"  [end]
loss_mask:    0      0      0      0      0      0      0      1      1      1      1      1
```
assistant 内容是下标 7..11（"床前明月" + 结束符），只有这段 `loss_mask=1`。

**1. 采样 anchor**

候选条件 `loss_mask[p]==1 && loss_mask[p+1]==1`，候选范围 `p ∈ [0, 10]`：

|p|7|8|9|10|其他|
|:---|:---|:---|:---|:---|:---|
|有效|✓|✓|✓|✓|✗|

`valid_counts = 4 < num_anchors = 5`，全部选中，第 5 个槽位补哨兵后归零：

```python
anchor_positions = [[7, 8, 9, 10, 0]]
block_keep_mask  = [[T, T, T, T,  F]]    ← 第 4 块是 dummy
```
**2. 噪声块生成**

噪声序列长 `5 × 3 = 15`，块首放真实 anchor token，其余填 `[M]`。dummy 块首位也填 `[M]`：

```python
块0 (anchor=7):  [ "床", [M], [M] ]
块1 (anchor=8):  [ "前", [M], [M] ]
块2 (anchor=9):  [ "明", [M], [M] ]
块3 (anchor=10): [ "月", [M], [M] ]
块4 (dummy):     [ [M],  [M], [M] ]
```
块0 的语义就是：「已经生成到'床'，接下来 3 个 token 是什么？」

3. **position_ids 生成**

```python
上下文: [0,1,2,3,4,5,6,7,8,9,10,11]                        长 12
草稿:   [7,8,9 | 8,9,10 | 9,10,11 | 10,11,12 | 0,1,2]      长 15
拼接:   长 27
```
块3 出现了位置 12，超出了 `seq_len-1=11`——这没问题，它是那个 token 本该占据的位置，RoPE 需要这个值才正确。

4. **注意力 mask 生成**

Q_LEN=15，KV_LEN=27。以块0（anchor=7）为例，它的 3 个 query 能看到：

```python
上下文 kv_idx < 7  →  {0,1,2,3,4,5,6}
本块   kv_idx ∈ {12,13,14}          ← 偏移 12 = seq_len
```
注：

1. 是不含上下文位置 7（条件是 `kv_idx < anchor_pos` 严格小于），anchor token 是通过噪声块首位提供的，不走上下文。
2. 是块内双向可见——`mask_draft = is_draft & (q_block_id == kv_block_id)` 没有任何因果约束，`Qwen3DSparkAttention.is_causal = False`。所以这是 mask-filling / 扩散式的并行填空，不是自回归。

块4 因 `is_valid_block = False`，整行全屏蔽。

> 这块原理具体可见后面 block 的并行分析
5. **主干前向**

```python
output_hidden      [1, 15, H]
output_hidden_4d   [1, 5, 3, H]
```
6. **标签确定**

`label_indices = anchor + [1,2,3]`，越界 clamp 到 11，dummy 块归零：

|**块**|**anchor**|**label_indices**|**safe**|**target_ids**|
|:---|:---|:---|:---|:---|
|0|7|8, 9, 10|8, 9, 10|前, 明, 月|
|1|8|9, 10, 11|9, 10, 11|明, 月, [end]|
|2|9|10, 11, **12**|10, 11, 11|月, [end], [end]|
|3|10|11, **12, 13**|11, 11, 11|[end], [end], [end]|
|4|0|1, 2, 3|0, 0, 0|[sys], [sys], [sys]|

块2、块3 尾部的重复值和块4 的整行都是垃圾，靠 `eval_mask` 而不是靠这里清理。

7. **蒸馏用的 teacher logits 确认**

`target_pred_indices = safe_label_indices - 1`：

|**块**|**target_pred_indices**|**含义**|
|:---|:---|:---|
|0|7, 8, 9|target 在"床/前/明"处的输出 hidden|
|1|8, 9, 10||
|2|9, 10, 10||
|3|10, 10, 10||
|4|0, 0, 0||

在这些位置 gather `target_last_hidden_states`，过共享的 `lm_head`，得到 target 模型的分布 `p_target(x_k | x_<k)`。draft 就是去逼近它，而不只是逼近 one-hot 标签。这也是为什么 `compute_logits` 被单独抽成方法。

8. **eval mask 确认**

```python
target_valid = label_indices < 12:
  块0 [T,T,T]   块1 [T,T,T]   块2 [T,T,F]   块3 [T,F,F]   块4 [T,T,T]

loss_mask[safe] > 0.5:
  块0 [1,1,1]   块1 [1,1,1]   块2 [1,1,1]   块3 [1,1,1]   块4 [0,0,0]

& block_keep_mask, 再 cumprod:
  块0 [1,1,1]   块1 [1,1,1]   块2 [1,1,0]   块3 [1,0,0]   块4 [0,0,0]
```
有效监督位共 `3+3+2+1+0 = 9` 个。`cumprod` 在这个例子里没改变结果（本来就是前缀形），但如果 assistant 段中间断开（多轮对话里 user 插话）就会生效：某位失效后，该块后续位置一律不监督，因为投机解码是前缀式接受的。

9. 获取前一个token id

```python
anchor_token_ids      = [ 床,   前,   明,    月,   [sys] ]
target_ids[:,:,:-1]:
  块0 [前, 明]      块1 [明, 月]      块2 [月, [end]]    块3 [[end],[end]]   块4 [[sys],[sys]]

拼接 → prev_token_ids:
  块0 [ 床,   前,   明   ]
  块1 [ 前,   明,   月   ]
  块2 [ 明,   月,  [end] ]
  块3 [ 月,  [end], [end]]
  块4 [[sys],[sys],[sys] ]

```
10. 马尔可夫头修正，输出 logtis

```python
lm_head(output_hidden) → [1, 5, 3, V]
     ↓ markov_head.apply_block_logits(token_ids=prev_token_ids, hidden_states=output_hidden_4d)
draft_logits [1, 5, 3, V]
```
11. 返回数据



## block 并行预测和flex attention 

### attention mask 

（沿用上面的例子）

并行不是靠循环或分批，而是把 15 个互不相干的场景塞进同一条 query 序列，**靠 mask 把它们彼此隔离**。

先看张量形状怎么对上：

```python
q = self.q_proj(hidden_states)                        # 只来自噪声块 → q_len = 15
k = torch.cat([k_ctx, k_noise], dim=1)                # ctx_len + q_len = 12 + 15 = 27
```

所以 attention 矩阵是 15 × 27，正好对应

```python
 create_block_mask(..., Q_LEN=num_blocks*block_size, KV_LEN=seq_len + num_blocks*block_size)
```
RoPE 也是这么对齐的：position_embeddings 由长度 27 的 full_position_ids 算出，k 用全部 27 个，q 用尾部 15 个——这就是 modeling.py 重写 `apply_rotary_pos_emb`（其中 `cos[..., -q_len:, :]`）的原因。

画出这个 15 × 27 矩阵：

```
                上下文 kv 0..11              草稿 kv 12..26
                0 1 2 3 4 5 6 7 8 9 10 11 │ 块0    块1    块2    块3    块4
q0  ┐          ─────────────────────────  │ ─────  ─────  ─────  ─────  ─────
q1  ├块0 A=7    1 1 1 1 1 1 1 · · ·  ·  · │ 1 1 1  · · ·  · · ·  · · ·  · · ·
q2  ┘           1 1 1 1 1 1 1 · · ·  ·  · │ 1 1 1  · · ·  · · ·  · · ·  · · ·
q3  ┐           1 1 1 1 1 1 1 1 · ·  ·  · │ · · ·  1 1 1  · · ·  · · ·  · · ·
q4  ├块1 A=8    1 1 1 1 1 1 1 1 · ·  ·  · │ · · ·  1 1 1  · · ·  · · ·  · · ·
q5  ┘           1 1 1 1 1 1 1 1 · ·  ·  · │ · · ·  1 1 1  · · ·  · · ·  · · ·
q6  ┐           1 1 1 1 1 1 1 1 1 ·  ·  · │ · · ·  · · ·  1 1 1  · · ·  · · ·
q7  ├块2 A=9    1 1 1 1 1 1 1 1 1 ·  ·  · │ · · ·  · · ·  1 1 1  · · ·  · · ·
q8  ┘           1 1 1 1 1 1 1 1 1 ·  ·  · │ · · ·  · · ·  1 1 1  · · ·  · · ·
q9  ┐           1 1 1 1 1 1 1 1 1 1  ·  · │ · · ·  · · ·  · · ·  1 1 1  · · ·
q10 ├块3 A=10   1 1 1 1 1 1 1 1 1 1  ·  · │ · · ·  · · ·  · · ·  1 1 1  · · ·
q11 ┘           1 1 1 1 1 1 1 1 1 1  ·  · │ · · ·  · · ·  · · ·  1 1 1  · · ·
q12 ┐           · · · · · · · · · ·  ·  · │ · · ·  · · ·  · · ·  · · ·  · · ·
q13 ├块4 dummy  · · · · · · · · · ·  ·  · │ · · ·  · · ·  · · ·  · · ·  · · ·
q14 ┘           · · · · · · · · · ·  ·  · │ · · ·  · · ·  · · ·  · · ·  · · ·

```

右半边是严格块对角：mask_draft = is_draft & (q_block_id == kv_block_id)。块0 完全看不到块1/2/3 的任何东西，反之亦然。所以这 5 个块虽然在同一条序列里、同一次前向、共享同一份权重和上下文 KV，但计算结果完全独立——等价于跑了 5 次前向。这就是「并行」的全部含义。

左半边是阶梯形：mask_context = kv_idx < anchor_pos。每个块只看到它自己 anchor 之前的上下文。这一步保证了训练和推理的一致性：推理时在 anchor 处，KV cache 里只有 < anchor 的内容。

注：

1. 块内是双向的。 

mask_draft 里没有任何 q_idx >= kv_idx 之类的因果约束，且 Qwen3DSparkAttention.`__init__` 里 `self.is_causal = False`。所以块0 的 3 个 query 互相全见。DSpark 是 **mask-filling** 式的并行预测（类似扩散 LM），一次性把 [M],[M] 全填出来，而不是自回归逐位生成。这正是它比逐位 draft 快的地方。

2. anchor 本身不在上下文里可见。 

条件是 kv_idx < anchor_pos 严格小于，所以块0 看不到上下文位置 7。anchor token "床" 是通过噪声块首位（kv 列 12）提供的。这个安排让 anchor 的 hidden 由草稿模型自己算，而不是白拿 target 的。

### 并行预测的依据

回归 draft（EAGLE 那一类）严格分解：

```python
p(x₁..x₇ | ctx) = p(x₁|ctx) · p(x₂|ctx,x₁) · p(x₃|ctx,x₁,x₂) · ...
```
每一项都依赖前面**实际采出来的 token**，所以必须串行 7 次完整前向。

DSpark 换成条件独立近似：

```python
p(x₁..x₇ | ctx) ≈ p(x₁|ctx) · p(x₂|ctx) · ... · p(x₇|ctx)
```
7 个位置各自只条件于**上下文**，互不依赖对方的取值，于是可以一次算完。这是**精度换速度**——近似必然有损，接受率会掉，但一次前向出 7 个 token。

关键：

1. **双向 attention 在这里的作用：**

条件独立不等于「完全孤立地猜」。虽然位置 3 不知道位置 2 最终采出哪个 token，但双向 attention 让它能看到位置 2 的 **hidden state**。7 个 `[M]` 位置在每一层互相交换信息、协同细化，这就是 BERT / 扩散 LM 的一次性去噪。位置 3 的 hidden 里编码了「整块大致长什么样」，比独立预测强得多。

这为什么不是标签泄漏？因为块内除首位外**输入全是 `mask_token_id`**，没有任何真实 token。位置 3 看位置 2，看到的是「从 ctx + `[M]` 算出来的 hidden」，信息源头始终只有上下文。所以双向是安全的，而且必要。

2. **markov head 是补偿条件独立的手段**

条件独立会产生「各自最优但组合不合法」的输出。补救办法是在 **logits 层**加一个依赖前驱 token 的低秩 bias。

```python
prev_token_ids = first_prev_token_ids
for step_idx in range(proposal_len):
    step_logits = self.apply_step_logits(base_logits[:, step_idx, :], token_ids=prev_token_ids, ...)
    next_token_ids = sample_tokens(step_logits, temperature)
    prev_token_ids = next_token_ids        # ← 串行依赖
```
`markov_w1: Embedding(V, 256)` → `markov_w2: Linear(256, V)`，即一个 rank-256 的 `V × V` 转移矩阵分解。

推理时 `sample_block_tokens` 确实是**串行** 7 步的：

```python
自回归 draft:  7 × (完整 backbone 前向 + attention + KV cache 读写)
DSpark:        1 × 完整 backbone  +  7 × (embedding lookup + 一个 rank-256 线性映射)
```
`base_logits` 已经全部算好了，循环里只在 logits 上叠 bias。这个 head 没有 attention、没有 KV cache 访问、没有 5 层的堆叠。串行依赖从「整个网络」降级成「一个小 head」，这才是速度来源。

3. **推理侧的对应**

```python
block_hidden = model._forward_backbone(
    noise_embedding=model.embed_tokens(draft_input_ids),
    attention_mask=None,          # ← 不需要 mask
    is_causal=False,
)
```
推理时一次只处理一个块，KV 就是 `cache(ctx) + 本块`，全可见恰好就是正确语义，所以 `attention_mask=None`。训练时那套复杂 mask 存在的唯一理由，是要把 512 个块塞进同一次前向。

### flex attention 

`torch.nn.attention.flex_attention`，PyTorch 2.5 起提供的 attention API。它解决的问题是：任意 mask 形状 × FlashAttention 的效率，两者以前只能选一个。

你只需要定义：

1. **`score_mod`**：对 softmax 之前的注意力分数做逐点修改（ALiBi 偏置、softcap 裁剪等）
2. **`mask_mod`**：布尔掩码，把某些位置设为 `-inf`（因果掩码、滑动窗口、跨文档隔离）

编译器（torch.compile/Inductor）自动把这个 Python 函数编译融合成 Triton / CuTe / FlashAttention 内核，自动生成前向 + 反向，避免显式存储完整 \(QK^T\) 大矩阵，省显存。

三种做法的对比：

|**做法**|**显存**|**能否跳过空区域**|**灵活性**|
|:---|:---|:---|:---|
|dense mask 传给 SDPA|`O(B·H·Q·KV)`，必须物化|不能|任意|
|手写 Triton kernel|`O(1)`|能|每种 mask 写一遍|
|flex_attention|`O(1)`|能|任意（写个谓词函数）|

没有 compile 后端时会 fallback 到 eager 模式执行：

非 dynamo 环境下它用 `backend="eager"` 的 `torch.compile` 只做 trace、不生成 Triton，最终 HOP 分发到 `math_attention` 参考实现。

区别在性能，不在正确性：

|**对比项**|**fused (inductor)**|**eager fallback**|
|:---|:---|:---|
|score 矩阵|不物化|**物化** `[B,H,Q,KV]`|
|空 tile 跳过|有|**无**|
|结果正确性|✓|✓|

对本项目来说，就是 `3584 × 7680` 的 float 矩阵每层都要物化，且 0.2% 稀疏度的块对角区域被老老实实算完——跑得动，但慢且吃显存。

# 验证部分

## eval 

### eval 总体结构

BaseEvaluator 的结构主要是：1. init 时 build_model；2. 数据指标的处理；3. evaluate 进行投机解码验证。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image7.png" alt="eval 总体结构（一）" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">eval 总体结构（一）</p>
</div>




<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image8.png" alt="eval 总体结构（二）" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">eval 总体结构（二）</p>
</div>


最终调用的是 base_evaluator.py 的 `generate_decoding_sample` 函数，该函数通过回调接口进行 callback，不挑投机解码算法：

**回调契约**

|**回调**|**时机**|**DSpark 的实现**|
|:---|:---|:---|
|`init_context`|prefill 之后一次|建 draft 的 `DynamicCache`，从 prefill 的 `hidden_states` 抽 `target_layer_ids` 拼成 `target_hidden_states`|
|`propose`|每轮开头|构造 `[anchor, M, M, ...]` 噪声块，跑 draft，采样 + confidence 截断|
|`update`|每轮验证后|把 target 的 hidden 切到 `[:accepted+1]`，作为下一轮 draft 的上下文|
|`post_verify`|可选诊断钩子|喂 `ConfidenceHeadRecorder` 做校准统计|



generate_decoding_sample  函数：

它是一个 batch-size=1 的投机解码引擎：目标模型 prefill 出首 token → 循环中草稿模型提议 k 个 token、目标模型一次并行验证、rejection sampling 决定接受前缀并保证分布等价、直到达到长度上限或命中 EOS。对混合注意力目标额外做「快照+重跑」以维持递归状态一致性。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image9.png" alt="generate_decoding_sample 流程（一）" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">generate_decoding_sample 流程（一）</p>
</div>




<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image10.png" alt="generate_decoding_sample 流程（二）" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">generate_decoding_sample 流程（二）</p>
</div>


其中构建草稿的 `init_context`、更新草稿状态的 `update`、草稿生成的 `propose` 以及置信头校准都是调用的回调函数。



verify_draft_tokens 函数：

一次目标前向 → 对 k 个草稿 token 逐位置做 `min(1, p/q)` 拒绝采样并取连续接受前缀 → 被拒处用残差分布、全接受则用尾位置正常采样得到下一个 token → 打包接受段 + 新 token 返回。数学上保证最终输出与「直接用目标模型自回归采样」分布完全一致，同时把接受的部分省成了并行验证。

关键：

`sample_residual` 就是算 `max(0, p-q)` 归一化后采样。数学上可证明：**「以 min(1,p/q) 接受草稿 token」+「拒绝时从残差分布采样」= 直接从 p 采样**。这就是投机解码「无损加速」的保证 —— 输出分布和目标模型自回归完全一致。

直觉：草稿在这个 token 上「过度自信」的那部分概率质量被接受步扣掉了，残差分布再把目标模型「相对更看好、而草稿低估」的那部分补回来。

> **残差分布采样**（residual distribution sampling）是投机解码里，当草稿 token 被拒绝时用来重新采样下一个 token 的方法。它的目的只有一个：**让「接受草稿 + 拒绝后补采」的整体过程，在数学上严格等价于直接从目标分布采样**。
### dspark 草稿信息构建与生成

**草稿信息构建与update**

`init_context` 和 `_update` 是 DSpark 草稿状态的「建立」与「推进」，两者围绕同一个 `context` 对象工作 —— `_init_context` 在主循环前建一次，`_update` 每轮接受后更新一次。它们维护的核心就是喂给草稿模型的 target 隐藏态特征和草稿自己的 KV cache。

init_context :

```python
def _init_context(self, *, initial_output, **kwargs):
    return SimpleNamespace(
        past_key_values_draft=DynamicCache(),
        target_hidden_states=extract_context_feature(
            initial_output.hidden_states,
            self.draft_model.target_layer_ids,
        ),
    )
```
在 prefill 那次 target forward 之后调用，建立两样跨轮状态：

**1. `past_key_values_draft`** —— 空的 `DynamicCache`，草稿 backbone 自己的 KV 缓存（与 target 的缓存完全独立）。每轮 `_propose` 里 forward 时用它、用完 `crop(start)` 回退。

**2. `target_hidden_states`** —— DSpark 生成草稿的核心输入特征，由 `extract_context_feature` 从 prefill 输出里抽取。看它的实现:

```python
def extract_context_feature(hidden_states, layer_ids):
    return torch.cat(
        [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )
```
* `hidden_states` 是 HF 的 `output_hidden_states` 元组，共 `num_layers+1` 项：索引 0 是 embedding 输出，索引 `k+1` 是第 `k` 层的输出。
* 所以 `layer_id + 1` 的偏移是把「解码器层号 k」映射到元组下标 `k+1`；`layer_id == -1` 特指 embedding 输出（下标 0）。
* 对 `target_layer_ids`（如 `[1,9,17,25,33]`）逐个取出对应层隐藏态，在特征维拼接（`dim=-1`）。

结果形状约 `[1, seq, K·H]`（K 个层拼接）。第一轮 `_propose` 就用这个 prefill 全序列的多层特征作为草稿输入。

_update:

```python
def _update(self, context, verification):
    verified_target_hidden = extract_context_feature(
        verification.target_output.hidden_states,   # 本轮验证前向的 hidden states
        self.draft_model.target_layer_ids,
    )
    context.target_hidden_states = verified_target_hidden[
        :,
        : verification.accepted_draft_tokens + 1,    # 只保留已接受 + 1
        :,
    ]
```
每轮验证并接受后调用，刷新 `target_hidden_states`，供下一轮 propose 使用：

* 数据来源换成本轮验证前向的 hidden states（`verification.target_output.hidden_states`），同样用 `extract_context_feature` 抽同样的层拼接。验证前向覆盖了 `[tok0, d1..dk]` 这 k+1 个位置。
* 切片 `: accepted_draft_tokens + 1` 是关键：只保留「已接受的草稿 token + 那 1 个 bonus/校正 token」对应的隐藏态，丢弃被拒位置的。因为被拒 token 不会进入最终序列，它们的 target 特征对下一轮无意义。切片长度正好对应主循环里 `start += accepted_draft_tokens + 1` 推进的 token 数。

注意 `_update` 只更新了 `target_hidden_states`，没碰 `past_key_values_draft` —— 草稿 KV 的推进在 `_propose` 内部 forward + crop 里处理。

**init_context  和 _update 的闭环拼接**

```python
_init_context:  target_hidden_states ← prefill 的多层特征
                past_key_values_draft ← 空

每轮:
  _propose:  用 context.target_hidden_states + mask 块 → 草稿 backbone → k 个候选
  verify:    target 验证，接受 j 个 + 1 个 next_token
  _update:   target_hidden_states ← 本轮验证 hidden states 的前 (j+1) 个位置
             （下一轮 propose 就基于这些"已确认 token 的 target 特征"继续）
```


**_post_verify 和 propose 中置信度截断的关系**

置信度截断

```python
confidence_logits = _predict_confidence_logits(
    model,
    proposal_hidden_states=proposal_hidden_states,
    draft_input_ids=draft_input_ids,
    sampled_tokens=sampled_tokens,
    block_size=block_size,
)

proposal_draft_tokens = _confident_prefix_length(
    confidence_logits,
    block_size=block_size,
    threshold=float(confidence_threshold),
)
```
propose 里的置信度截断——是决策，会改变解码流程：

```python
proposal_draft_tokens = _confident_prefix_length(confidence_logits, threshold=confidence_threshold)
```
用置信度在第一个「自估会被拒」的位置提前截断草稿块，少提没把握的尾部 token，从而少让目标模型做无效验证、提升效率。这是 confidence head 的使用/应用。

post_verify（generate_decoding_sample 里的流程）——是观测，不改流程：

```python
self.confidence_head_recorder.observe(proposal=proposal, verification=verification)
```
把「模型预测的置信度」和「目标实际接受结果」配对收集，事后算 ECE/Brier，评估这个置信度预测得准不准。这是 confidence head 的校准/验证。

一句话：propose 里是「信任并使用置信度去截断」，post_verify 是「核对置信度值到底靠不靠谱」。

* 跑校准评测时：propose 不截断（threshold=0），靠 post_verify 收集校准数据；
* 跑加速部署时：propose 按阈值截断，post_verify 空转。



**_post_verify**

`_post_verify` 本身很薄，真正的流程在它调用的 `ConfidenceHeadRecorder.observe`里。整条链路是：每轮验证后，把 confidence head 的预测和真实接受结果配对，累积成校准直方图，评测结束时算 ECE/AUC/Brier

* **ECE** = `Σ|平均预测 − 平均实际| · 权重 / 总权重`（各桶预测置信度与实际接受率的加权偏差）。
* **AUROC** = 从正/负细桶直方图算排序判别力。
* **Brier** = 均方误差。
* **reliability** = 每桶 (avg_pred, avg_target) 对，用于画可靠性图。

`_post_verify` 每轮把 confidence head 的**单步预测累乘成前缀接受概率**、与验证得到的**真实前缀接受掩码**配对，按 block 位置流式累积成分桶直方图；评测结束后汇总算出每个位置的 ECE / AUROC / Brier 和可靠性曲线 —— 用来衡量「模型自估的接受概率」和「目标模型真实接受率」有多吻合，从而判断 confidence head 是否可信、部署时阈值该怎么设。全程对解码结果零影响。



**propose**

`_propose` = 以「当前已确认 token + mask 块 + target 隐藏态」为输入，让草稿 backbone 一次非因果前向出整块隐藏态（并立即 crop 回退草稿缓存避免污染），再逐位置采样出 k 个候选 token、可选地按置信度截断，最后打包成含 `verify_input_ids` 和草稿分布 `draft_probs` 的提议，供 `verify_draft_tokens` 验证。它是 DSpark「并行块生成」原理的具体落地点。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image11.png" alt="propose 流程" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">propose 流程</p>
</div>




## 马尔可夫头修正

### 马尔可夫头和置信头

||**马尔可夫头 (markov_head)**|**置信头 (confidence_head)**|
|:---|:---|:---|
|输出|词表上的 logits 偏置 → 影响采样哪个 token|一个标量 logit → 预测该 token 会被接受的概率|
|目的|修正草稿 token 的质量（提升接受率）|估计草稿 token 的可信度（用于截断/校准）|
|结构|`markov_w1`(Embedding) + `markov_w2`(Linear)，可选 gate/RNN|`AcceptRatePredictor`（MLP）|
|构建开关|`markov_rank > 0`|`enable_confidence_head`|
|缺省|可为 None（退化成直接采样）|可为 None|

它们在代码里是分别构建的（modeling.py:252 建 markov_head，modeling.py:262-267 建 confidence_head）。

**唯一的联系**：置信头可以选择性地**复用**马尔可夫头的 token embedding 作为输入特征。看 `confidence_head_with_markov` 分支（[modeling.py:299-306](https://webview-17cv2b0ke0m5askjtrgmemu4sai1bfpne5bc6a0hr1i5rbruus00.icoding.baidu-int.com/icoding/webview/comate/stable/1.108.2/deepspec/modeling/dspark/qwen3/modeling.py#L299-L306)）：

```python
if self.confidence_head_with_markov:
    prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids)  # 借用 markov_w1
    features = torch.cat([hidden_states, prev_embeddings], dim=-1)
    return self.confidence_head(features)
return self.confidence_head(hidden_states)   # 不借用时只用 hidden state
```
即便如此，也只是「置信头借了马尔可夫头的 embedding 当输入」，两者职责、输出、训练目标都不同。可以类比：马尔可夫头管「猜得准不准」，置信头管「知不知道自己猜得准不准」。



马尔可夫的核心实现在 deepspec/modeling/dspark/markov_head.py。「修正」的本质是：在 backbone 算出的 base_logits 上加一个依赖「前一个 token」的偏置，把并行块生成里缺失的 token 间**依赖补回来**。

**修正公式**

```python
def apply_step_logits(self, logits, *, token_ids, hidden_states):
    return logits + self.compute_step_bias(token_ids, hidden_states)
```
其中偏置（VanillaMarkov，[L26-32](https://webview-17cv2b0ke0m5askjtrgmemu4sai1bfpne5bc6a0hr1i5rbruus00.icoding.baidu-int.com/icoding/webview/comate/stable/1.108.2/deepspec/modeling/dspark/markov_head.py#L26-L32)）：

```python
def compute_step_bias(self, token_ids, hidden_states):
    return self.markov_w2(self.markov_w1(token_ids))   # W2 · W1[prev_token]
```
即 `corrected_logits = base_logits + W2·W1[前一个token]`。`markov_w1` 是低秩 embedding（rank=`markov_rank`），`markov_w2` 投回词表 —— 一个低秩的「上一个 token → 词表偏置」映射，建模 bigram/马尔可夫依赖。



### 推理时在哪触发

链路：`_propose` → `build_dspark_proposal` → model.sample_draft_tokens → `markov_head.sample_block_tokens`。

关键在 `sample_block_tokens`（markov_head.py:55-90）—— 它把「并行块」内部改成**逐位置自回归地应用修正**：

```python
prev_token_ids = first_prev_token_ids            # 起始锚点（当前已确认 token）
for step_idx in range(proposal_len):
    step_logits = self.apply_step_logits(          # ← 修正：base + markov bias
        base_logits[:, step_idx, :],
        token_ids=prev_token_ids,                  # 用上一步采出的 token
        hidden_states=step_hidden,
    )
    next_token = sample_tokens(step_logits, temperature)
    sampled_tokens.append(next_token)
    prev_token_ids = next_token                    # ← 递推：这一步的输出成为下一步的输入

```
注意这里的巧妙之处：**backbone 的 `base_logits` 是一次并行前向出的（块内位置无因果依赖），但马尔可夫修正在采样阶段用一个便宜的循环把 token 间依赖串起来** —— 每一步的偏置依赖上一步实际采出的 token。这样既保留了 DSpark「并行出隐藏态」的速度，又通过轻量 markov 头补上了「位置 k 的 token 应该条件于位置 k-1 的 token」这个自回归信息，从而提升草稿质量和接受率。

### 三种马尔可夫头变体

|类型|修正方式|状态|
|:---|:---|:---|
|`VanillaMarkov`|`W2·W1[prev_token]`，纯 bigram|无记忆|
|`GatedMarkovHead` ([L93-122](https://webview-17cv2b0ke0m5askjtrgmemu4sai1bfpne5bc6a0hr1i5rbruus00.icoding.baidu-int.com/icoding/webview/comate/stable/1.108.2/deepspec/modeling/dspark/markov_head.py#L93-L122))|用 hidden 算 gate 门控 embedding 再投影|无记忆|
|`RNNHead` ([L125-284](https://webview-17cv2b0ke0m5askjtrgmemu4sai1bfpne5bc6a0hr1i5rbruus00.icoding.baidu-int.com/icoding/webview/comate/stable/1.108.2/deepspec/modeling/dspark/markov_head.py#L125-L284))|GRU 式递归状态，能看到整个前缀 `x_{<k}`|有跨位置递归状态|

### 训练时的修正

在前向里（[modeling.py:488-489](https://webview-17cv2b0ke0m5askjtrgmemu4sai1bfpne5bc6a0hr1i5rbruus00.icoding.baidu-int.com/icoding/webview/comate/stable/1.108.2/deepspec/modeling/dspark/qwen3/modeling.py#L488-L489)）用 `apply_block_logits` 做 teacher-forced 的整块修正（用真实的前一 token，可并行/展开计算），与推理的自回归采样对应。

DSpark 损失 = `α_ce·CE + α_l1·L1 + α_conf·Confidence`：CE 让草稿 token 逼近目标真值，L1 拉近草稿与目标的完整分布（直接优化接受率），Confidence 用 BCE 让置信头去预测「1−TV」形式的接受率。所有项都按 `eval_mask`（可加位置衰减）加权，并采用**本地分子 / 全局分母 × world_size** 的分布式归一化，保证多卡梯度严格等价于单卡全量训练。

# 损失计算

## loss 计算

DeepSpec DSpark 的损失由 loss.py 计算，是三项加权和：CE（主损失）+ L1（分布对齐）+ Confidence（置信头 BCE）。核心在于分布式下分子/分母分离的归一化。

**输入：DSparkForwardOutput**
关键字段（形状约定 [B, num_blocks, block_size, V]）：

* draft_logits：草稿模型每个块内位置的预测 logits。
* target_ids：CE 的监督标签（目标真实 token）。
* aligned_target_logits：目标模型对齐后的 logits（用于 L1 和接受率估计，可为 None）。
* eval_mask：哪些位置参与损失（有效预测位）。
* block_keep_mask：哪些块有效。
* confidence_pred：置信头预测（可为 None）。

顶层入口：**compute_dspark_loss**
三步：

* _collect_local_terms：本卡算出各损失项的分子和分母（num/den）。
* _all_reduce_loss_denominators：把三个分母跨卡求和得全局分母。
* _build_loss：用「本地分子 / 全局分母」组合成反传损失。

### 三项损失

1. CE 损失（主损失）

```python
loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
ce_loss_num = (loss_per_token * flat_weights).sum()   # 分子：加权求和
ce_loss_den = flat_weights.sum()                       # 分母：权重和
```
* 逐 token 交叉熵，让草稿预测逼近目标真实 token。
* `flat_weights` 来自 `loss_weight_mask`，即 `eval_mask × 可选衰减`。

2. L1 分布对齐损失

```python
draft_probs = softmax(draft_logits)
target_probs = softmax(aligned_target_logits)
l1_dist_per_token = (draft_probs - target_probs).abs().sum(dim=-1)   # L1 距离 = 2·TV
l1_loss_num = (l1_dist_per_token * loss_weight_mask).sum()
```
直接拉近草稿分布和目标分布（不只是 argmax token，而是整个分布）。这与投机解码接受率直接相关 —— 分布越接近，接受率越高。

3. Confidence 损失

```python
confidence_targets = accept_rate_3d.detach()          # 监督目标 = 估计的接受率
confidence_errors = F.binary_cross_entropy_with_logits(
    confidence_pred, confidence_targets, reduction="none") * loss_weight_mask
```
* 用 BCE 训练置信头去预测每个位置的接受率。
* 监督信号 `accept_rate_3d` 是拿草稿分布和目标分布算出来的，`.detach()` 保证它只作为标签、不回传梯度到主干

### 关键辅助量

**接受率估计 `accept_rate_3d`**

```python
accept_rate_3d = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
```
这是投机解码接受率的理论值：`1 - TV(draft, target)`（TV = 总变差距离 = ½·L1）。它同时是 confidence 损失的监督目标，也用于训练指标。

**损失权重掩码与衰减**

```python
ce_loss = loss_terms["ce_loss_num"] / (global_denominators["ce_loss_den"] + 1e-6)
...
return (ce_loss_alpha*ce_loss + l1_loss_alpha*l1_loss + confidence_head_alpha*confidence_loss) * world_size
```
可选的**位置衰减**：块内越靠后的位置（越难预测、越可能被拒）损失权重越低，把训练重心放在前面几个更关键的位置。

### 分布式归一化

```python
ce_loss = loss_terms["ce_loss_num"] / (global_denominators["ce_loss_den"] + 1e-6)
...
return (ce_loss_alpha*ce_loss + l1_loss_alpha*l1_loss + confidence_head_alpha*confidence_loss) * world_size
```
* 分子用**本卡的** `ce_loss_num`，分母用**全局的** `ce_loss_den`（所有卡求和）。
* 最后乘 `world_size`。

原因：各卡有效 token 数不同，简单 all-reduce 平均会给「有效 token 少的卡」错误权重。这里的做法等价于「全局分子 / 全局分母」的精确加权平均：DDP 会自动对各卡梯度求平均（除以 world_size），而 `本地分子/全局分母 × world_size` 经 DDP 平均后 = `Σ本地分子 / 全局分母` = 全局逐 token 平均损失。这样无论 token 在各卡如何分布，得到的梯度都严格等价于单卡跑全部数据。

`local_loss` 只用本地 num/den，仅供打日志展示，不参与反传。

整体训练流

```python
DSparkForwardOutput
  ├─ draft_logits vs target_ids ──────── CE (主损失, num/den)
  ├─ draft_probs vs target_probs ──┬──── L1 分布对齐 (num/den)
  │                                └──── accept_rate_3d = 1 - ½·L1
  │                                          │
  │                            (detach 作标签) ▼
  └─ confidence_pred ─────────────── BCE ── Confidence (num/den)

各项 num/den ──► all_reduce(den) ──► 本地num/全局den × world_size
                                          │
                       ce_α·CE + l1_α·L1 + conf_α·Conf  ──► backward_loss

```
# dflash 和 eagle3 算法

DeepSpec 里 DFlash / Eagle3 和 DSpark 走的是**同一套三段式流水线**，唯一区别就是换 config 文件——算法由 config 里的 `trainer_cls`（训练）和 draft 的 `architectures[0]`（评测）来分发，`train.py` / `eval.py` 入口完全通用。

## DFlash

关键点：DFlash **复用了 DSpark 的 trainer 和 draft 模型**，只是把额外的 head 关掉、只用 CE loss。

两种配置对比：



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image12.png" alt="DFlash 两种配置对比" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">DFlash 两种配置对比</p>
</div>


## Eagle3 

Eagle3 用独立的 trainer 和 modeling，训练范式是 TTT（test-time-training，多步展开），没有 markov/confidence 那套。

Qwen3Eagle3Trainer 和 Qwen3DSparkTrainer 一样，都继承自 BaseTrainer，实现的方法都是 `_build_draft_model` 和 `run_batch`，只是底下调用的子方法不一样。

### Qwen3Eagle3Model eagle3 的 forward 实现



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_deepspec/image13.png" alt="Qwen3Eagle3Model forward" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">Qwen3Eagle3Model forward</p>
</div>


1. position_ids

|**维度**|**Eagle3**|**DSpark**|
|:---|:---|:---|
|生成方式|`arange(past_seen, past_seen+seq_len)`，连续自然序列|context 段 `arange(seq_len)` + draft 段 `anchor_pos + [0..block_size)`，拼成 `full_position_ids`|
|语义|逐步 TTT，每步整段位置自然递增；`rope_cache_step_offset` 时按 `past_seen//q_len` 步数额外偏移，模拟"接在已接受 token 后面"|draft token 的位置直接**对齐到它在原序列里应处的绝对位置**（anchor 之后 block_size 个），因此每个 block 的 pos 都是不连续的、跳跃的|
|关键点|位置随 cache 步进平移|位置由 anchor 采样决定，与 KV context 的绝对位置严格对齐|

2. cache

|**维度**|**Eagle3**|**DSpark**|
|:---|:---|:---|
|训练 forward|TTT 展开时逐步 `update`/`get_seq_length`；还有 `extend_draft_cache`做单步推进|训练 forward **不走 cache**，一次性算完所有 block|
|KV 来源|标准自注意力，K/V 来自自身 hidden（拼了 input_embeds，维度 `2*hidden`）逐步累积到 cache|无 cache 累积；K/V 由 target_hidden_states(context) 与 noise(hidden) **当场拼接**，context 部分就相当于"固定 KV"，但不落 cache|
|cache_position|显式计算并传入|不涉及|

核心区别：Eagle3 是**串行、有状态**（cache 随 TTT 步增长），DSpark 训练是**并行、无状态**（一次前向覆盖所有 anchor block）。

3. attention 

|**维度**|**Eagle3**|**DSpark**|
|:---|:---|:---|
|形状|Q=q_len, KV=kv_len(=past_seen+q_len)，随 TTT 步增长|Q=`num_blocks*block_size`，KV=`seq_len + num_blocks*block_size`（context + 所有 draft）|
|规则|`causal`（下三角，且 kv 在有效长度内）或 `suffix_mask`：TTT 后续步能看到前面每个 chunk 里"同相位"的 token（`(kv_idx-q_idx)%q_len==0`）；再叠加 padding 有效长度|每个 query 属于某个 block（`q_idx//block_size`）→ 取该 block 的 `anchor_pos`；可见 = context 中 `kv_idx < anchor_pos`（只看 anchor 之前的上下文）或 同 block 内的 draft token（`q_block_id==kv_block_id`）；再乘 `block_keep_mask`|
|因果性|本质是跨 TTT chunk 的因果 + 对角相位可见|非因果的块内全连接 + 对 context 的前缀可见；block 之间互不可见|
|padding 处理|用 `attention_mask.sum` 得每个样本有效长度 `seq_len`，`- lck` 补偿 cache 步|靠 `block_keep_mask` 屏蔽无效 anchor block（整块置 False）|

### 为什么 Eagle3 训练时也用 flex_attention

关键点：Eagle3 训练时的 TTT 不是串行 for 循环跑 7 次，而是把 7 步打包进一次前向并行算的，这就需要一个非标准的 mask。

* `causal_mask`：普通下三角，看原始 context。
* `suffix_mask`：跨 TTT 步的"同相位"可见——第 k 步的 query 能看到前面各 chunk 里位置对应（`(kv_idx - q_idx) % q_len == 0`）的 token。

这个"因果 + 相位对齐"的 mask 标准 SDPA 根本写不出来，所以必须用 flex_attention 的 `mask_mod`。它的目的正是把本来要串行展开的 TTT 步并行化成一次 attention 调用。所以 Eagle3 训练其实也是"并行"的，只是并行的对象是 TTT 步，不是 DSpark 那种 anchor block。

### eval 实现

关键方法

* **类属性**：`EVAL_ATTN_IMPLEMENTATION = "sdpa"`、`draft_model_cls = Qwen3Eagle3Model`。
* **`__init__`**：断言 `draft_num_hidden_layers == 1`。原因写在注释里：`_update` 会用无 causal mask 的方式一次性把多个已提交 token 塞进 draft cache，只有单层 draft 才安全（cache 的 K/V 直接从每 token 输入投影，不受上一层双向 attention 混合影响）。

> 对于单层，cache 里的 K/V 只依赖每 token 输入。若是多层，第 2 层的 K/V 来自第 1 层的双向 attention 输出（已混入未来），那第 2 层缓存就被污染了。所以正是"批量、无 mask 写 cache 但要保证 cache 正确"这个诉求，反过来要求 draft 必须单层。
* **`_init_context`**：prefill。抽取 prompt 的多层 target 特征，构造 **shifted prompt ids**（token i 对齐 hidden i，符合训练约定），用 `extend_draft_cache` 预填 draft KV cache，返回 `draft_cache / draft_hidden / position_ids / current_pos / cache_len_before`。
* **`_propose`**：**自回归展开 `ttt_length` 步**。循环里：`compute_logits(hidden)` → 采样 next_token → 把 token 喂回 `draft_model(...)` 拿下一步 hidden，命中 stop token 就提前停。这是真·串行 TTT 循环。
* **`_update`**：verify 后，把 draft cache `crop` 回提议前长度，再用**已提交的 target 特征 + committed tokens** 通过 `extend_draft_cache` 重新推进 cache，更新 `draft_hidden` 和 `current_pos`。
* **`generate_one_sample`**：把上面几个钩子传给 `generate_decoding_sample`（没有 `post_verify`）。



|**维度**|**Qwen3Eagle3Evaluator**|**Qwen3DSparkEvaluator**|
|:---|:---|:---|
|提议 token 数|`ttt_length`|`block_size`|
|`_propose` 机制|**串行自回归循环**（跑 ttt_length 步，逐步喂回 hidden）|**一次并行 block**（block_size 个 mask token 单次前向）|
|draft cache|核心状态：prefill + 每步 update/crop/extend，逐步累积|几乎不用：只存空 cache，propose 无状态、不 crop|
|`_init_context`|重：extend_draft_cache 预填 + shifted ids + 位置对齐|轻：空 cache + target 特征|
|`_update`|crop draft cache 回提议前，再 extend 已提交 token|只截 target 特征到 accepted+1|
|confidence head|无|有：recorder + `_post_verify` + evaluate 覆写|
|markov head|无（`draft_sample` 纯采样）|有（`build_dspark_proposal` 里叠加）|

## MTP

MTP = Multi-Token Prediction（多 token 预测），代表工作是 Meta 的“Better & Faster LLMs via Multi-token Prediction”，以及 DeepSeek-V3 里的 MTP 模块。核心特征：

* MTP 是 target 模型自身的能力。它在**基座模型（预）训练阶段就加进去**：模型除了预测下一个 token，还带若干"MTP 头/模块"同时预测未来第 2、3… 个 token。
* DeepSeek-V3 的 MTP 是串行模块：每个 MTP module 有自己的 transformer block，共享主模型的 embedding 和 output head，用主模型的 hidden + 上一个预测 token 递推预测下一个。
* 推理时这些 MTP 头可直接当 draft 用（自推测），**无需另一个独立模型**。

和 Eagle / DFlash / DSpark 的本质区别

|**维度**|**MTP**|**Eagle3 / DFlash / DSpark**|
|:---|:---|:---|
|draft 来自哪|**target 自带**的 MTP 模块（模型的一部分）|**独立训练的外挂 draft 模型**|
|何时训练|基座模型（预）训练时**联合训练**|target **冻结**后，事后单独训练 draft|
|是否需要改 target|需要（改架构 + 重训基座）|不需要，target 原样冻结|
|依赖|需要基座训练权限/算力 + target 发布时带 MTP 权重|只需 target 的中间层 hidden state（hook/cache 抓取）|
|适用对象|只有"生来带 MTP"的模型（如 DeepSeek-V3）|**任意**已有 target（Qwen3-4B/8B/14B、Gemma 等）|
|特征输入|MTP 模块吃主干 hidden|draft 吃 target 多层 hidden 拼接（`target_layer_ids`）|

一句话：MTP 是“target 内建、随基座一起练出来的原生多 token 预测能力”；Eagle/DFlash/DSpark 是“对冻结 target 事后加练的外挂 draft”。前者是模型自带属性，后者是可插拔适配层。