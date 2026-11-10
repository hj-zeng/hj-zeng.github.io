---
title: "BERT 论文解读"
date: 2025-12-17
draft: false
weight: 500
tags: ["论文笔记"]
math: true
toc: true
summary: "BERT 论文（Pre-training of Deep Bidirectional Transformers for Language Understanding）要点：双向编码器架构、MLM + NSP 预训练任务、80/10/10 掩码策略、预训练与微调的掩码不一致及其缓解、参数规模计算与下游任务微调。子标签：BERT、Transformer、预训练、MLM、NSP。"
---

BERT 是谷歌提出的基于双向 Transformer 编码器的预训练语言模型，通过掩码语言模型（MLM）和下一句预测（NSP）两大预训练任务，在大规模无标注文本上学习通用的上下文语义表征。预训练阶段采用 80/10/10 的软掩码策略缓解预训练-微调的掩码不一致问题；微调阶段可通过接入轻量级任务头快速适配分类、问答、NER 等下游任务。其双向建模能力突破了传统单向语言模型的局限，成为自然语言处理领域的里程碑模型。

> paper: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

# 论文关键信息

**论文**：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**关键**：预训练、双向、语言模型。

**训练过程**：预训练 + 微调

**网络结构**：多层 Transformer，encoder-only 架构

**可扩展参数**：Transformer block、Hidden size、head dim

**参数计算**

* 嵌入层：$30K \times H$（hidden size）
* QKV：矩阵为 $H \times H$，输入 QKV，输出还有一个线性矩阵，共 $4H^2$
* MLP 层：输入 $H$，中间层 $4H$，输出层 $H$，共 $8H^2$（$H \to 4H \to H$）

总参数量（$L$ 为层数）：

<div style="text-align:center">

$$P = 30K \times H + 12 \times H^2 \times L \qquad (1)$$

</div>

BERT-H768-L12（$H = 768$，$L = 12$）：

<div style="text-align:center">

$$30K \times 768 + 12 \times 768^2 \times 12 = 107{,}974{,}656 \approx 110\text{M} \approx 0.11\text{B} \qquad (2)$$

</div>

BERT-H1024-L24（$H = 1024$，$L = 24$）：

<div style="text-align:center">

$$30K \times 1024 + 12 \times 1024^2 \times 24 = 332{,}709{,}888 \approx 330\text{M} \approx 0.33\text{B} \qquad (3)$$

</div>

**训练数据**

* **输入**：序列，可以是一个句子，也可以是两个句子。
* **切词**：WordPiece

  > WordPiece 是 BERT、GPT 等主流预训练模型采用的**子词分词算法**，核心是将词汇拆分为更小的“子词单元”（Subword Unit），在词汇表大小和语义完整性之间取得平衡，同时限制词表大小，防止可学习参数都集中在嵌入层。

* **序列开头**：总是以 [CLS] 开头，可用于后续分类任务的输出。
* **句子间切分**：句子间加入 [SEP]，并加入可学习的 segment 向量来区分句子 A/B。

<div style="text-align:center; margin:1em 0;">
<img src="/images/paper_bert/image1.png" alt="BERT 输入表示示意图" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">BERT 输入表示示意图<br>Image source: <a href="https://arxiv.org/abs/1810.04805">https://arxiv.org/abs/1810.04805</a></p>
</div>

# 预训练与微调的区别

BERT 在预训练阶段使用 Masked Language Model（MLM）对部分 token 进行随机掩码，而微调阶段（如分类、问答、NER 等任务）完全没有掩码操作，这就存在预训练-微调的掩码不一致问题（也被称为“预训练-微调的分布偏移”）。

BERT 的设计者通过两大核心策略缓解了这一不一致性。

**不一致来源**

预训练和微调的掩码差异主要体现在**输入分布**和**任务目标**两个层面：

1. **输入分布差异**：
   * 预训练：输入中约 15% 的 token 被掩码（[MASK]）、替换为随机词或保留原词，输入包含大量特殊的 [MASK] 符号；
   * 微调：输入是自然文本（仅含 [CLS]、[SEP]、[PAD] 等特殊符号），无 [MASK]，模型面对的是“完整的 token 序列”。

2. **任务目标差异**：
   * 预训练：MLM 的目标是预测被掩码的 token，属于“局部语义恢复任务”；
   * 微调：目标是任务特定的预测（如分类的标签、问答的答案区间），属于“全局语义理解任务”。

**解决方法**

**策略 1：减少 [MASK] 的特殊性，让预训练输入更接近自然文本**

BERT 的 MLM 并非简单将 15% 的 token 替换为 [MASK]，而是采用三种替换策略（被称为“非纯掩码”），核心目的是让模型不依赖 [MASK] 符号本身，而是学习基于上下文的语义表征，从而让预训练的输入分布更贴近微调的自然文本。

具体的 15% 掩码规则：

1. 80% 的概率：将选中的 token 替换为 **[MASK]**；
2. 10% 的概率：将选中的 token 替换为随机的其他 token；
3. 10% 的概率：保留原 token。

**策略 2：NSP 任务增强全局表征，弱化 MLM 的局部性**

BERT 的预训练包含两个任务：MLM（局部语义）+ Next Sentence Prediction（NSP，下一句预测），其中 NSP 任务的设计直接弥补了 MLM 的局部性缺陷，让模型学习句子级/篇章级的全局语义，而这种全局表征正是微调任务（如分类、问答、句对匹配）所需要的。

NSP 任务的作用：

* 任务目标：判断两个句子是否为连续的上下文；
* 学习到的能力：模型需要通过 [CLS] token 整合整个句对的语义信息，从而做出判断，这种全局语义整合能力与微调任务的需求高度一致；
* 缓解掩码依赖：NSP 任务不涉及任何掩码操作，其训练目标让模型在预训练阶段就开始学习“无掩码输入的语义理解”，直接缩小了预训练和微调的任务差异。

# 下游任务微调

做法：加输出层 + softmax 变换。用第一个 [CLS] token 的输出做分类，也可以取对应词元的输出。

* GLUE：多类分类，直接使用 [CLS] 输出加分类层；
* SQuAD v1.1：QA 数据集，给定一段文档和问题，找出答案（答案包含在输入中）；通过定义两个向量，判断每个词元是否为答案的开头或结尾来实现。
