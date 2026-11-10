---
title: "Vision Transformer（ViT）论文笔记"
date: 2025-12-23
draft: false
weight: 500
tags: ["论文笔记"]
math: true
toc: true
summary: "ViT 论文（An Image is Worth 16x16 Words）要点总结：patch 化与位置编码、cls token 设计、归纳偏置与 CNN 的对比、关键实验结论与微调细节。。子标签：ViT、Transformer、视觉模型、归纳偏置。"
---

Vision Transformer（ViT）是 2020 年提出的突破性视觉模型，其核心创新是将 Transformer 的全局自注意力机制直接应用于图像任务：先将图像分割为固定尺寸的图像块（Patch），把每个块映射为一维嵌入向量并添加位置编码，再送入由多头自注意力和前馈网络组成的标准 Transformer 编码器，最后通过分类头完成视觉任务。

ViT 完全抛弃了 CNN 依赖的局部相关性、平移不变性等归纳偏置，转而通过自注意力无差别地捕获图像全局与局部的依赖关系。在大规模数据集（如 ImageNet-21K）上预训练后，它在图像分类、目标检测等任务上展现出超越 CNN 的性能；其优势是建模长距离依赖更灵活，劣势是对数据量和计算资源要求更高，且在小数据集上泛化能力不如 CNN。

> paper: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

# 论文关键信息

**先前结合 Transformer 相关的视觉工作**：

1. 图片输入 + 稀疏注意力
2. 轴向 self-attention：先在 H 方向做 self-attn，再在 W 方向做 self-attn
3. 使用 2×2 的 patch 将图片输入到 Transformer

（技术上就是 ViT，只是 ViT 使用更大的 patch 和更多的数据）

**论文结论**：

如果数据足够多，直接使用 Transformer（不需要 CNN），也能拥有很好的性能。

<div style="text-align:center; margin:1em 0;">
<img src="/images/paper_vit/image1.png" alt="ViT 算法流程图" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">ViT 算法流程图<br>Image source: <a href="https://arxiv.org/abs/2010.11929">https://arxiv.org/abs/2010.11929</a></p>
</div>

**关键要点**：

* 目标：在不改动 Transformer 架构的基础上适配视觉任务（结构可扩展复用）
* 2D 图片 patch 化
    * patch：16×16，224×224 → 14×14 = 196 个
    * patch embedding 是一个全连接层（`linear(197, 768)`）
    * 将 patch 化的维度映射到 Transformer 的输入维度
    * 也可以使用 CNN 让图片变成与 patch 化同样尺寸的特征图
* 有监督任务
    * NLP 通常是自监督训练，视觉任务上使用自监督，也可以获取很好的性能，如文章 MAE（**Masked Autoencoders**）
* 归纳偏置（inductive biases）
    * CNN 的归纳偏置特性决定了其不需要特别多的数据，就可以训练出比较好的模型；而 ViT 没有归纳偏置特性，需要很多的数据才能训练好
* 分类字符 cls token（类似 BERT 的做法）
    * 在 patch 化的基础上 cat 一个分类字符，它的位置编码 id = 0。由于这个 token 会和所有 token 做交互，所以它的 embedding 可以表示其他 token 的有用信息，进而只用它对应的输出作为分类特征
    * patch 14×14 + 1 = 197
    * 文章指出也可以使用 N 个输出做 GAP（global average pooling）得到全局特征，使用 cls token 还是为了保持和 Transformer 完全的一致
* pos embedding
    * 文章使用可学习的标准 1D 位置编码，并发现 2D 和 1D 的区别不大：因为 patch 化后每个 patch 表示一个维度，且 patch 的数目很少，所以使用哪一种类型的位置编码区别不大

**一些实验**

**cls token 有效性实验**：

<div style="text-align:center; margin:1em 0;">
<img src="/images/paper_vit/image2.png" alt="cls / GAP 有效性实验" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">cls / GAP 有效性实验<br>Image source: <a href="https://arxiv.org/abs/2010.11929">https://arxiv.org/abs/2010.11929</a></p>
</div>

结论：cls 和 GAP 都是有效的。

**数据规模的影响**

<div style="text-align:center; margin:1em 0;">
<img src="/images/paper_vit/image3.png" alt="数据规模对效果的影响" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">数据规模对效果的影响<br>Image source: <a href="https://arxiv.org/abs/2010.11929">https://arxiv.org/abs/2010.11929</a></p>
</div>

结论：小数据集上 ResNet 的效果更好，大数据集上 ViT 的效果更好。

**位置编码的有效性**

<div style="text-align:center; margin:1em 0;">
<img src="/images/paper_vit/image4.png" alt="位置编码有效性示意图" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">位置编码有效性示意图<br>Image source: <a href="https://arxiv.org/abs/2010.11929">https://arxiv.org/abs/2010.11929</a></p>
</div>

模型学到了位置信息。

**自注意力的有效性**

<div style="text-align:center; margin:1em 0;">
<img src="/images/paper_vit/image5.png" alt="自注意力有效性示意图" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">自注意力有效性示意图<br>Image source: <a href="https://arxiv.org/abs/2010.11929">https://arxiv.org/abs/2010.11929</a></p>
</div>

每一层上的点表示每一个 head，ViT-L/16 的 head dim = 16。y 轴表示平均注意力距离。即使在第 0 层也有很大的注意力距离，说明了全局注意力的有效性。

**微调关键问题**：

当输入图像尺寸发生变化时，位置编码应该如何处理？

预训练的位置编码和输入图像的位置编码不对应，文章中使用**插值**解决这个问题，但是可能会损失一些性能。

# CNN 归纳偏置

核心是抓住一个关键点：**CNN 通过架构设计，内置了对自然数据的先验假设，这些假设让模型在训练时无需从零学习所有规律，从而降低计算复杂度、提升泛化能力**。

**归纳偏置（Inductive Bias）的本质**

CNN 的归纳偏置由卷积层、池化层的设计逻辑决定，具体可拆解为三个核心假设。

**CNN 的三大核心归纳偏置**

1. **空间平移不变性（Spatial Translation Invariance）**

   * **定义**：如果一个特征在图像的某个位置出现（比如左上角的边缘），那么它在其他位置出现时（比如右下角），模型也应该用相同的方式去识别它。
   * **实现机制**：共享卷积核参数。

2. **局部空间相关性（Local Spatial Correlation）**

   * **定义**：图像中一个像素的特征，只与它周围局部区域的像素强相关，与远处的像素相关性较弱。
   * **实现机制**：局部感受野。

3. **空间下采样偏置（Spatial Downsampling Bias）**

   * **定义**：图像的高级语义特征（比如“猫的脸”）是低分辨率的，而低级特征（比如“边缘”“纹理”）是高分辨率的。模型可以通过降低特征图的分辨率，来聚合局部信息、提取更抽象的特征。
   * **实现机制**：池化层（Pooling Layer）或步幅大于 1 的卷积。
      * 最大池化（Max Pooling）：取局部感受野内的最大值，保留最显著的特征；
      * 平均池化（Average Pooling）：取局部感受野内的平均值，平滑特征。

**归纳偏置的“双刃剑”效应**

CNN 的归纳偏置是为自然图像量身定做的，但在其他数据上可能会成为限制：

1. **优势场景**：自然图像、视频等具有强局部相关性和平移不变性的数据。
2. **劣势场景**：不满足上述假设的数据。

   * 棋盘格图像：每个像素的相关性是全局的，局部感受野无法捕获全局规律。
   * 文本序列：文本的相关性是序列式的。

**扩展：CNN 归纳偏置的演变**

随着 CNN 的发展，研究者通过修改架构来**调整归纳偏置的强度**：

1. **空洞卷积**：扩大感受野，降低“局部相关性”的偏置强度。
2. **可变形卷积**：让卷积核的采样位置自适应于输入图像，打破“固定局部感受野”的偏置，适用于目标变形较大的场景（如姿态估计）。
3. **Vision Transformer**：完全抛弃 CNN 的归纳偏置，用自注意力机制捕获全局相关性，但需要大量数据才能训练出好的模型（因为没有先验假设，需要从零学习）。
