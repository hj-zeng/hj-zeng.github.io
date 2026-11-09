---
title: "DFlash 2 技术报告"
date: 2026-11-03
draft: false
weight: 96
tags: ["投机解码", "DFlash", "LLM"]
math: true
toc: true
summary: "DFlash 2 技术报告解读：正确的 token 一直在 top-k 里、轻量级路径选择器（selector）、suffix decay 是局部问题与轻量级局部卷积、完整对比与发布信息。"
---

# DFlash 2：保持 Drafting 并行

> 原文：[DFlash 2 — Keep Drafting Parallel](https://inco.ai/blog/dflash2/)

Speculative decoding（投机解码）是现代推理栈的核心组件：一个轻量 draft 模型猜出一个 token 块，target 模型用一次 forward pass 验证整块。猜对了，一次 pass 就能产出多个 token；猜错了，直接丢弃。但多年来 draft 本身始终是**自回归**的——一次只出一个 token。DFlash 把它也做成了单次 pass：整块、每个位置**并行**预测。DFlash 2 在这个基础上进一步改进。

## 思路分析-1：正确的 token 一直在 top-k 里

DFlash 并行、独立地预测每个位置，单个预测单独看都合理，但没有任何机制让它们组合后保持一致——不协调的块在验证时会被截短。近期方法如 [Domino](https://arxiv.org/abs/2605.29707) 和 [DSpark](https://arxiv.org/abs/2607.05147) 用序列化 head 重写每个位置在整个词表上的分布，以此换取协调性。但这么昂贵的自回归修正真的必要吗？

以第一个位置为例：DFlash 的 top-1 命中率是 85.4%，但正确 token 落在 top 16 候选里的概率高达 99.5%——即使 top-1 猜错，正确 token 通常也还在候选列表里。

|指标|0|1|2|3|4|5|6|Acceptance length|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|Recall@1|85.4%|80.3%|79.4%|78.3%|77.5%|75.9%|72.9%|4.27|
|Recall@16|99.5%|97.3%|94.8%|92.6%|90.8%|89.4%|87.8%|6.79|

> 各 draft 位置的 Recall@1（top-1 命中率）和 Recall@16（正确 token 落在 top 16 的概率）



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_dflash2/image1.png &quot;selector 工作示意图 (Image source: https://inco.ai/blog/dflash2/" alt="selector 工作示意图" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">selector 工作示意图<br>Image source: <a href="https://inco.ai/blog/dflash2/">https://inco.ai/blog/dflash2/</a></p>
</div>

")

> 图示：一个解码周期内的 selector。只用 DFlash 时，每个位置保留自己的 top-1；这里两个相邻位置挑了同一个词，验证时卡住而终止。DFlash 2 在每个位置保留多个 top 候选，由 selector 在其中找出一条连贯路径；这里整个块都通过了验证。

## 处理方法-1：轻量级路径选择器

协调性本质上是局部的：一个候选是否合适，主要取决于它的前一个 token，所以给相邻对打分就够了。DFlash 2 在每个位置保留 top 16 候选，并对所有相邻对打分：对前驱 $a$ 和当前候选 $b$，

<div style="text-align:center">

$$S_t(a,b) = U_t(b) + \langle A(a) \odot H(h_t), B(b) \rangle.$$

</div>

打分由两部分组成。第一部分 $U_t(b)$ 是 DFlash 自己的 logit，代表 drafter 对 $b$ 的原始偏好；第二部分衡量 $b$ 接在 $a$ 之后是否自然：$A$、$B$ 为每个 token 各学习一个紧凑的 256 维 embedding，两者经上下文门 $H(h_t)$ 调制后做匹配，门控决定匹配中哪些维度起作用。本质上，这是对相邻候选做的一次低秩双线性注意力。

整个打分过程保持完全并行：每个位置的所有相邻对一次性打完分，无需额外的 backbone 或 LM-head pass。唯一需要串行的，是最后在预计算好的分数上走一遍——从上一个被验证的 token 出发，**贪心**地选择每步的最优后继；采样同样基于这些分数，并用拒绝采样恢复精确的 target 分布。

|方法|参数量|延迟|T = 0|T = 1|
|:---|:---|:---|:---|:---|
|DFlash|—|—|4.27|3.78|
|+ DSpark correction|+77.8M|+9.6%|4.49|4.08|
|+ path selection（本文）|+2.0M|+0.6%|**4.61**|**4.25**|

> 仅路径选择（不含卷积）带来的 acceptance length 指标，相比于 DSpark，成本更小、接受长度更长。

selector 在 $T=0$ 下为 DFlash 提升 **0.34** token，在 $T=1$ 下提升 **0.47** token，两个设定下都优于 DSpark correction，而参数量约少 40×、延迟开销约低 16×。**选择比预测便宜！**

## 思路分析-2：Suffix Decay 是局部问题

从表 1 可以看出，Recall@1 和 Recall@16 都在块尾出现下滑，Recall@16 准确率从第一个位置的 99.5% 掉到最后一个位置的 87.8%。任何 selector 都修不了这一点，因为候选本身就已枯竭。这种现象称为 **suffix decay**，它是 backbone 自身的问题。

一个可疑因素是容量：五层 backbone 可能太小，不足以承载跨块的依赖。如果这个猜想成立，加深模型应该对靠后位置帮助最大。事实正是如此——3 层、5 层、15 层 DFlash 在第一个位置几乎一样，越靠后差距越大。但加深深度是"一刀切"的：多加十个 attention block 给所有位置都加容量，包括本就提升空间有限的早期位置，同时把 DFlash 引以为傲的效率优势消磨掉大半。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_dflash2/image2.png &quot;不同参数量模型的接受率变化曲线 (Image source: https://inco.ai/blog/dflash2/" alt="不同参数量模型的接受率变化曲线" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">不同参数量模型的接受率变化曲线<br>Image source: <a href="https://inco.ai/blog/dflash2/">https://inco.ai/blog/dflash2/</a></p>
</div>

")

> 不同参数量 DFlash 的条件接受率变化曲线（Qwen3-4B，GSM8K，T=0），数据对应下表。

|Draft position|0|1|2|3|4|5|6|
|:---|:---|:---|:---|:---|:---|:---|:---|
|DFlash 3L|85.21%|79.26%|77.18%|75.75%|73.96%|70.4%|64.97%|
|DFlash 5L|85.39%|80.31%|79.39%|78.27%|77.39%|76.03%|72.86%|
|DFlash 15L（3× 参数量）|86.42%|81.61%|80.68%|80.34%|80.59%|79.66%|78.73%|
|DFlash 5L + conv（+3% 参数量）|85.83%|80.94%|79.98%|79.68%|79.73%|79.43%|77.61%|

DFlash 的 attention 已经指明了修复方向。它干两件事：读**块之前的上下文**，以及建模**块内的依赖**——但它在第二件事上花的注意力越来越少：块内 attention 占比从**第 1 层的 30% 掉到第 5 层的 8%**，剩下的还集中在越来越少的几个 head 上。于是我们把这两件事拆开：用一个专门的**模块承担块内工作**，让 attention 专心读上下文。



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_dflash2/image3.png &quot;各 head 的块内 attention (Image source: https://inco.ai/blog/dflash2/" alt="各 head 的块内 attention" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">各 head 的块内 attention<br>Image source: <a href="https://inco.ai/blog/dflash2/">https://inco.ai/blog/dflash2/</a></p>
</div>

")

> 五层 Qwen3-4B DFlash 各 head 的块内 attention。格子越亮表示该 head 在 draft 块上花的 attention 越多；在后面的层里，块内 attention 的量减少并集中到少数 head 上。

## 处理方法-2：轻量级局部卷积

块内工作本来就是短程的：一个块只跨 4 到 16 个 token，最紧的依赖在相邻位置之间。最自然的算子是短卷积：两个抽头（tap），一个看当前位置，一个回望上一个位置，权重随输入内容自适应。两抽头动态深度卷积：

<div style="text-align:center">

$$\operatorname{Conv}_k(x)_t = k_{t,0} \odot x_t + k_{t,1} \odot x_{t-1}.$$

</div>



<div style="text-align:center; margin:1em 0;">
<img src="/images/speculative_decoding_dflash2/image4.png &quot;两抽头动态卷积 (Image source: https://inco.ai/blog/dflash2/" alt="两抽头动态卷积" style="display:block; margin:0 auto; max-width:100%; height:auto;">
<p style="color:#888; font-size:0.92em; margin:0.25em 0 0;">两抽头动态卷积<br>Image source: <a href="https://inco.ai/blog/dflash2/">https://inco.ai/blog/dflash2/</a></p>
</div>

")

> 两抽头动态卷积。它位于 drafter 每层的每个 attention 和 MLP 子层前后。卷积内部，每个位置把自己的表示与前驱的表示混合，第一个位置读取上一个已验证 token。

卷积是块内局部且无状态的，所以它直接插进 DFlash，不改 attention、LM head 或验证。

卷积模块只增加 **16.5M 参数（3%）**，五层 DFlash + 卷积的效果就接近 15 层 DFlash，大幅缓解了 suffix decay。卷积只给 draft–verify 周期延迟增加 **0.7%**，而多加十个 Transformer 层要增加 15.2%。第 4、5 层的平均块内 attention 也从 **9.4% 掉到 0.5%**，与「卷积吸收了局部工作、attention 转回去读上下文」的预期一致。一个只回望一格的 kernel 就找回了多加十层的绝大部分收益：suffix decay 主要是**局部**问题。

## 总体结果

|Qwen3.5-4B|数据集|MTP|DFlash|DSpark|DFlash 2|
|:---|:---|:---|:---|:---|:---|
||GSM8K|4.78|4.99|5.69|**6.20**|
||MATH-500|5.04|5.42|6.20|**6.76**|
||HumanEval|4.84|5.43|5.80|**6.28**|
||MBPP|4.16|4.49|4.96|**5.41**|
||MT-Bench|3.90|4.26|4.77|**5.20**|
||平均|4.54|4.92|5.49|**5.97**|

> Qwen3.5-4B 每请求平均 acceptance length。采样：thinking 开启，temperature 1.0，top-p 0.95，top-k 20，presence penalty 1.5，无损 rejection sampling。

DFlash 2 在每个 benchmark 上都领先。跨 benchmark 取平均，它**比 DFlash 多 1.05 token（21%）**、**比 DSpark 多 0.48 token**。这次升级依然很便宜：selector 和卷积合计只给五层 DFlash 的 draft–verify 周期延迟增加 **1.3%**。
