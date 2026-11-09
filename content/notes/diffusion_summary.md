---
title: "扩散模型总结"
date: 2026-11-06
draft: false
weight: 102
tags: ["扩散模型", "DDPM", "Score Matching", "Flow Matching", "CFG"]
math: true
toc: true
summary: "扩散模型总结：SDE 建模与三种预测模型（噪声 / score / 速度）、Diffusion 与 Flow Matching 的统一视角、离散与连续扩散模型及其类型与训练范式选择、条件扩散（分类器引导与 CFG）、加速采样（DPM-Solver）。"
---


本篇总结扩散模型的基本原理与方法，包含几个重要问题：

1. 扩散模型的原理与建模方式
2. diffusion（score matching）与 flow matching 的关系
3. 离散和连续的扩散模型建模方式
   1. 离散和连续的选择
   2. 不同建模类型的选择
   3. 不同训练范式的选择
4. 条件控制
5. 加速采样方法

## 扩散模型建模

扩散模型在生成阶段执行的是正向加噪扩散过程的逆过程：从高斯噪声逐步还原出数据。从概率演化的角度看，它属于流式生成模型的一种。

### 建模

扩散模型可以建模为（随机微分方程 **SDE**）：

<div style="text-align:center">

$$dx = f(x, t)\,dt + g(t)\,dw_t \qquad (1)$$

</div>

相关变量如下：

1. $x(t)$ 是 $t$ 时刻的样本，在扩散模型里：
   * $x(0)$：真实数据
   * $x(T)$：接近高斯噪声

   所以整个扩散过程就是：$x(0) \to x(t) \to x(T)$

2. $dw_t$ 表示布朗运动：$w_t$ 是标准布朗运动，$dw_t \sim \mathcal{N}(0, dt)$，表示每一个极小时间步 $dt$ 都会注入独立的随机高斯噪声，这正是"扩散"这个词的来源。

> 什么是布朗运动：布朗运动 = 连续时间的高斯随机游走

3. $g(t)\,dw_t$：扩散项。$g(t)$ 控制噪声强度，一般只依赖时间，不依赖 $x$；$t$ 越大，噪声越强，样本越"乱"。

4. $f(x, t)\,dt$：漂移项。这是**确定性部分**，表示在没有噪声时样本"应该往哪儿走"；在扩散模型的正向过程中，$f(x, t)$ 是人为设定的，目的是让数据分布逐渐趋向于简单分布（高斯）。

### 正向扩散

含义：从真实数据出发，在时间轴上连续地"加噪 + 收缩"，最终得到一个已知的噪声分布。这是固定的、不训练的纯数学过程。

### 逆时间 SDE

<div style="text-align:center">

$$dx = \left[f(x, t) - g(t)^2\nabla_x \log p_t(x)\right]dt + g(t)\,d\hat{w}_t \qquad (2)$$

</div>

这里出现了关键量 score function：$\nabla_x \log p_t(x)$，扩散模型学的就是这个 score function。

score function 表示：在当前噪声尺度 $t$ 下，$x$ 往"高概率区域"走的方向；你可以把它理解为"去噪方向"或者"密度上升最快的方向"。

### 从 SDE 到实际生成的流程

1. 从 $x_T \sim \mathcal{N}(0, I)$ 采样：逆过程从高斯噪声开始
2. 数值解逆 SDE
3. 用神经网络近似 score function
4. 一步步还原到 $x_0$

这就是熟悉的 DDPM（离散）或连续时间的 SDE 解法。

|连续时间 SDE|离散 DDPM|
|:---|:---|
|$dx = f(x, t)\,dt + g(t)\,dw_t$|$x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, \epsilon$|
|逆 SDE：$dx = \left[f(x, t) - g(t)^2\nabla_x \log p_t(x)\right]dt + g(t)\,d\hat{w}_t$|去噪步骤：$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - (1 - \alpha_t)s_\theta(x_t, t)\right) + \sigma_t z$|

扩散的本质是将**高密度分布**转换为某种**低密度分布（高斯分布）**。保持扩散过程中每个时刻的整体概率密度分布不变，从时间维度逆转上述运动过程，即可通过扩散过程的逆过程，将某种**低密度分布**转换为**高密度分布**。

由此可见，扩散模型是一种将扩散过程的逆过程作为其生成过程的流生成模型。

### 三种常见的预测模型

DDPM 可以看成是连续时间扩散 SDE: $dx = f(x, t)\,dt + g(t)\,dw_t$ 的一种时间离散化实现。

> 但是注意，DDPM 不是唯一的离散化方式，只是其中一种。这里以 DDPM 作为引子。

DDPM 的正向扩散可以表示为：

<div style="text-align:center">

$$x_t = \alpha_t x_0 + \sigma_t \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I) \qquad (3)$$

</div>

* **噪声模型（ε-model）**

  模型直接预测 $\epsilon_\theta(x_t, t) = \epsilon$，这是 DDPM 原始做法。

* **Score 模型**

  score 定义：$s_\theta(x_t, t) \approx \nabla_x \log p_t(x)$；对于高斯扰动，有解析关系 $\nabla_x \log p_t(x) = -\frac{1}{\sigma_t}\epsilon$。

  > 高斯扰动指的是：在数据上叠加一个服从正态分布的随机变量。

  所以可以推导得到：$s_\theta = -\frac{1}{\sigma_t}\epsilon_\theta$。

  所以 **ε-model ↔ score-model 完全等价**。

* **速度模型（v-model）**

  定义：$v = \alpha_t \epsilon - \sigma_t x_0$，可以反解：$\epsilon = \alpha_t v + \sigma_t x_t$，$x_0 = \alpha_t x_t - \sigma_t v$。

  可以看到预测 $v$ 等价于同时掌握 $x_0$ 和 $\epsilon$，因此 $v \leftrightarrow \epsilon \leftrightarrow$ score 是线性可逆变换的。

综上，由于每一个扩散模型都有一个确定的扩散过程与之对应，因此对于同一个数据集，不同建模方式的扩散模型的**数学本质是等价的**。

## Diffusion 与 Flow 的统一视角

从生成流的角度看，score-based diffusion 模型与 flow matching 模型可以视为同一概率流方程的两种训练方式：

* 前者通过 **score matching** 间接学习概率梯度
* 后者通过 **flow matching** 直接学习生成速度场

### Score Matching

Score matching 先学 score，再通过 reverse SDE 间接得到速度场。

> 注：DDPM（ε / $x_0$ / v）本质都在做 score matching

正向扩散：$x_t = x_0 + \sigma(t)\epsilon$

目标函数：

<div style="text-align:center">

$$L_{DSM} = \mathbb{E}_{x_0, t, \epsilon}\left[\left\|s_\theta(x_t, t) - \nabla_{x_t} \log p(x_t \mid x_0)\right\|^2\right] \qquad (4)$$

</div>

对于高斯扰动：

<div style="text-align:center">

$$\nabla_{x_t} \log p(x_t \mid x_0) = -\frac{x_t - x_0}{\sigma(t)^2} = -\frac{1}{\sigma(t)}\epsilon \qquad (5)$$

</div>

所以目标函数可以表示为：

<div style="text-align:center">

$$L_{DSM} = \mathbb{E}_{x_0, t, \epsilon}\left[\left\|s_\theta(x_t, t) + \frac{1}{\sigma(t)}\epsilon\right\|^2\right] \qquad (6)$$

</div>

### Flow Matching

定义一条路径：$x_t = \psi(x_0, z)$，其中 $x_0 \sim p_{data}$，$z \sim p_{noise}$。

根据路径定义目标速度场：$v^*(x_t, t) = \frac{d}{dt}\psi(x_0, z)$。

目标函数：

<div style="text-align:center">

$$L_{FM} = \mathbb{E}_{x_t, t}\left[\left\|v_\theta(x_t, t) - v^*(x_t, t)\right\|^2\right] \qquad (7)$$

</div>

特点：不涉及概率密度、不涉及 score，是一个纯回归问题；模型预测速度场，生成时直接解 ODE。

### 两者的关系

在扩散的 probability flow ODE 中有如下关系：

<div style="text-align:center">

$$\frac{dx}{dt} = f(x, t) - \frac{1}{2}g(t)^2\nabla_x \log p_t(x) \qquad (8)$$

</div>

如果用 score matching 预测得到 $\nabla_x \log p_t(x)$，那么根据上面的公式，也可以得到一个速度场。

对比前面的内容可以看到，这个公式也是推理（反向扩散）的建模方式：

<div style="text-align:center">

$$dx = v_\theta(x, t)\,dt = \left(f(x, t) - \frac{1}{2}g(t)^2\nabla_x \log p_t(x)\right)dt \qquad (9)$$

</div>

对上式注入噪声，就等价于最开始的建模方式了：

<div style="text-align:center">

$$dx = v_\theta(x, t)\,dt + g(t)\,dw'_t = \left(f(x, t) - \frac{1}{2}g(t)^2\nabla_x \log p_t(x)\right)dt + g(t)\,dw'_t \qquad (10)$$

</div>

关于为什么要注入噪声：

原因：如果扩散模型的建模和训练是理想的，则理论上使用任意一种生成路径所得到的采样分布将是一致且有效的。但在具体实践中，扩散模型的建模不可能是完美的，训练无法完全做到收敛和正确。在生成过程中注入噪声等价于**破坏前期生成路径中的相关信息**，从而后期必须在生成路径中使用**更强的来自得分函数的引导**。假如前期生成路径的得分函数训练不充分，而后期生成路径的得分函数训练充分，则在生成过程中添加一定的噪声可以有效地纠正生成路径。

## 离散和连续的扩散模型加噪过程

### 离散时间扩散

设定 $T$ 个时间点，在每个整数时间点上进行采样和训练，这样的扩散模型称为离散时间扩散模型。

假设扩散过程的终点为一个标准高斯分布，则扩散过程的逆过程可以描述为一种**概率分布转移**：

<div style="text-align:center">

$$p_\theta(x_{0:T}) = p(x_T)\prod_{i=1}^{T}p(x_{i-1} \mid x_i) \qquad (11)$$

</div>

扩散过程逆过程的概率分布转移可以建模为：

<div style="text-align:center">

$$p(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) \qquad (12)$$

</div>

扩散过程的前向过程则定义为 $q(x_{1:T} \mid x_0)$，它遵循某种设定的规则：

<div style="text-align:center">

$$q(x_{1:T} \mid x_0) = \prod_{i=1}^{T}q(x_i \mid x_{i-1}) \qquad (13)$$

</div>

在 DDPM 中，前向过程的概率分布转移 $q(x_t \mid x_{t-1})$ 用一组参数描述：$[\beta_0, \dots, \beta_T]$。

前向过程表示：

<div style="text-align:center">

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I) \qquad (14)$$

</div>

因此前向过程的概率分布 $q(x_t \mid x_0)$ 可以表示为：

<div style="text-align:center">

$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t)I) \qquad (15)$$

</div>

其中 $\bar{\alpha}_t = \prod_{i=0}^{t}(1-\beta_i)$。

另外 DDPM 的反向过程可以表示为（这里省略了推导过程）：

<div style="text-align:center">

$$p(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I) \qquad (16)$$

</div>

可以将其看作是连续反向（$dx = \left[f(x, t) - g(t)^2\nabla_x \log p_t(x)\right]dt + g(t)\,d\hat{w}_t$）的一种离散形式。

### 连续时间扩散

连续时间扩散模型一般将扩散过程设定在 $[0, 1]$ 连续区间，并让其训练和推理在整个连续区间内进行。

连续时间扩散模型 = 用连续时间 $t \in [0, 1]$ 描述"数据 → 噪声"的随机过程，并学习其反向生成过程。

连续时间扩散模型建模中，不同的建模方式，区别在于正向过程怎么定、网络学什么、反向怎么走。

正向扩散（Forward Process）：

<div style="text-align:center">

$$dx = f(x, t)\,dt + g(t)\,dw_t \qquad (17)$$

</div>

它决定了 $x_t = \alpha(t)x_0 + \sigma(t)\epsilon$；不同的模型有不同的 $f, g$，进而有不同的 $\alpha, \sigma$。

**常见的连续扩散模型**

**1. 方差守恒型（VP）**

<div style="text-align:center">

$$x_t = \alpha(t)x_0 + \sigma(t)\epsilon, \qquad \alpha^2(t) + \sigma^2(t) = 1 \qquad (18)$$

</div>

**特点**：数据方差不变、噪声逐步"替换"信号

**2. 方差爆炸型（VE）**

<div style="text-align:center">

$$x_t = x_0 + \sigma(t)\epsilon, \qquad \alpha(t) = 1 \qquad (19)$$

</div>

**特点**：信号不衰减、噪声随时间爆炸、score 学习更直接

**3. 线性扩散模型**

<div style="text-align:center">

$$x_t = (1-t)x_0 + tx_1, \qquad x_0 \sim p_{noise}, x_1 \sim p_{data} \qquad (20)$$

</div>

**4. 广义方差守恒**

<div style="text-align:center">

$$x_t = \cos\left(\frac{1}{2}\pi t\right)x_0 + \sin\left(\frac{1}{2}\pi t\right)\epsilon \qquad (21)$$

</div>

### 离散和连续类型的选择

理论上，对于连续时间扩散模型的训练方法是否比离散时间扩散模型的训练方法具有更好的性能，目前还没有基于严格证明的确切结论。但有论文揭示了扩散模型的时间区间的数学含义——本质上是扩散后**噪声幅值**的重参数化。因此，对于扩散模型而言，时间步 $t$ 就对应噪声 $\sigma_t$。

由于扩散过程的噪声传播本质上是连续的，客观上要求模型在各个噪声幅值下都能习得重建数据的方法，因此需要在连续的时间中优化扩散模型，并在连续的时间中采样生成。这样一来，诸如得分函数匹配等算法就可以在每一个时间步进行求解，而不是在每一个离散的时间步进行求解，从而使得模型的训练更加连续和平滑，这有利于提高训练后的模型质量。

### 不同建模类型的选择（方差守恒 vs 直线扩散）

根据扩散过程的定义 $x_t \mid x_{data} = \alpha_t x_{data} + \sigma_t \epsilon$ 可知，$\alpha_t$ 与 $\sigma_t$ 的数学形式决定了扩散过程的类型。相关论文统一了不同类型的扩散过程的数学形式：它揭示了所有类型的扩散过程都是同一个过程 $dx = \sqrt{2t}\,dw_t$ 在时间上的重参数化（$\sigma = t \to \sigma = \sigma(t)$）以及在数值上的缩放（$x_t \to \frac{x_t}{s(t)}$）。

虽然可以认为不同类型的扩散过程在本质上都是同一个过程，并不存在绝对数学意义上的优劣之分，但不同的参数化形式还是会在经验意义上**影响训练的稳定性和生成质量**。

相关论文统计：线性扩散过程和广义方差守恒优于方差守恒（VP-SDE）。

### 不同训练范式的选择（score vs flow）

扩散模型的表现除了受到扩散过程的数学形式的影响外，也会受到扩散模型训练方式的影响，具体来说，包括建模方式和模型训练目标。

定义好一种扩散过程之后，不同的建模对象之间存在着确定性的数学转换关系。因此，不同的建模类型虽然没有数学本质上的差异，但会在神经网络的建模数值稳定性和拟合精度上产生差别，毕竟任何神经网络的拟合能力都是有限的。

不同的建模和训练方式会对扩散模型的训练稳定性和生成质量产生影响。若建模噪声函数并采用得分函数匹配的训练方式，则无法计算 $t = 0$ 时刻的训练目标，因此需要额外的技巧来处理这个问题。通常可以采用一个缩放的区间，比如 $t \in [0.001, 1]$，来避免这个问题。

对于得分函数匹配，在不同的实践中可能需要使用关于时间 $t$ 的权重系数 $\lambda(t)$ 来侧重训练不同时间点的函数权重，通常而言这个系数默认设置为 1。此外，一些学者还发现，相比简单一阶得分函数匹配法的训练目标，使用更高阶的得分函数匹配法可以改善扩散模型的训练稳定性和生成质量。

## 条件扩散

### 分类器引导采样

分类器引导采样通过利用预先训练好的分类器来控制生成过程。

<div style="text-align:center">

$$\log p(x \mid c) = \log p(c \mid x) + \log p(x) - \log p(c) \qquad (22)$$

</div>

对两边求导并加权重可以得到：

<div style="text-align:center">

$$\nabla_x \log p(x \mid c) = \nabla_x \log p(c \mid x) + w\,\nabla_x \log p(x) \qquad (23)$$

</div>

其中 $w$ 控制条件的权重。

DDPM 中，每一步的分类器引导采样可以描述为：

<div style="text-align:center">

$$x_{t-1} \sim \mathcal{N}\left(\mu_\theta(x_t, t) + w\,\Sigma_\theta(x_t, t)\nabla_x \log p(x \mid c), \Sigma_\theta\right) \qquad (24)$$

</div>

这里对比原始的 DDPM，可以看做对均值做了一个偏移：

<div style="text-align:center">

$$\mu_\theta(x_t, t) \to \mu_\theta(x_t, t) + w\,\Sigma_\theta(x_t, t)\nabla_x \log p(x \mid c) \qquad (25)$$

</div>

分类器引导在采样时需要图像分类器作为一个判别模型，会忽略掉其中的大量细节，这种方法无法采样出难以用标签描述的结果；另外，理想的 $p(x \mid c)$ 需要能够在任意一个时间步描述 $x$ 属于每个类别的概率，这对于被噪声破坏过的图片 $x$，是难以训练的。

### 无分类器引导采样

无分类器引导采样（Classifier-Free Guidance，CFG）可以表示为：

<div style="text-align:center">

$$\log p(c \mid x) = \log p(x \mid c) + \log p(c) - \log p(x) \qquad (26)$$

</div>

这要求预先训练两个生成模型，分别为 $p(x \mid c)$ 和 $p(x)$。然而在实践中，我们可以采用共享同一个扩散模型参数的方法来拟合 $p(x \mid c)$ 和 $p(x)$。要做到这一点，只需要在训练过程中将条件 $c$ 随机置 NULL 即可。训练结束后，我们可以利用这个隐式的分类器得到条件 score function：

<div style="text-align:center">

$$\begin{aligned}
\nabla_x\left[\log p(x \mid c) + w\,\log p(c \mid x)\right]
&= \nabla_x\left[\log p(x \mid c) + w\left(\log p(x \mid c) - \log p(x)\right)\right] \\
&= \nabla_x\left[(1 + w)\log p(x \mid c) - w\,\log p(x)\right]
\end{aligned} \qquad (27)$$

</div>

对于 DDPM，CFG 预测的损失函数可以表示为：

<div style="text-align:center">

$$L(\theta) = (1 - p_{uncond})\mathbb{E}\left[\left\|\epsilon_\theta(x_t, t, c) - \epsilon\right\|^2\right] + p_{uncond}\,\mathbb{E}\left[\left\|\epsilon_\theta(x_t, t, \text{NULL}) - \epsilon\right\|^2\right] \qquad (28)$$

</div>

## 加速采样

training-free 加速采样方法独立于扩散模型的训练过程，即仅改变标准扩散模型的采样过程。DPM-Solver 通过将推断过程由随机微分 SDE 转换为常微分方程 ODE，并通过平衡 ODE 求解精度和采样速度，实现采样过程的加速（DDIM 就是一阶 DPM-Solver）。

### DPM-Solver

根据上面的分析，连续时间的扩散模型可以建模为 SDE 方程：

加噪过程：

<div style="text-align:center">

$$dx = f(x, t)\,dt + g(t)\,dw_t \qquad (29)$$

</div>

去噪过程：

<div style="text-align:center">

$$dx = \left[f(x, t) - g(t)^2\nabla_x \log p_t(x)\right]dt + g(t)\,d\hat{w}_t \qquad (30)$$

</div>

> DDPM 中的去噪函数 $\epsilon_\theta$ 实际上预测的就是 $-\sigma_t \nabla_x \log p_t(x)$

相关论文证明了，去噪过程和下式 ODE 是等价的：

<div style="text-align:center">

$$dx = f(x, t)\,dt - \frac{1}{2}g^2(t)\nabla_x \log p_t(x)\,dt \qquad (31)$$

</div>

采样两种方式都能采样出图片的真正分布，但是 ODE 相比 SDE 更容易求解，DPM-Solver 就在于求解这个 ODE。

代入 $\epsilon_\theta = -\sigma_t \nabla_x \log p_t(x)$，上式 ODE 的求解可以表示为：

<div style="text-align:center">

$$dx = f(x, t)\,dt + \frac{g^2(t)}{\sigma_t}\epsilon_\theta\,dt \qquad (32)$$

</div>

根据上式，对于给定结果 $x_{t_{i-1}}$，$x_{t_i}$ 的结果可以表示为（省略了求解过程）：

<div style="text-align:center">

$$x_{t_i} = \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}x_{t_{i-1}} - \alpha_{t_i}\int_{\lambda_{i-1}}^{\lambda_i} e^{-\lambda}\hat{\epsilon}_\theta(\hat{x}_\lambda, \lambda)\,d\lambda \qquad (33)$$

</div>

对 $\hat{\epsilon}_\theta(\hat{x}_\lambda, \lambda)$ 求近似解，进行泰勒展开，阶数越高，结果越准确。

说明：

1. 采样次数越多，两个时间戳越接近，结果越准确，同时速度越慢
2. $\lambda_t = \log \frac{\alpha_t}{\sigma_t}$，均匀划分 $\lambda_t$，比均匀划分 $[0, T]$ 可以带来较大的性能提升
