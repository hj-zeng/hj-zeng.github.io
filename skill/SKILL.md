---
name: 修复半成品文档
description: 把一份写得匆忙、半成品状态的笔记/技术文档打磨成规范稿。无论用户说"优化这份文档""帮我把这篇笔记整理一下""修一下这份草稿""这篇 md 有错别字和拗口的地方帮我改改""把流程画成流程图""公式标准化一下""补个 frontmatter"，只要意图是修订一篇已有的 Markdown 文档（尤其是 Hugo 笔记、技术笔记、算法/RL/工程类 md），都应触发本 skill。它按固定流程做四件事：改错别字与语法、标准化 LaTeX 公式、把描述性的流程/步骤文字转成 mermaid 流程图、润色拗口句子，并补齐与同目录文档一致的 frontmatter，最后用构建验证。
---

# 修复半成品文档

把一份匆忙写就的 Markdown 文档（常见于 Hugo 笔记站、技术笔记）打磨成规范、可读、可渲染的成稿。核心是四类修订 + 一步验证，**不改变作者的技术观点和信息，只提升表达质量和可渲染性**。

## 适用场景

- 文档里有错别字、漏字、中英文标点混用、语病；
- 公式写得随意（如 `s_t+1`、`theta_{t+1}`、缺括号的 `log(\pi_\theta(a_t|s_t)`、下标没写全）；
- 用纯文本/箭头描述了一个流程或步骤（如 `A ---> B ---> C`、"第一步…第二步…"），适合画成流程图；
- 句子拗口、啰嗦、信息重复；
- 缺少 frontmatter，或 frontmatter 与同目录其他文档不一致。

## 工作流程

按顺序执行。每一步都先理解作者原意，再动手。

### 0. 先摸清上下文（动手前必做）

1. 读完整篇目标文档，建立对结构、技术主题、作者意图的认知。
2. 读同目录（或同一内容仓库）里 **2–3 篇已成稿的文档**，提取本仓库的既有约定：
   - frontmatter 字段和写法（`title / date / draft / weight / tags / math / toc / summary` 等）；
   - 公式用什么分隔符（`$...$` 行内、`$$...$$` 块级）；
   - mermaid 流程图的画法（见下面"流程图规范"）。
3. 确认渲染链路：查 Hugo 主题的 `layouts/_partials/extend_head.html`，确认 mermaid 和数学公式（KaTeX/MathJax）是否已接入。**如果文档用了大量 LaTeX 但页面没有接入 KaTeX/MathJax，公式不会渲染** —— 要如实告诉用户，并说明这属于站点配置问题（可另行接入），不要默默改动 layout。

### 1. 改错别字与语法

- 修明显错字、漏字（例如"略梯度定理"→"策略梯度定理"、"各负号解释"→"各符号解释"）。
- 统一标点：正文用中文全角标点，**公式和代码内部用半角**（常见错误是把 `，` 混进 LaTeX 或 mermaid 里）。
- 修语病、主谓不搭、重复啰嗦，但**保持作者语气和信息量**，不要改写成另一种风格。

### 2. 标准化公式（LaTeX）

目标：凡是数学量，都用规范 LaTeX 表示，前后符号一致。

- 行内量用 `$...$`，独立成行的重要公式用 `$$...$$`；正文引用改成"公式 (n)"。
- **公式编号用 `\qquad (n)`，不要用 `\tag{n}`**。例：`$$A = Q - V \qquad (1)$$`。原因：`\tag` 在 KaTeX 下常把编号绝对定位到右侧、在窄容器里压到公式上（"公式和编号重合"），而 `\qquad (n)` 是纯数学语法、任何渲染器都不会重叠。多行 `aligned`/`cases` 块把 `\qquad (n)` 接在最后一行末尾即可。
- **块级公式居中**：每个 `$$...$$`（含多行块）用居中 `div` 包一层，确保在任何渲染器下都居中显示（本仓库 `config.yml` 已开 `goldmark.renderer.unsafe: true`，原始 HTML 可透传）。

  ```markdown
  <div style="text-align:center">

  $$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \qquad (1)$$

  </div>

  公式后面紧接着的正文……
  ```

  **四处空行一个都不能少**：`<div>` 后、`$$` 前、`$$` 后、`</div>` 前各留一个空行；尤其是 **`</div>` 之后、下一段正文之前必须空一行**。否则 goldmark 会把 `</div>` 之后的那段文字当成原始 HTML 块的一部分，导致该段里的行内公式 `$...$` 和 `**加粗**` 不被解析、直接显示成源码（常见症状："某行的 `$\delta_t=...$` 没显示成公式"）。
- 下标写全：`s0` → `s_0`，`s_t+1` → `s_{t+1}`，`a_t`、`\theta`、`\pi_\theta`。
- 括号配平：`log(\pi_\theta(a_t|s_t)` → `\nabla_\theta \log \pi_\theta(a_t \mid s_t)`。
- 函数名加反斜杠：`log` → `\log`；乘号用 `\cdot`；条件概率竖线用 `\mid`。
- 期望、集合等用标准写法：`E` → `\mathbb{E}`；修掉手滑的重复符号（如 `p(\tau;(;\theta))` → `p(\tau;\theta)`）。
- 把散落在正文里"用自然语言描述的数学量"也补成 LaTeX（用户常要求"有公式/符号的都表示一下"）。

### 3. 把流程/步骤转成 mermaid 流程图

**识别信号**：纯文本箭头（`A ---动作--> B ---> C`）、"第一步/第二步"式的顺序描述、带条件分支的过程、两个过程的对比（如"监督学习 vs 强化学习"）、架构数据流、求解流程、工程实现 pipeline。

把这些替换成 mermaid 代码块，**遵循本仓库既有画法**（务必先看样例文档再动手）：

- 用 ```` ```mermaid ````围栏，`flowchart TB`（竖）或 `flowchart LR`（横）。
- 对比/分组用 `subgraph ... end` + `direction TB`。
- 节点文字用双引号包裹，换行用 `<br/>`。节点内**不要用中文标点**；含 `|`、`()` 的文字务必放进双引号里，避免解析报错。
- 数学符号在节点里用 Unicode（`θ`、`α`、`∇`、`π_θ`、`τ`），不要在 mermaid 节点里写 `$...$`。
- 配色用 `classDef` + `class`，沿用仓库里的 Google 配色（见下）。

**仓库标准配色（classDef 调色板）**：

```
classDef blue   fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1a1a1a
classDef green  fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#1a1a1a
classDef yellow fill:#FFF8E1,stroke:#F9AB00,stroke-width:2px,color:#1a1a1a
classDef red    fill:#FCE4EC,stroke:#DB4437,stroke-width:2px,color:#1a1a1a
```

**对比型流程图示例**（两个并列过程）：

```mermaid
flowchart TB
    subgraph A["过程甲"]
        direction TB
        A1["步骤1"] --> A2["步骤2"] --> A3["步骤3"]
    end
    subgraph B["过程乙"]
        direction TB
        B1["步骤1"] --> B2["步骤2"]
    end
    classDef blue fill:#E8F0FE,stroke:#4285F4,stroke-width:2px,color:#1a1a1a
    classDef green fill:#E6F4EA,stroke:#34A853,stroke-width:2px,color:#1a1a1a
    class A1,A2,A3 blue
    class B1,B2 green
```

**数据流/闭环示例**（带边标签、回路）：

```mermaid
flowchart LR
    P["策略模型 π_θ"]
    E["环境"]
    P -- "动作 a" --> E
    E -- "状态、奖励" --> P
```

> 保留原流程里的信息（箭头上的标签、分支条件），只是换成图形表达；不要新增作者没写的步骤。

### 4. 润色拗口/繁琐的句子

- 长句拆短，去掉冗余的"的的""进行""通过…的方式"。
- 对比关系讲清楚（如"梯度上升 vs 梯度下降"的差异只在符号），但不堆砌排比和华丽辞藻。
- 术语前后统一，一个概念只用一个词。

### 5. 补齐/校正 frontmatter

若文档缺 frontmatter 或与同目录不一致，按仓库约定补上。用了公式就 `math: true`，用了流程图保持与同类文档一致的字段。`date` 用当天日期，`summary` 一句话概括全文。

### 6. 验证

- 如果仓库是 Hugo 站点，跑一次构建确认不报错：
  ```bash
  hugo --quiet --renderToMemory
  ```
- 复查：公式括号配平、公式编号用 `\qquad (n)` 而非 `\tag{n}`、块级公式用居中 `div` 包裹且 **`</div>` 后留空行**（否则下一段的行内 `$...$`/`**加粗**` 不渲染）、mermaid 能解析（节点标点、引号、`|`）、frontmatter YAML 合法、没有残留的编辑占位符。
- 向用户如实汇报做了哪几类改动；若发现站点未接入数学渲染等**超出文档本身**的问题，单独提示而不擅自改配置。

## 原则

- **只改表达，不改观点**：错字、公式、流程图、句子通顺度是修订对象；作者的技术判断和信息不动。
- **跟随仓库约定**，而不是套用通用模板——先读样例，后动手。
- **可渲染优先**：改完要能被 Hugo + mermaid（+ 数学渲染）正确渲染，这是"成稿"的硬标准。
