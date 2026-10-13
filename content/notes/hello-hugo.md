---
title: "用 Hugo + PaperMod 搭建笔记站"
date: 2026-10-13
draft: false
tags: ["hugo", "博客", "教程"]
categories: ["工具"]
math: true
toc: true
summary: "记录如何用 Hugo + PaperMod 主题 + GitHub Pages 搭建一个静态笔记博客，含数学公式、代码高亮、搜索、自动部署。"
---

这是一篇示例笔记，演示本站支持的常用功能。把它删掉，换成你自己的笔记即可。

## 1. 代码高亮

```python
import torch

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

print(softmax(torch.tensor([1.0, 2.0, 3.0])))
```

## 2. 数学公式

行内公式 $E = mc^2$，块级公式：

$$
\hat{\theta} = \arg\min_{\theta} \frac{1}{N}\sum_{i=1}^{N} \ell(f_\theta(x_i), y_i) + \lambda \|\theta\|_2^2
$$

## 3. 列表与引用

- 静态站点生成器：Hugo
- 主题：PaperMod
- 托管：GitHub Pages

> 笔记即 Markdown，Git 即数据库。

## 4. 表格

| 组件 | 作用 |
|---|---|
| `content/` | Markdown 正文 |
| `static/` | 图片、附件 |
| `themes/PaperMod/` | 主题（submodule） |
| `.github/workflows/` | 自动部署 |

## 5. 图片

把图片放到 `static/images/` 下，用 `![](/images/xxx.png)` 引用即可。

---

下一篇笔记直接在 `content/notes/` 下新建 `.md` 文件，`hugo server` 会自动热更新。
