---
title: Hello World - MetaUniverse博客首篇
description: 欢迎来到MetaUniverse！这是一个展示八大特性系统的示例文章。
date: 2026-02-04
tags:
  - 入门
  - VitePress
wikiLinks:
  - Transformer架构
  - 注意力机制
  - 深度学习基础
---

# Hello World

> 发布日期: 2026-02-04

欢迎来到 **MetaUniverse** 博客！这是一篇展示八大特性系统的示例文章。

## 🎯 八大特性概览

本博客系统包含以下核心特性：

| 特性 | 说明 | 状态 |
|------|------|------|
| 深度链接 | URL状态持久化 | ✅ |
| 双模布局 | 扫读/实验模式 | ✅ |
| RAG搜索 | FlexSearch本地搜索 | ✅ |
| 知识图谱 | Wiki链接可视化 | ✅ |
| 热力图 | 阅读行为追踪 | ✅ |
| Git注解 | 段落级评论 | 📋 |
| 张量可视化 | 3D渲染 | 📋 |
| WASM沙箱 | 代码执行 | 📋 |

## 📖 双模布局演示

看到屏幕右下角的切换器了吗？点击可以在 **扫读模式** 和 **实验模式** 之间切换！

::: tip 扫读模式
适合快速浏览，只显示核心内容。
:::

::: warning 实验模式
显示详细推导、代码示例和深入解析。
:::

## 🔗 Wiki风格链接

本文与以下主题相关联（查看右侧知识图谱）：

- [[Transformer架构]] - 大模型的核心架构
- [[注意力机制]] - Self-Attention详解
- [[深度学习基础]] - 前置知识

## 🧮 数学公式

Attention机制的核心公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q$ = Query矩阵
- $K$ = Key矩阵  
- $V$ = Value矩阵
- $d_k$ = Key的维度

## 💻 代码示例

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """
    缩放点积注意力
    
    Args:
        Q: Query [batch, heads, seq_len, d_k]
        K: Key [batch, heads, seq_len, d_k]
        V: Value [batch, heads, seq_len, d_v]
    
    Returns:
        注意力输出和权重
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

## 🚀 下一步

- [ ] 探索更多文章
- [ ] 尝试切换阅读模式
- [ ] 体验本地搜索 (Ctrl/Cmd + K)
- [ ] 查看知识图谱

---

感谢阅读！欢迎在 GitHub 上关注这个项目。 ✨
