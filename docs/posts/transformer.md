---
title: Transformer 架构深度解析
description: 深入理解 Transformer 模型的核心架构和数学原理
date: 2026-02-04
tags:
  - Transformer
  - Deep Learning
  - NLP
wikiLinks:
  - 注意力机制
  - 深度学习基础
  - GPT
  - BERT
---

# Transformer 架构深度解析

> 本文深入解析 Transformer 模型的核心架构，包括数学推导和代码实现。

![Transformer 架构图](/images/transformer_diagram.png)

## 1. 核心组件概览

Transformer 架构主要由以下核心组件构成：

| 组件 | 作用 | 复杂度 |
|------|------|--------|
| Self-Attention | 捕获序列内部依赖关系 | $O(n^2 \cdot d)$ |
| Multi-Head Attention | 并行学习不同表示子空间 | $O(n^2 \cdot d)$ |
| Feed-Forward Network | 非线性变换 | $O(n \cdot d^2)$ |
| Layer Normalization | 稳定训练过程 | $O(n \cdot d)$ |

## 2. 数学公式

### 2.1 缩放点积注意力 (Scaled Dot-Product Attention)

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$ 是 Query 矩阵
- $K \in \mathbb{R}^{m \times d_k}$ 是 Key 矩阵
- $V \in \mathbb{R}^{m \times d_v}$ 是 Value 矩阵
- $d_k$ 是 Key 的维度

### 2.2 多头注意力 (Multi-Head Attention)

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

其中每个注意力头：

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### 2.3 位置编码 (Positional Encoding)

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

## 3. 代码实现

### 3.1 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch, heads, seq_len, d_k]
            K: [batch, heads, seq_len, d_k]
            V: [batch, heads, seq_len, d_v]
            mask: [batch, 1, 1, seq_len] or None
        
        Returns:
            output: [batch, heads, seq_len, d_v]
            attention: [batch, heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数: [batch, heads, seq_q, seq_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 归一化
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 加权求和
        output = torch.matmul(attention, V)
        
        return output, attention


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        residual = query
        
        # 线性投影并分割为多头
        Q = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        output, attention = self.attention(Q, K, V, mask)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(output)
        
        # 残差连接 + LayerNorm
        output = self.norm(residual + self.dropout(output))
        
        return output, attention
```

### 3.2 使用示例

```python
# 初始化模型
mha = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.1)

# 创建输入
batch_size, seq_len = 32, 100
x = torch.randn(batch_size, seq_len, 512)

# 前向传播
output, attention_weights = mha(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attention_weights.shape}")
```

## 4. 关键洞见

::: tip 为什么需要缩放？
缩放因子 $\frac{1}{\sqrt{d_k}}$ 是为了防止当 $d_k$ 较大时，点积结果过大导致 softmax 梯度消失。
:::

::: warning 计算复杂度
Self-Attention 的 $O(n^2)$ 复杂度在处理长序列时可能成为瓶颈，这催生了 Linformer、Longformer 等高效注意力变体。
:::

## 5. 相关资源

- [[注意力机制]] - 注意力机制的演进历史
- [[GPT]] - 基于 Decoder 的语言模型
- [[BERT]] - 基于 Encoder 的预训练模型

---

*感谢阅读！如有疑问欢迎留言讨论。*
