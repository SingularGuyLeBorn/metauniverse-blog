---
title: DeepSeek-V2 vs GPT-4 Comparison
date: 2024-05-20
---

# DeepSeek-V2: Strong, Economical, and Efficient

> [!TIP]
> This article is a reproduction of the core analysis from the DeepSeek-V2 paper, used to demonstrate the **Semantic Heatmap** and **Mermaid** capabilities of the MetaUniverse blog.

[[TOC]]

## 1. Introduction

DeepSeek-V2 is a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It introduces the **Multi-head Latent Attention (MLA)** mechanism which significantly reduces Key-Value (KV) cache usage during generation.

### Key Features
- **MLA**: Compression of KV cache.
- **DeepSeekMoE**: Fine-grained expert routing.
- **236B Parameters**: Only 21B active parameters per token.

## 2. Architecture Analysis

### 2.1 Multi-head Latent Attention (MLA)

The core innovation is the low-rank compression of KV heads.

$$
\mathbf{c}_{KV} = \mathbf{W}_{DKV} \mathbf{h}
$$

$$
[\mathbf{k}_{C, 1}, \cdots, \mathbf{k}_{C, H}] = \mathbf{W}_{UK} \mathbf{c}_{KV}
$$

This allows DeepSeek-V2 to handle 128K context length efficiently.

### 2.2 Model Architecture Diagram

```mermaid
graph TD
    Input[Input Embeddings] --> MLA[Multi-head Latent Attention]
    MLA --> MoE[DeepSeekMoE Layer]
    MoE --> Output[Output Projection]
    
    subgraph "MLA Mechanism"
    MLA_Compress[Low-Rank Compression] --> MLA_Decompress[Up Projection]
    end
    
    subgraph "DeepSeekMoE"
    SharedExp[Shared Experts]
    RoutedExp[Routed Experts (Fine-grained)]
    end
```

## 3. Performance Benchmark

| Model | MMLU | GSM8K | HumanEval | Active Params |
| :--- | :---: | :---: | :---: | :---: |
| Llama 3 70B | 79.5 | 85.0 | 81.7 | 70B |
| **DeepSeek-V2** | **78.5** | **92.2** | **81.1** | **21B** |
| Mixtral 8x22B | 77.8 | 87.6 | 78.6 | 39B |

DeepSeek-V2 achieves comparable performance to top-tier dense models while using significantly fewer active parameters.

## 4. Conclusion

DeepSeek-V2 represents a significant step forward in open-source LLMs, particularly in the balance between performance and inference cost.

See also: [[Transformer Architecture]] and [[PPO vs GRPO]] for more on efficient training.
