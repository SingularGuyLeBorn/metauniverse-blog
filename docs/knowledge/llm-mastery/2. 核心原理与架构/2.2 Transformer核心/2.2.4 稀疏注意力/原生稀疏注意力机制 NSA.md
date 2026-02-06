---
title: 原生稀疏注意力机制 NSA
date: 2024-03-20
---

# 原生稀疏注意力机制 NSA

## 性能优化之道

针对 **原生稀疏注意力机制 NSA** 的极致优化，是降低部署成本的关键。

### 性能对比

| 方法 | Latency (ms) | VRAM (GB) | Throughput (token/s) |
|------|--------------|-----------|----------------------|
| Baseline | 120 | 24 | 150 |
| Optimized | 45 | 12 | 480 |

### CUDA Kernel 伪代码

```cpp
__global__ void 原生稀疏注意力机制_nsa_kernel(float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __expf(x[idx]);
    }
}
```

