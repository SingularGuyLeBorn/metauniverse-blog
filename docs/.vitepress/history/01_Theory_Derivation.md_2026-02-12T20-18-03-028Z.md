# 第5章：近端策略优化 (Proximal Policy Optimization, PPO) [​](#第5章-近端策略优化-proximal-policy-optimization-ppo)

**论文信息**：

- **标题**：Proximal Policy Optimization Algorithms
- **作者**：John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov OpenAI
- **年份**：2017
- **arXiv**：1707.06347
- **PDF**：见 papers/ 目录

**前置知识**：策略梯度定理（第2章）、REINFORCE（第3章）

## 0. 本章目标 [​](#_0-本章目标)

PPO是**现代深度强化学习的标准算法**，被广泛应用于：

- OpenAI Five (Dota 2)
- ChatGPT训练 (RLHF)
- 机器人控制
- 各种LLM对齐任务

本章将：

- 解释REINFORCE/TRPO的局限性，以及PPO如何解决这些问题
- 详细推导**重要性采样 (Importance Sampling)** 的数学原理
- 逐步推导**PPO-Clip目标函数**
- 解释裁剪机制的直观含义和数学意义
- 介绍完整的PPO算法流程

## 1. REINFORCE和TRPO的问题 [​](#_1-reinforce和trpo的问题)

### 1.1 REINFORCE的问题回顾 [​](#_1-1-reinforce的问题回顾)

回忆第3章，REINFORCE的更新规则是：

$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) \cdot G_t$

修正后：
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) \cdot G_t$$

**公式符号详解**：

| 符号 | 含义 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| $\theta$ | 参数 | 向量 | 策略网络的参数 |
| $\alpha$ | 学习率 | 标量 | 控制更新步长 |
| $\nabla_\theta$ | 梯度 | 算子 | 计算损失对参数的偏导 |
| $\pi_\theta(a|s)$ | 策略概率 | 标量 | 在状态 $s$ 下选择动作 $a$ 的概率 |
| $G_t$ | 累计回报 | 标量 | $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ |

**问题1：On-Policy限制**

每次更新 $\theta$ 后，必须丢弃所有旧数据，重新采样。

**原因**：策略梯度公式中的期望 $\mathbb{E}_{\tau \sim \pi_\theta}[\ldots]$ 要求轨迹必须来自**当前策略** $\pi_\theta$。

**后果**：样本效率极低，需要大量环境交互。

**问题2：更新步长敏感**

- 步长太大 → 策略可能"跳"得太远，性能突然崩溃
- 步长太小 → 学习太慢

### 1.2 TRPO：一种解决方案 [​](#_1-2-trpo-一种解决方案)

**信任区域策略优化 (Trust Region Policy Optimization, TRPO)** 通过限制策略更新的幅度来保证单调改进。

TRPO的优化问题：

$$\max_\theta \quad \mathbb{E}_{s \sim \rho^{\pi_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\text{old}}}(s, a) \right]$$

$$\text{s.t.} \quad D_{KL}\left(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s)\right) \leq \delta$$

| 符号 | 含义 | 说明 |
| :--- | :--- | :--- |
| $\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ | 概率比率 | 衡量新旧策略差异 |
| $A^{\pi_{\text{old}}}(s, a)$ | 优势函数 | $A = Q(s,a) - V(s)$ |
| $D_{KL}$ | KL 散度 | 衡量两个概率分布的差异 |
| $\delta$ | 信任区域半径 | 控制更新幅度 |

**TRPO的问题**：

- 求解带约束的优化问题需要二阶导数（Hessian矩阵）
- 实现复杂（共轭梯度、线搜索等）
- 计算代价高

### 1.3 PPO的动机 [​](#_1-3-ppo的动机)

> **PPO的核心思想**：用简单的**裁剪 (Clipping)** 机制替代TRPO的KL约束，获得类似的效果但实现更简单。



## 2. 重要性采样 (Importance Sampling) [​](#_2-重要性采样-importance-sampling)

### 2.1 动机：复用旧数据 [​](#_2-1-动机-复用旧数据)

我们想用旧策略 πθold\pi_{\theta_{\text{old}}}πθold​​ 采集的数据来更新新策略 πθ\pi_\thetaπθ​。

**问题**：策略梯度公式要求 τ∼πθ\tau \sim \pi_\thetaτ∼πθ​，但我们只有 τ∼πθold\tau \sim \pi_{\theta_{\text{old}}}τ∼πθold​​。

**解决方案**：重要性采样。

### 2.2 重要性采样的数学原理 [​](#_2-2-重要性采样的数学原理)

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

| 符号 | 含义 | 说明 |
| :--- | :--- | :--- |
| $\mathbb{E}_{x \sim p}$ | 目标期望 | 我们想要计算的目标 |
| $\frac{p(x)}{q(x)}$ | 重要性权重 | 校正采样分布的偏差 |

**推导**：

Ex∼p[f(x)]=∫p(x)f(x)dx\mathbb{E}_{x \sim p}[f(x)] = \int p(x) f(x) dx Ex∼p​[f(x)]=∫p(x)f(x)dx

=∫q(x)⋅p(x)q(x)⋅f(x)dx= \int q(x) \cdot \frac{p(x)}{q(x)} \cdot f(x) dx =∫q(x)⋅q(x)p(x)​⋅f(x)dx

=Ex∼q[p(x)q(x)f(x)]= \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right] =Ex∼q​[q(x)p(x)​f(x)]

**关键**：p(x)q(x)\frac{p(x)}{q(x)}q(x)p(x)​ 称为**重要性权重 (Importance Weight)**。

$$J(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s, a)\right]$$

定义**概率比**：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

**公式符号详解**：

符号含义取值范围含义解释rt(θ)r_t(\theta)rt​(θ)时刻 ttt 的**概率比率**(0,+∞)(0, +\infty)(0,+∞)新旧策略选择该动作的概率之比πθ(at∣st)\pi_\theta(a_t|s_t)πθ​(at​∣st​)**新策略**选择动作 ata_tat​ 的概率(0,1)(0, 1)(0,1)-πθold(at∣st)\pi_{\theta_{\text{old}}}(a_t|s_t)πθold​​(at​∣st​)**旧策略**选择动作 ata_tat​ 的概率(0,1)(0, 1)(0,1)-rt=1r_t = 1rt​=1新旧策略**相同**-概率没有变化rt>1r_t > 1rt​>1新策略**更可能**选择该动作-概率增加了rt<1r_t < 1rt​<1新策略**更不可能**选择该动作-概率减少了rt→∞r_t \to \inftyrt​→∞新策略**极其偏好**该动作-危险！可能不稳定rt→0r_t \to 0rt​→0新策略**几乎不选**该动作-危险！可能不稳定则目标变为：

J(θ)=Et[rt(θ)At]J(\theta) = \mathbb{E}_t\left[r_t(\theta) A_t\right] J(θ)=Et​[rt​(θ)At​]

### 2.4 重要性权重的问题 [​](#_2-4-重要性权重的问题)

**问题**：当 πθ\pi_\thetaπθ​ 和 πθold\pi_{\theta_{\text{old}}}πθold​​ 差异很大时，rt(θ)r_t(\theta)rt​(θ) 可能变得非常大或非常小，导致：

- **方差爆炸**：梯度估计不稳定
- **策略崩溃**：一次大步更新可能使策略变得很差

**这就是PPO裁剪机制要解决的问题**。

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

**公式各项逐一详解**：

符号含义类型说明LCLIP(θ)L^{CLIP}(\theta)LCLIP(θ)**PPO裁剪目标函数**标量需要**最大化**（损失函数取负）Et\mathbb{E}_tEt​对所有**时间步 ttt** 的期望期望遍历采样的所有状态-动作对min⁡(⋅,⋅)\min(\cdot, \cdot)min(⋅,⋅)取两个值的**最小值**函数"悲观"估计，防止过度乐观rt(θ)r_t(\theta)rt​(θ)时刻 ttt 的**概率比率**标量rt=πθ(at∣st)πθold(at∣st)r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}rt​=πθold​​(at​∣st​)πθ​(at​∣st​)​AtA_tAt​时刻 ttt 的**优势函数**标量通常用GAE估计，At=Q(st,at)−V(st)A_t = Q(s_t, a_t) - V(s_t)At​=Q(st​,at​)−V(st​)clip(r,a,b)\text{clip}(r, a, b)clip(r,a,b)**裁剪函数**，将 rrr 限制在 [a,b][a, b][a,b] 范围函数clip(r,a,b)=max⁡(a,min⁡(b,r))\text{clip}(r, a, b) = \max(a, \min(b, r))clip(r,a,b)=max(a,min(b,r))1−ϵ1-\epsilon1−ϵ裁剪的**下界**标量当 ϵ=0.2\epsilon=0.2ϵ=0.2 时为0.81+ϵ1+\epsilon1+ϵ裁剪的**上界**标量当 ϵ=0.2\epsilon=0.2ϵ=0.2 时为1.2ϵ\epsilonϵ**裁剪超参数**标量通常 0.1∼0.20.1 \sim 0.20.1∼0.2，控制信任区域大小### 3.2 clip函数的定义 [​](#_3-2-clip函数的定义)

clip(r,1−ϵ,1+ϵ)={1−ϵif r<1−ϵrif 1−ϵ≤r≤1+ϵ1+ϵif r>1+ϵ\text{clip}(r, 1-\epsilon, 1+\epsilon) = \begin{cases} 1-\epsilon & \text{if } r < 1-\epsilon \\ r & \text{if } 1-\epsilon \leq r \leq 1+\epsilon \\ 1+\epsilon & \text{if } r > 1+\epsilon \end{cases} clip(r,1−ϵ,1+ϵ)=⎩⎪⎪⎨⎪⎪⎧​1−ϵr1+ϵ​if r<1−ϵif 1−ϵ≤r≤1+ϵif r>1+ϵ​

**公式符号详解**：

条件输出含义r<1−ϵr < 1-\epsilonr<1−ϵ1−ϵ1-\epsilon1−ϵ概率比过小，**截断到下界**1−ϵ≤r≤1+ϵ1-\epsilon \leq r \leq 1+\epsilon1−ϵ≤r≤1+ϵrrr概率比在安全范围，**保持不变**r>1+ϵr > 1+\epsilonr>1+ϵ1+ϵ1+\epsilon1+ϵ概率比过大，**截断到上界**### 图解：PPO裁剪目标函数 [​](#图解-ppo裁剪目标函数)

![PPO裁剪目标函数](/knowledge/rl-math-principle/05_PPO/images/clipping_objective.png)

**图片详细说明**：

此图展示了PPO裁剪目标函数 LCLIPL^{CLIP}LCLIP 如何随概率比 rrr 变化而变化。

**图片结构**：

- **横轴：Policy Ratio rrr（概率比）**

- 表示 rt(θ)=πθ(at∣st)πθold(at∣st)r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}rt​(θ)=πθold​​(at​∣st​)πθ​(at​∣st​)​
- r=1r = 1r=1 表示新旧策略相同
- r>1r > 1r>1 表示新策略更偏好该动作
- r<1r < 1r<1 表示新策略更不偏好该动作


- **纵轴：Objective Value（目标值）**

- 表示 LCLIPL^{CLIP}LCLIP 的值
- 正值表示"应该增加该动作的概率"
- 负值表示"应该减少该动作的概率"


- **蓝色实线：原始目标 r⋅Ar \cdot Ar⋅A**

- 未裁剪的目标函数
- 线性关系：rrr 越大，目标值越高（当 A>0A > 0A>0 时）
- **问题**：rrr 可以无限大，导致不稳定


- **红色线段（平坦区域）：裁剪后的区域**

- 当 r>1+ϵr > 1 + \epsilonr>1+ϵ 时（图中约1.2），目标值**不再增加**
- 当 r<1−ϵr < 1 - \epsilonr<1−ϵ 时（图中约0.8），目标值**不再减少**
- **效果**：梯度变为0，阻止进一步更新


- **灰色阴影区域 [1−ϵ,1+ϵ][1-\epsilon, 1+\epsilon][1−ϵ,1+ϵ]**

- "信任区域"或"安全区"
- 在此范围内，策略可以自由更新
- 超出此范围，梯度被"切断"


- **两种情况的对比**：

- **A>0A > 0A>0（好动作）**：我们想增大 rrr，但 r>1+ϵr > 1+\epsilonr>1+ϵ 时停止
- **A<0A < 0A<0（坏动作）**：我们想减小 rrr，但 r<1−ϵr < 1-\epsilonr<1−ϵ 时停止



**关键理解**：

- PPO通过裁剪限制了策略更新的幅度
- 保证新策略不会偏离旧策略太远
- 实现了与TRPO类似的"信任区域"效果，但计算更简单

### 3.3 为什么用min而不是直接clip？ [​](#_3-3-为什么用min而不是直接clip)

**关键洞察**：min⁡\minmin 的作用取决于 AtA_tAt​ 的符号。

**情况1：At>0A_t > 0At​>0（好动作）**

目标是**增大**动作的概率。

- 原始目标 r⋅Atr \cdot A_tr⋅At​ 希望 rrr 越大越好
- 但 clip(r,1−ϵ,1+ϵ)⋅At\text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot A_tclip(r,1−ϵ,1+ϵ)⋅At​ 限制了 rrr 最大只能是 1+ϵ1+\epsilon1+ϵ
- min⁡\minmin 取两者较小值 = 裁剪后的值
- **效果**：r>1+ϵr > 1+\epsilonr>1+ϵ 时梯度消失，防止过度增大概率

**情况2：At<0A_t < 0At​<0（坏动作）**

目标是**减小**动作的概率。

- 原始目标 r⋅Atr \cdot A_tr⋅At​ 希望 rrr 越小越好（因为 At<0A_t < 0At​<0）
- clip\text{clip}clip 限制了 rrr 最小只能是 1−ϵ1-\epsilon1−ϵ
- min⁡\minmin 在 At<0A_t < 0At​<0 时取到的是较大（负得更少）的那个 = 裁剪后的值
- **效果**：r<1−ϵr < 1-\epsilonr<1−ϵ 时梯度消失，防止过度减小概率

**总结**：无论 AtA_tAt​ 正负，min⁡\minmin 都起到"保守更新"的作用。

### 图解：PPO信任区域视角 [​](#图解-ppo信任区域视角)

![PPO信任区域](/knowledge/rl-math-principle/05_PPO/images/trust_region.png)

**图片详细说明**：

此图从"信任区域"角度解释PPO的工作原理。

**图片结构**：

- **蓝色曲线：旧策略 πθold\pi_{\theta_{\text{old}}}πθold​​**

- 表示采样数据时使用的策略
- 在动作空间上的概率分布
- 这是我们的"起点"


- **绿色曲线：新策略 πθ\pi_{\theta}πθ​**

- **表示经过优化后的策略**
- 我们希望它比旧策略更好
- 但不能偏离太远


- **阴影区域：信任区域**

- 表示允许的策略变化范围
- 由 DKL(πold∥πnew)≤δD_{KL}(\pi_{\text{old}} \| \pi_{\text{new}}) \leq \deltaDKL​(πold​∥πnew​)≤δ 或 r∈[1−ϵ,1+ϵ]r \in [1-\epsilon, 1+\epsilon]r∈[1−ϵ,1+ϵ] 定义
- 新策略必须落在这个区域内


- **虚线箭头：策略更新方向**

- 梯度希望把策略往某个方向更新
- 但被信任区域约束



**关键理解**：

- TRPO通过显式的KL约束 DKL≤δD_{KL} \leq \deltaDKL​≤δ 定义信任区域
- PPO通过裁剪 r∈[1−ϵ,1+ϵ]r \in [1-\epsilon, 1+\epsilon]r∈[1−ϵ,1+ϵ] 隐式实现类似效果
- 两者目标相同：防止策略剧烈变化导致性能崩溃

## 4. 完整的PPO算法 [​](#_4-完整的ppo算法)

### 4.1 PPO-Clip伪代码 [​](#_4-1-ppo-clip伪代码)


```
算法: PPO-Clip

初始化:
  - 策略网络 π_θ
  - 价值网络 V_φ
  - 设置 ε = 0.2, γ = 0.99, λ = 0.95

重复 N 次迭代:
  1. 采集数据:
     使用当前策略 π_θ_old 采集 T 步数据
     存储 (s_t, a_t, r_t, s_{t+1}, log π_θ_old(a_t|s_t))
     
  2. 计算优势:
     使用 GAE 计算 Â_t
     计算目标价值 V_target = Â_t + V(s_t)
     
  3. 优化:
     for k = 1, 2, ..., K 次迭代:
       # 策略损失
       r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
       L_clip = min(r_t Â_t, clip(r_t, 1-ε, 1+ε) Â_t)
       L_π = -mean(L_clip)
       
       # 价值损失
       L_V = mean((V_φ(s_t) - V_target)²)
       
       # 熵正则化（鼓励探索）
       L_entropy = mean(Entropy(π_θ(·|s_t)))
       
       # 总损失
       L = L_π + c_1 * L_V - c_2 * L_entropy
       
       # 梯度更新
       θ, φ ← θ, φ - α * ∇L
     
  4. 更新旧策略:
     θ_old ← θ
```
12345678910111213141516171819202122232425262728293031323334353637

### 4.2 关键超参数详解 [​](#_4-2-关键超参数详解)

参数符号典型值含义与作用裁剪系数ϵ\epsilonϵ0.1~0.2**控制策略更新范围，越小越保守**折扣因子γ\gammaγ0.99控制未来奖励的权重，越大越重视长期GAE参数λ\lambdaλ0.95偏差-方差权衡，越大方差越大但偏差越小优化轮数KKK3~10每批数据的复用次数，太多可能过拟合价值系数c1c_1c1​0.5价值损失的权重，平衡策略和价值学习熵系数c2c_2c2​0.01**熵正则化的权重，鼓励探索**## 5. 本章总结 [​](#_5-本章总结)

### 5.1 核心公式汇总 [​](#_5-1-核心公式汇总)

概念公式关键符号说明概率比rt(θ)=πθ(at∣st)πθold(at∣st)r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}rt​(θ)=πθold​​(at​∣st​)πθ​(at​∣st​)​πθ\pi_\thetaπθ​=新策略，πold\pi_{\text{old}}πold​=旧策略PPO-ClipL=E[min⁡(rtAt,clip(rt,1±ϵ)At)]L = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t)]L=E[min(rt​At​,clip(rt​,1±ϵ)At​)]ϵ\epsilonϵ=裁剪范围，AtA_tAt​=优势重要性采样Ep[f]=Eq[pqf]\mathbb{E}_{p}[f] = \mathbb{E}_{q}[\frac{p}{q} f]Ep​[f]=Eq​[qp​f]pq\frac{p}{q}qp​=重要性权重### 5.2 PPO的贡献 [​](#_5-2-ppo的贡献)

- **简单高效**：只需一阶梯度，无需复杂约束求解
- **稳定可靠**：裁剪机制保证策略不会剧烈变化
- **样本高效**：可以多次复用同一批数据
- **广泛适用**：从游戏到机器人到LLM

## 6. 开源实现参考 [​](#_6-开源实现参考)

### 6.1 官方实现 [​](#_6-1-官方实现)

- **OpenAI Baselines**: [https://github.com/openai/baselines](https://github.com/openai/baselines)
- **Spinning Up**: [https://spinningup.openai.com/](https://spinningup.openai.com/)

### 6.2 常用库 [​](#_6-2-常用库)

- **Stable-Baselines3**: [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- **TRL (Transformer RL)**: [https://github.com/huggingface/trl](https://github.com/huggingface/trl)
- **CleanRL**: [https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)

**下一章预告**：[第6章：RLHF与人类反馈对齐](./../06_RLHF/01_Theory_Derivation)