# CS336 Lecture 17: 鎵嬫妸鎵嬭瑙RPO (Hands-on Explanation of GRPO)

> **缂栬緫钃濆浘 (Editorial Blueprint)**
> 
> **鏍稿績涓婚**: 鏈搴ф槸CS336璇剧▼RL绯诲垪鐨勬敹瀹樹箣浣滐紝鎻愪緵浜?*GRPO绠楁硶鐨勫畬鏁翠唬鐮佸疄鐜?*銆備粠绛栫暐姊害鐨勬暟瀛︽帹瀵硷紝鍒板熀绾跨殑鐩磋鐞嗚В锛屽啀鍒板畬鏁寸殑璁粌寰幆锛岄厤鍚堜竴涓畝鍗曠殑鎺掑簭浠诲姟杩涜婕旂ず銆?
> 
> **鐭ヨ瘑缁撴瀯**: 
> - 绗竴閮ㄥ垎锛歊L鍦ㄨ瑷€妯″瀷涓殑璁惧畾锛堢姸鎬併€佸姩浣溿€佸鍔憋級
> - 绗簩閮ㄥ垎锛氱瓥鐣ユ搴︽暟瀛︽帹瀵?
> - 绗笁閮ㄥ垎锛氬熀绾夸笌浼樺娍鍑芥暟
> - 绗洓閮ㄥ垎锛氬畬鏁碐RPO瀹炵幇涓庡疄楠?
> 
> **绮捐嫳琛ュ厖绗旇**: 鏃狅紙鏈搴ф湰韬氨鏄疄鐜扮骇鍒殑娣卞害璁茶В锛?

---

## 涓€銆佽瑷€妯″瀷RL璁惧畾 (RL Setup for Language Models)

### 1.1 鍩烘湰瀹氫箟

| 姒傚康 | 璇█妯″瀷涓婁笅鏂?|
|------|---------------|
| **鐘舵€?(State)** $s$ | Prompt + 宸茬敓鎴愮殑response |
| **鍔ㄤ綔 (Action)** $a$ | 鐢熸垚涓嬩竴涓猼oken |
| **濂栧姳 (Reward)** $R$ | Response鐨勫ソ鍧忕▼搴?|
| **绛栫暐 (Policy)** $\pi$ | 璇█妯″瀷 $P_\theta(a|s)$ |
| **杞ㄨ抗 (Trajectory)** | $s \to a_1 \to a_2 \to ... \to R$ |

### 1.2 鏈绋嬭仛鐒︾殑璁惧畾

鎴戜滑鍏虫敞**缁撴灉濂栧姳 (Outcome Rewards)**锛?
- 濂栧姳鏄暣涓猺esponse鐨勫嚱鏁?
- 濂栧姳鏄?*鍙獙璇佺殑**锛堜笉鏄涔犵殑锛?

**绀轰緥**: 鏁板闂
```
Prompt: "2 + 3 * 4 = ?"
Response: "Let me think... 2 + 3 = 5, then 5 * 4 = 20... 
           Wait, order of operations! 3 * 4 = 12, then 2 + 12 = 14.
           Therefore, the answer is 14."
           
Reward Function: Extract "14", compare to ground truth 鈫?R = 1
```

### 1.3 璇█妯″瀷RL鐨勭壒娈婃€?

**杞Щ鍔ㄦ€佺‘瀹氭€?*: $T(s'|s,a) = \delta(s' = s + a)$

杩欐剰鍛崇潃锛?
- 鍙互杩涜**瑙勫垝/娴嬭瘯鏃惰绠?*锛堟満鍣ㄤ汉鍋氫笉鍒帮級
- 鐘舵€佹槸"铏氭瀯鐨?锛坱oken搴忓垪锛岃€岄潪鐗╃悊鐘舵€侊級
- 浠讳綍鐘舵€侀兘鍙揪锛堝彧瑕佸啓鍑簍okens锛?
- 鎸戞垬涓嶆槸"鍒拌揪"鐘舵€侊紝鑰屾槸"姝ｇ‘"鐘舵€?

### 1.4 鐩爣

鏈€澶у寲鏈熸湜濂栧姳锛?

$$J(\theta) = \mathbb{E}_{s \sim p(s), a \sim \pi_\theta(a|s)}[R(s, a)]$$

---

## 浜屻€佺瓥鐣ユ搴?(Policy Gradient)

### 2.1 绗﹀彿绠€鍖?

涓轰簡绠€鍖栵紝浠?$a$ 琛ㄧず**鏁翠釜response**锛堣€岄潪鍗曚釜token锛夈€?

鍦ㄧ粨鏋滃鍔辫瀹氫笅锛岃繖鏄悎鐞嗙殑鈥斺€斿彲浠ヨ涓轰竴娆℃€х敓鎴愭暣涓洖澶嶃€?

### 2.2 姊害鎺ㄥ

鏈熸湜濂栧姳锛?
$$J(\theta) = \int p(s) \pi_\theta(a|s) R(s,a) \, ds \, da$$

瀵?$\theta$ 姹傛搴︼細
$$\nabla J(\theta) = \int p(s) \nabla \pi_\theta(a|s) R(s,a) \, ds \, da$$

浣跨敤 $\nabla \log f = \frac{\nabla f}{f}$ 鍙樻崲锛?
$$\nabla J(\theta) = \int p(s) \pi_\theta(a|s) \nabla \log \pi_\theta(a|s) R(s,a) \, ds \, da$$

鍐欐垚鏈熸湜褰㈠紡锛?
$$\nabla J(\theta) = \mathbb{E}_{s,a}[\nabla \log \pi_\theta(a|s) \cdot R(s,a)]$$

### 2.3 鏈寸礌绛栫暐姊害

```python
def naive_policy_gradient(model, prompts, reward_fn, lr):
    """鏈寸礌绛栫暐姊害涓€姝ユ洿鏂?""
    # 1. 浠庡綋鍓嶇瓥鐣ラ噰鏍?
    responses = model.generate(prompts)
    
    # 2. 璁＄畻濂栧姳
    rewards = [reward_fn(p, r) for p, r in zip(prompts, responses)]
    
    # 3. 璁＄畻姊害骞舵洿鏂?
    # 鈭囄?鈫?E[R 路 鈭噇og 蟺_胃(a|s)]
    log_probs = model.log_prob(prompts, responses)
    loss = -(log_probs * torch.tensor(rewards)).mean()
    loss.backward()
    optimizer.step()
```

### 2.4 涓嶴FT鐨勫叧绯?

绛栫暐姊害 = **濂栧姳鍔犳潈鐨凷FT**

$$\nabla J \propto R \cdot \nabla \log \pi_\theta(y|x)$$

- 濡傛灉 $R > 0$锛氭彁楂樿response鐨勬鐜?
- 濡傛灉 $R < 0$锛氶檷浣庤response鐨勬鐜?
- 濡傛灉 $R = 0$锛氫笉鏇存柊

### 2.5 绋€鐤忓鍔遍棶棰?

鑰冭檻浜屽厓濂栧姳 $R \in \{0, 1\}$锛?

**闂**: 濡傛灉绛栫暐寰堝樊锛屽ぇ閮ㄥ垎response閮藉緱鍒?$R=0$
- 姊害澶у涓洪浂
- 鍑犱箮娌℃湁瀛︿範淇″彿
- 妯″瀷"鍗′綇"

**瀵规瘮RLHF**: 濂栧姳妯″瀷缁欏嚭杩炵画鍒嗘暟锛屼俊鍙锋洿涓板瘜

---

## 涓夈€佸熀绾夸笌鏂瑰樊鍑忓皯 (Baselines and Variance Reduction)

### 3.1 楂樻柟宸棶棰?

鐩磋渚嬪瓙锛?

| 鐘舵€?| 鍔ㄤ綔 | 濂栧姳 |
|------|------|------|
| s1 | a1 | 11 |
| s1 | a2 | 9 |
| s2 | a1 | 0 |
| s2 | a2 | 2 |

- 鏈€浼樼瓥鐣? s1鈫抋1, s2鈫抋2
- 浣?R(s1,a2)=9 > R(s2,a2)=2
- 鍗曠湅濂栧姳浼氳瀵硷紒

**鏍规湰闂**: 涓嶅悓鐘舵€佺殑濂栧姳灏哄害涓嶅悓

### 3.2 鍩虹嚎鐨勬暟瀛?

**鍏抽敭瀹氱悊**: 瀵逛簬浠绘剰鍙緷璧栫姸鎬佺殑鍑芥暟 $b(s)$锛?

$$\mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot b(s)] = 0$$

**璇佹槑**:
$$\mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot b(s)] = \int p(s) b(s) \left[\int \nabla \pi_\theta(a|s) da\right] ds$$

鑰?$\int \nabla \pi_\theta(a|s) da = \nabla \int \pi_\theta(a|s) da = \nabla 1 = 0$

**缁撹**: 鍙互鍑忓幓浠绘剰 $b(s)$ 鑰屼笉鏀瑰彉姊害鏈熸湜锛?
$$\nabla J = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot (R - b(s))]$$

### 3.3 鏂瑰樊鍑忓皯鏁堟灉

鍥炲埌渚嬪瓙锛岃 $b(s_1)=10, b(s_2)=1$锛?

| 鐘舵€?| 鍔ㄤ綔 | 鍘熷鍔?| 鍩虹嚎鍚庡鍔?|
|------|------|--------|-----------|
| s1 | a1 | 11 | +1 |
| s1 | a2 | 9 | -1 |
| s2 | a1 | 0 | -1 |
| s2 | a2 | 2 | +1 |

```python
import torch

# 鍘熷濂栧姳鏂瑰樊
raw_rewards = torch.tensor([11., 9., 0., 2.])
raw_variance = torch.var(raw_rewards)  # 绾?2.7

# 鍩虹嚎鍚庢柟宸?
baselined_rewards = torch.tensor([1., -1., -1., 1.])
baseline_variance = torch.var(baselined_rewards)  # 绾?.3

print(f"鏂瑰樊鍑忓皯: {raw_variance:.1f} 鈫?{baseline_variance:.1f}")
```

### 3.4 鏈€浼樺熀绾?

鐞嗚涓婃渶浼樼殑鍩虹嚎锛?
$$b^*(s) = \frac{\mathbb{E}[(\nabla \log \pi)^2 \cdot R | s]}{\mathbb{E}[(\nabla \log \pi)^2 | s]}$$

瀹為檯涓毦浠ヨ绠楋紝甯哥敤**鍚彂寮?*:
$$b(s) \approx \mathbb{E}[R|s] = V(s)$$

杩欏氨鏄?*浠峰€煎嚱鏁?*鐨勬潵婧愶紒

### 3.5 浼樺娍鍑芥暟 (Advantage Function)

瀹氫箟锛?
- **浠峰€煎嚱鏁?*: $V(s) = \mathbb{E}[R|s]$
- **Q鍑芥暟**: $Q(s,a) = \mathbb{E}[R|s,a]$锛堝湪缁撴灉濂栧姳涓?= R锛?
- **浼樺娍鍑芥暟**: $A(s,a) = Q(s,a) - V(s)$

**鐩磋**: 浼樺娍琛￠噺"鍔ㄤ綔 $a$ 姣斿钩鍧囨按骞冲ソ澶氬皯"

浣跨敤浼樺娍浣滀负鏇存柊鏉冮噸锛?
$$\nabla J = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot A(s,a)]$$

---

## 鍥涖€丟RPO瀹炵幇 (GRPO Implementation)

### 4.1 缁勭粨鏋?(Group Structure)

璇█妯″瀷鐨?*鐗规畩浼樺娍**锛氬鍚屼竴涓猵rompt鍙互鐢熸垚澶氫釜responses锛?

```python
prompts = ["What is 2+2?"]
responses_per_prompt = [
    ["4", "Let me calculate... 4", "It's 4", "The answer is 4"]
]
```

杩欎簺responses褰㈡垚涓€涓?*缁?*锛屽彲浠ョ敤缁勫唴鍧囧€间綔涓哄熀绾匡紒

### 4.2 GRPO浼樺娍璁＄畻

$$A_i = \frac{R_i - \text{mean}(R_1, ..., R_G)}{\text{std}(R_1, ..., R_G) + \epsilon}$$

```python
def compute_deltas(rewards: torch.Tensor, mode: str) -> torch.Tensor:
    """
    璁＄畻GRPO鐨刣elta锛堜紭鍔夸及璁★級
    
    Args:
        rewards: [batch, num_responses] 姣忎釜response鐨勫鍔?
        mode: "rewards" | "centered_rewards" | "normalized_rewards"
    
    Returns:
        deltas: [batch, num_responses] 鐢ㄤ簬鏇存柊鐨勬潈閲?
    """
    if mode == "rewards":
        # 鏈寸礌绛栫暐姊害
        return rewards
    
    if mode == "centered_rewards":
        # 鍑忓幓缁勫唴鍧囧€?
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        return rewards - mean_rewards
    
    if mode == "normalized_rewards":
        # 鍑忓幓鍧囧€硷紝闄や互鏍囧噯宸?
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        std_rewards = rewards.std(dim=-1, keepdim=True)
        centered = rewards - mean_rewards
        return centered / (std_rewards + 1e-5)
    
    raise ValueError(f"Unknown mode: {mode}")
```

### 4.3 绠€鍗曚换鍔★細鏁板瓧鎺掑簭

```python
def sort_inclusion_ordering_reward(prompt: list[int], response: list[int]) -> float:
    """
    璇勪及鎺掑簭response鐨勫鍔?
    
    缁欏垎瑙勫垯:
    1. 姣忎釜prompt涓殑鏁板瓧鍑虹幇鍦╮esponse涓?鈫?+1
    2. 姣忓鐩搁偦鏁板瓧鏄崌搴?鈫?+1
    """
    # 鍖呭惈濂栧姳
    inclusion_reward = sum(1 for x in prompt if x in response)
    
    # 鎺掑簭濂栧姳
    ordering_reward = sum(1 for i in range(len(response)-1) 
                          if response[i] <= response[i+1])
    
    return inclusion_reward + ordering_reward

# 绀轰緥
prompt = [3, 1, 0, 2]
correct_response = [0, 1, 2, 3]  # 濂栧姳 = 4(鍖呭惈) + 3(鎺掑簭) = 7
wrong_response = [7, 2, 2, 5]    # 濂栧姳 = 1(鍖呭惈) + 2(鎺掑簭) = 3
```

### 4.4 绠€鍗曟ā鍨?

```python
class Model(nn.Module):
    """绠€鍖栫殑闈炶嚜鍥炲綊鎺掑簭妯″瀷"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 prompt_length: int, response_length: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 姣忎釜浣嶇疆鏈夌嫭绔嬬殑缂栫爜/瑙ｇ爜鏉冮噸
        self.encode_weights = nn.Parameter(
            torch.randn(prompt_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim)
        )
        self.decode_weights = nn.Parameter(
            torch.randn(response_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim)
        )
    
    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompts: [batch, prompt_length]
        Returns:
            logits: [batch, response_length, vocab_size]
        """
        # 宓屽叆
        embeddings = self.embedding(prompts)  # [batch, pos, dim]
        
        # 缂栫爜锛氬prompt浣嶇疆鍔犳潈姹傚拰
        encoded = einsum(embeddings, self.encode_weights, 
                        "batch pos dim1, pos dim1 dim2 -> batch dim2")
        
        # 瑙ｇ爜锛氫负姣忎釜response浣嶇疆鐢熸垚鍚戦噺
        decoded = einsum(encoded, self.decode_weights,
                        "batch dim2, pos dim2 dim1 -> batch pos dim1")
        
        # 杞崲涓簂ogits锛堣緭鍏ヨ緭鍑哄叡浜玡mbedding锛?
        logits = einsum(decoded, self.embedding.weight,
                       "batch pos dim1, vocab dim1 -> batch pos vocab")
        
        return logits
```

### 4.5 鐢熸垚Responses

```python
def generate_responses(prompts: torch.Tensor, model: Model, 
                       num_responses: int) -> torch.Tensor:
    """
    涓烘瘡涓猵rompt鐢熸垚澶氫釜responses
    
    Returns:
        responses: [batch, num_responses, response_length]
    """
    logits = model(prompts)  # [batch, pos, vocab]
    
    # 閲囨牱
    batch_size = prompts.shape[0]
    flattened_logits = rearrange(logits, "batch pos vocab -> (batch pos) vocab")
    flattened_responses = torch.multinomial(
        F.softmax(flattened_logits, dim=-1), 
        num_samples=num_responses, 
        replacement=True
    )
    responses = rearrange(
        flattened_responses, 
        "(batch pos) trial -> batch trial pos", 
        batch=batch_size
    )
    
    return responses
```

### 4.6 璁＄畻Log姒傜巼

```python
def compute_log_probs(prompts: torch.Tensor, responses: torch.Tensor, 
                      model: Model) -> torch.Tensor:
    """
    璁＄畻responses鐨刲og姒傜巼
    
    Returns:
        log_probs: [batch, num_responses, response_length]
    """
    logits = model(prompts)  # [batch, pos, vocab]
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, pos, vocab]
    
    # 鎵╁睍浠ュ尮閰峳esponses缁村害
    num_responses = responses.shape[1]
    log_probs = repeat(log_probs, "batch pos vocab -> batch trial pos vocab", 
                       trial=num_responses)
    
    # 绱㈠紩鑾峰彇瀹為檯閫夋嫨鐨則oken鐨刲og姒傜巼
    log_probs = log_probs.gather(dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)
    
    return log_probs
```

### 4.7 鎹熷け璁＄畻

```python
def compute_loss(log_probs: torch.Tensor, deltas: torch.Tensor, 
                 mode: str, old_log_probs: torch.Tensor = None) -> torch.Tensor:
    """
    璁＄畻绛栫暐姊害鎹熷け
    
    Args:
        log_probs: [batch, trial, pos] 褰撳墠绛栫暐鐨刲og姒傜巼
        deltas: [batch, trial] 浼樺娍/濂栧姳
        mode: "naive" | "clipped"
        old_log_probs: [batch, trial, pos] 鏃х瓥鐣ョ殑log姒傜巼锛堢敤浜庤鍓級
    """
    if mode == "naive":
        # 鏈寸礌绛栫暐姊害: -E[未 路 log 蟺]
        loss = -einsum(log_probs, deltas, 
                       "batch trial pos, batch trial -> batch trial pos").mean()
        return loss
    
    if mode == "clipped":
        epsilon = 0.1
        # 璁＄畻姒傜巼姣?
        ratios = log_probs / old_log_probs  # 娉ㄦ剰锛氳繖閲屽簲璇ユ槸exp(log宸?
        
        unclipped = einsum(ratios, deltas, 
                          "batch trial pos, batch trial -> batch trial pos")
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        clipped = einsum(clipped_ratios, deltas,
                        "batch trial pos, batch trial -> batch trial pos")
        
        return -torch.minimum(unclipped, clipped).mean()
    
    raise ValueError(f"Unknown mode: {mode}")
```

### 4.8 KL鎯╃綒

```python
def compute_kl_penalty(log_probs: torch.Tensor, 
                       ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    璁＄畻KL鏁ｅ害鎯╃綒
    
    浣跨敤浣庢柟宸及璁?
    KL(p||q) = E_p[q/p - log(q/p) - 1]
    """
    # ref/current鐨勬鐜囨瘮
    ratio = torch.exp(ref_log_probs - log_probs)
    
    # 浣庢柟宸甂L浼拌
    kl = ratio - (ref_log_probs - log_probs) - 1
    
    return kl.sum(dim=-1).mean()
```

### 4.9 瀹屾暣璁粌寰幆

```python
def run_policy_gradient(num_epochs: int = 100,
                        num_steps_per_epoch: int = 10,
                        num_responses: int = 10,
                        deltas_mode: str = "centered_rewards",
                        kl_penalty: float = 0.0):
    """瀹屾暣鐨凣RPO璁粌寰幆"""
    
    # 鏁版嵁
    prompts = torch.tensor([[1, 0, 2], [3, 2, 4], [1, 2, 3]])
    vocab_size = prompts.max() + 1
    
    # 妯″瀷
    model = Model(vocab_size=vocab_size, embedding_dim=10,
                  prompt_length=3, response_length=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 鍙傝€冩ā鍨嬶紙鐢ㄤ簬KL鎯╃綒锛?
    ref_model = None
    
    for epoch in range(num_epochs):
        # 瀹氭湡鏇存柊鍙傝€冩ā鍨?
        if kl_penalty != 0 and epoch % 10 == 0:
            ref_model = copy.deepcopy(model)
            for p in ref_model.parameters():
                p.requires_grad = False
        
        # 鐢熸垚responses
        responses = generate_responses(prompts, model, num_responses)
        
        # 璁＄畻濂栧姳
        rewards = compute_reward(prompts, responses, sort_inclusion_ordering_reward)
        
        # 璁＄畻delta锛堜紭鍔匡級
        deltas = compute_deltas(rewards, mode=deltas_mode)
        
        # 淇濆瓨鏃х殑log姒傜巼锛堢敤浜庤鍓級
        with torch.no_grad():
            old_log_probs = compute_log_probs(prompts, responses, model)
        
        # 鍙傝€冩ā鍨媗og姒傜巼锛堢敤浜嶬L锛?
        if ref_model is not None:
            with torch.no_grad():
                ref_log_probs = compute_log_probs(prompts, responses, ref_model)
        
        # 鍐呭眰寰幆锛氬姝ユ洿鏂?
        for step in range(num_steps_per_epoch):
            # 褰撳墠log姒傜巼
            log_probs = compute_log_probs(prompts, responses, model)
            
            # 绛栫暐姊害鎹熷け
            loss = compute_loss(log_probs, deltas, mode="naive")
            
            # KL鎯╃綒
            if kl_penalty != 0:
                loss += kl_penalty * compute_kl_penalty(log_probs, ref_log_probs)
            
            # 鏇存柊
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 鎵撳嵃杩涘害
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: mean_reward = {rewards.mean():.2f}")
```

![GRPO绠楁硶浼唬鐮乚(./images/l17-grpo-algorithm.png)

---

## 浜斻€佸疄楠岃瀵?(Experimental Observations)

### 5.1 鏈寸礌濂栧姳 vs 涓績鍖栧鍔?

**鏈寸礌濂栧姳闂**:
- 濡傛灉鎵€鏈塺esponse鑾峰緱鐩稿悓濂栧姳锛堝閮芥槸3锛?
- 浠嶇劧浼氬仛鏇存柊锛堜笉搴旇锛侊級

**涓績鍖栧鍔辫В鍐虫柟妗?*:
- 鍑忓幓鍧囧€煎悗锛岀浉鍚屽鍔?鈫?delta = 0 鈫?涓嶆洿鏂?
- 鍙湪鏈夊樊寮傛椂鎵嶅涔?

```python
# 绀轰緥
rewards = torch.tensor([3., 3., 3., 3.])

# 鏈寸礌
deltas_naive = rewards  # [3, 3, 3, 3] 鈫?浼氭洿鏂帮紒

# 涓績鍖?
deltas_centered = rewards - rewards.mean()  # [0, 0, 0, 0] 鈫?涓嶆洿鏂?
```

### 5.2 鏍囧噯宸綊涓€鍖?

**鏁堟灉**: 浣垮緱鎵€鏈塸rompt鐨勬搴﹀昂搴︿竴鑷?

**娼滃湪闂** (Dr. GRPO璁烘枃):
- 绠€鍗?鍥伴毦闂鐨勬爣鍑嗗樊灏?鈫?姊害琚斁澶?
- 鍙兘瀵艰嚧鍦ㄧ畝鍗曢涓婃氮璐规洿鏂?

### 5.3 鎹熷け鏇茬嚎鐨勮瀵兼€?

**閲嶈璀﹀憡**: RL涓殑鎹熷け鏇茬嚎**涓嶅儚鐩戠潱瀛︿範閭ｆ牱鏈夋剰涔?*锛?

```
姣忎釜epoch:
- 鐢熸垚鏂扮殑responses
- 璁＄畻鏂扮殑濂栧姳鍜宒elta
- 鎹熷け鏄拡瀵?鏂?鏁版嵁璁＄畻鐨?

鉄?鎹熷け涓嶆槸鍦ㄥ悓涓€鍒嗗竷涓婅绠楃殑锛屼笉鑳界洿鎺ユ瘮杈?
```

**搴旇鐪?*: 骞冲潎濂栧姳锛堣€岄潪鎹熷け锛?

### 5.4 瀹為獙缁撹

浠庢帓搴忎换鍔＄殑瀹為獙锛?
1. **涓績鍖栨湁甯姪**: 閬垮厤鍦ㄦ棤淇″彿鏍锋湰涓婃洿鏂?
2. **鏍囧噯宸綊涓€鍖栨晥鏋滄湁闄?*: 鍦ㄨ繖涓缃笅宸紓涓嶅ぇ
3. **瀹规槗闄峰叆灞€閮ㄦ渶浼?*: 閮ㄥ垎姝ｇ‘浣嗕笉瀹屽叏姝ｇ‘
4. **閮ㄥ垎濂栧姳鏄弻鍒冨墤**: 缁欏お澶氶儴鍒嗗垎鍙兘闃荤杩涙

---

## 鍏€佸伐绋嬫敞鎰忎簨椤?(Engineering Considerations)

### 6.1 姊害璁＄畻鐨勯櫡闃?

**鍏抽敭**: 鍖哄垎"甯告暟"鍜?鍙橀噺"

```python
# 鉂?閿欒锛氬涓や釜閮芥眰姊害
w = torch.tensor(2., requires_grad=True)
p = torch.sigmoid(w)
p_old = torch.sigmoid(w)  # 杩欎篃鏈夋搴︼紒
ratio = p / p_old
ratio.backward()
print(w.grad)  # 0! 鍥犱负姊害鐩告秷

# 鉁?姝ｇ‘锛氬喕缁損_old
w = torch.tensor(2., requires_grad=True)
p = torch.sigmoid(w)
with torch.no_grad():
    p_old = torch.sigmoid(w)  # 浣滀负甯告暟
ratio = p / p_old
ratio.backward()
print(w.grad)  # 闈為浂
```

### 6.2 澶氭ā鍨嬬鐞?

GRPO闇€瑕佺鐞嗗涓ā鍨嬬姸鎬侊細

| 妯″瀷 | 鐢ㄩ€?| 鏇存柊棰戠巼 |
|------|------|----------|
| $\pi_\theta$ | 褰撳墠绛栫暐 | 姣忔鏇存柊 |
| $\pi_{old}$ | 閲嶈鎬ч噰鏍?| 姣廵poch鏇存柊 |
| $\pi_{ref}$ | KL姝ｅ垯鍖?| 姣廚 epoch鏇存柊 |

**鎶€宸?*: $\pi_{old}$ 涓嶉渶瑕佸瓨鍌ㄦā鍨嬶紝鍙渶瀛樺偍log_probs

### 6.3 绯荤粺澶嶆潅鎬?

鐪熷疄GRPO绯荤粺闇€瑕侊細
- **鎺ㄧ悊宸ヤ綔鑺傜偣**: 涓撻棬鍋氱敓鎴愶紙GPU瀵嗛泦锛?
- **璁粌宸ヤ綔鑺傜偣**: 涓撻棬鍋氭搴︽洿鏂帮紙GPU瀵嗛泦锛?
- **濂栧姳璁＄畻**: 鍙兘闇€瑕佹墽琛屼唬鐮併€佹煡鏁版嵁搴撶瓑
- **妯″瀷鏉冮噸鍚屾**: 璁粌鍚庡悓姝ュ埌鎺ㄧ悊鑺傜偣
- **闀緾oT澶勭悊**: 涓嶅潎鍖€batch鐨勮礋杞藉潎琛?

---

## 涓冦€佸叧閿鐐规€荤粨 (Key Takeaways)

### 鏍稿績鍏紡

$$\nabla J = \mathbb{E}\left[\nabla \log \pi_\theta(a|s) \cdot \underbrace{(R - b(s))}_{\text{鍩虹嚎鍚庡鍔眪}\right]$$

### GRPO鐗硅壊

1. **鏃犻渶浠峰€煎嚱鏁?*: 鐢ㄧ粍鍐呭潎鍊兼浛浠?
2. **鍒╃敤LM缁撴瀯**: 澶歳esponse閲囨牱鎻愪緵鍩虹嚎
3. **绠€鍗曟湁鏁?*: 宸茶R1銆並1.5绛夐獙璇?

### 瀹炶返寤鸿

- **鎬绘槸浣跨敤鍩虹嚎**: 鍑忓皯鏂瑰樊
- **鐩戞帶濂栧姳锛屼笉鍙槸鎹熷け**: 鎹熷け涓嶅彲姣?
- **娉ㄦ剰灞€閮ㄦ渶浼?*: RL瀹规槗闄峰叆
- **濂栧姳璁捐鏄叧閿?*: 姣旂畻娉曟洿閲嶈

### RL鐨勬湰璐?

> "If you can measure it, you can optimize it."
> 
> 濡傛灉浣犺兘琛￠噺瀹冿紝浣犲氨鑳戒紭鍖栧畠銆?

浣嗗叧閿槸锛?
1. 琛￠噺鏍囧噯鏄惁鍙潬锛燂紙RLHF鐨勬寫鎴橈級
2. 浼樺寲杩囩▼鏄惁绋冲畾锛燂紙鏂瑰樊鐨勬寫鎴橈級
3. 濂栧姳鏄惁鍙硾鍖栵紵锛堣繃鎷熷悎鐨勬寫鎴橈級

---

## 闄勫綍锛氬畬鏁翠唬鐮佸弬鑰?

瀹屾暣瀹炵幇璇峰弬鑰冭绋嬩唬鐮侊細
`spring2025-lectures/lecture_17.py`

鍖呭惈锛?
- 鎺掑簭浠诲姟瀹氫箟
- 绠€鍖栨ā鍨嬪疄鐜?
- GRPO璁粌寰幆
- 澶氱delta妯″紡瀵规瘮瀹為獙
- 鍙鍖栬缁冩洸绾?

---

## 鍙傝€冭祫鏂?

1. **Policy Gradient Theorem**: Sutton & Barto, Reinforcement Learning: An Introduction, Chapter 13
2. **GRPO**: Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning
3. **Dr. GRPO**: Liu et al. (2025). Understanding R1-Zero-Like Training: A Critical Perspective
4. **CS224R**: Stanford Deep Reinforcement Learning, Lecture Notes on Policy Gradients

