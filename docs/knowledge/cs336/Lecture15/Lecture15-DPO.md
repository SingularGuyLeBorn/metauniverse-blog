# 娣卞叆鎺㈣: DPO鏁板鎺ㄥ

鏈枃鏄疞ecture 15鐨勭簿鑻辫ˉ鍏呯瑪璁帮紝瀹屾暣鎺ㄥDPO (Direct Preference Optimization) 鐨勬暟瀛﹀師鐞嗭紝瑙ｉ噴涓轰綍鍙互缁曡繃鏄惧紡濂栧姳妯″瀷鐩存帴浠庡亸濂芥暟鎹紭鍖栫瓥鐣ャ€?

---

## 涓€銆丷LHF鐨勬爣鍑嗙洰鏍?

### 1.1 鐩爣鍑芥暟

鍦≧LHF涓紝鎴戜滑甯屾湜鎵惧埌绛栫暐 $\pi_\theta$ 鏈€澶у寲:

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) \right] - \beta D_{KL}(\pi_\theta(\cdot|x) || \pi_{ref}(\cdot|x))$$

鍏朵腑:
- $r(x, y)$: 濂栧姳鍑芥暟锛堥€氬父浠庡亸濂芥暟鎹涔狅級
- $\pi_{ref}$: 鍙傝€冪瓥鐣ワ紙閫氬父鏄疭FT妯″瀷锛?
- $\beta$: KL鎯╃綒绯绘暟

### 1.2 Bradley-Terry鍋忓ソ妯″瀷

鍋囪浜虹被鍋忓ソ鐢辨綔鍦ㄥ鍔卞喅瀹?

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$$

鍏朵腑 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 鏄痵igmoid鍑芥暟銆?

### 1.3 鏍囧噯RLHF娴佺▼

1. 浠庡亸濂芥暟鎹缁冨鍔辨ā鍨?$r_\phi$
2. 浣跨敤PPO鏈€澶у寲 $J(\theta)$

**DPO鐨勯棶棰?*: 鑳藉惁璺宠繃姝ラ1锛岀洿鎺ヤ粠鍋忓ソ鏁版嵁璁粌绛栫暐锛?

---

## 浜屻€丏PO鏍稿績鎺ㄥ

### 2.1 闈炲弬鏁版渶浼樼瓥鐣?

**鍏抽敭娲炲療**: 瀵逛簬鍥哄畾鐨勫鍔?$r$锛屾垜浠彲浠ュ啓鍑烘渶浼樼瓥鐣ョ殑**瑙ｆ瀽褰㈠紡**銆?

灏嗙洰鏍囧嚱鏁板啓鎴愮Н鍒嗗舰寮?

$$J(\theta) = \int p(x) \left[ \int \pi_\theta(y|x) \left( r(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right) dy \right] dx$$

瀵逛簬姣忎釜 $x$锛屽唴閮ㄤ紭鍖栨槸鍏充簬鍒嗗竷 $\pi_\theta(\cdot|x)$ 鐨勪紭鍖栥€?

### 2.2 鍙樺垎鎺ㄥ

鍥哄畾 $x$锛屽 $\pi = \pi_\theta(\cdot|x)$ 姹傛瀬鍊?

$$\max_\pi \mathbb{E}_{y \sim \pi} \left[ r(x,y) - \beta \log \frac{\pi(y)}{\pi_{ref}(y|x)} \right]$$

鏀瑰啓涓?

$$\max_\pi \mathbb{E}_{y \sim \pi} \left[ r(x,y) \right] - \beta D_{KL}(\pi || \pi_{ref})$$

杩欐槸涓€涓?**甯L姝ｅ垯鍖栫殑鏈€澶у寲闂**銆?

### 2.3 鏈€浼樿В

閫氳繃鍙樺垎娉曟垨鎷夋牸鏈楁棩涔樺瓙娉曪紝鍙互璇佹槑鏈€浼樺垎甯冧负:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

鍏朵腑閰嶅垎鍑芥暟:

$$Z(x) = \sum_y \pi_{ref}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

### 2.4 璇佹槑

璁炬媺鏍兼湕鏃ラ噺:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}_{y \sim \pi}[r(x,y)] - \beta D_{KL}(\pi || \pi_{ref}) - \lambda \left( \sum_y \pi(y) - 1 \right)$$

瀵?$\pi(y)$ 姹傚骞朵护鍏朵负0:

$$\frac{\partial \mathcal{L}}{\partial \pi(y)} = r(x,y) - \beta \log \frac{\pi(y)}{\pi_{ref}(y|x)} - \beta - \lambda = 0$$

瑙ｅ緱:

$$\pi(y) = \pi_{ref}(y|x) \exp\left( \frac{r(x,y) - \lambda - \beta}{\beta} \right)$$

褰掍竴鍖栧悗寰楀埌涓婅堪鏈€浼樿В褰㈠紡銆?

---

## 涓夈€佷粠绛栫暐鍙嶆帹濂栧姳

### 3.1 鍏抽敭杞崲

浠庢渶浼樼瓥鐣ュ叕寮?$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left( \frac{r(x,y)}{\beta} \right)$

鍙栧鏁板苟鏁寸悊:

$$\log \pi^*(y|x) = \log \pi_{ref}(y|x) + \frac{r(x,y)}{\beta} - \log Z(x)$$

瑙ｅ嚭濂栧姳:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

### 3.2 濂栧姳鐨勯噸鍙傛暟鍖?

**鏍稿績鍙戠幇**: 濂栧姳鍙互鐢ㄧ瓥鐣ョ殑瀵规暟姣旂巼琛ㄧず锛堝姞涓婁竴涓彧渚濊禆浜?$x$ 鐨勯」锛夈€?

$$r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

瀵逛簬鏈€浼樼瓥鐣?$\pi = \pi^*$锛岃繖涓瓑寮忔垚绔嬨€?

---

## 鍥涖€佷唬鍏radley-Terry

### 4.1 鍋忓ソ姒傜巼

灏嗛噸鍙傛暟鍖栫殑濂栧姳浠ｅ叆Bradley-Terry妯″瀷:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$$

浠ｅ叆 $r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$:

$$P(y_1 \succ y_2 | x) = \sigma\left( \beta \log \frac{\pi(y_1|x)}{\pi_{ref}(y_1|x)} + \cancel{\beta \log Z(x)} - \beta \log \frac{\pi(y_2|x)}{\pi_{ref}(y_2|x)} - \cancel{\beta \log Z(x)} \right)$$

### 4.2 $Z(x)$鐩告秷

娉ㄦ剰 $\beta \log Z(x)$ 椤圭浉娑堬紒

$$P(y_1 \succ y_2 | x) = \sigma\left( \beta \log \frac{\pi(y_1|x)}{\pi_{ref}(y_1|x)} - \beta \log \frac{\pi(y_2|x)}{\pi_{ref}(y_2|x)} \right)$$

**杩欐剰鍛崇潃**: 鍋忓ソ姒傜巼瀹屽叏鐢辩瓥鐣ヤ笌鍙傝€冪瓥鐣ョ殑姣旂巼鍐冲畾锛屼笉闇€瑕侀厤鍒嗗嚱鏁帮紒

---

## 浜斻€丏PO鎹熷け鍑芥暟

### 5.1 鏈€澶т技鐒朵及璁?

缁欏畾鍋忓ソ鏁版嵁闆?$\mathcal{D} = \{(x, y_w, y_l)\}$锛?y_w$鏄亸濂界殑response锛夛紝鏈€澶у寲浼肩劧:

$$\max_\theta \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log P_\theta(y_w \succ y_l | x) \right]$$

### 5.2 DPO鎹熷け

浠ｅ叆鍋忓ソ姒傜巼鍏紡:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

### 5.3 绠€鍖栬鍙?

瀹氫箟闅愬紡濂栧姳:

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

鍒?

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E} \left[ \log \sigma(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)) \right]$$

**杩欎笌濂栧姳妯″瀷璁粌鐨勫舰寮忓畬鍏ㄧ浉鍚?*锛?

---

## 鍏€丏PO姊害鍒嗘瀽

### 6.1 姊害姹傚

瀵?$\mathcal{L}_{DPO}$ 姹傚叧浜?$\theta$ 鐨勬搴?

$$\nabla_\theta \mathcal{L}_{DPO} = -\mathbb{E} \left[ \sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)) \cdot \beta \cdot \left( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right) \right]$$

### 6.2 姊害瑙ｉ噴

$$\nabla_\theta \mathcal{L}_{DPO} \propto -\underbrace{w(\theta)}_{\text{鑷€傚簲鏉冮噸}} \cdot \left( \underbrace{\nabla \log \pi_\theta(y_w)}_{\text{鎻愰珮 } y_w} - \underbrace{\nabla \log \pi_\theta(y_l)}_{\text{闄嶄綆 } y_l} \right)$$

鍏朵腑 $w(\theta) = \sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))$銆?

**鏉冮噸鐨勫惈涔?*:
- 褰?$\hat{r}_\theta(y_l) > \hat{r}_\theta(y_w)$锛堟ā鍨嬭涓?$y_l$ 鏇村ソ锛夋椂锛?w$ 杈冨ぇ
- 姝ゆ椂姊害鏇村己锛屾洿绉瀬鍦扮籂姝ｉ敊璇垽鏂?
- 杩欐槸涓€绉?*闅愬紡鐨勫洶闅炬牱鏈寲鎺?*

### 6.3 涓庣瓥鐣ユ搴︾殑鑱旂郴

DPO姊害鍙互鐪嬩綔涓€绉?*瀵规瘮绛栫暐姊害**:
- 姝ｆ牱鏈?$y_w$ 鍚戜笂鎺?
- 璐熸牱鏈?$y_l$ 鍚戜笅鎺?
- 鏉冮噸鏍规嵁褰撳墠妯″瀷鐨勯敊璇▼搴﹀姩鎬佽皟鏁?

---

## 涓冦€丏PO vs RLHF瀵规瘮

### 7.1 娴佺▼瀵规瘮

| 姝ラ | RLHF | DPO |
|------|------|-----|
| 1. SFT | 鉁?| 鉁?|
| 2. 濂栧姳妯″瀷 | 璁粌鍗曠嫭鐨凴M | 鉂?|
| 3. 閲囨牱 | 闇€瑕佸湪绾块噰鏍?| 鉂?|
| 4. 浼樺寲 | PPO锛堝鏉傦級 | 鐩戠潱瀛︿範锛堢畝鍗曪級 |

### 7.2 鐞嗚绛変环鎬?

鍦ㄤ互涓嬪亣璁句笅锛孌PO涓嶳LHF绛変环:
1. Bradley-Terry鍋忓ソ妯″瀷鎴愮珛
2. 绛栫暐绫昏冻澶熺伒娲伙紙闈炲弬鏁板亣璁撅級
3. 閰嶅垎鍑芥暟鍙互琚惛鏀?

### 7.3 瀹為檯宸紓

| 鏂归潰 | RLHF | DPO |
|------|------|-----|
| 瀹炵幇澶嶆潅搴?| 楂?| 浣?|
| 鍦ㄧ嚎閲囨牱 | 闇€瑕?| 涓嶉渶瑕?|
| 鍐呭瓨闇€姹?| 楂橈紙澶氭ā鍨嬶級 | 浣?|
| 鎺㈢储鑳藉姏 | 鏈?| 鏃?|
| 绋冲畾鎬?| 闇€瑕佽皟鍙?| 杈冪ǔ瀹?|

---

## 鍏€丏PO鐨勫眬闄愪笌鍙樹綋

### 8.1 绂荤嚎鐨勫眬闄?

DPO鏄绾跨畻娉曪細
- 涓嶄粠褰撳墠绛栫暐閲囨牱
- 鍙兘鏃犳硶鍙戠幇鏂扮殑濂藉洖澶?
- 鍒嗗竷鍋忕Щ闂

### 8.2 IPO (Identity Preference Optimization)

瑙ｅ喅DPO鐨勮繃鎷熷悎闂:

$$\mathcal{L}_{IPO} = \mathbb{E}\left[ \left( \hat{r}_\theta(y_w) - \hat{r}_\theta(y_l) - \frac{1}{\beta} \right)^2 \right]$$

### 8.3 SimPO

鍘婚櫎鍙傝€冩ā鍨嬶紝娣诲姞闀垮害褰掍竴鍖?

$$\mathcal{L}_{SimPO} = -\log \sigma\left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l) - \gamma \right)$$

### 8.4 KTO (Kahneman-Tversky Optimization)

鍙渶瑕佸崟涓洖澶嶇殑濂?鍧忔爣绛撅紝涓嶉渶瑕佹垚瀵规瘮杈?

$$\mathcal{L}_{KTO} = \mathbb{E}_{y_w}[1 - \sigma(\hat{r}_\theta(y_w))] + \mathbb{E}_{y_l}[\sigma(\hat{r}_\theta(y_l))]$$

---

## 涔濄€佷唬鐮佸疄鐜?

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_logprobs_w, policy_logprobs_l, 
             ref_logprobs_w, ref_logprobs_l, beta=0.1):
    """
    璁＄畻DPO鎹熷け
    
    Args:
        policy_logprobs_w: [B] 褰撳墠绛栫暐瀵箇inner鐨刲og姒傜巼
        policy_logprobs_l: [B] 褰撳墠绛栫暐瀵筶oser鐨刲og姒傜巼
        ref_logprobs_w: [B] 鍙傝€冪瓥鐣ュwinner鐨刲og姒傜巼
        ref_logprobs_l: [B] 鍙傝€冪瓥鐣ュloser鐨刲og姒傜巼
        beta: KL姝ｅ垯鍖栫郴鏁?
    
    Returns:
        loss: 鏍囬噺
    """
    # 璁＄畻闅愬紡濂栧姳
    implicit_reward_w = beta * (policy_logprobs_w - ref_logprobs_w)
    implicit_reward_l = beta * (policy_logprobs_l - ref_logprobs_l)
    
    # DPO鎹熷け = -log(sigmoid(r_w - r_l))
    loss = -F.logsigmoid(implicit_reward_w - implicit_reward_l).mean()
    
    return loss

# 浣跨敤绀轰緥
# 鍋囪宸茶绠楀ソ鍚刲og姒傜巼
loss = dpo_loss(
    policy_logprobs_w=torch.tensor([-2.5, -3.0, -2.8]),
    policy_logprobs_l=torch.tensor([-3.5, -4.0, -3.8]),
    ref_logprobs_w=torch.tensor([-2.8, -3.2, -3.0]),
    ref_logprobs_l=torch.tensor([-3.2, -3.8, -3.5]),
    beta=0.1
)
```

---

## 鍙傝€冭祫鏂?

1. Rafailov, R. et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model
2. Azar, M. G. et al. (2023). A General Theoretical Paradigm to Understand Learning from Human Preferences
3. Ethayarajh, K. et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization
4. Meng, Y. et al. (2024). SimPO: Simple Preference Optimization with a Reference-Free Reward

