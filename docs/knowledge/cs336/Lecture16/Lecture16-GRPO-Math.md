# 娣卞叆鎺㈣: GRPO鏁板缁嗚妭

鏈枃鏄疞ecture 16鐨勭簿鑻辫ˉ鍏呯瑪璁帮紝鍒嗘瀽GRPO绠楁硶鐨勬暟瀛︾粏鑺傦紝鍖呮嫭鏍囧噯宸綊涓€鍖栫殑娼滃湪闂鍜孌r. GRPO璁烘枃鐨勬敼杩涘缓璁€?

---

## 涓€銆丟RPO鍥為【

### 1.1 鏍稿績鍏紡

GRPO鐨勪紭鍔夸及璁?

$$A_i = \frac{R_i - \text{mean}(R_1, ..., R_G)}{\text{std}(R_1, ..., R_G) + \epsilon}$$

鍏朵腑 $G$ 鏄瘡涓猵rompt鐢熸垚鐨剅esponse鏁伴噺銆?

### 1.2 涓嶱PO鐨勫叧绯?

| 缁勪欢 | PPO | GRPO |
|------|-----|------|
| 鍩虹嚎 | 瀛︿範鐨刅(s) | 缁勫唴鍧囧€?|
| 鏍囧噯鍖?| GAE澶勭悊 | 缁勫唴std褰掍竴鍖?|
| 浠峰€煎嚱鏁?| 闇€瑕佽缁?| 涓嶉渶瑕?|

---

## 浜屻€佹爣鍑嗗樊褰掍竴鍖栫殑闂

### 2.1 绛栫暐姊害瀹氱悊鍥為【

鏍囧噯鐨勭瓥鐣ユ搴﹀畾鐞嗗厑璁告垜浠?*鍑忓幓**浠绘剰鍙緷璧栫姸鎬佺殑鍩虹嚎:

$$\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot (R - b(s))]$$

**鍏抽敭**: 鍙湁**鍑忔硶**鏄悊璁轰繚璇佺殑锛?*闄ゆ硶**涓嶅湪瀹氱悊鑼冨洿鍐呫€?

### 2.2 闄ゆ硶甯︽潵鐨勯棶棰?

璁句袱涓猵rompt:
- Prompt A锛氱畝鍗曪紝鎵€鏈塺esponse閮芥帴杩戞纭紝$\text{std}(R) \approx 0.01$
- Prompt B锛氬洶闅撅紝response璐ㄩ噺宸紓澶э紝$\text{std}(R) \approx 1.0$

褰掍竴鍖栧悗:
- Prompt A鐨勬搴﹁**鏀惧ぇ100鍊?*
- Prompt B鐨勬搴︽甯?

**缁撴灉**: 绠€鍗曢棶棰樿幏寰楄繃澶х殑姊害鏉冮噸锛?

### 2.3 鏁板鍒嗘瀽

璁?$\delta = R - \bar{R}$锛孏RPO浣跨敤 $\tilde{\delta} = \delta / (\sigma + \epsilon)$銆?

**闂1**: $\sigma$ 鎺ヨ繎0鏃?
$$\tilde{\delta} \approx \frac{\delta}{\epsilon} \quad \text{(琚珆\epsilon\text{闄愬埗锛屼絾浠嶅彲鑳藉緢澶?}$$

**闂2**: 涓嶆弧瓒崇瓥鐣ユ搴﹀畾鐞?
$$\mathbb{E}[\nabla \log \pi \cdot \tilde{\delta}] \neq \mathbb{E}[\nabla \log \pi \cdot \delta] / \text{constant}$$

鍥犱负 $\sigma$ 鏈韩渚濊禆浜庨噰鏍风殑actions銆?

### 2.4 瀹為獙璇佹嵁

Dr. GRPO璁烘枃鐨勫疄楠?

```
璁剧疆: 绠€鍗曟帓搴忎换鍔?
姣旇緝: 
  - GRPO (甯td褰掍竴鍖?
  - GRPO-centered (鍙噺鍧囧€硷紝涓嶉櫎std)
  - GRPO-raw (鐩存帴浣跨敤reward)

缁撴灉:
  - GRPO-centered 鏀舵暃鏇寸ǔ瀹?
  - GRPO 鍦ㄦ煇浜涙儏鍐典笅鎸崱
  - 瀵逛簬浜屽厓reward (0/1)锛屽樊鍒渶鏄庢樉
```

---

## 涓夈€侀暱搴﹀綊涓€鍖栫殑闂

### 3.1 鍘熷GRPO鐨勯暱搴﹀綊涓€鍖?

DeepSeep Math璁烘枃鐨凣RPO瀵筶oss杩涜闀垮害褰掍竴鍖?

$$L = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi(y_t | y_{<t}, x) \cdot A$$

### 3.2 闂鍒嗘瀽

**鍦烘櫙1**: 绛旈敊鏃?($A < 0$)
- 鏇撮暱鐨剅esponse 鈫?姣弔oken璐熸搴︽洿灏?
- 鏈€浼樼瓥鐣? 鐢熸垚**灏藉彲鑳介暱**鐨勯敊璇洖绛旓紙绋€閲婃儵缃氾級

**鍦烘櫙2**: 绛斿鏃?($A > 0$)
- 鏇撮暱鐨剅esponse 鈫?姣弔oken姝ｆ搴︽洿灏?
- 鏈€浼樼瓥鐣? 鐢熸垚**灏藉彲鑳界煭**鐨勬纭洖绛旓紙闆嗕腑濂栧姳锛?

**缁撴灉**:
- 妯″瀷瀛︿細闀跨瘒搴熻瘽锛堥敊璇椂锛?
- 妯″瀷瀛︿細绠€鐭洖绛旓紙姝ｇ‘鏃讹級
- 涓庢湡鏈涜涓虹浉鍙?

### 3.3 瀹為獙璇佹嵁

R1-Zero瀹為獙涓瀵熷埌:
- 鎬濈淮閾鹃暱搴︽寔缁闀?
- 鍙兘涓嶆槸"娣卞害鎬濊€?锛岃€屾槸闀垮害鍋忓樊

---

## 鍥涖€丏r. GRPO鐨勬敼杩?

### 4.1 寤鸿1: 绉婚櫎鏍囧噯宸綊涓€鍖?

鍙娇鐢╟entered rewards:

$$A_i = R_i - \text{mean}(R_1, ..., R_G)$$

**浼樼偣**:
- 婊¤冻绛栫暐姊害瀹氱悊
- 閬垮厤绠€鍗昿rompt鐨勬搴︾垎鐐?
- 瀹炵幇鏇寸畝鍗?

### 4.2 寤鸿2: 绉婚櫎闀垮害褰掍竴鍖?

鐩存帴浣跨敤搴忓垪绾у埆鐨刲oss:

$$L = -\sum_{t=1}^{|y|} \log \pi(y_t | y_{<t}, x) \cdot A$$

鎴栬€呬娇鐢╰oken绾у埆浣嗕笉褰掍竴鍖?

$$L = -\sum_{t=1}^{|y|} \log \pi(y_t | y_{<t}, x) \cdot A_t$$

### 4.3 寤鸿3: 鏄惧紡闀垮害濂栧姳

濡傛灉闇€瑕佹帶鍒堕暱搴︼紝浣跨敤鏄惧紡濂栧姳鑰岄潪闅愬紡褰掍竴鍖?

```python
def compute_reward(response, ground_truth):
    accuracy = 1.0 if is_correct(response, ground_truth) else 0.0
    
    # 鏄惧紡闀垮害鎯╃綒锛堝彲璋冿級
    length_penalty = -0.001 * len(response)
    
    return accuracy + length_penalty
```

---

## 浜斻€佺悊璁鸿瑙?

### 5.1 鍩虹嚎鐨勬湰璐?

鍩虹嚎鐨勭洰鐨勬槸**鍑忓皯鏂瑰樊**锛屼笉鏀瑰彉鏈熸湜:

$$\text{Var}[\nabla \log \pi \cdot (R - b)] < \text{Var}[\nabla \log \pi \cdot R]$$

鏈€浼樺熀绾?
$$b^*(s) = \frac{\mathbb{E}[(\nabla \log \pi)^2 \cdot R | s]}{\mathbb{E}[(\nabla \log \pi)^2 | s]}$$

### 5.2 GRPO鍩虹嚎鐨勪紭鍔?

GRPO鐨勭粍鍐呭潎鍊兼槸瀵?$V(s)$ 鐨?*鏃犲亸浼拌**:

$$\bar{R} = \frac{1}{G} \sum_{i=1}^G R_i \approx \mathbb{E}[R | s]$$

褰?$G$ 瓒冲澶ф椂锛岃繖涓及璁″緢鍑嗙‘銆?

### 5.3 涓轰綍涓嶉渶瑕佷环鍊煎嚱鏁?

浼犵粺PPO闇€瑕佷环鍊煎嚱鏁版槸鍥犱负:
1. 鍙兘閲囨牱涓€涓猼rajectory
2. 闇€瑕佷粠V(s)浼拌鏈熸湜鍥炴姤

GRPO鐨勭壒娈婄粨鏋?
1. 鍚屼竴涓猵rompt鍙互閲囨牱澶氫釜response
2. 缁勫唴鍧囧€肩洿鎺ヤ及璁℃湡鏈?
3. 鏃犻渶鍗曠嫭鐨勫嚱鏁拌繎浼?

---

## 鍏€佸疄璺靛缓璁?

### 6.1 浣曟椂浣跨敤鏍囧噯鍖?

**寤鸿浣跨敤**:
- 涓嶅悓prompt闅惧害宸紓澶?
- 闇€瑕佹墍鏈塸rompt鏈夌浉浼肩殑瀛︿範璐＄尞
- reward鍒嗗竷宸紓澶?

**寤鸿涓嶄娇鐢?*:
- 浜屽厓reward (0/1)
- prompt闅惧害鐩歌繎
- 闇€瑕佺悊璁轰繚璇?

### 6.2 瓒呭弬鏁伴€夋嫨

```python
class GRPOConfig:
    # 缁勫ぇ灏?
    group_size: int = 8  # 姣弍rompt鐢熸垚8涓猺esponse
    
    # 褰掍竴鍖?
    use_std_normalization: bool = False  # Dr. GRPO寤鸿
    epsilon: float = 1e-5  # 濡傛灉浣跨敤std褰掍竴鍖?
    
    # 闀垮害
    use_length_normalization: bool = False  # Dr. GRPO寤鸿
    length_penalty: float = 0.0  # 濡傞渶鏄惧紡鎯╃綒
    
    # 鍏朵粬
    clip_epsilon: float = 0.2
    kl_penalty: float = 0.01
```

### 6.3 璋冭瘯寤鸿

1. **鐩戞帶姊害**: 妫€鏌ヤ笉鍚宲rompt鐨勬搴﹀箙搴︽槸鍚﹀悎鐞?
2. **鐩戞帶闀垮害**: 妫€鏌esponse闀垮害鏄惁寮傚父澧為暱/缂╃煭
3. **鍒嗗眰鍒嗘瀽**: 鍒嗗埆鍒嗘瀽绠€鍗?鍥伴毦闂鐨勫涔犳洸绾?
4. **娑堣瀺瀹為獙**: 瀵规瘮鏈?鏃犲綊涓€鍖栫殑鏁堟灉

---

## 涓冦€佷唬鐮佸疄鐜?

### 7.1 鍘熷GRPO

```python
def grpo_loss_original(log_probs, rewards, group_size):
    """鍘熷GRPO瀹炵幇锛堝甫std褰掍竴鍖栵級"""
    batch_size = rewards.shape[0]
    num_groups = batch_size // group_size
    
    # reshape涓虹粍
    rewards = rewards.view(num_groups, group_size)
    log_probs = log_probs.view(num_groups, group_size, -1)
    
    # 缁勫唴褰掍竴鍖?
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-5)
    
    # 闀垮害褰掍竴鍖栫殑loss
    seq_lengths = (log_probs != 0).sum(dim=-1)
    normalized_log_probs = log_probs.sum(dim=-1) / seq_lengths
    
    loss = -(normalized_log_probs * advantages).mean()
    return loss
```

### 7.2 Dr. GRPO

```python
def grpo_loss_dr(log_probs, rewards, group_size):
    """Dr. GRPO瀹炵幇锛堢Щ闄ゅ綊涓€鍖栵級"""
    batch_size = rewards.shape[0]
    num_groups = batch_size // group_size
    
    # reshape涓虹粍
    rewards = rewards.view(num_groups, group_size)
    log_probs = log_probs.view(num_groups, group_size, -1)
    
    # 鍙噺鍧囧€硷紝涓嶉櫎std
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - mean_rewards
    
    # 涓嶅仛闀垮害褰掍竴鍖?
    total_log_probs = log_probs.sum(dim=-1)
    
    loss = -(total_log_probs * advantages).mean()
    return loss
```

### 7.3 甯︽樉寮忛暱搴︽儵缃?

```python
def grpo_loss_with_length_penalty(log_probs, rewards, lengths, 
                                   group_size, length_penalty=0.001):
    """甯︽樉寮忛暱搴︽儵缃氱殑GRPO"""
    # 鍦╮eward涓姞鍏ラ暱搴︽儵缃?
    adjusted_rewards = rewards - length_penalty * lengths
    
    # 浣跨敤Dr. GRPO鐨刲oss
    return grpo_loss_dr(log_probs, adjusted_rewards, group_size)
```

---

## 鍏€佹€荤粨

### 鍏抽敭缁撹

1. **鏍囧噯宸綊涓€鍖栦笉鏄悊璁轰繚璇佺殑**锛屽湪鏌愪簺鎯呭喌涓嬪彲鑳芥湁瀹?
2. **闀垮害褰掍竴鍖栦細瀵艰嚧鍙嶅悜婵€鍔?*锛堥暱閿欒銆佺煭姝ｇ‘锛?
3. **绠€鍗曠殑centered rewards寰€寰€鏁堟灉鏇村ソ**
4. **濡傞渶闀垮害鎺у埗锛屼娇鐢ㄦ樉寮忓鍔?*

### 瀹炶返checklist

- [ ] 绉婚櫎std褰掍竴鍖栵紙鎴栬嚦灏戝仛娑堣瀺瀹為獙锛?
- [ ] 绉婚櫎闀垮害褰掍竴鍖?
- [ ] 鐩戞帶response闀垮害鍙樺寲
- [ ] 妫€鏌ユ搴﹀湪涓嶅悓prompt涓婄殑鍒嗗竷
- [ ] 鑰冭檻鏄惧紡闀垮害濂栧姳锛堝鏋滈渶瑕侊級

---

## 鍙傝€冭祫鏂?

1. Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning
2. Liu et al. (2025). Dr. GRPO: Understanding R1-Zero-Like Training
3. Schulman et al. (2017). Proximal Policy Optimization Algorithms
4. Greensmith et al. (2004). Variance Reduction Techniques for Gradient Estimates in RL


