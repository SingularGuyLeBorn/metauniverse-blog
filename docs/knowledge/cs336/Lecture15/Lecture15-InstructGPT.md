# 娣卞叆鎺㈣: InstructGPT娴佹按绾?

鏈枃鏄疞ecture 15鐨勭簿鑻辫ˉ鍏呯瑪璁帮紝璇︾粏鍓栨瀽InstructGPT璁烘枃涓殑涓夐樁娈靛悗璁粌娴佺▼锛岃繖鏄悊瑙ｇ幇浠ｅぇ妯″瀷瀵归綈鎶€鏈殑鍩虹銆?

---

## 涓€銆佽儗鏅細浠嶨PT-3鍒癐nstructGPT

### 1.1 GPT-3鐨勯棶棰?

GPT-3 铏界劧鑳藉姏寮哄ぇ锛屼絾瀛樺湪鏍规湰鎬ч棶棰橈細
- **涓嶉伒寰寚浠?*: 鐢ㄦ埛璇?鍐欎竴棣栬瘲"锛屾ā鍨嬪彲鑳借緭鍑?璇楁槸浠€涔堬紵璇楁湁寰堝绉?.."
- **涓嶅畨鍏?*: 鍙兘杈撳嚭鏈夊銆佸亸瑙佸唴瀹?
- **涓嶅彲鎺?*: 琛屼负闅句互棰勬祴

### 1.2 InstructGPT鐨勭洰鏍?

璁╂ā鍨嬪仛鍒颁笁涓狧:
- **Helpful (鏈夊府鍔?**: 閬靛惊鐢ㄦ埛鎰忓浘锛屾彁渚涙湁鐢ㄥ洖澶?
- **Honest (璇氬疄)**: 涓嶇紪閫犱簨瀹烇紝鎵胯涓嶇‘瀹氭€?
- **Harmless (鏃犲)**: 鎷掔粷鏈夊璇锋眰锛屼笉杈撳嚭鍗遍櫓鍐呭

---

## 浜屻€佺涓€闃舵锛氱洃鐫ｅ井璋?(SFT)

### 2.1 鏁版嵁鏀堕泦

**鏉ユ簮**: 闆囦剑绾?0鍚嶆爣娉ㄥ憳锛岀紪鍐欓珮璐ㄩ噺prompt-response瀵?

**鏁版嵁瑙勬ā**: 绾?3,000鏉＄ず鑼?

**鏁版嵁鏍煎紡**:
```json
{
  "prompt": "Explain quantum entanglement to a 10-year-old",
  "response": "Imagine you have two magical coins that are best friends..."
}
```

### 2.2 鏍囨敞鎸囧崡鎽樿

**Helpful 鍘熷垯**:
- 浣跨敤娓呮櫚銆佺畝娲佺殑璇█
- 鐩存帴鍥炵瓟闂
- 鑰冭檻鍥介檯鍖栵紙濡?football"鍙兘鎸囦笉鍚岃繍鍔級
- 蹇呰鏃惰姹傛緞娓?

**Honest 鍘熷垯**:
- 璇村嚭鐪熷疄鎯虫硶
- 涓嶇紪閫犳潵婧愭垨寮曠敤
- 瀵逛笉纭畾鐨勫唴瀹硅〃杈句笉纭畾

**Harmless 鍘熷垯**:
- 涓嶄井杈便€佽船浣庣敤鎴?
- 涓嶈緭鍑烘€ф殫绀恒€佹毚鍔涘唴瀹?
- 閬垮厤鏁忔劅璇濋鐨勬瀬绔珛鍦?

### 2.3 SFT璁粌缁嗚妭

```python
# 浼唬鐮?
for epoch in range(num_epochs):
    for batch in dataloader:
        prompts, responses = batch
        
        # 鏍囧噯璇█妯″瀷鎹熷け
        loss = -log_prob(responses | prompts)
        
        loss.backward()
        optimizer.step()
```

**瓒呭弬鏁?*:
- 瀛︿範鐜? 9.65e-6锛堜綑寮﹁“鍑忥級
- Batch size: 32
- Epochs: 16
- 浣跨敤Dropout閬垮厤杩囨嫙鍚?

### 2.4 SFT鐨勫眬闄?

SFT鑳借妯″瀷"鍍忔牱"鍦板洖绛旈棶棰橈紝浣?
- 鏁版嵁鏀堕泦鎴愭湰楂?
- 鏍囨敞鍛樿兘鍔涙湁闄愶紙涓嶅妯″瀷鐨勬渶浣宠〃鐜帮級
- 鍙兘鏁欎細妯″瀷缂栭€狅紙濡傛灉璁粌鏁版嵁瓒呭嚭妯″瀷鑳藉姏锛?

---

## 涓夈€佺浜岄樁娈碉細濂栧姳妯″瀷璁粌 (RM)

### 3.1 浠庣ず鑼冨埌鍋忓ソ

**娲炲療**: 璁╀汉绫?*姣旇緝**涓や釜鍥炲姣?*缂栧啓**鍥炲鏇村鏄撲笖璐ㄩ噺鏇撮珮

**鏁版嵁鏀堕泦娴佺▼**:
1. 缁欏畾prompt锛岃SFT妯″瀷鐢熸垚澶氫釜鍥炲
2. 鏍囨敞鍛樺鍥炲杩涜鎺掑簭锛堜笉鍙槸鎴愬姣旇緝锛?
3. 浠庢帓搴忕敓鎴愭墍鏈夋垚瀵瑰亸濂?

### 3.2 鏁版嵁瑙勬ā

- 绾?3,000涓猵rompts
- 姣忎釜prompt绾?-9涓洖澶嶆帓搴?
- 鐢熸垚绾?00,000涓垚瀵规瘮杈?

### 3.3 Bradley-Terry妯″瀷

鍋囪姣忎釜鍥炲鏈夋綔鍦ㄥ垎鏁?$r(x, y)$锛屽亸濂芥鐜囦负:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$$

鍏朵腑 $\sigma$ 鏄痵igmoid鍑芥暟銆?

### 3.4 濂栧姳妯″瀷鏋舵瀯

濂栧姳妯″瀷 = GPT-3鏋舵瀯 + 鏍囬噺澶?

```python
class RewardModel(nn.Module):
    def __init__(self, gpt_model):
        self.backbone = gpt_model  # 鍏变韩GPT鏋舵瀯
        self.reward_head = nn.Linear(hidden_dim, 1)  # 杈撳嚭鏍囬噺
    
    def forward(self, prompt, response):
        # 缂栫爜prompt + response
        hidden = self.backbone.encode(prompt + response)
        # 鍙栨渶鍚庝竴涓猼oken鐨刪idden state
        last_hidden = hidden[:, -1, :]
        # 杈撳嚭鏍囬噺濂栧姳
        reward = self.reward_head(last_hidden)
        return reward
```

### 3.5 RM璁粌

浣跨敤鎴愬姣旇緝鎹熷け:

$$\mathcal{L}_{RM} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

```python
for batch in dataloader:
    prompt, winner, loser = batch
    
    r_winner = reward_model(prompt, winner)
    r_loser = reward_model(prompt, loser)
    
    loss = -torch.log(torch.sigmoid(r_winner - r_loser)).mean()
    
    loss.backward()
    optimizer.step()
```

### 3.6 RM璁粌鎶€宸?

**鍚屼竴鎺掑簭鐨勫瀵逛竴璧疯缁?*:
- 濡傛灉鏈夋帓搴?A > B > C > D
- 涓€娆℃€ц绠?(A,B), (A,C), (A,D), (B,C), (B,D), (C,D) 鐨勬崯澶?
- 鏇撮珮鏁堬紝閬垮厤閲嶅鍓嶅悜浼犳挱

**姝ｅ垯鍖?*:
- 妯″瀷浠嶨PT-3鍒濆鍖栵紝鍙兘鏃犻渶浠庡ご瀛︿範
- 鏃╂湡鍋滄闃叉杩囨嫙鍚?

---

## 鍥涖€佺涓夐樁娈碉細PPO寮哄寲瀛︿範

### 4.1 鐩爣鍑芥暟

$$\max_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta D_{KL}(\pi_\theta(y|x) || \pi_{SFT}(y|x)) \right]$$

**瑙ｉ噴**:
- $r_\phi$: 濂栧姳妯″瀷鎵撳垎
- $\beta D_{KL}$: 鎯╃綒鍋忕SFT妯″瀷澶繙锛堥槻姝eward hacking锛?

### 4.2 KL鎯╃綒鐨勯噸瑕佹€?

**娌℃湁KL鎯╃綒**:
- 妯″瀷鍙兘鎵惧埌"娆洪獥"濂栧姳妯″瀷鐨勬柟寮?
- 鐢熸垚楂樺垎浣嗘棤鎰忎箟鐨勮緭鍑?
- 璇█鑳藉姏閫€鍖?

**瀹為獙鍙戠幇**:
- $\beta$ 澶皬: reward hacking涓ラ噸
- $\beta$ 澶ぇ: 鍑犱箮涓嶆洿鏂帮紝绛変簬SFT妯″瀷
- 鏈€浼?$\beta$ 闇€瑕佽皟鍙?

### 4.3 PPO瀹炵幇缁嗚妭

#### 4.3.1 鍥涗釜妯″瀷

| 妯″瀷 | 浣滅敤 | 鏇存柊 |
|------|------|------|
| Policy $\pi_\theta$ | 鐢熸垚鍥炲 | 姣忔鏇存柊 |
| Value $V_\psi$ | 浼拌鏈熸湜濂栧姳 | 姣忔鏇存柊 |
| Reward $r_\phi$ | 璇勪及鍥炲璐ㄩ噺 | 鍐荤粨 |
| Reference $\pi_{SFT}$ | KL鎯╃綒鍙傝€?| 鍐荤粨 |

#### 4.3.2 鍗曟娴佺▼

```python
def ppo_step(prompts, policy, value, reward_model, ref_policy):
    # 1. 鐢熸垚鍥炲
    responses = policy.generate(prompts)
    
    # 2. 璁＄畻濂栧姳锛堝鍔辨ā鍨?- KL鎯╃綒锛?
    rm_scores = reward_model(prompts, responses)
    log_probs = policy.log_prob(responses | prompts)
    ref_log_probs = ref_policy.log_prob(responses | prompts)
    kl_penalty = beta * (log_probs - ref_log_probs)
    rewards = rm_scores - kl_penalty
    
    # 3. 璁＄畻浼樺娍
    values = value(prompts, responses)
    advantages = rewards - values  # 绠€鍖栫増
    
    # 4. PPO鏇存柊
    # ... (瑁佸壀绛栫暐姊害)
    
    # 5. 鏇存柊浠峰€煎嚱鏁?
    value_loss = (values - rewards) ** 2
```

#### 4.3.3 姣弔oken濂栧姳 vs 缁撴灉濂栧姳

InstructGPT灏咾L鎯╃綒**鍒嗛厤鍒版瘡涓猼oken**:

$$r_t = -\beta \cdot (\log \pi_\theta(a_t|s_t) - \log \pi_{SFT}(a_t|s_t))$$

鍙湁鏈€鍚庝竴涓猼oken鑾峰緱RM鍒嗘暟:

$$r_T = r_\phi(x, y) + r_T^{KL}$$

### 4.4 PreTrain娣峰悎

涓洪槻姝㈣瑷€鑳藉姏閫€鍖栵紝InstructGPT娣峰悎棰勮缁冪洰鏍?

$$\mathcal{L} = \mathcal{L}_{PPO} + \gamma \mathcal{L}_{pretrain}$$

鍏朵腑 $\mathcal{L}_{pretrain}$ 鏄湪GPT-3棰勮缁冩暟鎹笂鐨勮瑷€妯″瀷鎹熷け銆?

---

## 浜斻€佽瘎浼版柟娉?

### 5.1 浜虹被鍋忓ソ璇勪及

- 闅忔満閲囨牱prompts
- InstructGPT vs GPT-3 鐢熸垚鍥炲
- 鏍囨敞鍛橀€夋嫨鍋忓ソ

**缁撴灉**: InstructGPT鑾疯儨鐜?~85%

### 5.2 TruthfulQA

娴嬭瘯妯″瀷鏄惁浼氱紪閫犺櫄鍋囦俊鎭€?

**缁撴灉**: InstructGPT鐪熷疄鎬ф樉钁楁彁楂?

### 5.3 姣掓€ц瘎浼?

浣跨敤RealToxicityPrompts鏁版嵁闆嗐€?

**缁撴灉**: InstructGPT姣掓€ф樉钁楅檷浣?

### 5.4 "瀵归綈绋? (Alignment Tax)

瀵归綈鍙兘鎹熷鏌愪簺鑳藉姏:
- 鍦ㄤ竴浜汵LP benchmark涓婄暐鏈変笅闄?
- 浣嗗浜庡疄闄呭簲鐢ㄥ満鏅槸鍊煎緱鐨勬潈琛?

---

## 鍏€佸叧閿彂鐜颁笌鏁欒

### 6.1 瑙勬ā鍖栫殑閲嶈鎬?

- 1.3B鍙傛暟鐨処nstructGPT浼樹簬175B鐨凣PT-3
- 瀵归綈姣斿崟绾澶ц妯℃洿閲嶈

### 6.2 鏁版嵁璐ㄩ噺 > 鏁版嵁鏁伴噺

- SFT鍙敤13K绀鸿寖
- 鍏抽敭鏄爣娉ㄥ憳鐨勮川閲忓拰鎸囧崡鏄庣‘鎬?

### 6.3 浜虹被鍙嶉鐨勫眬闄?

- 鏍囨敞鍛樹細鐘敊锛堜簨瀹炴牳鏌ユ椂闂翠笉瓒筹級
- 鏍囨敞鍛樻湁鍋忚锛堟枃鍖栥€佽瑷€鑳屾櫙锛?
- 澶嶆潅浠诲姟闅句互璇勪及

### 6.4 杩唬鏀硅繘

InstructGPT鏄凯浠ｄ骇鐗?
- 鐢ㄥ綋鍓嶆ā鍨嬬敓鎴愭暟鎹?
- 鏀堕泦鍙嶉
- 璁粌鏂版ā鍨?
- 閲嶅

---

## 涓冦€佸悗缁彂灞?

### 7.1 Constitutional AI (Anthropic)

鐢ˋI鍙嶉鏇夸唬閮ㄥ垎浜虹被鍙嶉:
- 瀹氫箟"瀹硶"鍘熷垯
- 璁〢I鑷垜鎵硅瘎鍜屼慨姝?
- 鍑忓皯浜虹被鏍囨敞闇€姹?

### 7.2 RLHF鐨勭畝鍖?

- DPO: 鏃犻渶璁粌鍗曠嫭鐨勫鍔辨ā鍨?
- SLiC: 绫讳技SFT浣嗕娇鐢ㄥ亸濂芥帓搴?
- 鍚勭*PO鍙樹綋

### 7.3 鍙獙璇佸鍔辩殑鍏磋捣

- 鏁板銆佷唬鐮佺瓑棰嗗煙
- 涓嶉渶瑕佷汉绫诲亸濂斤紝鐩存帴楠岃瘉姝ｇ‘鎬?
- DeepSeek R1 绛?

---

## 鍙傝€冭祫鏂?

1. Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback
2. Christiano, P. et al. (2017). Deep reinforcement learning from human preferences
3. Stiennon, N. et al. (2020). Learning to summarize with human feedback
4. Bai, Y. et al. (2022). Training a Helpful and Harmless Assistant with RLHF

