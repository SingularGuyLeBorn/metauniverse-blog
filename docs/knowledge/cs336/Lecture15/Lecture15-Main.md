# CS336 Lecture 15: 璇﹁ВSFT涓嶳LHF (Deep Dive into SFT and RLHF)

> **缂栬緫钃濆浘 (Editorial Blueprint)**
> 
> **鏍稿績涓婚**: 鏈搴ф爣蹇楃潃CS336璇剧▼浠庨璁粌杩涘叆鍚庤缁?(Post-training) 闃舵銆傛繁鍏ヨ瑙ｅ浣曞皢棰勮缁冪殑澶фā鍨嬶紙濡侴PT-3锛夎浆鍙樹负鏈夌敤鐨勩€佸彲鎺х殑鍔╂墜锛堝ChatGPT锛夈€傛牳蹇冩柟娉曞寘鎷?*鐩戠潱寰皟 (SFT)**鍜?*鍩轰簬浜虹被鍙嶉鐨勫己鍖栧涔?(RLHF)**銆?
> 
> **鐭ヨ瘑缁撴瀯**: 
> - 绗竴閮ㄥ垎锛氱洃鐫ｅ井璋?(SFT) - 鏁版嵁绫诲瀷銆佽川閲忔潈琛°€佸畨鍏ㄥ井璋冦€丮id-training
> - 绗簩閮ㄥ垎锛歊LHF - 鎴愬鍋忓ソ鏁版嵁鏀堕泦銆丅radley-Terry妯″瀷銆佸鍔辨ā鍨嬨€丳PO绠楁硶
> - 绗笁閮ㄥ垎锛欴PO - 鐩存帴鍋忓ソ浼樺寲锛孯LHF鐨勭畝鍖栨浛浠ｆ柟妗?
> 
> **绮捐嫳琛ュ厖绗旇**:
> - **[娣卞叆鎺㈣: InstructGPT娴佹按绾縘(./Lecture15-InstructGPT.md)** - 瀹屾暣鐨勪笁闃舵鍚庤缁冩祦绋?
> - **[娣卞叆鎺㈣: DPO鏁板鎺ㄥ](./Lecture15-DPO.md)** - 浠嶳LHF鐩爣鍒癉PO鎹熷け鍑芥暟

---

## 涓€銆佸悗璁粌鐨勬剰涔?(The Significance of Post-Training)

### 1.1 浠嶨PT-3鍒癈hatGPT鐨勮浆鍙?

```
GPT-3 (棰勮缁?          鈫?         ChatGPT (鍚庤缁?
- 濉厖鏂囨湰              鈫?         - 閬靛惊鎸囦护
- 涓嶅彲鎺?              鈫?         - 瀹夊叏鍙帶  
- 闅句互鐩存帴浣跨敤          鈫?         - 浜у搧绾у姪鎵?
```

**鍏抽敭娲炲療**: 棰勮缁冩ā鍨嬪凡缁?鐭ラ亾"寰堝鑳藉姏锛堟帹鐞嗐€佸洖绛旈棶棰橈級锛屼絾杩欎簺鑳藉姏琚煁钘忓湪鍙傛暟涓€傚悗璁粌鐨勭洰鐨勬槸**婵€娲诲拰寮曞**杩欎簺鑳藉姏銆?

### 1.2 鐜颁唬鎸囦护閬靛惊鑳藉姏鐨勫己澶?

Example from Sebastian Bubeck's "Sparks of AGI" Paper (2023):
- 妯″瀷鍙互鍚屾椂閬靛惊10+鏉″祵濂楀鍚堟寚浠?
- 缁撳悎缂栫▼鑳藉姏闆舵牱鏈敓鎴恗atplotlib浠ｇ爜
- 杩欏湪浠ュ墠鐨勫彲鎺х敓鎴愭柟娉曚腑鏄笉鍙兘鐨?

### 1.3 瀹夊叏涓庡唴瀹瑰鏍?

鍚庤缁冧篃鏄坊鍔犲畨鍏ㄦ姢鏍忕殑鍏抽敭闃舵锛?
- 闃叉妯″瀷琚互鐢紙璇堥獥銆佽櫄鍋囦俊鎭級
- 鍐呭瀹℃牳锛堥伩鍏嶆湁瀹宠緭鍑猴級
- ChatGPT鎴愬姛鐨勯噸瑕佸師鍥犱箣涓€鏄叾鏄捐憲鐨勫畨鍏ㄦ姢鏍?

---

## 浜屻€佺洃鐫ｅ井璋?(Supervised Fine-Tuning, SFT)

### 2.1 SFT鐨勫熀鏈€濇兂

SFT鏈川涓婂氨鏄湪**涓撳绀鸿寖鏁版嵁**涓婅繘琛屾搴︿笅闄嶏細

$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log P_\theta(y|x) \right]$$

- $x$: 鎸囦护/鎻愮ず
- $y$: 鏈熸湜鐨勫洖澶?
- $\mathcal{D}$: 鎸囦护璺熼殢鏁版嵁闆?

### 2.2 涓夌鍏稿瀷鐨勬寚浠ゅ井璋冩暟鎹?

#### 2.2.1 FLAN (浠诲姟鑱氬悎鍨?

**鏋勫缓鏂规硶**: 鑱氬悎鐜版湁NLP浠诲姟鏁版嵁闆嗭紝杞寲涓烘寚浠ゆ牸寮?

```python
# 鍘熷鏁版嵁
{"text": "The quick brown fox...", "label": "positive"}

# FLAN鏍煎紡
{
    "instruction": "Classify the sentiment of the following text:",
    "input": "The quick brown fox...",
    "output": "positive"
}
```

**鐗圭偣**:
- 鉁?鏁版嵁閲忓ぇ锛屽厤璐?
- 鉂?鏍煎紡涓嶈嚜鐒讹紙澶氶€夐銆佺煭鍥炵瓟锛?
- 鉂?涓庣敤鎴峰疄闄呬氦浜掑樊寮傚ぇ

#### 2.2.2 Stanford Alpaca (AI鐢熸垚鍨?

**鏋勫缓鏂规硶**: 浣跨敤璇█妯″瀷鐢熸垚鎸囦护鍜屽洖澶?

```python
# 娴佺▼:
# 1. 浜虹被缂栧啓绉嶅瓙鎸囦护闆?
# 2. 鐢℅PT-3.5鐢熸垚鏇村鎸囦护
# 3. 鐢℅PT-3.5鐢熸垚瀵瑰簲鍥炲
```

**鐗圭偣**:
- 鉁?鏇村儚鑷劧瀵硅瘽
- 鉁?闀挎牸寮忓洖澶?
- 鉂?鎸囦护澶氭牱鎬ф湁闄?
- 鉂?鍙兘缁ф壙妯″瀷鍋忚

#### 2.2.3 OpenAssistant (浜虹被浼楀寘鍨?

**鏋勫缓鏂规硶**: 鍦ㄧ嚎鐖卞ソ鑰呯ぞ鍖鸿嚜鎰跨紪鍐?

**鐗圭偣**:
- 鉁?楂樿川閲忋€佽缁嗗洖澶?
- 鉁?鏈夋椂鍖呭惈寮曠敤
- 鉂?鎴愭湰楂樸€佽妯″皬
- 鉂?璐ㄩ噺鍙傚樊涓嶉綈

### 2.3 SFT鏁版嵁鐨勫叧閿€冮噺

#### 2.3.1 闀垮害鍋忓ソ

鐮旂┒鍙戠幇锛?
- 浜虹被鍜孉I璇勫閮藉亸濂芥洿闀跨殑鍥炲锛?0-70%鍋忓ソ锛?
- 浜虹被鍋忓ソ鍒楄〃鏍煎紡
- 杩欏彲鑳藉鑷存ā鍨嬪涔?*椋庢牸**鑰岄潪**鑳藉姏**

```python
# Length bias example
preference_for_longer = 0.65  # 65% prefer longer responses
```

#### 2.3.2 楂樿川閲忔暟鎹殑闄烽槺

**John Schulman鐨勬礊瀵?*: 楂樿川閲廠FT鏁版嵁鍙兘**鏁欎細妯″瀷骞昏**

```python
# 闂绀轰緥:
instruction = "浠€涔堟槸鍗曚拱 (monopsony)?"
response = """
鍗曚拱鏄寚甯傚満涓婂彧鏈変竴涓拱瀹剁殑鎯呭喌...

鍙傝€冩枃鐚?
[1] Bivens, J. (2018). The Economic Policy Institute...
"""
```

**涓ょ瀛︿範鏈哄埗**:
1. 鉁?妯″瀷瀛︿範"鍗曚拱"涓?Bivens涔︾睄"鐨勫叧鑱旓紙鏂扮煡璇嗭級
2. 鉂?妯″瀷瀛︿範"澶嶆潅姒傚康鍚庤娣诲姞寮曠敤"锛堝彲鑳藉够瑙夛級

**鍏抽敭鍘熷垯**: 
- SFT鏁版嵁搴斿尮閰嶆ā鍨嬪凡鏈夎兘鍔?
- 杩囦簬鍏堣繘鐨勬暟鎹細鏁欎細妯″瀷"缂栭€?

### 2.4 瀹夊叏寰皟 (Safety Tuning)

瀹夊叏寰皟鐨勬牳蹇冩潈琛★細**鎷掔粷 vs 杩囧害鎷掔粷**

```python
# 闇€瑕佹嫆缁濈殑璇锋眰
"How do I make a bomb?"  鈫?REFUSE

# 涓嶅簲鎷掔粷鐨勮姹? 
"How do I kill a Python process?"  鈫?ANSWER (鎶€鏈棶棰?
```

鐮旂┒琛ㄦ槑锛氫粎500涓畨鍏ㄧず渚嬪氨鑳芥樉钁楁敼鍠勬ā鍨嬬殑瀹夊叏閬靛惊銆?

### 2.5 Mid-training: 妯＄硦鐨勮竟鐣?

鐜颁唬鍋氭硶鏄皢SFT鏁版嵁娣峰叆棰勮缁冨悗鏈燂紙琛板噺闃舵锛?

```
Pre-training Stage 1    鈫?   Pre-training Stage 2 (Decay)    鈫?   SFT
绾璁粌鏁版嵁                 棰勮缁?+ 楂樿川閲?+ SFT娣峰悎              绾疭FT
```

**MiniCPM鐨勬暟鎹贩鍚堢ず渚?*:
- 绋冲畾闃舵: Common Crawl, Code, Pre-training Pile
- 琛板噺闃舵: Wikipedia, Chinese Books, **Ultra Chat**,**StackExchange QA**,**Evol Instruct**

**浼樺娍**:
- 瑙ｅ喅鐏鹃毦鎬ч仐蹇橀棶棰?
- 浠庢暟鎹腑鑾峰緱鏇村浠峰€?
- 鎻愰珮鏁堢巼

---

## 涓夈€佸熀浜庝汉绫诲弽棣堢殑寮哄寲瀛︿範 (RLHF)

### 3.1 涓轰粈涔堥渶瑕丷LHF?

#### 鍘熷洜1: SFT鏁版嵁鏀堕泦鏄傝吹

```
鎴愭湰瀵规瘮:
- SFT: 涓撳缂栧啓璇︾粏鍥炲 鈫?姣忔潯鏁版嵁鎴愭湰楂?
- RLHF: 浜虹被鍙渶姣旇緝涓や釜鍥炲 鈫?姣忔潯鏁版嵁鎴愭湰浣?
```

#### 鍘熷洜2: 鐢熸垚鍣?楠岃瘉鍣ㄥ樊璺?

鐮旂┒鍙戠幇锛氫汉绫?*楠岃瘉**鑳藉姏鍙兘浼樹簬**鐢熸垚**鑳藉姏

```python
# 瀹為獙缁撹
annotator_prefers_own_summary = 0.35  # 35%鏇村枩娆㈣嚜宸卞啓鐨?
annotator_prefers_AI_summary = 0.65   # 65%鏇村枩娆I鐢熸垚鐨?

# 鍘熷洜: "鎴戝啓鐨勬椂鍊欒寰楅渶瑕佹洿姝ｅ紡锛屼絾AI鐨勮璧锋潵鏇存祦鐣?
```

### 3.2 鎴愬鍋忓ソ鏁版嵁鏀堕泦

#### InstructGPT鐨勬爣娉ㄦ寚鍗?

涓夊ぇ鍘熷垯:
1. **Helpful (鏈夊府鍔?**: 娓呮櫚璇█銆佸洖绛旈棶棰樸€佸浗闄呭寲鏁忔劅
2. **Truthful (鐪熷疄)**: 涓嶅够瑙?
3. **Harmless (鏃犲)**: 涓嶆瘨鎬с€佷笉鏆村姏

#### 鏍囨敞鐨勭幇瀹炴寫鎴?

**璇惧爞浜掑姩瀹為獙**:
- 缁欏鐢?鍒嗛挓姣旇緝涓や釜AI鍥炲
- 缁撴灉: 澶у鏁颁汉鏃犳硶鏍稿疄鎵€鏈変簨瀹炲拰鏁板
- 杈冮暱鍥炲鑾峰緱鏇村鎶曠エ锛屽敖绠″寘鍚够瑙?

**瀹為檯鏍囨敞闂**:
- 鏃堕棿闄愬埗锛堝Google Bard鏍囨敞鍛樻瘡棰樺彧鏈?鍒嗛挓锛?
- 鏍囨敞鍛樺彲鑳戒娇鐢℅PT-4浣滅瓟
- 鎴愭湰vs璐ㄩ噺鏉冭　

### 3.3 Bradley-Terry鍋忓ソ妯″瀷

鍋囪姣忎釜鍥炲鏈夋綔鍦ㄦ爣閲忓鍔?$r(x, y)$锛屼汉绫诲亸濂藉缓妯′负:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2)) = \frac{1}{1 + e^{-(r(x,y_1) - r(x,y_2))}}$$

杩欐剰鍛崇潃锛?
- 濂栧姳宸秺澶э紝鍋忓ソ姒傜巼瓒婇珮
- 濂栧姳鐩稿悓鏃讹紝鍋忓ソ姒傜巼涓?0%

### 3.4 濂栧姳妯″瀷璁粌

浠庢垚瀵瑰亸濂芥暟鎹缁冨鍔辨ā鍨?$r_\theta(x, y)$:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

鍏朵腑:
- $y_w$: 鍋忓ソ鐨勶紙鑾疯儨锛夊洖澶?
- $y_l$: 涓嶅亸濂界殑锛堝け璐ワ級鍥炲

### 3.5 InstructGPT鐩爣鍑芥暟

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\phi(x, y) \right] - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

绗竴椤? 鏈€澶у寲濂栧姳
绗簩椤? 涓嶈鍋忕鍙傝€冩ā鍨嬪お杩滐紙闃叉reward hacking锛?

### 3.6 PPO绠楁硶绠€浠?

**PPO (Proximal Policy Optimization)** 鏄疪LHF鐨勬牳蹇冪畻娉曘€?

#### 绛栫暐姊害

$$\nabla J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla \log \pi_\theta(a|s) \cdot R \right]$$

#### PPO鐨勫叧閿敼杩?

1. **浼樺娍鍑芥暟**: 浣跨敤 $A(s,a)$ 浠ｆ浛 $R$ 鍑忓皯鏂瑰樊
2. **閲嶈鎬ч噰鏍?*: 鍏佽鍦ㄦ棫绛栫暐鏍锋湰涓婂娆℃洿鏂?
3. **瑁佸壀**: 闄愬埗绛栫暐鏇存柊骞呭害

$$L^{CLIP}(\theta) = \mathbb{E} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

鍏朵腑 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

#### PPO鐨勫鏉傛€?

PPO瀹炵幇闈炲父澶嶆潅锛屾湁璁烘枃鎬荤粨浜?7鏉″疄鐜扮粏鑺傘€傝繖婵€鍙戜簡瀵规洿绠€鍗曟浛浠ｆ柟妗堢殑鐮旂┒銆?

---

## 鍥涖€佺洿鎺ュ亸濂戒紭鍖?(DPO)

### 4.1 DPO鐨勫姩鏈?

**闂**: PPO澶鏉傦紙闇€瑕佸鍔辨ā鍨嬨€佷环鍊煎嚱鏁般€佸湪绾块噰鏍?..锛?

**DPO鐨勮В鍐虫柟妗?*: 缁曡繃鏄惧紡濂栧姳妯″瀷锛岀洿鎺ヤ粠鍋忓ソ鏁版嵁浼樺寲绛栫暐

### 4.2 DPO鎺ㄥ

#### Step 1: 鏈€浼樼瓥鐣ョ殑褰㈠紡

瀵逛簬KL姝ｅ垯鍖栫殑濂栧姳鏈€澶у寲闂锛屾渶浼樼瓥鐣ヤ负:

$$\pi^*(y|x) \propto \pi_{ref}(y|x) \cdot \exp\left(\frac{1}{\beta} r(x,y)\right)$$

#### Step 2: 浠庣瓥鐣ュ弽鎺ㄥ鍔?

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

#### Step 3: 浠ｅ叆Bradley-Terry妯″瀷

灏嗕笂寮忎唬鍏ュ亸濂芥鐜囷紝$Z(x)$椤圭浉娑?

$$P(y_1 \succ y_2 | x) = \sigma\left(\beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}\right)$$

#### Step 4: DPO鎹熷け鍑芥暟

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right]$$

### 4.3 DPO姊害鐨勭洿瑙傜悊瑙?

DPO姊害褰㈠紡:

$$\nabla \mathcal{L}_{DPO} \propto -\underbrace{w}_{\text{鏉冮噸}} \cdot \left( \underbrace{\nabla \log \pi_\theta(y_w|x)}_{\text{鎻愰珮濂藉洖澶嶆鐜噠} - \underbrace{\nabla \log \pi_\theta(y_l|x)}_{\text{闄嶄綆鍧忓洖澶嶆鐜噠} \right)$$

鍏朵腑 $w$ 鍦ㄩ殣鍚鍔变及璁￠敊璇椂鏇村ぇ锛堢被浼间簬鍥伴毦鏍锋湰鎸栨帢锛夈€?

### 4.4 DPO鐨勪紭鍔?

- 鉁?鏃犻渶璁粌鍗曠嫭鐨勫鍔辨ā鍨?
- 鉁?鏃犻渶鍦ㄧ嚎閲囨牱/rollout
- 鉁?瀹炵幇绠€鍗曪紙绫讳技浜嶴FT锛?
- 鉁?鍦ㄥ紑婧愭ā鍨嬩腑骞挎硾浣跨敤

### 4.5 DPO鐨勫眬闄?

- 鏈川涓婃槸绂荤嚎绠楁硶锛堝師濮嬪舰寮忥級
- 瀵逛簬鍙獙璇佸鍔憋紙濡傛暟瀛﹂锛変笉澶€傜敤
- 鍙兘浠嶇劧瀛樺湪闀垮害鍋忓樊闂

---

## 浜斻€丷LHF鐨勬寫鎴樹笌娉ㄦ剰浜嬮」

### 5.1 杩囧害浼樺寲 (Over-optimization)

闅忕潃RL璁粌杩涜:
- 浠ｇ悊濂栧姳锛堝鍔辨ā鍨嬪垎鏁帮級鎸佺画涓婂崌
- 鐪熷疄浜虹被鍋忓ソ鍏堝崌鍚庨檷

```
                鐪熷疄鍋忓ソ
                   鈻?
                   |      ___
                   |   __/
                   |  /
                   | /
                   |/
    鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈻?RL姝ユ暟
                   
    鍘熷洜: 濂栧姳妯″瀷涓嶅畬缇庯紝妯″瀷瀛︿細"娆洪獥"瀹?
```

### 5.2 鏍″噯闂

RLHF鍚庣殑妯″瀷寰€寰€**杩囧害鑷俊**:
- 棰勬祴鐨勭疆淇″害涓庡疄闄呮纭巼涓嶅尮閰?
- 鍦ㄦ俯搴?1鏃舵洿鏄庢樉
- 杩欐槸鍥犱负RL浼樺寲鐨勬槸濂栧姳锛屼笉鏄垎甯?

### 5.3 鏍囨敞鍛樺亸宸?

鐮旂┒鍙戠幇:
- 缇庡浗鏍囨敞鍛樺崰17%
- 鑿插緥瀹?瀛熷姞鎷夊浗鏍囨敞鍗犳瘮楂?
- 妯″瀷鍙兘鍋忓悜杩欎簺缇や綋鐨勪环鍊艰

```python
# 鏍囨敞鍛樺叧娉ㄧ偣宸紓
expert_annotators: [
    {"focus": "factuality", "weight": 0.5},
    {"focus": "helpfulness", "weight": 0.3},
    {"focus": "formatting", "weight": 0.2}
]

crowdworkers: [
    {"focus": "formatting", "weight": 0.5},  # 鏇村叧娉ㄦ牸寮?
    {"focus": "helpfulness", "weight": 0.3},
    {"focus": "factuality", "weight": 0.2}  # 杈冨皯鍏虫敞浜嬪疄
]
```

### 5.4 AI鍙嶉鐨勫叴璧?

鐢变簬浜虹被鏍囨敞鐨勫眬闄愶紝AI鍙嶉锛圧LAIF锛夎秺鏉ヨ秺娴佽:
- Constitutional AI (Anthropic)
- Ultra Feedback (寮€婧?
- Tulu 3 (AI2)

濂藉:
- 鎴愭湰浣?
- 涓€鑷存€ч珮
- 瑙勬ā澶?

椋庨櫓:
- AI鑷垜鍋忓ソ
- 鍚岃川鍖?
- 浠嶆湁闀垮害鍋忓樊

---

## 鍏€佸叧閿鐐规€荤粨 (Key Takeaways)

### SFT瑕佺偣

1. **鎯婁汉鐨勯珮鏁?*: 灏戦噺鏁版嵁灏辫兘浜х敓鏄捐憲鏁堟灉
2. **鏁版嵁璐ㄩ噺澶嶆潅**: "楂樿川閲?涓嶄竴瀹氬妯″瀷濂斤紙骞昏闂锛?
3. **Mid-training鏄秼鍔?*: SFT鏁版嵁娣峰叆棰勮缁冨悗鏈?

### RLHF瑕佺偣

1. **楠岃瘉姣旂敓鎴愪究瀹?*: 鎴愬鍋忓ソ鏁版嵁鏀堕泦鎴愭湰浣庝簬涓撳绀鸿寖
2. **浣嗕粛鏈夋寫鎴?*: 鏃堕棿闄愬埗銆佹爣娉ㄥ憳鍋忓樊銆佷簨瀹炴牳鏌ュ洶闅?
3. **杩囧害浼樺寲鏄湡瀹為闄?*: 闇€瑕佸钩琛′唬鐞嗗鍔卞拰鐪熷疄鍋忓ソ

### DPO瑕佺偣

1. **绠€鍖栦簡RLHF**: 鏃犻渶濂栧姳妯″瀷鍜屽湪绾块噰鏍?
2. **骞挎硾閲囩敤**: 鎴愪负寮€婧愮ぞ鍖虹殑棣栭€夋柟娉?
3. **浠嶆湁灞€闄?*: 绂荤嚎銆侀暱搴﹀亸宸瓑闂

### 鏁翠綋娴佹按绾?

```
Pre-training 鈫?Mid-training 鈫?SFT 鈫?RLHF/DPO 鈫?Deployment
   鑳藉姏         楂樿川閲忔敞鍏?     鎸囦护璺熼殢   瀹夊叏/鍋忓ソ瀵归綈    浜у搧
```

---

## 鍙傝€冭祫鏂?

1. **InstructGPT**: Ouyang et al. (2022). Training language models to follow instructions with human feedback
2. **RLHF Tutorial**: Lambert et al. (2022). Illustrating Reinforcement Learning from Human Feedback
3. **DPO**: Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model
4. **Constitutional AI**: Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback
5. **Sparks of AGI**: Bubeck et al. (2023). Sparks of Artificial General Intelligence

