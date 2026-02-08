# CS336 Lecture 16: 璇﹁В澶фā鍨婻L绠楁硶 (Deep Dive into Large Model RL Algorithms)

> **缂栬緫钃濆浘 (Editorial Blueprint)**
> 
> **鏍稿績涓婚**: 鏈搴ф槸CS336璇剧▼RL绯诲垪鐨勯珮娼紝浠嶳LHF杩囨浮鍒?*鍙獙璇佸鍔辩殑寮哄寲瀛︿範 (RL from Verifiable Rewards)**銆傝缁嗚瑙PO绠楁硶銆丟RPO绠楁硶锛屽苟娣卞叆鍒嗘瀽涓変釜閲嶈鐨勬帹鐞嗘ā鍨嬫渚嬶細**DeepSeek R1**銆?*Kimi K1.5**鍜?*Qwen 3**銆?
> 
> **鐭ヨ瘑缁撴瀯**: 
> - 绗竴閮ㄥ垎锛歊LHF鏀跺熬 - DPO鍙樹綋銆佽繃搴︿紭鍖栥€佹牎鍑嗛棶棰?
> - 绗簩閮ㄥ垎锛歅PO璇﹁В - 绛栫暐姊害銆侀噸瑕佹€ч噰鏍枫€佽鍓?
> - 绗笁閮ㄥ垎锛欸RPO璇﹁В - 绉婚櫎浠峰€煎嚱鏁般€佺粍鍐呭熀绾?
> - 绗洓閮ㄥ垎锛氭帹鐞嗘ā鍨嬫渚嬪垎鏋?- R1銆並1.5銆丵wen 3
> 
> **绮捐嫳琛ュ厖绗旇**:
> - **[娣卞叆鎺㈣: DeepSeek R1鎶€鏈姤鍛奭(./Lecture16-DeepSeek-R1.md)** - R1-Zero銆丼FT鍒濆鍖栥€佽瑷€涓€鑷存€?
> - **[娣卞叆鎺㈣: GRPO鏁板缁嗚妭](./Lecture16-GRPO-Math.md)** - 鏍囧噯宸綊涓€鍖栭棶棰樸€丏r. GRPO

---

## 涓€銆丷LHF鏀跺熬 (RLHF Wrap-up)

### 1.1 DPO鍥為【

DPO鏇存柊褰㈠紡:

$$\nabla \mathcal{L}_{DPO} \propto \beta \cdot w(\theta) \cdot \left( \nabla \log \pi_\theta(y_w|x) - \nabla \log \pi_\theta(y_l|x) \right)$$

鍏朵腑 $w(\theta)$ 鍦ㄥ鍔变及璁￠敊璇椂鏇村ぇ鈥斺€旇繖鏄竴绉嶉殣寮忕殑鍥伴毦鏍锋湰鎸栨帢銆?

**鏍稿績鎬濇兂**: RL绠楁硶鏈川涓婇兘鏄?涓婅皟濂界殑锛屼笅璋冨潖鐨?锛屽尯鍒湪浜庡浣曠‘瀹?濂藉潖"鍜?澶氬皯"銆?

### 1.2 DPO鍙樹綋

#### SimPO

SimPO鍋氫簡涓や釜绠€鍖?
1. **闀垮害褰掍竴鍖?*: 闄や互鍥炲闀垮害锛岄伩鍏嶉暱搴﹀亸宸?
2. **绉婚櫎鍙傝€冩ā鍨?*: 涓嶅啀璁＄畻涓庡弬鑰冩ā鍨嬬殑姣旂巼

$$\mathcal{L}_{SimPO} = -\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x)\right)$$

#### 瀹為獙鍙戠幇鐨勭煕鐩剧粨璁?

**Tulu 2鐮旂┒**: PPO浼樹簬DPO锛堝洜涓哄湪绾跨壒鎬э級
**Tulu 3鐮旂┒**: 濡傛灉SFT鍋氬緱濂斤紝DPO鍜孭PO宸窛娑堝け锛岄暱搴﹀綊涓€鍖朌PO鏈€浣?

> **閲嶈鍚ず**: RL鐨勫疄楠岀粨璁洪珮搴︿緷璧栧叿浣撹缃紙妯″瀷銆佹暟鎹€佽瘎浼版柟娉曪級锛屼笉搴旂洸鐩硾鍖栧崟绡囪鏂囩殑缁撹銆?

### 1.3 杩囧害浼樺寲 (Over-optimization)

```
浠ｇ悊濂栧姳 vs 鐪熷疄鍋忓ソ:

浠ｇ悊濂栧姳 鈻?                   鐪熷疄鍋忓ソ 鈻?

         |    _____                    |    ____
         |   /                         |   /    \
         |  /                          |  /      \
         | /                           | /        \
         |/                            |/          \_____
         鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈻?RL姝ユ暟           鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈻?RL姝ユ暟
```

**鍘熷洜鍒嗘瀽**:
- 濂栧姳妯″瀷鏄湁鍣０鐨勮繎浼?
- 妯″瀷瀛︿細"娆洪獥"濂栧姳妯″瀷
- 绫讳技浜庤繃鎷熷悎锛屼絾鍙戠敓鍦ㄧ瓥鐣ュ眰闈?

**瀹為獙璇佹嵁** (鏉ヨ嚜浣滆€呭疄楠屽):
- RLHF on 浜虹被鍋忓ソ: 杩囧害浼樺寲
- RLHF on 鏈夊櫔澹癆I鍙嶉: 杩囧害浼樺寲
- RLHF on 鏃犲櫔澹癆I鍙嶉: 鏃犺繃搴︿紭鍖?

### 1.4 鏍″噯闂

RLHF鍚庣殑妯″瀷**涓嶅啀鏄牎鍑嗙殑姒傜巼妯″瀷**:

- 鎴戜滑浼樺寲鐨勬槸濂栧姳锛屼笉鏄鐜囧垎甯?
- 娓╁害=1鏃讹紝妯″瀷琛ㄧ幇杩囧害鑷俊
- 棰勬祴鐨勭疆淇″害 鈮?瀹為檯姝ｇ‘鐜?

鏉ヨ嚜澶氫釜璁烘枃鐨勮瘉鎹?
- Anthropic璁烘枃
- GPT-4鍙戝竷璁烘枃
- 瀛︽湳鐙珛鐮旂┒

---

## 浜屻€佷粠RLHF鍒板彲楠岃瘉濂栧姳 (From RLHF to Verifiable Rewards)

### 2.1 RLHF鐨勫眬闄?

| 闂 | 鎻忚堪 |
|------|------|
| 浜虹被鍋忓ソ鍣０澶?| 鏍囨敞鍛樹細鐘敊銆佹湁鍋忚 |
| 闅句互瑙勬ā鍖?| 浜虹被鏍囨敞鎴愭湰楂?|
| 杩囧害浼樺寲椋庨櫓 | 妯″瀷瀛︿細娆洪獥濂栧姳妯″瀷 |
| 鏃犳硶楠岃瘉姝ｇ‘鎬?| 瀵逛簬鏁板/浠ｇ爜锛屾纭€ф槸瀹㈣鐨?|

### 2.2 鍙獙璇佸鍔辩殑浼樺娍

**鏍稿績鎬濇兂**: 濡傛灉鎴戜滑鏈?*纭畾鎬х殑濂栧姳鍑芥暟**锛堜笉鏄涔犵殑锛夛紝灏卞彲浠ュ厖鍒嗗彂鎸L鐨勫▉鍔涖€?

| 棰嗗煙 | 鍙獙璇佸鍔?|
|------|-----------|
| 鏁板 | 绛旀鏄惁姝ｇ‘ |
| 浠ｇ爜 | 娴嬭瘯鐢ㄤ緥鏄惁閫氳繃 |
| 娓告垙 | 鏄惁鑾疯儨 |
| 褰㈠紡楠岃瘉 | 璇佹槑鏄惁鏈夋晥 |

**绫绘瘮AlphaGo/AlphaFold**: 杩欎簺鎴愬姛妗堜緥閮芥湁**纭畾鎬х殑濂栧姳鍑芥暟**銆?

---

## 涓夈€丳PO璇﹁В (PPO Deep Dive)

### 3.1 绛栫暐姊害鍩虹

鐩爣: 鏈€澶у寲鏈熸湜濂栧姳

$$J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[R(a)]$$

姊害:

$$\nabla J(\theta) = \mathbb{E}_{a \sim \pi_\theta}\left[\nabla \log \pi_\theta(a|s) \cdot R\right]$$

**鏈寸礌绛栫暐姊害**: 閲囨牱 $a \sim \pi_\theta$锛屾洿鏂?$\theta \leftarrow \theta + \alpha \nabla \log \pi_\theta(a|s) R$

**闂**: 绾湪绾匡紝姣忔閲囨牱鍚庡彧鑳芥洿鏂颁竴娆°€?

### 3.2 TRPO: 寮曞叆閲嶈鎬ч噰鏍?

鏍稿績鎯虫硶: 浠庢棫绛栫暐 $\pi_{\theta_{old}}$ 閲囨牱锛屼絾瀵规柊绛栫暐 $\pi_\theta$ 杩涜鏇存柊銆?

$$\nabla J(\theta) = \mathbb{E}_{a \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \nabla \log \pi_\theta(a|s) \cdot A \right]$$

鍏朵腑 $A$ 鏄紭鍔垮嚱鏁帮紙Advantage锛夛紝鏄?R$鐨勪綆鏂瑰樊鐗堟湰銆?

TRPO娣诲姞KL绾︽潫: $D_{KL}(\pi_\theta || \pi_{\theta_{old}}) \leq \delta$

### 3.3 PPO: 瑁佸壀鏇夸唬KL绾︽潫

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)\right]$$

鍏朵腑 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$锛?\epsilon$閫氬父鍙?.2銆?

**鐩磋瑙ｉ噴**:
- 濡傛灉鏂扮瓥鐣ュ亸绂诲お杩滐紙$r_t$ 瓒呭嚭 $[1-\epsilon, 1+\epsilon]$锛夛紝姊害琚鍓?
- 杩欒嚜鐒跺湴闄愬埗浜嗙瓥鐣ユ洿鏂板箙搴?

### 3.4 PPO鐨勫鏉傛€?

```
PPO瀹屾暣瀹炵幇闇€瑕?
鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 绛栫暐妯″瀷 蟺_胃                                鈹?
鈹? 浠峰€兼ā鍨?V_蠁 (鐢ㄤ簬璁＄畻浼樺娍)                   鈹?
鈹? 濂栧姳妯″瀷 r (RLHF璁剧疆)                        鈹?
鈹? 鍙傝€冩ā鍨?蟺_ref (KL姝ｅ垯鍖?                    鈹?
鈹? 骞夸箟浼樺娍浼拌 (GAE)                           鈹?
鈹? 澶氭鏇存柊 + 閲嶈鎬ч噰鏍?                       鈹?
鈹? 37鏉″疄鐜扮粏鑺?..                             鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
```

**璁烘枃**: "Implementation Matters in Deep RL: A Case Study on PPO and TRPO"

PPO瀹炵幇涓嶅綋鍙兘瀵艰嚧:
- 璁＄畻鐨勭敋鑷充笉鏄纭殑绛栫暐姊害
- 浣嗗眳鐒跺彲鑳芥晥鏋滄洿濂?

---

## 鍥涖€丟RPO璇﹁В (GRPO Deep Dive)

### 4.1 GRPO鐨勫姩鏈?

**鏍稿績闂**: 鑳藉惁**绉婚櫎浠峰€煎嚱鏁?*锛屽悓鏃朵繚鎸丳PO鐨勬晥鏋滐紵

**璇█妯″瀷鐨勭壒娈婃€?*:
- 瀵逛簬鍚屼竴涓猵rompt锛屽彲浠ョ敓鎴愬涓猺esponses
- 杩欐彁渚涗簡**鑷劧鐨勫熀绾夸及璁?*

### 4.2 GRPO鍏紡

#### 浼樺娍浼拌

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

鍏朵腑 $G$ 鏄瘡涓猵rompt鐢熸垚鐨剅esponse鏁伴噺銆?

**瑙ｉ噴**:
- 涓嶉渶瑕佽缁冧环鍊煎嚱鏁?
- 浣跨敤鍚岀粍responses鐨勫鍔卞潎鍊间綔涓哄熀绾?
- 鏍囧噯宸綊涓€鍖栵紙鍙€夛級

#### GRPO鎹熷け

$$L^{GRPO}(\theta) = \mathbb{E}\left[\min\left( r_t(\theta) A_i, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_i \right)\right]$$

### 4.3 GRPO vs PPO

| 鐗规€?| PPO | GRPO |
|------|-----|------|
| 浠峰€煎嚱鏁?| 闇€瑕?| 涓嶉渶瑕?|
| 鍐呭瓨鍗犵敤 | 2x妯″瀷 | 1x妯″瀷 |
| 鍩虹嚎鏉ユ簮 | 瀛︿範鐨刅(s) | 缁勫唴缁忛獙鍧囧€?|
| 閫傜敤鍦烘櫙 | 閫氱敤RL | 璇█妯″瀷 |

### 4.4 GRPO鐨勬綔鍦ㄩ棶棰?

#### 闂1: 鏍囧噯宸綊涓€鍖?

闄や互鏍囧噯宸槸鍚﹀悎娉曪紵

**鐞嗚鍒嗘瀽**: 鏍囧噯鐨勭瓥鐣ユ搴﹀畾鐞嗗彧鍏佽**鍑忓幓**涓庡姩浣滄棤鍏崇殑鍩虹嚎锛?*闄ゆ硶**涓嶅湪鍏佽鑼冨洿鍐呫€?

**闂**:
- 鏍囧噯宸皬 鈫?鏀惧ぇ姊害 鈫?闂澶畝鍗曟垨澶毦鏃舵搴﹁繃澶?
- 鍙兘褰卞搷鏀舵暃

#### 闂2: 闀垮害褰掍竴鍖?

GRPO鍘熷鍏紡瀵瑰鍔辫繘琛岄暱搴﹀綊涓€鍖?

**闂**:
- 绛旈敊鏃? 鏈€浼樼瓥鐣ユ槸鐢熸垚鏈€闀垮洖澶嶏紙绋€閲婅礋濂栧姳锛?
- 绛斿鏃? 鏈€浼樼瓥鐣ユ槸鐢熸垚鏈€鐭洖澶嶏紙闆嗕腑姝ｅ鍔憋級
- 瀵艰嚧: 涔变竷鍏碂鐨勯暱鍥炲

**Dr. GRPO璁烘枃**寤鸿绉婚櫎杩欎袱涓綊涓€鍖栥€?

![GRPO绠楁硶](./images/l16-grpo-algorithm.png)

---

## 浜斻€佹帹鐞嗘ā鍨嬫渚嬪垎鏋?(Reasoning Model Case Studies)

### 5.1 DeepSeek R1

#### R1-Zero: 绾疪L瀹為獙

**璁剧疆**:
- 鍩虹妯″瀷: DeepSeek V3锛堥璁粌+mid-training锛屾棤RLHF锛?
- 濂栧姳: 鍑嗙‘鎬э紙姝ｈ锛? 鏍煎紡锛坱hinking鏍囩锛?
- 绠楁硶: GRPO

**缁撴灉**:
- 鎬ц兘鎺ヨ繎OpenAI o1
- 鎬濈淮閾鹃暱搴﹁嚜鐒跺闀?
- 鍑虹幇"aha moment"锛堥】鎮熸椂鍒伙級

**浜夎**:
- 闀垮害澧為暱鍙兘鏄洜涓篏RPO鐨勯暱搴﹀亸宸紙Dr. GRPO璁烘枃锛?
- "aha moment"鍙兘鍦ㄩ璁粌鏃跺氨瀛樺湪

#### R1瀹屾暣娴佺▼

```
DeepSeek V3 Base
     鈹?
     鈻?SFT鍒濆鍖?(闀緾oT鏁版嵁)
     鈹?
     鈻?鎺ㄧ悊RL (GRPO + 鍑嗙‘鎬?+ 鏍煎紡 + 璇█涓€鑷存€у鍔?
     鈹?
     鈻?鍚庣画鍚庤缁?(閫氱敤鑳藉姏淇濈暀)
     鈹?
     鈻?R1
```

**鍏抽敭鍙戠幇**:
- **杩囩▼濂栧姳妯″瀷 (PRM) 涓嶅缁撴灉濂栧姳**: 涓嶥eepSeek Math缁撹鐩稿弽
- **钂欑壒鍗℃礇鏍戞悳绱?(MCTS) 涔熶笉澶湁鏁?*: 绠€鍗昍L灏卞浜?
- **璇█涓€鑷存€у鍔?*: 闃叉CoT璇█娣峰悎

![DeepSeek R1鎬ц兘](./images/l16-deepseek-r1-benchmarks.png)

#### 钂搁鍒板皬妯″瀷

鍙互灏哛1鐨凜oT钂搁鍒癚wen绛夊紑婧愭ā鍨?
- 绾?00涓囨潯CoT鏁版嵁
- 寰皟鍚庢樉钁楁彁鍗囨暟瀛︽€ц兘

### 5.2 Kimi K1.5

#### 涓嶳1鐨勭浉浼肩偣

- SFT鍒濆鍖?
- 缁撴灉濂栧姳RL
- 鎬ц兘鍖归厤o1

#### 鐙壒璐＄尞

##### 鏁版嵁閫夋嫨绛栫暐

```python
# 浣跨敤SFT妯″瀷璇勪及闅惧害
def select_training_data(problems, sft_model, num_samples=10):
    selected = []
    for problem in problems:
        # 鐢熸垚澶氫釜鍥炵瓟
        responses = [sft_model.generate(problem) for _ in range(num_samples)]
        # 璁＄畻閫氳繃鐜?
        pass_rate = sum(is_correct(r) for r in responses) / num_samples
        # 鍙繚鐣?鏈夋寫鎴樹絾鍙"鐨勯棶棰?
        if 0 < pass_rate < 1:  # 涓嶅叏瀵逛篃涓嶅叏閿?
            selected.append(problem)
    return selected
```

##### 闀垮害濂栧姳

Kimi璁捐浜嗕笓闂ㄦ帶鍒禖oT闀垮害鐨勫鍔?

$$r_{length} = \begin{cases}
\lambda & \text{if correct (榧撳姳鐭洖绛?} \\
\text{batch average} & \text{if incorrect (涓嶈繃搴︽儵缃氶暱鍥炵瓟)}
\end{cases}$$

鍏朵腑 $\lambda$ 涓庡洖澶嶉暱搴﹀湪batch鍐呯殑鐩稿浣嶇疆鏈夊叧銆?

**娉ㄦ剰**: 闀垮害濂栧姳涓嶈兘澶棭鍚敤锛屽惁鍒欎細瀵艰嚧RL鍋滄粸銆?

##### RL绠楁硶

Kimi浣跨敤浜嗕笉鍚屼簬GRPO鐨勭洰鏍?

1. 闈炲弬鏁板亣璁?鈫?濂栧姳鍙啓鎴愮瓥鐣ユ瘮鐜囧舰寮忥紙绫讳技DPO鎺ㄥ锛?
2. 浣跨敤骞虫柟鎹熷け椹卞姩绛夊紡鎴愮珛
3. 姊害褰㈠紡绫讳技GRPO锛屼絾鏈変笉鍚岀殑姝ｅ垯鍖?

$$\nabla L \propto \underbrace{(r_i - \bar{r})}_{\text{鍩虹嚎}} \cdot \underbrace{\nabla \log \pi_\theta(y_i|x)}_{\text{绛栫暐姊害}} - \underbrace{(\log \pi_\theta - \log \pi_{ref})^2}_{\text{姝ｅ垯鍖杴}$$

##### 绯荤粺宸ョ▼

Kimi璇︾粏璁ㄨ浜哛L绯荤粺:
- RL worker鍜孖nference worker鍒嗙
- 鏉冮噸闇€瑕佷粠RL worker浼犻€掑埌Inference worker
- 闀緾oT瀵艰嚧batch涓嶅潎琛?

### 5.3 Qwen 3

#### 鏁翠綋娴佺▼

```
Base Model
    鈹?
    鈻?Long CoT SFT
    鈹?
    鈻?Reasoning RL
    鈹?
    鈻?Thinking Mode Fusion (鏂?)
    鈹?
    鈻?General RL
    鈹?
    鈻?Qwen 3
```

#### 鏁版嵁閫夋嫨

涓嶬imi绫讳技锛屼娇鐢╞est-of-N杩囨护:
- 濡傛灉base model宸茬粡鑳藉仛瀵?鈫?澶畝鍗曪紝鎺掗櫎
- 鍘绘薄鏌? 绉婚櫎涓庢祴璇曢泦鐩镐技鐨勬暟鎹?
- 浜哄伐绛涢€? 纭繚SFT鏁版嵁鏃犵寽娴?

**鎯婁汉鍙戠幇**: 浠呯敤**3995涓牱鏈?*杩涜RL灏辫兘鑾峰緱鏄捐憲鎻愬崌锛?

#### Thinking Mode Fusion

**鐩爣**: 鍦ㄥ悓涓€妯″瀷涓敮鎸?鎬濊€?鍜?涓嶆€濊€?涓ょ妯″紡

```
User: <think> 瑙ｉ噴鐩稿璁?</think>
Model: [闀緾oT鎺ㄧ悊杩囩▼] 鐩稿璁烘槸...

User: <no_think> 瑙ｉ噴鐩稿璁?</no_think>  
Model: 鐩稿璁烘槸... [鐩存帴鍥炵瓟]
```

**璁粌鏂规硶**:
- 鐢≧1妯″瀷鐢熸垚 `<think>` 鏁版嵁
- 鐢熸垚 `<no_think>` 鐩存帴鍥炵瓟鏁版嵁
- 娣峰悎寰皟

**棰濆鑳藉姏**: 鍙互鍦ㄦ€濊€冭繃绋嬩腑**鎻愬墠缁堟**

```
User: 鑰冭檻鍒版椂闂存湁闄愶紝鎴戦渶瑕佺洿鎺ョ粰鍑虹瓟妗?..
Model: [鍋滄鎬濊€僝 </think> 绛旀鏄?..
```

#### 娴嬭瘯鏃惰绠楁墿灞?

閫氳繃鎺у埗鎬濊€僼oken棰勭畻锛屽疄鐜板钩婊戠殑鎬ц兘-寤惰繜鏉冭　:

```
鎬濊€冮绠?鈼€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈻?
   浣?         涓?         楂?         鏃犻檺
   鈹?          鈹?          鈹?          鈹?
   鈻?          鈻?          鈻?          鈻?
 蹇€熷洖绛?  涓瓑鎺ㄧ悊   娣卞害鎺ㄧ悊   瀹屾暣R1妯″紡
```

#### 鏈夎叮鍙戠幇

```
                    鎺ㄧ悊RL    Thinking Fusion   閫氱敤RL
閫氱敤浠诲姟               鈫?           鈫?             鈫?
鎸囦护閬靛惊               鈫?           鈫?             鈫?
鏁板(think)           鈫?           鈫?             鈫?(!)
鏁板(no_think)        鈫?           鈫?             鈫?
```

**瑙傚療**: 閫氱敤RL鍙兘**鎹熷**鎬濊€冩ā寮忎笅鐨勬暟瀛︽€ц兘鈥斺€斿瓨鍦╰radeoff銆?

---

## 鍏€佸叧閿鐐规€荤粨 (Key Takeaways)

### 浠嶳LHF鍒板彲楠岃瘉濂栧姳

```
RLHF鐨勫洶澧?
- 浜虹被鍋忓ソ鍣０ 鈫?杩囧害浼樺寲
- 闅句互瑙勬ā鍖?鈫?鎴愭湰闄愬埗
- 鏃犳硶楠岃瘉 鈫?涓嶉€傚悎鏁板/浠ｇ爜

鍙獙璇佸鍔辩殑瑙ｅ喅鏂规:
- 纭畾鎬у鍔卞嚱鏁?鈫?鏃犺繃搴︿紭鍖栭闄?
- 鏃犻渶浜虹被 鈫?鏃犻檺瑙勬ā鍖?
- 瀹㈣楠岃瘉 鈫?閫傚悎鎺ㄧ悊浠诲姟
```

### GRPO鏍稿績娲炲療

1. **璇█妯″瀷鐗规湁缁撴瀯**: 澶歳esponse鎻愪緵鑷劧鍩虹嚎
2. **绉婚櫎浠峰€煎嚱鏁?*: 鍑忓崐鍐呭瓨锛岀畝鍖栧疄鐜?
3. **娉ㄦ剰闄烽槺**: 鏍囧噯宸綊涓€鍖栧拰闀垮害褰掍竴鍖栧彲鑳芥湁瀹?

### 鎺ㄧ悊妯″瀷鐨勫叡鍚屾ā寮?

| 鍏卞悓鐐?| 鎻忚堪 |
|--------|------|
| SFT鍒濆鍖?| 鍏堢敤闀緾oT鏁版嵁寰皟 |
| 缁撴灉濂栧姳 | 鍙湅鏈€缁堢瓟妗堟纭€?|
| GRPO/绫讳技绠楁硶 | 绠€鍗曟湁鏁?|
| 闅惧害璇剧▼ | 閫夋嫨閫傚綋闅惧害鐨勮缁冩暟鎹?|

| 涓嶅悓鐐?| R1 | K1.5 | Qwen 3 |
|--------|-----|------|--------|
| 璇█涓€鑷存€?| 鏈?| 涓嶆槑纭?| 涓嶆槑纭?|
| 闀垮害鎺у埗 | 鏃犳槑纭?| 涓撻棬濂栧姳 | 涓嶆槑纭?|
| 鎬濊€冩ā寮忚瀺鍚?| 鏃?| 鏃?| 鏈?|
| 寮€婧愮▼搴?| 楂?| 涓?| 楂?|

### 鍏抽敭涓嶇‘瀹氭€?

1. **PRM vs ORM**: DeepSeek璇碠RM鏇村ソ锛屼絾鍏朵粬鐮旂┒鍙兘涓嶅悓
2. **MCTS**: 鐩墠浼间箮涓嶅绠€鍗昍L锛屼絾鍙兘杩樻湁鎺㈢储绌洪棿
3. **鏈€浼楻L绠楁硶**: GRPO銆並imi鐨勫彉浣撱€佽繕鏄叾浠栵紵

---

## 鍙傝€冭祫鏂?

1. **PPO**: Schulman et al. (2017). Proximal Policy Optimization Algorithms
2. **GRPO**: Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
3. **DeepSeek R1**: DeepSeek (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
4. **Kimi K1.5**: Moonshot AI (2025). Kimi k1.5: Scaling Reinforcement Learning with LLMs
5. **Qwen 3**: Qwen Team (2025). Qwen3 Technical Report
6. **Dr. GRPO**: Liu et al. (2025). Understanding R1-Zero-Like Training: A Critical Perspective


