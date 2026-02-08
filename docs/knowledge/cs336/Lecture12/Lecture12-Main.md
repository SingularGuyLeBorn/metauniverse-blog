# Lecture 12: 妯″瀷璇勪及璇﹁В (Deep Dive into Model Evaluation)

**璇剧▼**: CS336 路**涓婚**: 璇█妯″瀷璇勪及鐨勫摬瀛︺€佹鏋朵笌瀹炶返

## 0. 璇剧▼寮€鍦? 璇勪及鍗辨満

> **Andre Karpathy**: "There is an evaluation crisis."

![Karpathy 璇勪及鍗辨満鎺ㄦ枃](images/l12-karpathy-crisis.png)

**璇勪及**(Evaluation) 鐪嬩技绠€鍗曗€斺€旂粰瀹氫竴涓浐瀹氭ā鍨? 闂畠鏈夊"濂?? 浣嗚甯堝紑瀹楁槑涔夊湴鎸囧嚭: 杩欏叾瀹炴槸涓€涓?*鏋佸叾娣卞埢涓斿鏉?*鐨勮瘽棰? 瀹冧笉浠呮槸涓€涓満姊拌繃绋?(鎶涘嚭 Prompt, 璁＄畻鎸囨爣), 鏇存槸**鍐冲畾璇█妯″瀷鏈潵鍙戝睍鏂瑰悜**鐨勫叧閿姏閲? 鍥犱负椤剁骇妯″瀷寮€鍙戣€呴兘鍦ㄨ拷韪繖浜涜瘎浼版寚鏍? 鑰屼綘杩借釜浠€涔? 灏变細浼樺寲浠€涔?

---

## 1. 浣犳墍鐪嬪埌鐨? 璇勪及鏁版嵁鐨勫鍏冮潰璨?

褰撲綘鐪嬪埌涓€涓瑷€妯″瀷鍙戝竷鏃? 閫氬父浼氱湅鍒板悇绉?鍩哄噯鍒嗘暟":

### 1.1 瀹樻柟鍩哄噯鎶ュ憡

![DeepSeek-R1 鍩哄噯鍒嗘暟](images/l12-deepseek-r1-benchmarks.png)
![Llama 4 鍩哄噯鍒嗘暟](images/l12-llama4-benchmarks.png)

杩戞湡鐨勮瑷€妯″瀷 (DeepSeek-R1, Llama 4, OLMo2 绛? 閫氬父鍦?*鐩镐技浣嗕笉瀹屽叏鐩稿悓**鐨勫熀鍑嗕笂璇勪及, 鍖呮嫭 MMLU, MATH, GPQA 绛? 浣嗚繖浜涘熀鍑嗗埌搴曟槸浠€涔? 杩欎簺鏁板瓧鍒板簳鎰忓懗鐫€浠€涔?

### 1.2 鑱氬悎鎺掕姒?

![HELM Capabilities 鎺掕姒淽(images/l12-helm-capabilities-leaderboard.png)

**[HELM](https://crfm.stanford.edu/helm/capabilities/latest/)** 鏄柉鍧︾寮€鍙戠殑缁熶竴璇勪及妗嗘灦, 灏嗗绉嶆爣鍑嗗熀鍑嗚仛鍚堝湪涓€璧? 鎻愪緵鏇村叏闈㈢殑瑙嗚.

### 1.3 鎴愭湰鏁堢泭鍒嗘瀽

![Artificial Analysis 甯曠疮鎵樺墠娌縘(images/l12-artificial-analysis.png)

**[Artificial Analysis](https://artificialanalysis.ai/)**鎻愪緵浜嗕竴涓湁瓒ｇ殑瑙嗚:**鏅鸿兘鎸囨暟 vs 姣?Token 浠锋牸**鐨勫笗绱墭鍓嶆部. 渚嬪, O3 鍙兘闈炲父寮哄ぇ, 浣嗕篃闈炲父鏄傝吹; 鑰屾煇浜涙ā鍨嬪彲鑳芥€т环姣旀洿楂?

> **鍏抽敭娲炲療**: 鍙湅鍑嗙‘鐜囨槸涓嶅鐨? 杩樿鑰冭檻**鎴愭湰**!

### 1.4 鐢ㄦ埛閫夋嫨鏁版嵁

![OpenRouter 娴侀噺鎺掑悕](images/l12-openrouter.png)

鍙︿竴绉嶆€濊矾: 濡傛灉浜轰滑鎰挎剰**浠樿垂浣跨敤**鏌愪釜妯″瀷, 閭ｅ畠鍙兘灏辨槸"濂界殑".**[OpenRouter](https://openrouter.ai/rankings)** 鍩轰簬瀹為檯娴侀噺 (鍙戦€佸埌鍚勬ā鍨嬬殑 Token 鏁? 鐢熸垚鎺掑悕.

### 1.5 浜虹被鍋忓ソ鎺掑悕

![Chatbot Arena 鎺掕姒淽(images/l12-chatbot-arena-leaderboard.png)

**[Chatbot Arena](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)** 鏄洰鍓嶆渶娴佽鐨勪汉绫诲亸濂借瘎浼板钩鍙颁箣涓€, 閫氳繃閰嶅姣旇緝璁＄畻 ELO 璇勫垎.

### 1.6 绀句氦濯掍綋姘涘洿 (Vibes)

闄や簡纭寚鏍? 杩樻湁"姘涘洿"鈥斺€斾汉浠湪 X (Twitter) 涓婂垎浜殑閰风偒绀轰緥. 杩欎篃鏄瘎浼版ā鍨嬭兘鍔涚殑涓€绉嶉潪姝ｅ紡鏁版嵁鏉ユ簮.

---

## 2. 濡備綍鎬濊€冭瘎浼? 鐩殑鍐冲畾鏂规硶

璁插笀寮鸿皟: **璇勪及娌℃湁"鍞竴鐪熺悊" (No One True Evaluation)**, 瀹冨彇鍐充簬浣犺瘯鍥惧洖绛斾粈涔堥棶棰?

### 2.1 璇勪及鐨勪笉鍚岀洰鐨?

| 璇勪及鑰呰韩浠?               | 璇勪及鐩殑       | 绀轰緥闂                                      |
| ------------------------- | -------------- | --------------------------------------------- |
| **鐢ㄦ埛/浼佷笟**       | 璐拱鍐崇瓥       | Claude vs Gemini vs O3, 鍝釜閫傚悎鎴戠殑瀹㈡湇鍦烘櫙? |
| **鐮旂┒鑰?*          | 娴嬮噺鍘熷鑳藉姏   | 琛￠噺"鏅鸿兘"姘村钩, AI 鏄惁鍦ㄨ繘姝?                |
| **鏀跨瓥鍒跺畾鑰?浼佷笟** | 鐞嗚В鏀剁泭涓庨闄?| 妯″瀷甯︽潵鐨勪环鍊煎拰鍗卞鍒嗗埆鏄粈涔?               |
| **妯″瀷寮€鍙戣€?*      | 鑾峰彇鏀硅繘鍙嶉   | 璇勪及浣滀负寮€鍙戝惊鐜殑鍙嶉淇″彿                    |

姣忕鎯呭喌涓? 閮芥湁涓€涓?*鎶借薄鐩爣**(Abstract Goal) 闇€瑕佽杞寲涓?*鍏蜂綋璇勪及** (Concrete Evaluation). 閫夋嫨浠€涔堣瘎浼版柟寮? 鍙栧喅浜庝綘鐨勭洰鏍?

### 2.2 璇勪及妗嗘灦: 鍥涗釜鏍稿績闂

1. **杈撳叆 (Inputs)**: Prompt 浠庡摢閲屾潵? 瑕嗙洊鍝簺鐢ㄤ緥? 鏄惁鍖呭惈鍥伴毦鐨勯暱灏炬儏鍐? 杈撳叆鏄惁闇€瑕侀€傚簲妯″瀷 (濡傚杞璇??
2. **妯″瀷璋冪敤 (How to Call LM)**: 浣跨敤浠€涔?Prompting 绛栫暐 (Zero-shot, Few-shot, Chain-of-Thought)? 鏄惁浣跨敤宸ュ叿銆丷AG? 璇勪及鐨勬槸**妯″瀷鏈韩**杩樻槸**鏁翠釜 Agent 绯荤粺**?
3. **杈撳嚭璇勪及 (How to Evaluate Outputs)**: 鍙傝€冪瓟妗堟槸鍚︽棤璇? 浣跨敤浠€涔堟寚鏍?(Pass@k)? 濡備綍鑰冭檻鎴愭湰? 濡備綍澶勭悊涓嶅绉伴敊璇?(濡傚尰鐤楀満鏅殑骞昏)? 濡備綍璇勪及寮€鏀惧紡鐢熸垚?
4. **缁撴灉瑙ｈ (How to Interpret)**: 91% 鎰忓懗鐫€浠€涔? 鑳藉惁閮ㄧ讲? 濡備綍璇勪及娉涘寲鑳藉姏 (鑰冭檻璁粌-娴嬭瘯閲嶅彔)? 璇勪及鐨勬槸妯″瀷杩樻槸鏂规硶?

> **鎬荤粨**: 鍋氳瘎浼版椂鏈?*澶ч噺闂**闇€瑕佹€濊€? 瀹冪粷涓嶄粎浠呮槸"璺戜釜鑴氭湰"閭ｄ箞绠€鍗?

---

## 3. 鍥版儜搴?(Perplexity) 璇勪及

### 3.1 鍩烘湰姒傚康

鍥為【: 璇█妯″瀷鏄?Token 搴忓垪涓婄殑姒傜巼鍒嗗竷 $p(x)$.

**鍥版儜搴?(Perplexity)** 琛￠噺妯″瀷鏄惁瀵规煇涓暟鎹泦 $D$ 鍒嗛厤浜嗛珮姒傜巼:

$$
\text{Perplexity} = \left( \frac{1}{p(D)} \right)^{1/|D|}
$$

鍦ㄩ璁粌涓? 鎴戜滑**鏈€灏忓寲璁粌闆嗙殑鍥版儜搴?*. 鑷劧鍦? 璇勪及鏃舵垜浠?*娴嬮噺娴嬭瘯闆嗙殑鍥版儜搴?*.

### 3.2 浼犵粺璇█寤烘ā璇勪及

鍦?2010 骞翠唬, 璇█寤烘ā鐮旂┒鐨勬爣鍑嗘祦绋嬫槸:

1. 閫夋嫨涓€涓爣鍑嗘暟鎹泦 (Penn Treebank, WikiText-103, 1 Billion Word Benchmark)
2. 鍦ㄦ寚瀹氳缁冮泦涓婅缁?
3. 鍦ㄦ寚瀹氭祴璇曢泦涓婅瘎浼板洶鎯戝害

杩欐槸 N-gram 鍚戠缁忕綉缁滆繃娓℃湡鐨勪富瑕佽寖寮? 2016 骞?Google 鐨勮鏂囧睍绀轰簡绾?CNN+LSTM 妯″瀷鍦?1 Billion Word Benchmark 涓婂皢鍥版儜搴︿粠 **51.3 闄嶅埌 30.0**.

### 3.3 GPT-2 鐨勮寖寮忚浆鍙?

![GPT-2 鍥版儜搴﹀疄楠宂(images/l12-gpt2-perplexity.png)

GPT-2 鏀瑰彉浜嗘父鎴忚鍒?

- 鍦?**40GB 鐨?WebText** 涓婅缁?(Reddit 閾炬帴鐨勭綉绔?
- **闆跺井璋?(Zero-shot)**, 鐩存帴鍦ㄤ紶缁熷洶鎯戝害鍩哄噯涓婅瘎浼?
- 杩欐槸**鍒嗗竷澶栬瘎浼?* (Out-of-Distribution), 浣嗙悊蹇垫槸璁粌鏁版嵁瓒冲骞挎硾

**缁撴灉**: 鍦ㄥ皬鏁版嵁闆?(Penn Treebank) 涓婅秴瓒?SOTA, 浣嗗湪澶ф暟鎹泦 (1 Billion Words) 涓婁粛钀藉悗鈥斺€旇繖璇存槑褰撴暟鎹泦瓒冲澶ф椂, 鐩存帴鍦ㄨ鏁版嵁闆嗕笂璁粌浠嶇劧鏇村ソ.

鑷?GPT-2/3 浠ユ潵, 璇█妯″瀷璁烘枃鏇村杞悜**涓嬫父浠诲姟鍑嗙‘鐜?* (Downstream Task Accuracy).

### 3.4 鍥版儜搴︿负浣曚粛鐒舵湁鐢?

1. **姣斾笅娓镐换鍔″噯纭巼鏇村钩婊?*: 鎻愪緵姣忎釜 Token 鐨勭粏绮掑害姒傜巼, 閫傚悎鎷熷悎 Scaling Law
2. **鍏锋湁鏅亶鎬?*: 鍏虫敞姣忎竴涓?Token, 鑰屼换鍔″噯纭巼鍙兘閬楁紡鏌愪簺缁嗚妭 (鍙兘"瀵逛簡浣嗙悊鐢遍敊浜?)
3. **鍙敤浜庝笅娓镐换鍔＄殑 Scaling Law**: 鍙互娴嬮噺**鏉′欢鍥版儜搴?* (Conditional Perplexity), 鐩存帴閽堝鐗瑰畾浠诲姟鎷熷悎鏇茬嚎

**娉ㄦ剰**: 鍥版儜搴︿篃鍙互閫氳繃鏉′欢鏂瑰紡搴旂敤浜庝笅娓镐换鍔♀€斺€旂粰瀹?Prompt, 璁＄畻绛旀鐨勬鐜?

### 3.5 鍥版儜搴︾殑闄烽槺 (Leaderboard 璀﹀憡)

濡傛灉浣犲湪杩愯惀涓€涓帓琛屾:

- 瀵逛簬**浠诲姟鍑嗙‘鐜?*: 鍙渶浠庨粦鐩掓ā鍨嬭幏鍙栫敓鎴愯緭鍑? 鐒跺悗鐢ㄤ綘鐨勪唬鐮佽瘎浼?
- 瀵逛簬**鍥版儜搴?*: 闇€瑕佹ā鍨嬫彁渚涙鐜? 骞?*淇′换瀹冧滑鍔犺捣鏉ョ瓑浜?1**

濡傛灉妯″瀷鏈?Bug (姣斿瀵规墍鏈?Token 閮借緭鍑烘鐜?0.8), 瀹冧細鐪嬭捣鏉ラ潪甯稿ソ, 浣嗛偅涓嶆槸鏈夋晥鐨勬鐜囧垎甯?

### 3.6 鍥版儜搴︽渶澶у寲涓讳箟鑰?(Perplexity Maximalist) 鐨勮鐐?

璁剧湡瀹炲垎甯冧负 $t$, 妯″瀷涓?$p$:

- 鏈€浣冲洶鎯戝害鏄?$H(t)$ (淇℃伅鐔?, 褰撲笖浠呭綋 $p = t$
- 濡傛灉浣犺揪鍒颁簡 $t$, 浣犲氨瑙ｅ喅浜嗘墍鏈変换鍔?鈫?**AGI**
- 鍥犳, 閫氳繃涓嶆柇鍘嬩綆鍥版儜搴? 鏈€缁堜細杈惧埌 AGI

**鍙嶉┏**: 杩欏彲鑳戒笉鏄渶楂樻晥鐨勮矾寰? 鍥犱负浣犲彲鑳藉湪鍘嬩綆鍒嗗竷涓?涓嶉噸瑕?鐨勯儴鍒?

### 3.7 涓庡洶鎯戝害鐩歌繎鐨勪换鍔?

**LAMBADA** (瀹屽舰濉┖):

![LAMBADA 绀轰緥](images/l12-lambada.png)

缁欏畾闇€瑕侀暱涓婁笅鏂囩悊瑙ｇ殑鍙ュ瓙, 棰勬祴鏈€鍚庝竴涓瘝.

**HellaSwag** (甯歌瘑鎺ㄧ悊):

![HellaSwag 绀轰緥](images/l12-hellaswag.png)

缁欏畾鍙ュ瓙, 閫夋嫨鏈€鍚堢悊鐨勭画鍐? 鏈川涓婃槸姣旇緝鍊欓€夐」鐨?Likelihood.

> **娉ㄦ剰**: WikiHow 鏄竴涓綉绔? 铏界劧 HellaSwag 鏁版嵁缁忚繃澶勭悊, 浣嗗鏋滀綘璁块棶 WikiHow, 浼氱湅鍒颁笌鏁版嵁闆嗛潪甯哥浉浼肩殑鍐呭鈥斺€旇繖娑夊強**璁粌-娴嬭瘯閲嶅彔**闂.

---

## 4. 鐭ヨ瘑鍩哄噯 (Knowledge Benchmarks)

### 4.1 MMLU (Massive Multitask Language Understanding)

![MMLU 绀轰緥](images/l12-mmlu.png)

- **鍙戝竷骞翠唤**: 2020 (GPT-3 鍙戝竷鍚庝笉涔?
- **鍐呭**: 57 涓绉? 鍏ㄩ儴涓哄閫夐
- **鏁版嵁鏉ユ簮**: "鐢辩爺绌剁敓鍜屾湰绉戠敓浠庣綉涓婂厤璐硅祫婧愭敹闆?
- **褰撴椂璇勪及鏂瑰紡**: GPT-3 Few-shot Prompting
- **褰撴椂 SOTA**: GPT-3 绾?*45%**

> **璁插笀鍚愭Ы**: 灏界鍚嶅瓧鍙?璇█鐞嗚В", 浣嗗畠鏇村儚鏄湪娴嬭瘯**鐭ヨ瘑璁板繂** (Knowledge) 鑰岄潪璇█鑳藉姏. 鎴戠殑璇█鐞嗚В鑳藉姏涓嶉敊, 浣嗘垜鍙兘鍋氫笉濂?MMLU, 鍥犱负寰堝閮芥槸鎴戜笉鐭ラ亾鐨勪簨瀹?(濡傚浜ゆ斂绛?.

**Few-shot Prompt 鏍煎紡**:

```
The following are multiple choice questions (with answers) about [subject].

[Example 1]
...
Answer: A

[Example 2]
...
Answer: C

[Actual Question]
...
Answer:
```

**HELM 鍙鍖?*: [HELM MMLU](https://crfm.stanford.edu/helm/mmlu/latest/) 鍏佽浣犳煡鐪嬪悇妯″瀷鍦ㄥ悇瀛愪换鍔′笂鐨勮〃鐜? 骞舵繁鍏ュ埌鍏蜂綋闂鍜岄娴?

**褰撳墠鐘舵€?*: 椤剁骇妯″瀷 (Claude, O3) 宸茶揪**90%+**. 宸茶璁や负鎺ヨ繎**楗卞拰**, 鍙兘琚繃搴︿紭鍖?

> **閲嶈鍖哄垎**: MMLU 鏈€鍒濊璁＄敤浜庤瘎浼?*鍩虹妯″瀷**(Base Model), 鑰岀幇鍦ㄥ父鐢ㄤ簬璇勪及**鎸囦护寰皟妯″瀷**. 瀵逛簬鍩虹妯″瀷, 濡傛灉瀹冨湪娌℃湁涓撻棬璁粌鐨勬儏鍐典笅灏辫兘鍋氬ソ MMLU, 璇存槑瀹冨叿鏈夎壇濂界殑娉涘寲鑳藉姏. 浣嗗鏋滀綘涓撻棬閽堝杩?57 涓绉戣繘琛屽井璋? 楂樺垎鍙兘鍙弽鏄犱簡瀵瑰熀鍑嗙殑杩囧害鎷熷悎, 鑰岄潪鐪熸鐨勯€氱敤鏅鸿兘.

### 4.2 MMLU-Pro

![MMLU-Pro 绀轰緥](images/l12-mmlu-pro.png)

- **鍙戝竷骞翠唤**: 2024
- **鏀硅繘**:
  - 绉婚櫎浜?MMLU 涓殑鍣０/绠€鍗曢棶棰?
  - 閫夐」浠?4 涓鍔犲埌 **10 涓?*
  - 浣跨敤 **Chain-of-Thought** 璇勪及 (缁欐ā鍨嬫洿澶氭€濊€冩満浼?
- **缁撴灉**: 妯″瀷鍑嗙‘鐜囦笅闄?*16%-33%** (涓嶅啀楗卞拰)

**HELM 鍙鍖?*: [HELM MMLU-Pro](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro)

### 4.3 GPQA (Graduate-Level Google-Proof Q&A)

![GPQA 鏁版嵁鍒涘缓娴佺▼](images/l12-gpqa.png)

- **鍙戝竷骞翠唤**: 2023
- **鐗圭偣**:
  - 鐢?**61 浣?PhD 鎵垮寘鍟?* (鏉ヨ嚜 Upwork) 鎾板啓闂
  - 缁忚繃涓撳楠岃瘉銆侀潪涓撳娴嬭瘯鐨勫闃舵娴佺▼
  - **Google-Proof**: 闈炰笓瀹剁敤 Google 鎼滅储 30 鍒嗛挓涔熷彧鑳借揪鍒?~34% 鍑嗙‘鐜?
- **PhD 涓撳鍑嗙‘鐜?*:**65%**
- **褰撴椂 SOTA (GPT-4)**:**39%**
- **褰撳墠 SOTA (O3)**:**~75%** (涓€骞村唴鎻愬崌鏄捐憲!)

**HELM 鍙鍖?*: [HELM GPQA](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa)

> **璇惧爞闂瓟**:
>
> - Q: GPQA 鏄?Google-Proof, 浣嗘€庝箞鐭ラ亾 O3 娌℃湁鍋峰伔璋冪敤浜掕仈缃?
> - A: 闇€瑕佷娇鐢ㄦ槑纭鐢ㄦ悳绱㈢殑 API 绔偣, 骞朵俊浠昏繖鏄湡鐨?

### 4.4 HLE (Humanity's Last Exam)

<!-- ![HLE 绀轰緥](images/l12-hle-examples.png) -->

- **鍙戝竷骞翠唤**: 2025
- **鐗圭偣**:
  - **澶氭ā鎬?* (鍖呭惈鍥惧儚)
  - 2500 涓棶棰? 澶氬绉? 澶氶€夐 + 绠€绛旈
  - **$500K 濂栭噾姹?+ 璁烘枃缃插悕**婵€鍔卞嚭棰?
  - 鐢卞墠娌?LLM 杩囨护"澶畝鍗?鐨勯棶棰?
  - 澶氶樁娈靛鏍?

<!-- ![HLE 鏁版嵁娴佺▼](images/l12-hle-pipeline.png) -->

<!-- ![HLE 缁撴灉](images/l12-hle-results.png) -->

- **褰撳墠 SOTA (O3)**:**~20%**
- **鏈€鏂版帓琛屾**: [agi.safe.ai](https://agi.safe.ai/)

> **璇惧爞鎵硅瘎**: 鍏紑寰侀泦闂浼氬甫鏉?*鍋忚**鈥斺€斿搷搴旇€呭線寰€鏄偅浜涘凡缁忛潪甯哥啛鎮?LLM 鐨勪汉, 鍙兘鍑虹殑棰樼洰闈炲父"LLM-adversarial" (鍒绘剰閽堝 LLM 寮辩偣).

---

## 5. 鎸囦护閬靛惊鍩哄噯 (Instruction Following Benchmarks)

鍒扮洰鍓嶄负姝? 鎴戜滑璇勪及鐨勯兘鏄粨鏋勫寲浠诲姟 (澶氶€夐/绠€绛旈). 浣?ChatGPT 浠ユ潵, **鎸囦护閬靛惊** (Instruction Following) 鎴愪负鏍稿績鑳藉姏: 鐢ㄦ埛鐩存帴鎻忚堪浠诲姟, 妯″瀷鎵ц.

**鏍稿績鎸戞垬**: 濡備綍璇勪及**寮€鏀惧紡鍥炲**?

### 5.1 Chatbot Arena

<!-- ![Chatbot Arena 鎺掕姒淽(images/l12-chatbot-arena-leaderboard.png) -->

**宸ヤ綔鍘熺悊**:

1. 缃戠粶鐢ㄦ埛杈撳叆 Prompt
2. 鑾峰緱涓や釜**鍖垮悕**妯″瀷鐨勫洖澶?
3. 鐢ㄦ埛閫夋嫨鍝釜鏇村ソ
4. 鍩轰簬閰嶅鎺掑悕璁＄畻 **ELO 璇勫垎**

**浼樼偣**:

- **鍔ㄦ€?(Live)**: 涓嶆槸闈欐€佸熀鍑? 濮嬬粓鏈夋柊鏁版嵁娴佸叆
- **鏄撲簬娣诲姞鏂版ā鍨?*: ELO 绯荤粺澶╃劧鏀寔

**闂涓庝簤璁?*:

- **The Leaderboard Illusion**(璁烘枃): 鎻ず浜嗘煇浜涙ā鍨嬫彁渚涜€呰幏寰椾簡**鐗规潈璁块棶鏉冮檺** (澶氭鎻愪氦), 鍗忚瀛樺湪闂
- **鐢ㄦ埛鍒嗗竷鍋忚**: "缃戠粶涓婇殢鏈虹殑浜?浠ｈ〃浠€涔堝垎甯?

**[娣卞叆鎺㈣: Chatbot Arena 涓?ELO 璇勫垎绯荤粺](./Lecture12-Chatbot-Arena.md)**

### 5.2 IFEval (Instruction-Following Eval)

<!-- ![IFEval 绾︽潫绫诲瀷](images/l12-ifeval-categories.png) -->

- **宸ヤ綔鍘熺悊**: 缁欐寚浠ゆ坊鍔?*鍚堟垚绾︽潫** (濡?鍥炵瓟涓嶈秴杩?10 涓瘝", "蹇呴』鍖呭惈鏌愪釜璇?, "浣跨敤鐗瑰畾鏍煎紡")
- **浼樼偣**: 绾︽潫鍙互**鑷姩楠岃瘉** (鐢ㄧ畝鍗曡剼鏈?
- **缂虹偣**: 鍙瘎浼版槸鍚﹂伒寰害鏉?**涓嶈瘎浼板洖澶嶇殑璇箟璐ㄩ噺**; 绾︽潫鏈変簺浜轰负閫犱綔

**HELM 鍙鍖?*: [HELM IFEval](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/ifeval)

### 5.3 AlpacaEval

<!-- ![AlpacaEval 鎺掕姒淽(images/l12-alpacaeval-leaderboard.png) -->

- **鍐呭**: 805 鏉℃潵鑷笉鍚屾潵婧愮殑鎸囦护
- **鎸囨爣**: 鐢?*GPT-4 Preview 浣滀负瑁佸垽**, 璁＄畻鐩稿浜?GPT-4 Preview 鐨?*鑳滅巼**
- **娼滃湪鍋忚**: GPT-4 浣滀负瑁佸垽鍙兘鍋忓悜鑷繁

**鍘嗗彶瓒ｉ椈**: 鏃╂湡鐗堟湰琚?Gaming鈥斺€斾竴浜涘皬妯″瀷閫氳繃鐢熸垚鏇撮暱鐨勫洖澶嶈幏寰楅珮鍒? 鍥犱负 GPT-4 鍋忓ソ闀垮洖澶? 鍚庢潵寮曞叆浜?*闀垮害鏍℃**鐗堟湰.

**鐩稿叧鎬?*: 涓?Chatbot Arena 鐩稿叧鎬ц緝楂?(鎻愪緵绫讳技淇℃伅, 浣嗘洿鑷姩鍖?鍙鐜?.

### 5.4 WildBench

<!-- ![WildBench](images/l12-wildbench.png) -->

- **鏁版嵁鏉ユ簮**: 浠?100 涓囩湡瀹炰汉鏈哄璇濅腑鎶藉彇 1024 涓ず渚?
- **璇勪及鏂瑰紡**: GPT-4 Turbo 浣滀负瑁佸垽, 浣跨敤**Checklist** (绫讳技 CoT for Judging) 纭繚璇勪及鍏ㄩ潰
- **涓?Chatbot Arena 鐩稿叧鎬?*:**0.95** (闈炲父楂?

**HELM 鍙鍖?*: [HELM WildBench](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench)

> **鏈夎叮瑙傚療**: 璇勪及鍩哄噯鐨?璇勪及" (Evaluation of Evaluation) 浼间箮鏄湅瀹冧笌 Chatbot Arena 鐨勭浉鍏虫€?

---

## 6. Agent 鍩哄噯 (Agent Benchmarks)

璁稿浠诲姟闇€瑕?*宸ュ叿浣跨敤**(濡傝繍琛屼唬鐮併€佽闂簰鑱旂綉) 鍜?*澶氳疆杩唬**. 杩欏氨鏄?*Agent** 鐧诲満鐨勫湴鏂?

**Agent = 璇█妯″瀷 + Agent 鑴氭墜鏋?* (鍐冲畾濡備綍璋冪敤 LM 鐨勭▼搴忛€昏緫)

### 6.1 SWE-Bench

<!-- ![SWE-Bench](images/l12-swebench.png) -->

- **浠诲姟**: 缁欏畾浠ｇ爜搴?+ GitHub Issue 鎻忚堪, 鎻愪氦涓€涓?*Pull Request**
- **鏁版嵁**: 12 涓?Python 浠撳簱鐨?2294 涓换鍔?
- **璇勪及鎸囨爣**:**鍗曞厓娴嬭瘯鏄惁閫氳繃**

**娴佺▼**: Issue 鎻忚堪 鈫?Agent 闃呰浠ｇ爜 鈫?鐢熸垚 Patch 鈫?杩愯娴嬭瘯

### 6.2 CyBench (缃戠粶瀹夊叏)

<!-- ![CyBench 浠诲姟](images/l12-cybench.png) -->

- **浠诲姟**:**Capture The Flag (CTF)** 椋庢牸鐨勬笚閫忔祴璇?
- **鍐呭**: 40 涓?CTF 浠诲姟
- **闅惧害琛￠噺**: 浣跨敤浜虹被**棣栨瑙ｅ喅鏃堕棿 (First-Solve Time)** 浣滀负闅惧害鎸囨爣

<!-- ![CyBench Agent 鏋舵瀯](images/l12-cybench-agent.png) -->

**Agent 鏋舵瀯** (鏍囧噯妯″紡):

1. LM 鎬濊€冨苟鍒跺畾璁″垝
2. 鐢熸垚鍛戒护
3. 鎵ц鍛戒护
4. 鏇存柊 Agent 璁板繂
5. 杩唬鐩村埌鎴愬姛鎴栬秴鏃?

<!-- ![CyBench 缁撴灉](images/l12-cybench-results.png) -->

- **褰撳墠鍑嗙‘鐜?*: ~20%
- **浜偣**: O3 鑳借В鍐充汉绫诲洟闃熼渶瑕?42 鍒嗛挓鎵嶈兘瑙ｅ喅鐨勯棶棰?(鏈€闀挎寫鎴橀渶 24 灏忔椂)

**[娣卞叆鎺㈣: Agent 鍩哄噯: SWE-Bench 涓?CyBench](./Lecture12-Agent-Benchmarks.md)**

### 6.3 MLE-Bench (Kaggle)

<!-- ![MLE-Bench 浠诲姟](images/l12-mlebench.png) -->

- **浠诲姟**: 75 涓?Kaggle 绔炶禌, Agent 闇€瑕佺紪鍐欎唬鐮併€佽缁冩ā鍨嬨€佽皟鍙傘€佹彁浜?
- **璇勪及**: 鏄惁鑾峰緱濂栫墝 (鏌愪釜鎬ц兘闃堝€?

<!-- ![MLE-Bench 缁撴灉](images/l12-mlebench-results.png) -->

- **褰撳墠鍑嗙‘鐜?*: 鍗充娇鏈€濂界殑妯″瀷, 鑾峰緱浠讳綍濂栫墝鐨勫噯纭巼涔?*< 20%**

---

## 7. 绾帹鐞嗗熀鍑?(Pure Reasoning Benchmarks)

涔嬪墠鎵€鏈変换鍔￠兘闇€瑕?*璇█鐭ヨ瘑**鍜?*涓栫晫鐭ヨ瘑**. 鑳藉惁灏?*鎺ㄧ悊**浠庣煡璇嗕腑鍒嗙鍑烘潵?

> **璁虹偣**: 鎺ㄧ悊鎹曟崏浜嗕竴绉嶆洿绾补鐨?鏅鸿兘"褰㈠紡, 涓嶄粎浠呮槸璁板繂浜嬪疄.

### 7.1 ARC-AGI

**ARC-AGI** 鐢?Fran莽ois Chollet 浜?2019 骞存彁鍑?(鏃╀簬褰撳墠 LLM 娴疆).

**浠诲姟**: 缁欏畾杈撳叆杈撳嚭妯″紡, 鎺ㄦ柇瑙勫垯骞跺～鍏呮祴璇曟渚?

**鐗圭偣**:

- **鏃犺瑷€**, 绾瑙夋ā寮忚瘑鍒?
- 璁捐涓轰汉绫诲鏄撱€佹満鍣ㄥ洶闅?(涓庝紶缁熷熀鍑嗙浉鍙?

**ARC-AGI-1 缁撴灉**:

- 浼犵粺 LLM (GPT-4o): 鈮?**0%**
- O3: 鈮?**75%** (浣跨敤澶ч噺璁＄畻, 姣忎釜浠诲姟鍙兘鑺辫垂鏁扮櫨缇庡厓)

**ARC-AGI-2**: 鏇撮毦, 鐩墠鍑嗙‘鐜囦粛鐒跺緢浣?

> **ARC Prize 缃戠珯**: [arcprize.org](https://arcprize.org/arc-agi)

---

## 8. 瀹夊叏鍩哄噯 (Safety Benchmarks)

鍦ㄦ苯杞﹁涓氭湁纰版挒娴嬭瘯, 椋熷搧琛屼笟鏈夊崼鐢熻瘎绾? **AI 鐨勫畨鍏ㄨ瘎浼板簲璇ユ槸浠€涔堟牱鐨?**

鐩墠娌℃湁鏄庣‘绛旀鈥斺€擜I 杩樺お鏃╂湡, 浜轰滑杩樺湪鎺㈢储"瀹夊叏"鎰忓懗鐫€浠€涔?

### 8.1 HarmBench

- **鍐呭**: 510 绉嶈繚鍙嶆硶寰嬫垨瑙勮寖鐨?*鏈夊琛屼负**
- **璇勪及**: 妯″瀷鏄惁鎷掔粷鎵ц鏈夊鎸囦护

**HELM 鍙鍖?*: [HELM HarmBench](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench)

> **瑙傚療**: 涓嶅悓妯″瀷鎷掔粷鐜囧樊寮傚緢澶? 渚嬪, 鏌愪簺妯″瀷 (濡?DeepSeek V3) 鍦ㄦ煇浜涙湁瀹宠姹備笂閬典粠鐜囪緝楂?

### 8.2 AIR-Bench

- **鐗圭偣**: 灏?瀹夊叏"姒傚康閿氬畾鍦?*娉曡妗嗘灦鍜屽叕鍙告斂绛?*涓?
- **鍐呭**: 鍩轰簬娉曞緥鍜屾斂绛栨瀯寤?314 涓闄╃被鍒? 5694 涓?Prompt
- **浼樼偣**: 鏇存湁渚濇嵁 (Grounded), 鑰岄潪浠绘剰瀹氫箟"瀹夊叏"

**HELM 鍙鍖?*: [HELM AIR-Bench](https://crfm.stanford.edu/helm/air-bench/latest/)

### 8.3 瓒婄嫳 (Jailbreaking)

璇█妯″瀷琚缁冩嫆缁濇湁瀹虫寚浠? 浣嗗彲浠ヨ**缁曡繃**.

**GCG Attack (Greedy Coordinate Gradient)**:

<!-- ![GCG 鏀诲嚮绀轰緥](images/l12-gcg-examples.png) -->

- **鏂规硶**: 鑷姩浼樺寲 Prompt 鍚庣紑浠ョ粫杩囧畨鍏ㄦ満鍒?
- **鎯婁汉鍙戠幇**: 鍦ㄥ紑婧愭ā鍨?(Llama) 涓婁紭鍖栫殑鍚庣紑鍙互**杩佺Щ**鍒伴棴婧愭ā鍨?(GPT-4)

> **鎰忎箟**: 鍗充娇妯″瀷琛ㄩ潰涓婂畨鍏? 瓒婄嫳鏀诲嚮琛ㄦ槑搴曞眰**鑳藉姏**浠嶇劧瀛樺湪 (骞跺彲鑳借閲婃斁).

### 8.4 瀹夊叏 vs 鑳藉姏: 涓€涓鏉傜殑鍏崇郴

**涓や釜缁村害**:

- **鑳藉姏 (Capability)**: 妯″瀷鏄惁*鑳藉*鍋氭煇浜?
- **鍊惧悜 (Propensity)**: 妯″瀷鏄惁*鎰挎剰*鍋氭煇浜?

**API 妯″瀷**: 鍙渶鎺у埗 Propensity (鍙互鎷掔粷)
**寮€婧愭ā鍨?*: Capability 涔熼噸瑕? 鍥犱负瀹夊叏鎺柦鍙互閫氳繃**寰皟杞绘澗绉婚櫎**

**鍙岄噸鐢ㄩ€?(Dual-Use)**: CyBench 鏄畨鍏ㄨ瘎浼拌繕鏄兘鍔涜瘎浼?

- 鎭舵剰: 鐢?Agent 榛戝叆绯荤粺
- 鍠勬剰: 鐢?Agent 杩涜娓楅€忔祴璇曚繚鎶ょ郴缁?

### 8.5 棰勯儴缃叉祴璇?

缇庡浗鍜岃嫳鍥界殑 AI 瀹夊叏鐮旂┒鎵€涓庢ā鍨嬪紑鍙戣€?(Anthropic, OpenAI 绛? 寤虹珛浜?*鑷効鍗忚**:

- 鍏徃鍦ㄥ彂甯冨墠缁欏畨鍏ㄧ爺绌舵墍鏃╂湡璁块棶鏉冮檺
- 瀹夊叏鐮旂┒鎵€杩愯璇勪及骞剁敓鎴愭姤鍛?
- 鐩墠鏄?*鑷効鐨?*, 鏃犳硶寰嬪己鍒跺姏

---

## 9. 鐪熷疄鎬?(Realism)

璇█妯″瀷鍦ㄥ疄璺典腑琚ぇ閲忎娇鐢?

- OpenAI: 姣忓ぉ 1000 浜?Token
- Cursor: 10 浜胯浠ｇ爜

<!-- ![OpenAI 鏃ユ祦閲廬(images/openai-100b-tokens.png) -->

鐒惰€? 澶у鏁板熀鍑?(濡?MMLU) 涓?*鐪熷疄浣跨敤鍦烘櫙**鐩稿幓鐢氳繙.

### 9.1 涓ょ Prompt

1. **Quizzing (娴嬭瘯)**: 鐢ㄦ埛**鐭ラ亾绛旀**, 璇曞浘娴嬭瘯绯荤粺 (濡傛爣鍑嗗寲鑰冭瘯)
2. **Asking (璇㈤棶)**: 鐢ㄦ埛**涓嶇煡閬撶瓟妗?*, 璇曞浘浣跨敤绯荤粺鑾峰彇淇℃伅

**Asking** 鏇寸湡瀹? 鑳戒负鐢ㄦ埛甯︽潵浠峰€? 鏍囧噯鍖栬€冭瘯鏄剧劧涓嶅鐪熷疄.

### 9.2 Clio (Anthropic)

<!-- ![Clio 鍒嗘瀽琛╙(images/l12-clio-table4.png) -->

Anthropic 浣跨敤璇█妯″瀷鍒嗘瀽**鐪熷疄鐢ㄦ埛鏁版嵁**, 鎻ず浜轰滑瀹為檯浣跨敤 Claude 鐨勬柟寮?**缂栫爜**鏄渶甯歌鐨勭敤閫斾箣涓€.

> **鎰忎箟**: 涓€鏃﹂儴缃茬郴缁? 浣犲氨鏈変簡鐪熷疄鏁版嵁, 鍙互鍦ㄧ湡瀹炵敤渚嬩笂璇勪及.

### 9.3 MedHELM

浠ュ線鐨勫尰鐤楀熀鍑嗗熀浜庢爣鍑嗗寲鑰冭瘯. **MedHELM** 涓嶅悓:

- 浠?**29 浣嶄复搴婂尰鐢?*澶勫緛闆?*121 涓复搴婁换鍔?*
- 娣峰悎鍏紑鍜岀鏈夋暟鎹泦

**HELM 鍙鍖?*: [MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/)

> **鏉冭　**: 鐪熷疄鎬у拰闅愮鏈夋椂鏄煕鐩剧殑. 鐪熷疄鐨勫尰鐤楁暟鎹秹鍙婃偅鑰呴殣绉?

---

## 10. 鏈夋晥鎬?(Validity)

鎴戜滑濡備綍鐭ラ亾璇勪及鏄?*鏈夋晥鐨?*?

### 10.1 璁粌-娴嬭瘯閲嶅彔 (Train-Test Contamination)

**鏈哄櫒瀛︿範 101**: 涓嶈鍦ㄦ祴璇曢泦涓婅缁?

- **Pre-鍩虹妯″瀷鏃朵唬 (ImageNet, SQuAD)**: 鏈夋槑纭畾涔夌殑璁粌/娴嬭瘯鍒嗗壊
- **濡備粖**: 鍦ㄤ簰鑱旂綉涓婅缁? 涓嶅憡璇変綘鏁版嵁鏄粈涔?

**搴斿绛栫暐**:

**璺嚎 1: 浠庢ā鍨嬭涓烘帹鏂噸鍙?*

<!-- ![姹℃煋妫€娴嬫柟娉昡(images/l12-contamination-exchangeability.png) -->

鍒╃敤鏁版嵁鐐圭殑**鍙氦鎹㈡€?* (Exchangeability): 濡傛灉妯″瀷瀵规祴璇曢泦涓煇涓壒瀹氶『搴忚〃鐜板嚭鍋忓ソ (涓庢暟鎹泦椤哄簭鐩稿叧), 鍒欏彲鑳芥槸璁粌杩囦簡.

**璺嚎 2: 榧撳姳鎶ュ憡瑙勮寖**

灏卞儚璁烘枃搴旀姤鍛婄疆淇″尯闂翠竴鏍? 妯″瀷鍙戝竷鑰呭簲鎶ュ憡**璁粌-娴嬭瘯閲嶅彔妫€娴嬬粨鏋?*.

**[娣卞叆鎺㈣: 璁粌-娴嬭瘯姹℃煋 (Train-Test Contamination)](./Lecture12-Contamination.md)**

### 10.2 鏁版嵁闆嗚川閲?

璁稿鍩哄噯瀛樺湪**閿欒**:

- **SWE-Bench Verified**: OpenAI 淇浜?SWE-Bench 涓殑涓€浜涢敊璇?
- **Platinum 鐗堟湰**: 鍒涘缓楂樿川閲忔爣娉ㄧ殑"鐧介噾鐗?鍩哄噯

> **褰卞搷**: 濡傛灉浣犵湅鍒?MATH/GSM8K 鍑嗙‘鐜?90%+, 骞惰涓洪棶棰樺緢闅? 瀹為檯涓婂彲鑳芥湁涓€鍗婃槸鏍囩鍣０. 淇鍚庡垎鏁颁細涓婂崌.

---

## 11. 鎴戜滑鍒板簳鍦ㄨ瘎浼颁粈涔?

鎹㈠彞璇濊: **娓告垙瑙勫垯鏄粈涔?**

| 鏃朵唬                   | 璇勪及瀵硅薄                             | 瑙勫垯                              |
| ---------------------- | ------------------------------------ | --------------------------------- |
| **Pre-鍩虹妯″瀷**|**鏂规硶** (Methods)             | 鏍囧噯鍖栬缁?娴嬭瘯鍒嗗壊, 姣旇緝瀛︿範绠楁硶 |
| **濡備粖**|**妯″瀷/绯荤粺** (Models/Systems) | Anything goes                     |

**渚嬪 (榧撳姳绠楁硶鍒涙柊)**:

<!-- ![NanoGPT Speedrun](images/l12-karpathy-nanogpt-speedrun.png) -->

- **NanoGPT Speedrun**: 鍥哄畾鏁版嵁, 鏈€灏忓寲杈惧埌鐗瑰畾楠岃瘉 Loss 鐨勬椂闂?
- **DataComp-LM**: 缁欏畾鍘熷鏁版嵁闆? 浣跨敤鏍囧噯璁粌娴佺▼鑾峰緱鏈€浣冲噯纭巼 (姣旇緝**鏁版嵁閫夋嫨**绛栫暐)

**鍏抽敭鐐?*: 鏃犺璇勪及浠€涔? 閮介渶瑕?*鏄庣‘瀹氫箟娓告垙瑙勫垯**!

---

## 12. 鎬荤粨: 鏍稿績瑕佺偣

1. **娌℃湁鍞竴姝ｇ‘鐨勮瘎浼?*: 鏍规嵁浣犺娴嬮噺鐨勫唴瀹归€夋嫨璇勪及鏂瑰紡.
2. **濮嬬粓鏌ョ湅鍏蜂綋瀹炰緥鍜岄娴?*: 涓嶈鍙湅鏁板瓧, 娣卞叆鍒板叿浣撻棶棰?
3. **璇勪及鏈夊涓淮搴?*: 鑳藉姏銆佸畨鍏ㄣ€佹垚鏈€佺湡瀹炴€?
4. **鏄庣‘娓告垙瑙勫垯**: 浣犲湪璇勪及**鏂规硶**杩樻槸**妯″瀷/绯荤粺**?

---

## 闄勫綍: 閰嶅浠ｇ爜缁撴瀯 (`lecture_12.py`)

璇剧▼閰嶅浠ｇ爜 `lecture_12.py` 瀹氫箟浜嗘湰璁茬殑瀹屾暣缁撴瀯:

```python
def main():
    text("**Evaluation**: given a**fixed model**, how \"**good**\" is it?")
    what_you_see()              # 1. 浣犳墍鐪嬪埌鐨?(鍩哄噯鍒嗘暟銆佹帓琛屾銆佹皼鍥?
    how_to_think_about_evaluation()  # 2. 濡備綍鎬濊€冭瘎浼?
    perplexity()                # 3. 鍥版儜搴﹁瘎浼?
    knowledge_benchmarks()      # 4. 鐭ヨ瘑鍩哄噯 (MMLU, GPQA, HLE)
    instruction_following_benchmarks()  # 5. 鎸囦护閬靛惊鍩哄噯
    agent_benchmarks()          # 6. Agent 鍩哄噯
    pure_reasoning_benchmarks() # 7. 绾帹鐞嗗熀鍑?(ARC-AGI)
    safety_benchmarks()         # 8. 瀹夊叏鍩哄噯
    realism()                   # 9. 鐪熷疄鎬?
    validity()                  # 10. 鏈夋晥鎬?
    what_are_we_evaluating()    # 11. 鎴戜滑鍦ㄨ瘎浼颁粈涔?
    # 鎬荤粨...
```

浠ｇ爜涓紩鐢ㄧ殑鍏抽敭鍥剧墖鍜岄摼鎺ュ凡鍦ㄦ湰绗旇涓祵鍏?

---

## 鍙傝€冮摼鎺?

- **HELM**: https://crfm.stanford.edu/helm/
- **Chatbot Arena**: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard
- **Artificial Analysis**: https://artificialanalysis.ai/
- **OpenRouter Rankings**: https://openrouter.ai/rankings
- **ARC Prize**: https://arcprize.org/
- **HLE Leaderboard**: https://agi.safe.ai/



