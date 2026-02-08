# Elite Note: Maximal Update Parametrization (muP) - Theory & Derivation

**鏉ユ簮:** CS336 Lecture 11 Elite Extension
**鏍稿績姒傚康:** Tensor Programs, Spectral Conditions, Initialization, Learning Rate Scaling

---

## 1. 涓轰粈涔堟垜浠渶瑕?muP锛?

鍦ㄦ繁搴﹀涔犱腑锛屽綋鎴戜滑鏀瑰彉妯″瀷鐨勫搴︼紙Width, $n$锛夋椂锛屾ā鍨嬬殑璁稿琛屼负閮戒細鍙戠敓鍙樺寲銆傚鏋滀娇鐢ㄦ爣鍑嗙殑鍙傛暟鍖栨柟娉曪紙Standard Parametrization, SP锛夛紝鍗虫棤璁哄搴﹀浣曢兘浣跨敤鐩稿悓鐨勫垵濮嬪寲鏂瑰樊鍜屽涔犵巼锛屾垜浠細鍙戠幇锛?
1.  **婵€娲诲€肩垎鐐告垨娑堝け**锛氶殢鐫€ $n \to \infty$锛屽墠鍚戜紶鎾殑淇″彿鍙兘鍙樺緱鏋佸ぇ鎴栨瀬灏忋€?
2.  **鏇存柊閲忎笉绋冲畾**锛氭搴︽洿鏂板彲鑳藉鑷存潈閲嶅彉鍖栬繃澶э紝鐮村潖璁粌绋冲畾鎬с€?
3.  **瓒呭弬鏁版紓绉?*锛氭渶浼樺涔犵巼 $\eta^*$ 浼氶殢鐫€ $n$ 鐨勫彉鍖栬€屽墽鐑堢Щ鍔ㄣ€傝繖杩娇鎴戜滑鍦ㄨ缁冨ぇ妯″瀷锛堝 GPT-3, Llama锛夋椂锛屽繀椤诲湪鏄傝吹鐨勫ぇ瑙勬ā璁剧疆涓嬮噸鏂版悳绱㈣秴鍙傛暟銆?

**Maximal Update Parametrization (muP)**鐨勭洰鏍囨槸璁捐涓€濂楄鍒欙紝浣垮緱鍦ㄥ搴?$n \to \infty$ 鐨勬瀬闄愪笅锛屾ā鍨嬬殑**鐗瑰緛瀛︿範锛團eature Learning锛?*琛屼负淇濇寔鏈€澶у寲涓旂ǔ瀹氥€傝繖浣垮緱鎴戜滑鍙互浠庝竴涓皬妯″瀷锛圥roxy Model锛夋棤缂濊縼绉绘渶浼樿秴鍙傛暟鍒板ぇ妯″瀷銆?

## 2. 鏍稿績鏁板鍋囪锛氭繁搴︾嚎鎬х綉缁滀笌璋辨潯浠?

涓轰簡鎺ㄥ muP锛岃搴т腑绠€鍖栦簡涓€涓?*娣卞害绾挎€х綉缁?(Deep Linear Network)** 妯″瀷锛?
$$ h_l = W_l h_{l-1} $$
鍏朵腑 $W_l \in \mathbb{R}^{n_l \times n_{l-1}}$銆?

muP 鐨勬帹瀵煎熀浜庝袱涓牳蹇冪殑**璋辨潯浠?(Spectral Conditions)**锛岃繖瀹為檯涓婃槸鐗╃悊瀛︿腑鈥滈噸鏁村寲缇わ紙Renormalization Group锛夆€濇€濇兂鍦ㄧ缁忕綉缁滀腑鐨勫簲鐢細纭繚鐗╃悊閲忓湪灏哄害鍙樻崲涓嬩繚鎸佹湁闄愩€?

### 鏉′欢 A1锛氭縺娲诲€肩ǔ瀹氭€?(Stability of Activations)
**鐩爣**: 鏃犺瀹藉害 $n$ 濡備綍锛屾縺娲诲€煎悜閲忕殑鍧愭爣绾ф暟鍊煎簲淇濇寔 $O(1)$銆傝繖鎰忓懗鐫€婵€娲诲€煎悜閲忕殑 $L_2$ 鑼冩暟搴斾负 $O(\sqrt{n})$銆?

**鎺ㄥ**:
鍋囪杈撳叆 $h_0$ 婊¤冻 $\|h_0\|_2 = \Theta(\sqrt{n_0})$銆?
鏉冮噸 $W_l$ 鍒濆鍖栦负 $N(0, \sigma^2)$銆?
鏍规嵁闅忔満鐭╅樀鐞嗚锛岄珮鏂煩闃电殑绠楀瓙鑼冩暟锛圤perator Norm锛屽嵆鏈€澶у寮傚€硷級闆嗕腑鍦細
$$ \|W_l\|_* \approx \sigma (\sqrt{n_l} + \sqrt{n_{l-1}}) $$
鍓嶅悜浼犳挱鐨勮寖鏁板叧绯讳负锛?
$$ \|h_l\|_2 \le \|W_l\|_* \|h_{l-1}\|_2 $$
涓轰簡缁存寔褰掔撼鍋囪 $\|h_l\|_2 = \Theta(\sqrt{n_l})$锛屾垜浠渶瑕?$\|W_l\|_* = \Theta(1)$銆?
浠ｅ叆绠楀瓙鑼冩暟鍏紡锛?
$$ \sigma (\sqrt{n_l} + \sqrt{n_{l-1}}) = \Theta(1) \implies \sigma \propto \frac{1}{\sqrt{n}} $$
杩欒В閲婁簡涓轰粈涔堟爣鍑嗙殑 Xavier/Kaiming Initialization ($Var \propto 1/n$) 鏄纭殑锛屽洜涓哄畠淇濊瘉浜嗗墠鍚戜俊鍙风殑绋冲畾銆?

### 鏉′欢 A2锛氭洿鏂伴噺绋冲畾鎬?(Stability of Updates)
**鐩爣**: 鍦ㄤ竴姝ユ搴︽洿鏂板悗锛屾縺娲诲€肩殑鍙樺寲閲?$\Delta h_l$ 涔熷簲淇濇寔涓?$h_l$ 鐩稿悓鐨勯噺绾э紝鍗?$O(\sqrt{n})$銆傝繖琚О涓衡€淢aximal Update鈥濓紝鍗虫垜浠湪涓嶅鑷村彂鏁ｇ殑鍓嶆彁涓嬶紝灏藉彲鑳藉ぇ鍦版洿鏂版ā鍨嬨€?

**鎺ㄥ (SGD case)**:
SGD 鐨勬潈閲嶆洿鏂颁负绉?1 鏇存柊锛圧ank-1 Update锛夛細
$$ \Delta W_l = -\eta \nabla_{W_l} \ell = -\eta (\nabla_{h_l} \ell) h_{l-1}^T $$
婵€娲诲€肩殑鍙樺寲閲忥紙蹇界暐楂橀樁椤癸級涓猴細
$$ \Delta h_l \approx \Delta W_l h_{l-1} + W_l \Delta h_{l-1} $$
鎴戜滑闇€瑕佸叧娉ㄧ涓€椤?$\Delta W_l h_{l-1}$ 鐨勯噺绾с€?
浠ｅ叆 $\Delta W_l$锛?
$$ \Delta W_l h_{l-1} = -\eta (\nabla_{h_l} \ell) (h_{l-1}^T h_{l-1}) $$
娉ㄦ剰 $(h_{l-1}^T h_{l-1}) = \|h_{l-1}\|_2^2 = \Theta(n_{in})$銆?
鍥犳锛屾洿鏂伴噺鐨勯噺绾уぇ鑷翠负锛?
$$ \|\Delta h_l\| \propto \eta \cdot n_{in} \cdot \|\nabla_{h_l} \ell\| $$
涓轰簡璁?$\|\Delta h_l\|$ 淇濇寔 $O(\sqrt{n})$锛堝亣璁炬搴﹂」涔熸槸鑹€佺殑锛夛紝鎴戜滑闇€瑕侊細
$$ \eta \cdot n_{in} = \Theta(1) \implies \eta \propto \frac{1}{n_{in}} $$

**鍏抽敭淇 (The Adam Difference)**:
涓婅堪鎺ㄥ鏄熀浜?SGD 鐨勩€傚浜?**Adam**锛屾儏鍐靛畬鍏ㄤ笉鍚屻€?
Adam 鐨勬洿鏂版闀垮ぇ鑷翠粎鍙栧喅浜庢搴︾殑绗﹀彿锛堟垨鏍囧噯鍖栧悗鐨勬搴︼級锛屽畠娑堥櫎浜嗘搴﹀箙搴︾殑褰卞搷銆?
$$ \Delta W_{l, \text{Adam}} \approx -\eta \cdot \text{sign}(\nabla_{W_l} \ell) $$
瀵逛簬楂樻柉鐭╅樀锛岀煩闃靛厓绱犵殑鏇存柊涓嶅啀涓?$n$ 鎴愭姣旓紝鑰屾槸鏇村姞鍧囧寑銆傝缁嗙殑 Tensor Program 鍒嗘瀽琛ㄦ槑锛屼负浜嗚 $\Delta W_l h_{l-1}$ 淇濇寔绋冲畾锛屾垜浠渶瑕侊細
$$ \eta_{\text{Adam}} \propto \frac{1}{n} $$

## 3. 瀹炴柦缁嗚妭锛歋caling Table

鍦?Transformer 鐨勫叿浣撳疄鐜颁腑锛堝 Cerebras-GPT锛夛紝muP 鐨勭缉鏀捐鍒欏涓嬶細

| 鍙傛暟绫诲瀷 | 鍒濆鍖栨柟宸?(Init Var) | Adam 瀛︿範鐜?(LR) | 璇存槑 |
| :--- | :--- | :--- | :--- |
| **Embedding** | 1 (or specific scale) | 1 (Fixed) | Embedding 灞傞€氬父涓嶇缉鏀撅紝鍥犱负瀹冩槸 One-hot 鏌ユ壘 |
| **Matrix Weights** | $1/n$ (Fan-in) | $1/n$ (Fan-in) | 鏍稿績鏉冮噸灞?(Attention, MLP) |
| **Output/Readout** | $1/n^2$ | $1/n$ | 杈撳嚭灞傞€氬父闇€瑕佹洿灏忕殑鍒濆鍖栦互闃叉 Logits 鐖嗙偢 |

**娉ㄦ剰**: Cerebras-GPT 鐨勫疄鐜颁腑鐗瑰埆鎻愬埌锛孲tandard Parametrization (SP) 鐨勫涔犵巼鏄叏灞€甯告暟锛岃€?muP 瑕佹眰姣忓眰锛圥er-Layer锛夌殑瀛︿範鐜囨牴鎹叾杈撳叆缁村害 $n$ 杩涜 $1/n$ 鐨勭缉鏀俱€?

## 4. 涓轰粈涔?"A2" 琚О涓?"Maximal Update"?
濡傛灉瀛︿範鐜囨瘮 muP 寤鸿鐨勬洿灏忥紙渚嬪 $1/n^2$锛夛紝鍒?$\Delta h$ 浼氶殢鐫€ $n \to \infty$ 瓒嬪悜浜?0锛屾ā鍨嬪湪鍒濆鍖栭檮杩戞棤娉曟湁鏁堝涔狅紙Feature Learning 閫€鍖栦负 Kernel Regime/Neural Tangent Kernel锛夈€?
濡傛灉瀛︿範鐜囨瘮 muP 寤鸿鐨勬洿澶э紙渚嬪 $O(1)$锛夛紝鍒?$\Delta h$ 浼氱垎鐐革紝瀵艰嚧璁粌鍙戞暎銆?
鍥犳锛宮uP 瀹氫箟鐨勬槸**鍦ㄤ繚鎸佽缁冪ǔ瀹氱殑鍓嶆彁涓嬶紝鐞嗚涓婂厑璁哥殑鏈€澶у涔犵巼缂╂斁姣斾緥**锛屼粠鑰屾渶澶у寲鐗瑰緛瀛︿範鐨勬晥鐜囥€
