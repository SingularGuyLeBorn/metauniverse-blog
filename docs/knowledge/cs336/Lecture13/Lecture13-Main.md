# Lecture 13: 璁粌鏁版嵁绛栫暐 (Training Data Strategy)

**璇剧▼**: CS336 路**涓婚**: 棰勮缁冦€佷腑鏈熻缁冦€佸悗鏈熻缁冪殑鏁版嵁鏉ユ簮涓庡鐞?

---

## 0. 璇剧▼寮€鍦? 鏁版嵁鏄渶閲嶈鐨?

> **璁插笀 Hot Take**: 鏁版嵁鏄缁冭瑷€妯″瀷鏈€閲嶈鐨勪簨鎯?

涔嬪墠鐨勮搴ц璁轰簡**缁欏畾鏁版嵁**濡備綍璁粌妯″瀷 (鏋舵瀯銆佷紭鍖栧櫒銆乀okenization銆丼caling Laws銆佸苟琛岃绠?. 鐜板湪, 鎴戜滑璁ㄨ**璁粌浠€涔堟暟鎹?*.

**涓轰粈涔堟暟鎹渶閲嶈?** 鐪嬬湅鍏徃瀹為檯鎶湶浠€涔?

![Llama 3 鏁版嵁鎻忚堪 (鍑犱箮娌℃湁淇℃伅)](images/l13-llama3-data.png)

鍗充娇鏄紑婧愭ā鍨?(濡?Llama 3, DeepSeek), 涔熷畬鍏ㄦ姭闇叉灦鏋勭敋鑷宠缁冪粏鑺? 浣?*鍏充簬鏁版嵁鍑犱箮浠€涔堥兘涓嶈**:

> "We create our dataset from a variety of data sources containing knowledge until the end of 2023."

**淇濆瘑鍘熷洜**:

1. **绔炰簤鍔ㄦ€?*: 鏁版嵁鏄牳蹇冪珵浜夊姏
2. **鐗堟潈璐ｄ换**: 涓嶆兂琚捣璇?

---

## 1. 璁粌闃舵姒傝

鏁版嵁宸ヤ綔璐┛璁粌鐨勫悇涓樁娈? 浣嗕晶閲嶇偣涓嶅悓:

| 闃舵                               | 鏁版嵁鐗圭偣                                  | 鐩爣               |
| ---------------------------------- | ----------------------------------------- | ------------------ |
| **Pre-training (棰勮缁?**    | 澶ч噺浣庤川閲忓師濮嬫暟鎹?(閫氬父鏉ヨ嚜缃戠粶)         | 鑾峰緱骞挎硾鐨勮瑷€鑳藉姏 |
| **Mid-training (涓湡璁粌)**  | 杈冨皬閲忛珮璐ㄩ噺鏁版嵁 (濡傛暟瀛︺€佷唬鐮併€侀暱涓婁笅鏂? | 澧炲己鐗瑰畾鑳藉姏       |
| **Post-training (鍚庢湡璁粌)** | 鎸囦护璺熼殢鏁版嵁銆佸璇濇暟鎹€丷LHF              | 浣挎ā鍨嬪彲瀵硅瘽銆佸畨鍏?|

**鏈**:

- **Base Model (鍩虹妯″瀷)**: Pre-training + Mid-training 鍚庣殑妯″瀷
- **Instruct Model (鎸囦护妯″瀷)**: Post-training 鍚庣殑妯″瀷

### 1.1 绀轰緥: OLMo (AI2)

**Pre-training**:

![OLMo Pre-training 鏁版嵁娣峰悎](images/l13-olmo2-pretraining.png)

- DCLM Baseline: 3.7T tokens (涓讳綋)
- 浠ｇ爜銆佸鏈鏂囥€佹暟瀛︺€乄ikipedia

**Mid-training**:

![OLMo Mid-training 鏁版嵁娣峰悎](images/l13-olmo2-dolmino.png)

- 浠嶇劧鏈?DCLM Baseline, 浣嗕粠 3.7T 杩囨护鍒?700B
- 鏂板鍚堟垚鏁版嵁, 鐢氳嚦 GSM8K 璁粌闆?

**Post-training**:

![Tulu Post-training 鏁版嵁](images/l13-tulu.png)

- 鍚勭鏉ユ簮鐨勫璇濇暟鎹?
- 澶ч噺鍚堟垚鏁版嵁

> **鍏抽敭娲炲療**: 浠?澶ч噺浣庤川閲?鍒?灏戦噺楂樿川閲?, 鐣岄檺妯＄硦浣嗚秼鍔挎槑纭?

---

## 2. 棰勮缁冩暟鎹? 鍘嗗彶婕旇繘

### 2.1 BERT (2018): Books + Wikipedia

**BooksCorpus**:

- 鏉ユ簮: Smashwords (2008 骞存垚绔嬬殑鑷嚭鐗堝钩鍙?
- 鍐呭: 7,000 鏈厤璐硅嚜鍑虹増涔︾睄
- 鐜扮姸: 鍥犺繚鍙嶆湇鍔℃潯娆惧凡琚笅鏋?

**Wikipedia**:

- 2001 骞存垚绔? 鐩墠 6200 涓囩瘒鏂囩珷, 329 绉嶈瑷€
- **涓嶅寘鍚師鍒涙€濇兂**: 鏃犺鐐广€佹棤涓汉缃戦〉
- **鍩轰簬鍙煡璇佹€?*: 闇€瑕佸彲闈犳潵婧?
- **瀹氭湡 Dump**: 姣忛殧鍑犲懆鎻愪緵瀹屾暣鏁版嵁涓嬭浇

> **鏁版嵁鎶曟瘨璀﹀憡**: 鏀诲嚮鑰呭彲浠ュ湪 Wikipedia Dump 鍓嶆敞鍏ユ伓鎰忕紪杈? 鍦ㄥ洖婊氬墠琚敹褰? 杩欏凡琚敤浜庢搷绾佃瑷€妯″瀷鐨勬儏鎰熷垽鏂?(濡傚"iPhone"浜х敓璐熼潰鎯呯华).

### 2.2 GPT-2 (2019): WebText

**鏍稿績鎬濊矾**: Web 寰堝ぇ浣嗚川閲忎綆, 濡備綍蹇€熻幏寰楅珮璐ㄩ噺瀛愰泦?

**鏂规硶**: 鍒╃敤 Reddit 浣滀负"璐ㄩ噺杩囨护鍣?

- 鏀堕泦 Reddit 甯栧瓙涓?**karma 鈮?3** 鐨勫閾?
- 缁撴灉: 800 涓囬〉闈? 40GB 鏂囨湰

**OpenWebText**: WebText 鐨勫紑婧愬鍒剁増鏈?

### 2.3 Common Crawl: 瀛︽湳鐣岀殑"浜掕仈缃?

**[娣卞叆鎺㈣: Common Crawl 涓庣綉缁滅埇铏玗(./Lecture13-Common-Crawl.md)**

**鍩烘湰姒傚喌**:

- 2007 骞存垚绔嬬殑闈炶惀鍒╃粍缁?
- 姣忔湀杩涜涓€娆＄綉缁滅埇铏? 鑷充粖宸叉湁 ~100 娆?
- 鏈€鏂扮埇铏? 2025 骞?4 鏈?

**涓ょ鏍煎紡**:

- **WARC**: 鍘熷 HTTP 鍝嶅簲 (濡?HTML)
- **WET**: 杞崲涓虹函鏂囨湰 (鏈夋崯杩囩▼)

**HTML 鈫?鏂囨湰杞崲鍣?*鐨勫樊寮?

![DCLM: HTML 杞崲鍣ㄥ鍑嗙‘鐜囩殑褰卞搷](images/l13-dclm-wet.png)

浣跨敤 **trafilatura**姣斾娇鐢?WET 鏂囦欢楂?*4 涓櫨鍒嗙偣**!

> **娉ㄦ剰**: Common Crawl**涓嶆槸瀹屾暣鐨勪簰鑱旂綉**! 瀹冨埢鎰忎繚瀹堝拰绀艰矊. 鐢氳嚦涓嶆槸鎵€鏈?Wikipedia 鏂囩珷閮藉湪 Common Crawl 涓?

### 2.4 CCNet (2019): 鐢?Wikipedia 杩囨护 Common Crawl

**鐩爣**: 鑷姩鏋勫缓澶ц妯￠珮璐ㄩ噺澶氳瑷€鏁版嵁闆?

**娴佺▼**:

1. **鍘婚噸**: 鍩轰簬杞婚噺绾ц鑼冨寲绉婚櫎閲嶅娈佃惤
2. **璇█璇嗗埆**: fastText 鍒嗙被鍣? 鍙繚鐣欑洰鏍囪瑷€
3. **璐ㄩ噺杩囨护**: 淇濈暀鍦?*KenLM 5-gram 妯″瀷**涓嬬湅璧锋潵鍍?Wikipedia 鐨勬枃妗?

**鍏抽敭娲炲療**: Wikipedia 浣滀负"楂樿川閲?鐨勪唬鐞? 浣?Wikipedia 涓嶈鐩栨墍鏈夊唴瀹? 杩欎釜鏂规硶涔熶笉浼?

### 2.5 T5 / C4 (2019): 瑙勫垯杩囨护

**C4 (Colossal Clean Crawled Corpus)**:

- 浠庝竴涓?Common Crawl 蹇収 (2019 骞?4 鏈? 寮€濮? 1.4T tokens
- **绾鍒欒繃婊?* (鏃犳ā鍨?:
  - 淇濈暀浠ユ爣鐐圭粨灏俱€佲墺5 璇嶇殑琛?
  - 绉婚櫎 < 3 鍙ョ殑椤甸潰
  - 绉婚櫎鍖呭惈"鍧忚瘝"鐨勯〉闈?
  - 绉婚櫎鍖呭惈 `{` 鐨勯〉闈?(绉婚櫎浠ｇ爜!)
  - 鍙繚鐣欒嫳璇?(姒傜巼 鈮?0.99)
- 缁撴灉: 806 GB (156B tokens)

**瑙勫垯 vs 妯″瀷杩囨护鐨勬潈琛?*:

- **瑙勫垯**: 鏇村箍娉?(闈?Wikipedia 椋庢牸鐨勫ソ鍙ュ瓙涔熻兘淇濈暀), 浣嗗彲鑳藉寘鍚瀮鍦?
- **妯″瀷**: 鏇寸簿鍑? 浣嗗彧鑳藉鍒?姝ｄ緥"鐨勫垎甯?

### 2.6 GPT-3 (2020): 澶氭簮娣峰悎

- Common Crawl (澶勭悊鍚?
- WebText2 (WebText 鐨勬墿灞?
- 绁炵鐨勪功绫嶈鏂欏簱 (Books1, Books2)
- Wikipedia
- **鎬昏**: 400B tokens

**Common Crawl 澶勭悊**: 璁粌璐ㄩ噺鍒嗙被鍣? 鍖哄垎 {WebText, Wikipedia, Books} 涓庡叾浣?

### 2.7 The Pile (2021): 绀惧尯椹卞姩

涓轰簡瀵规姉 GPT-3 鐨勫皝闂? EleutherAI 绀惧尯鍦?Discord 涓婂崗璋? 浼楀寘鏋勫缓浜?**22 涓珮璐ㄩ噺棰嗗煙**:

- Pile-CC (Common Crawl, 浣跨敤 WARC + jusText)
- OpenWebText
- Wikipedia, arXiv
- **PubMed Central**: 500 涓囩瘒璁烘枃 (NIH 璧勫姪鐨勫繀椤诲叕寮€)
- **Enron Emails**: 50 涓囧皝閭欢 (鏉ヨ嚜 2002 骞磋皟鏌? 鍥犱负**鍑犱箮娌℃湁鍏朵粬鍏紑閭欢鏁版嵁闆?*)
- **Project Gutenberg**: 7.5 涓囨湰鍏増涔?
- **Books3**: 19.6 涓囨湰涔?(鏉ヨ嚜褰卞瓙搴? 鍥犵増鏉冮棶棰樺凡涓嬫灦)
- **Stack Exchange**: QA 椋庢牸, 鎺ヨ繎鎸囦护閬靛惊
- **GitHub**: 浠ｇ爜

**缁撴灉**: 825 GB (~275B tokens)

### 2.8 MassiveText / Gopher (2021): 瑙勫垯涓诲

**MassiveWeb** 杩囨护:

- 鍙繚鐣欒嫳璇?
- **鎵嬪姩瑙勫垯**杩囨护 (濡? 80% 鐨勮瘝鑷冲皯鍖呭惈涓€涓瓧姣嶅瓧绗?
- **Google SafeSearch** 杩囨护姣掓€?(闈炶瘝琛?

> **褰撴椂鐨勭悊鐢?*: 閬垮厤寮辨ā鍨嬬殑鍋忚. 浣嗚繖涓寖寮忓悗鏉ヨ DCLM 鎵撶牬.

**缁撴灉**: 10.5 TB 鏂囨湰, 浣?Gopher 鍙缁冧簡 300B tokens (12%)

### 2.9 LLaMA (2022): 缁煎悎鏂规

- Common Crawl + CCNet (鍒嗙被鍣? 鏄惁琚?Wikipedia **寮曠敤**)
- C4
- GitHub (淇濈暀瀹芥澗璁稿彲)
- Wikipedia, Project Gutenberg, **Books3** (鎯逛笂澶ч夯鐑?)
- arXiv, Stack Exchange
- **鎬昏**: 1.2T tokens

**澶嶅埗鐗堟湰**:

- **RedPajama v1** (Together): 寮€婧愬鍒?
- **SlimPajama** (Cerebras): 鍘婚噸鍚庣殑 627B 瀛愰泦

### 2.10 RefinedWeb (2023): Web Data is All You Need

**璁虹偣**: 濡傛灉杩囨护鍋氬緱濂?**鍙渶瑕佺綉缁滄暟鎹?*.

**鏂规硶**:

- trafilatura 鎻愬彇 (WARC 鑰岄潪 WET)
- Gopher 瑙勫垯杩囨护, **閬垮厤 ML 杩囨护**浠ラ伩鍏嶅亸瑙?
- MinHash 妯＄硦鍘婚噸
- **缁撴灉**: 5T tokens (鍙戝竷 600B)

**FineWeb** (HuggingFace): RefinedWeb 鐨勬敼杩涚増

- 95 涓?Common Crawl 蹇収
- Gopher + C4 瑙勫垯
- PII 鍖垮悕鍖?
- **缁撴灉**: 15T tokens (浠嶆槸**杞诲害杩囨护**, 閫傚悎杩涗竴姝ユā鍨嬭繃婊?

### 2.11 Dolma (2024): AI2 鐨勭患鍚堟暟鎹泦

- Common Crawl (璇█璇嗗埆 + 瑙勫垯杩囨护 + 鍘婚噸)
- Reddit (Pushshift 椤圭洰)
- PeS2o: 4000 涓囩瘒瀛︽湳璁烘枃 (Semantic Scholar)
- C4, Project Gutenberg, Wikipedia
- **缁撴灉**: 3T tokens

### 2.12 DCLM (2024): 妯″瀷杩囨护鐨勮儨鍒?

**[娣卞叆鎺㈣: DCLM 涓庢ā鍨嬪熀璐ㄩ噺杩囨护](./Lecture13-DCLM.md)**

**DataComp-LM**鐨勭洰鏍囨槸鍒涘缓涓€涓?*鏁版嵁澶勭悊绠楁硶鐨勬爣鍑嗙珵璧?*.

**DCLM-pool**: 澶勭悊鎵€鏈?Common Crawl 鈫?*240T tokens**

**DCLM-baseline**: 浣跨敤璐ㄩ噺鍒嗙被鍣ㄨ繃婊?鈫?*3.8T tokens** (鍙繚鐣?1.4%!)

![DCLM 杩囨护娴佺▼](images/l13-dclm-filter.png)

**妯″瀷杩囨护鏂规硶**:

- **姝ｄ緥** (20 涓?: OpenHermes-2.5 (GPT-4 鐢熸垚鐨勬寚浠ゆ暟鎹? + ELI5 (Reddit 瀛愮増鍧?
- **璐熶緥** (20 涓?: RefinedWeb 闅忔満鏍锋湰
- 璁粌 **fastText 鍒嗙被鍣?*

![DCLM 璐ㄩ噺鍒嗙被鍣ㄦ晥鏋淽(images/l13-dclm-quality.png)

> **鍏抽敭杞姌**: 杩欐墦鐮翠簡"閬垮厤 ML 杩囨护"鐨勬棫鑼冨紡. 浣跨敤妯″瀷杩囨护**鏄捐憲鎻愬崌**涓嬫父浠诲姟琛ㄧ幇.

### 2.13 Nemotron-CC (2024): 鏇村 Token

**闂**: DCLM 杩囨护澶縺杩?(240T 鈫?3.8T). 鎯宠鏇村 Token!

**鏂规硶**:

1. **HTML 鈫?鏂囨湰**: 浣跨敤 jusText (鑰岄潪 trafilatura), 鍥犱负淇濈暀鏇村 Token
2. **鍒嗙被鍣ㄩ泦鎴?*:
   - Nemotron-340B 璇勫垎鏁欒偛浠峰€? 钂搁鍒板揩閫熸ā鍨?
   - DCLM 鍒嗙被鍣?
   - 鎸夊垎鏁板垎妗? 浠庢瘡涓《閲囨牱 (淇濊瘉瑕嗙洊)
3. **鍚堟垚鏁版嵁鏀瑰啓**:
   - 浣庤川閲忔暟鎹? 鐢?LM 鏀瑰啓鎴愰珮璐ㄩ噺
   - 楂樿川閲忔暟鎹? 鐢?LM 鐢熸垚 QA 瀵?/ 鎽樿 / 鍏抽敭淇℃伅鎻愬彇

**缁撴灉**: 6.3T tokens (楂樿川閲忓瓙闆?1.1T)

![Nemotron-CC 鏁堟灉](images/l13-nemotron-results.png)

> **瀵规瘮**: Llama 3 璁粌 15T, Qwen 3 璁粌 36T (鍚妯℃€?.

---

## 3. 鐗堟潈娉曚笌鏁版嵁鍚堟硶鎬?

**[娣卞叆鎺㈣: 鐗堟潈娉曚笌 Fair Use](./Lecture13-Copyright.md)**

### 3.1 鐗堟潈娉曞熀纭€

- **鐩殑**: 婵€鍔辩煡璇嗕骇鍝佺殑鍒涢€?
- **鑼冨洿**: "鍥哄畾鍦ㄤ换浣曟湁褰㈣〃杈惧獟浠嬩腑鐨勫師鍒涗綔鍝?
- **鏃犻渶娉ㄥ唽**: 浣犵殑缃戠珯宸茬粡鏄増鏉冧綔鍝?(鍙槸璧疯瘔鍓嶉渶瑕佹敞鍐? $65)
- **鏈熼檺**: 75 骞村悗杩涘叆鍏増

> **鍏抽敭**: 浜掕仈缃戜笂鐨?*澶у鏁板唴瀹归兘鏄増鏉冧綔鍝?*.

### 3.2 濡備綍鍚堟硶浣跨敤鐗堟潈浣滃搧

**鏂瑰紡涓€: 鑾峰緱璁稿彲 (License)**

- 绛捐鍚堝悓 (濡?Google-Reddit, OpenAI-Shutterstock)
- Creative Commons 璁稿彲 (濡?Wikipedia, Khan Academy)

**鏂瑰紡浜? 鎻村紩 Fair Use**

鍥涗釜鍥犵礌:

1. **浣跨敤鐩殑**: 鏁欒偛 > 鍟嗕笟, 鍙橀潻鎬?> 澶嶅埗鎬?
2. **浣滃搧鎬ц川**: 浜嬪疄鎬?> 铏氭瀯鎬?
3. **浣跨敤閲?*: 鐗囨 > 鍏ㄩ儴
4. **甯傚満褰卞搷**: 涓嶆浛浠ｅ師浣滃搧

**LLM 璁粌鐨勬寫鎴?*:

- 澶嶅埗鏁版嵁 (璁粌绗竴姝? 鏈韩**鍙兘宸茶繚瑙?*, 鍗充娇浣犱粈涔堥兘涓嶅仛
- 鍙互璁鸿瘉 ML 璁粌鏄?*鍙橀潻鎬?*鐨?
- ML 绯荤粺鍏冲績鐨勬槸**鎯虫硶**(濡傚仠杞︽爣蹇?, 鑰岄潪**琛ㄨ揪** (鏌愬紶鍥剧殑鑹烘湳閫夋嫨)
- **浣?*: LLM 鏄庢樉褰卞搷甯傚満 (浣滃銆佽壓鏈)

### 3.3 鏈嶅姟鏉℃ (Terms of Service)

鍗充娇鏈夎鍙垨 Fair Use, **鏈嶅姟鏉℃鍙兘鏂藉姞棰濆闄愬埗**.

渚? YouTube 鐨勬湇鍔℃潯娆剧姝笅杞借棰? 鍗充娇瑙嗛鏈韩鏄?Creative Commons.

---

## 4. 涓湡璁粌涓庡悗鏈熻缁?

### 4.1 闀夸笂涓嬫枃鎵╁睍 (Long Context)

**闇€姹?*:

- DeepSeek v3: 128K tokens
- Claude 3.5: 200K tokens
- Gemini 1.5 Pro: 1.5M tokens

**闂**: Transformer 涓庡簭鍒楅暱搴﹀憟**浜屾鏂?*鍏崇郴, 棰勮缁冮樁娈典笉楂樻晥.

**瑙ｅ喅**: 鍦?Mid-training 闃舵娣诲姞闀夸笂涓嬫枃鑳藉姏

- **鏁版嵁鏉ユ簮**: 涔︾睄 (PG-19), 鏁板璇佹槑 (Proof-Pile)
- **鎶€鏈?*: Shifted sparse attention, Positional interpolation

### 4.2 浠诲姟/NLP 鏁版嵁闆?

**鎬濊矾**: 灏嗕紶缁?NLP 鏁版嵁闆嗚浆鎹负 Prompt 鏍煎紡

**Super-Natural Instructions (2022)**:

- 1,600+ 浠诲姟, 绀惧尯璐＄尞
- 寰皟 T5 鈫?Tk-Instruct

**Flan (2022-2023)**:

- 1,800+ 浠诲姟
- Zero-shot, Few-shot, Chain-of-Thought 鐗堟湰

> **闂**: Prompt 澶ā鏉垮寲, 涓嶅鑷劧.

### 4.3 鎸囦护閬靛惊涓庡璇濇暟鎹?

**Alpaca (2023)**:

- 浣跨敤 **Self-Instruct** 浠?text-davinci-003 鐢熸垚 52K 绀轰緥
- 寰皟 LLaMA 7B

**Vicuna**:

- 浣跨敤 ShareGPT (鐢ㄦ埛鍒嗕韩鐨?ChatGPT 瀵硅瘽, 宸插簾寮? 鐨?70K 瀵硅瘽
- 寰皟 LLaMA

**Baize**:

- GPT-3.5 鑷垜瀵硅瘽 (浠?Quora/StackOverflow 闂涓虹瀛?
- 111.5K 绀轰緥

**WizardLM**:

- **Evol-Instruct**: 璁╅棶棰?杩涘寲"浠ュ鍔犻毦搴?骞垮害

**MAmmoTH2**:

- 浠?Common Crawl 涓敤 fastText 璇嗗埆"娴嬮獙缃戠珯"
- 鐢?GPT-4/Mixtral 鎻愬彇 QA 瀵?
- 10M 鎸囦护

**OpenHermes 2.5**:

- 澶氫釜鏁版嵁闆嗙殑鑱氬悎
- 1M GPT-4 鐢熸垚鐨勭ず渚?

**Llama 2 Chat**:

- 27,540 鏉?*浜哄伐鏍囨敞**鐨勯珮璐ㄩ噺鎸囦护
- 澹扮О浼樹簬浣跨敤鏁扮櫨涓囧紑婧愮ず渚?

**Llama-Nemotron Post-training (2024)**:

- 浠庡叕寮€鏁版嵁闆?(WildChat 绛? 鎴栧悎鎴愮敓鎴?Prompt
- 浣跨敤 Llama, Mixtral, DeepSeek R1, Qwen 鐢熸垚鍥炲 (鍟嗕笟鍙敤, 涓嶅儚 GPT-4)
- 鍖呭惈鎺ㄧ悊杞ㄨ抗

---

## 5. 鎬荤粨: 鏍稿績瑕佺偣

1. **鏁版嵁涓嶄細浠庡ぉ涓婃帀涓嬫潵**: 闇€瑕佸ぇ閲忓伐浣滆幏鍙?

   - **Live Service 鈫?Raw Dump 鈫?Processed Data**
   - 娑夊強杞崲銆佽繃婊ゃ€佸幓閲?
2. **鏁版嵁鏄尯鍒嗚瑷€妯″瀷鐨勫叧閿?*: 鏋舵瀯宸茶秼鍚? 鏁版嵁鍐冲畾璐ㄩ噺
3. **娉曞緥鍜屼鸡鐞嗛棶棰?*: 鐗堟潈銆侀殣绉併€佹湇鍔℃潯娆?
4. **鐩墠涓€鍒囬兘鏄惎鍙戝紡鐨?*: 澶ч噺鏈轰細鏀硅繘!

---

## 闄勫綍: 閰嶅浠ｇ爜缁撴瀯 (`lecture_13.py`)

璇剧▼浠ｇ爜瀹氫箟浜嗗畬鏁寸殑璁插骇缁撴瀯:

```python
def main():
    introduction()              # 鏁版嵁鏈€閲嶈
    # Pre-training
    bert()                      # Wikipedia + Books (2019)
    gpt2_webtext()              # Reddit 閾炬帴 (2019)
    common_crawl()              # 缃戠粶鐖櫕
    ccnet()                     # Wikipedia 杩囨护 (2019)
    t5_c4()                     # 瑙勫垯杩囨护 (2019)
    gpt3()                      # 澶氭簮娣峰悎 (2020)
    the_pile()                  # 绀惧尯浼楀寘 (2021)
    gopher_massivetext()        # 瑙勫垯杩囨护 (2021)
    llama()                     # 缁煎悎鏂规 (2022)
    refinedweb()                # Web Only (2023)
    dolma()                     # AI2 缁煎悎 (2024)
    dclm()                      # 妯″瀷杩囨护 (2024)
    nemotron_cc()               # 鏇村 Token (2024)
    copyright()                 # 鐗堟潈娉?
    # Mid/Post-training
    long_context()              # 闀夸笂涓嬫枃
    tasks()                     # NLP 浠诲姟杞崲
    instruction_chat()          # 鎸囦护/瀵硅瘽鏁版嵁
```

---

## 鍙傝€冮摼鎺?

- **Common Crawl**: https://commoncrawl.org/
- **DCLM**: https://arxiv.org/abs/2406.11794
- **FineWeb**: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- **The Pile**: https://arxiv.org/abs/2101.00027
- **Nemotron-CC**: https://arxiv.org/abs/2412.xxxxx (寰呯‘璁?
- **CS324 鐗堟潈绗旇**: https://stanford-cs324.github.io/winter2022/lectures/legality/

