# 娣卞叆鎺㈣: KenLM涓嶯-gram璇█妯″瀷

鏈枃鏄疞ecture 14鐨勭簿鑻辫ˉ鍏呯瑪璁帮紝娣卞叆璁茶ВN-gram璇█妯″瀷鐨勬暟瀛﹀師鐞嗐€並neser-Ney骞虫粦鎶€鏈紝浠ュ強KenLM鐨勯珮鏁堝疄鐜般€?

---

## 涓€銆丯-gram璇█妯″瀷鍩虹

### 1.1 璇█妯″瀷鐨勭洰鏍?

璇█妯″瀷鐨勭洰鏍囨槸浼拌**搴忓垪姒傜巼** $P(w_1, w_2, ..., w_n)$銆?

鏍规嵁閾惧紡娉曞垯:
$$P(w_1, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})$$

**闂**: 鍘嗗彶 $w_1, ..., w_{i-1}$ 鍙兘闈炲父闀匡紝鏃犳硶鐩存帴浼拌銆?

**瑙ｅ喅鏂规**:**椹皵鍙か鍋囪** - 鍙€冭檻鍓?$n-1$ 涓瘝

$$P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-n+1}, ..., w_{i-1})$$

### 1.2 N-gram鐨勯樁鏁伴€夋嫨

| 闃舵暟 | 妯″瀷 | 鏉′欢 | 浼樼己鐐?|
|------|------|------|--------|
| n=1 | Unigram | $P(w_i)$ | 绠€鍗曚絾蹇界暐涓婁笅鏂?|
| n=2 | Bigram | $P(w_i \| w_{i-1})$ | 鎹曟崏灞€閮ㄤ緷璧?|
| n=3 | Trigram | $P(w_i \| w_{i-2}, w_{i-1})$ | 鏇翠赴瀵岀殑涓婁笅鏂?|
| n=5 | 5-gram | $P(w_i \| w_{i-4:i-1})$ | KenLM榛樿锛屽钩琛℃€ц兘鍜屾硾鍖?|

### 1.3 鏈€澶т技鐒朵及璁?

瀵逛簬Bigram:
$$P_{MLE}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

鍏朵腑 $C(\cdot)$ 鏄鏁板嚱鏁般€?

**闂**: 鏈杩囩殑n-gram姒傜巼涓?锛?

---

## 浜屻€佸钩婊戞妧鏈?(Smoothing)

### 2.1 涓轰粈涔堥渶瑕佸钩婊戯紵

璇枡搴撴槸鏈夐檺鐨勶紝蹇呯劧鏈夊ぇ閲忔湭瑙佽繃鐨刵-gram銆?

**Zipf's Law**: 璇嶉鍒嗗竷鏋佷笉鍧囧寑
- 灏戞暟璇嶅嚭鐜伴潪甯搁绻?
- 澶ч噺璇嶅彧鍑虹幇涓€涓ゆ
- 鏇村璇嶄粠鏈嚭鐜?

### 2.2 Add-One (Laplace) 骞虫粦

鏈€绠€鍗曠殑鏂规硶:
$$P_{add-1}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$$

鍏朵腑 $V$ 鏄瘝琛ㄥぇ灏忋€?

**闂**: 缁欐湭瑙乶-gram鍒嗛厤浜嗗お澶氭鐜囪川閲忥紒

### 2.3 Add-k 骞虫粦

鏀硅繘鐗?
$$P_{add-k}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + kV}$$

$k < 1$ 鍙互鍑忓皯瀵规湭瑙乶-gram鐨勮繃搴﹀垎閰嶃€?

### 2.4 Kneser-Ney骞虫粦

**鏍稿績鎬濇兂**: 涓嶅彧鐪嬮鐜囷紝杩樿鐪?*涓婁笅鏂囧鏍锋€?*銆?

#### 缁濆鎶樻墸 (Absolute Discounting)

棣栧厛锛屼粠姣忎釜璁℃暟涓噺鍘诲浐瀹氬€?$d$:
$$P_{abs}(w_i | w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - d, 0)}{C(w_{i-1})} + \lambda(w_{i-1}) P_{lower}(w_i)$$

鍏朵腑 $\lambda(w_{i-1})$ 鏄綊涓€鍖栧洜瀛愩€?

#### Continuation姒傜巼

Kneser-Ney鐨勫垱鏂帮細瀵逛簬浣庨樁鍒嗗竷锛屼娇鐢?*continuation姒傜巼**鑰岄潪棰戠巼:

$$P_{KN}(w) = \frac{|\{w' : C(w', w) > 0\}|}{|\{(w', w'') : C(w', w'') > 0\}|}$$

**鐩磋**: 涓€涓瘝鐨勬鐜囦笉鍙栧喅浜庡畠鍑虹幇澶氬皯娆★紝鑰屾槸瀹冭兘璺熷灏戜笉鍚岀殑鍓嶇紑鎼厤銆?

**渚嬪瓙**: "Francisco" 鍑犱箮鍙窡 "San" 鎼厤
- 楂橀鐜囷紝浣嗕綆continuation姒傜巼
- 浣滀负鍥為€€锛屼笉搴旇寰楀埌楂樻鐜?

#### Modified Kneser-Ney

浣跨敤澶氫釜鎶樻墸鍊?$d_1, d_2, d_{3+}$锛屾牴鎹師濮嬭鏁伴€夋嫨:

$$d = \begin{cases}
0 & \text{if } C = 0 \\
d_1 & \text{if } C = 1 \\
d_2 & \text{if } C = 2 \\
d_{3+} & \text{if } C \geq 3
\end{cases}$$

---

## 涓夈€佸洖閫€鏈哄埗 (Backoff)

### 3.1 Katz鍥為€€

褰撻珮闃秐-gram鏈杩囨椂锛屽洖閫€鍒颁綆闃?

$$P_{BO}(w_i | w_{i-n+1:i-1}) = \begin{cases}
P^*(w_i | w_{i-n+1:i-1}) & \text{if } C(w_{i-n+1:i}) > k \\
\alpha(w_{i-n+1:i-1}) \cdot P_{BO}(w_i | w_{i-n+2:i-1}) & \text{otherwise}
\end{cases}$$

鍏朵腑:
- $P^*$ 鏄姌鎵ｅ悗鐨勬鐜?
- $\alpha$ 鏄洖閫€鏉冮噸锛岀‘淇濇鐜囧綊涓€鍖?

### 3.2 鎻掑€?(Interpolation)

鍙︿竴绉嶆柟娉曟槸濮嬬粓娣峰悎涓嶅悓闃?

$$P_{interp}(w_i | w_{i-n+1:i-1}) = \lambda_n P_n + \lambda_{n-1} P_{n-1} + ... + \lambda_1 P_1$$

鍏朵腑 $\sum \lambda_i = 1$銆?

### 3.3 Kneser-Ney鐨勫畬鏁村叕寮?

缁撳悎鍥為€€鍜宑ontinuation姒傜巼:

$$P_{KN}(w_i | w_{i-n+1:i-1}) = \frac{\max(C(w_{i-n+1:i}) - d, 0)}{C(w_{i-n+1:i-1})} + \gamma(w_{i-n+1:i-1}) P_{KN}(w_i | w_{i-n+2:i-1})$$

鍏朵腑浣庨樁 $P_{KN}$ 浣跨敤continuation璁℃暟銆?

---

## 鍥涖€並enLM瀹炵幇缁嗚妭

### 4.1 楂樻晥瀛樺偍锛歍rie缁撴瀯

KenLM浣跨敤**鍘嬬缉Trie**瀛樺偍n-gram:

```
      ROOT
     / | \
   the  a   ...
   / \
  cat dog
  |
 sat
```

姣忎釜鑺傜偣瀛樺偍:
- **姒傜巼**: 閲忓寲鍚庣殑log姒傜巼
- **鍥為€€鏉冮噸**: 閲忓寲鍚庣殑鍥為€€鍊?
- **鎸囬拡**: 鎸囧悜瀛愯妭鐐?

### 4.2 閲忓寲 (Quantization)

涓轰簡鑺傜渷绌洪棿锛孠enLM瀵规鐜囧拰鍥為€€鏉冮噸杩涜閲忓寲:

```python
# 浼唬鐮?
def quantize(value, bits=8):
    # 灏嗚繛缁€兼槧灏勫埌256涓鏁ｇ骇鍒?
    min_val, max_val = get_range(all_values)
    level = int((value - min_val) / (max_val - min_val) * (2**bits - 1))
    return level
```

8-bit閲忓寲鍙互灏嗗瓨鍌ㄥ噺灏?鍊嶏紝绮惧害鎹熷け鍙拷鐣ャ€?

### 4.3 鏌ヨ娴佺▼

```python
def query_kenlm(sentence, model):
    """璁＄畻鍙ュ瓙鐨刲og姒傜巼"""
    tokens = tokenize(sentence)
    log_prob = 0.0
    state = model.begin_state()
    
    for token in tokens:
        # 灏濊瘯鏈€闀垮尮閰?
        prob, new_state = model.score(state, token)
        log_prob += prob  # log10姒傜巼
        state = new_state
    
    return log_prob

def model.score(state, token):
    """鏌ヨ鍗曚釜token鐨勬鐜?""
    context = get_context_from_state(state)
    
    for order in range(max_order, 0, -1):
        ngram = context[-(order-1):] + [token]
        if ngram in trie:
            prob = trie[ngram].log_prob
            new_state = update_state(context, token)
            return prob, new_state
        else:
            # 鍥為€€锛氫箻浠ュ洖閫€鏉冮噸锛屽皾璇曚綆闃?
            backoff = trie[context[-(order-1):]].backoff
            prob += backoff
            context = context[1:]  # 缂╃煭涓婁笅鏂?
    
    # 鍥為€€鍒皍nigram
    return unigram_prob[token], reset_state()
```

### 4.4 KenLM vs 鍏朵粬瀹炵幇

| 鐗规€?| KenLM | SRILM | 鑷畾涔塒ython |
|------|-------|-------|--------------|
| 閫熷害 | 鏋佸揩 | 蹇?| 鎱?|
| 鍐呭瓨 | 浣庯紙鍘嬬缉锛?| 涓?| 楂?|
| 鏈€澶ч樁鏁?| 鏃犻檺鍒?| 閫氬父5-7 | 鑷畾涔?|
| 鎵归噺鏌ヨ | 鏀寔 | 鏀寔 | 闇€鑷瀹炵幇 |

### 4.5 璁粌KenLM妯″瀷

```bash
# 1. 鍑嗗鏂囨湰鏁版嵁 (姣忚涓€涓彞瀛?
# corpus.txt

# 2. 浣跨敤lmplz璁粌
lmplz -o 5 \        # 5-gram
      -S 80% \      # 浣跨敤80%鍐呭瓨
      --discount_fallback \
      < corpus.txt \
      > model.arpa

# 3. 杞崲涓轰簩杩涘埗鏍煎紡 (鏇村揩鍔犺浇)
build_binary model.arpa model.binary
```

---

## 浜斻€佸湪鏁版嵁杩囨护涓殑搴旂敤

### 5.1 璐ㄩ噺璇勫垎

浣跨敤鍥版儜搴?(Perplexity) 璇勪及鏂囨湰璐ㄩ噺:

$$PPL(w_1, ..., w_n) = P(w_1, ..., w_n)^{-1/n}$$

**浣庡洶鎯戝害** = 妯″瀷棰勬祴濂?= 鏂囨湰"姝ｅ父"
**楂樺洶鎯戝害** = 妯″瀷棰勬祴宸?= 鏂囨湰"寮傚父"

```python
import kenlm

model = kenlm.Model("wiki.binary")

def perplexity_filter(text, threshold=500):
    """鍩轰簬鍥版儜搴﹁繃婊や綆璐ㄩ噺鏂囨湰"""
    ppl = model.perplexity(text)
    return ppl < threshold
```

### 5.2 DSIR涓殑搴旂敤

鍦―SIR涓紝浣跨敤涓や釜KenLM妯″瀷璁＄畻閲嶈鎬ф潈閲?

```python
target_model = kenlm.Model("high_quality.binary")  # 濡俉ikipedia
raw_model = kenlm.Model("raw_data.binary")         # 濡侰ommon Crawl

def importance_weight(text):
    log_p_target = target_model.score(text)
    log_p_raw = raw_model.score(text)
    return 10 ** (log_p_target - log_p_raw)  # 杞崲鍥炴鐜囨瘮
```

---

## 鍙傝€冭祫鏂?

1. Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling
2. Kneser, R., & Ney, H. (1995). Improved backing-off for m-gram language modeling
3. Heafield, K. (2011). KenLM: Faster and Smaller Language Model Queries
4. Jurafsky, D., & Martin, J. (2023). Speech and Language Processing, Chapter 3

