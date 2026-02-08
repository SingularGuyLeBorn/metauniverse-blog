# CS336 Lecture 14: 瀹炴垬鏁版嵁杩囨护鍜屽幓閲?(Practical Data Filtering and Deduplication)

> **缂栬緫钃濆浘 (Editorial Blueprint)**
> 
> **鏍稿績涓婚**: 鏈搴ф槸CS336璇剧▼涓渶鍏峰疄璺垫€х殑涓€璁诧紝娣卞叆璁茶В浜嗗ぇ瑙勬ā璇█妯″瀷棰勮缁冩暟鎹鐞嗘祦姘寸嚎鐨勪袱澶ф牳蹇冿細**鏁版嵁杩囨护 (Filtering)**鍜?*鏁版嵁鍘婚噸 (Deduplication)**銆備粠绠楁硶鍘熺悊鍒板伐绋嬪疄鐜帮紝浠庤瑷€璇嗗埆鍒版瘨鎬ц繃婊わ紝鍏ㄩ潰瑕嗙洊銆?
> 
> **鐭ヨ瘑缁撴瀯**: 
> - 绗竴閮ㄥ垎锛氳繃婊ょ畻娉曞熀纭€锛圢-gram璇█妯″瀷銆並enLM銆乫astText鍒嗙被鍣ㄣ€侀噸瑕佹€ч噸閲囨牱DSIR锛?
> - 绗簩閮ㄥ垎锛氳繃婊ゅ簲鐢ㄥ満鏅紙璇█璇嗗埆銆佽川閲忚繃婊ゃ€佹瘨鎬ц繃婊わ級
> - 绗笁閮ㄥ垎锛氬幓閲嶆妧鏈紙绮剧‘鍘婚噸銆佸竷闅嗚繃婊ゅ櫒銆佽繎浼煎幓閲嶃€丮inHash銆丩SH锛?
> 
> **绮捐嫳琛ュ厖绗旇**:
> - **[娣卞叆鎺㈣: KenLM涓嶯-gram璇█妯″瀷](./Lecture14-KenLM.md)** - N-gram鍥為€€鏈哄埗銆並neser-Ney骞虫粦
> - **[娣卞叆鎺㈣: MinHash涓嶭SH绠楁硶](./Lecture14-MinHash-LSH.md)** - 杩戜技鍘婚噸鐨勬暟瀛﹀師鐞嗕笌瀹炵幇

---

## 涓€銆佽繃婊ょ畻娉曞熀纭€ (Filtering Algorithms Fundamentals)

### 1.1 涓轰粈涔堥渶瑕佽繃婊わ紵

棰勮缁冩暟鎹殑璐ㄩ噺鐩存帴鍐冲畾妯″瀷鎬ц兘銆備簰鑱旂綉鏁版嵁锛堝Common Crawl锛夊寘鍚ぇ閲忥細
- **浣庤川閲忓唴瀹?*: 鍨冨溇閭欢銆佸箍鍛婃枃鏈€侀噸澶嶆ā鏉?
- **闈炵洰鏍囪瑷€**: 澶氳瑷€娣锋潅
- **鏈夊鍐呭**: 姣掓€ф枃鏈€佽繚娉曚俊鎭?

杩囨护鐨勭洰鏍囨槸锛?*浠庢捣閲忓櫔澹版暟鎹腑楂樻晥绛涢€夊嚭楂樿川閲忋€佺洰鏍囪瑷€銆佸畨鍏ㄧ殑鏂囨湰**銆?

### 1.2 N-gram 璇█妯″瀷 (N-gram Language Models)

N-gram妯″瀷鏄繃婊ゆ祦姘寸嚎鐨勫熀纭€宸ュ叿锛岀敤浜庤瘎浼版枃鏈川閲忋€?

#### 鏍稿績鎬濇兂

缁欏畾涓€涓猼oken搴忓垪 $x_1, x_2, ..., x_n$锛孨-gram妯″瀷灏嗗叾姒傜巼鍒嗚В涓猴細

$$P(x_1, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_{i-n+1}, ..., x_{i-1})$$

- **Unigram (n=1)**: $P(x_i)$ - 鍙湅褰撳墠璇?
- **Bigram (n=2)**: $P(x_i | x_{i-1})$ - 鐪嬪墠涓€涓瘝
- **Trigram (n=3)**: $P(x_i | x_{i-2}, x_{i-1})$ - 鐪嬪墠涓や釜璇?

#### Python瀹炵幇锛欱igram妯″瀷

```python
from collections import defaultdict
import math

class BigramModel:
    """绠€鍗曠殑Bigram璇█妯″瀷"""
    def __init__(self):
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.total_count = 0
    
    def train(self, texts: list[str]):
        """鍦ㄨ鏂欏簱涓婅缁冩ā鍨?""
        for text in texts:
            tokens = text.split()
            for i, token in enumerate(tokens):
                self.unigram_counts[token] += 1
                self.total_count += 1
                if i > 0:
                    prev_token = tokens[i - 1]
                    self.bigram_counts[prev_token][token] += 1
    
    def log_prob(self, text: str) -> float:
        """璁＄畻鏂囨湰鐨勫鏁版鐜?""
        tokens = text.split()
        log_p = 0.0
        for i, token in enumerate(tokens):
            if i == 0:
                # Unigram姒傜巼
                p = (self.unigram_counts[token] + 1) / (self.total_count + len(self.unigram_counts))
            else:
                prev_token = tokens[i - 1]
                # Bigram姒傜巼锛屼娇鐢ㄥ姞涓€骞虫粦
                count = self.bigram_counts[prev_token][token]
                total = sum(self.bigram_counts[prev_token].values())
                p = (count + 1) / (total + len(self.unigram_counts))
            log_p += math.log(p)
        return log_p
```

### 1.3 KenLM锛氶珮鏁圢-gram璇█妯″瀷

**KenLM** 鏄竴涓珮搴︿紭鍖栫殑N-gram璇█妯″瀷搴擄紝鏀寔:
- **淇敼Kneser-Ney骞虫粦**: 姣旂畝鍗曠殑鍔犱竴骞虫粦鏁堟灉鏇村ソ
- **鍥為€€鏈哄埗 (Backoff)**: 褰撻珮闃秐-gram鏈杩囨椂锛屽洖閫€鍒颁綆闃?
- **鍘嬬缉瀛樺偍**: 浣跨敤Trie缁撴瀯鍜岄噺鍖栨妧鏈?

#### 瀹夎涓庝娇鐢?

```bash
pip install kenlm
# 闇€瑕佸厛鐢╨mplz宸ュ叿璁粌.arpa妯″瀷
```

```python
import kenlm

# 鍔犺浇棰勮缁冪殑KenLM妯″瀷
model = kenlm.Model("wiki_en_5gram.arpa")

# 璁＄畻鍙ュ瓙鐨勫鏁版鐜囷紙浠?0涓哄簳锛?
text = "The quick brown fox jumps over the lazy dog"
log_prob = model.score(text)
perplexity = model.perplexity(text)

print(f"Log probability: {log_prob}")
print(f"Perplexity: {perplexity}")
```

#### 鍥為€€鏈哄埗璇﹁В

褰撻亣鍒版湭瑙佽繃鐨刵-gram鏃讹紝KenLM浣跨敤鍥為€€锛?

$$P(x_i | x_{i-n+1:i-1}) = \begin{cases} 
P_{MLE}(x_i | x_{i-n+1:i-1}) & \text{if count} > 0 \\
\alpha(x_{i-n+1:i-1}) \cdot P(x_i | x_{i-n+2:i-1}) & \text{otherwise}
\end{cases}$$

鍏朵腑 $\alpha$ 鏄洖閫€鏉冮噸锛岀‘淇濇鐜囧綊涓€鍖栥€?

### 1.4 fastText 鏂囨湰鍒嗙被鍣?

**fastText** 鏄疐acebook寮€鍙戠殑楂樻晥鏂囨湰鍒嗙被宸ュ叿锛岀壒鍒€傚悎锛?
- **璇█璇嗗埆 (Language Identification)**
- **涓婚鍒嗙被**
- **璐ㄩ噺璇勫垎**

#### 鏍稿績鎶€鏈?

1. **璇嶈妯″瀷 + N-gram鐗瑰緛**: 鎹曡幏璇嶅簭淇℃伅
2. **灞傛Softmax**: 鍔犻€熷ぇ璇嶈〃鍒嗙被
3. **瀛愯瘝琛ㄧず**: 浣跨敤瀛楃n-gram澶勭悊OOV璇?

```python
import fasttext

# 璁粌璇█璇嗗埆妯″瀷
# 璁粌鏁版嵁鏍煎紡: __label__en This is English text
# model = fasttext.train_supervised("train.txt")

# 浣跨敤棰勮缁冪殑璇█璇嗗埆妯″瀷
lang_model = fasttext.load_model("lid.176.bin")

text = "This is a sample text"
predictions = lang_model.predict(text, k=3)  # 杩斿洖top-3棰勬祴
# 杈撳嚭: (('__label__en', '__label__de', '__label__nl'), (0.95, 0.02, 0.01))

label, confidence = predictions[0][0], predictions[1][0]
print(f"Language: {label.replace('__label__', '')}, Confidence: {confidence:.2f}")
```

### 1.5 閲嶈鎬ч噸閲囨牱 (DSIR - Data Selection via Importance Resampling)

**DSIR**鏄竴绉嶆暟鎹€夋嫨鏂规硶锛屾牳蹇冩€濇兂鏄細**璁╅€変腑鐨勬暟鎹垎甯冩帴杩戠洰鏍囧垎甯?*銆?

#### 鏁板鍘熺悊

璁?
- $P_{raw}(x)$: 鍘熷鏁版嵁锛堝Common Crawl锛夌殑鍒嗗竷
- $P_{target}(x)$: 鐩爣鏁版嵁锛堝Wikipedia楂樿川閲忔枃鏈級鐨勫垎甯?

閲嶈鎬ф潈閲?
$$w(x) = \frac{P_{target}(x)}{P_{raw}(x)}$$

瀵逛簬鏂囨湰 $x$锛屼互姒傜巼 $\min(1, \lambda \cdot w(x))$ 閫夋嫨瀹冿紝鍏朵腑 $\lambda$ 鎺у埗閫夋嫨鐜囥€?

#### 瀹炵幇锛氫娇鐢↘enLM璁＄畻閲嶈鎬ф潈閲?

```python
import kenlm
import math
import random

class DSIRFilter:
    """鍩轰簬DSIR鐨勬暟鎹€夋嫨鍣?""
    
    def __init__(self, target_model_path: str, raw_model_path: str):
        self.target_model = kenlm.Model(target_model_path)
        self.raw_model = kenlm.Model(raw_model_path)
    
    def compute_importance_weight(self, text: str) -> float:
        """璁＄畻閲嶈鎬ф潈閲?w(x) = P_target(x) / P_raw(x)"""
        # KenLM杩斿洖log10姒傜巼锛岄渶瑕佽浆鎹?
        log_p_target = self.target_model.score(text)
        log_p_raw = self.raw_model.score(text)
        
        # 杞崲涓哄鏁版潈閲嶏紙闃叉鏁板€兼孩鍑猴級
        log_weight = (log_p_target - log_p_raw) * math.log(10)
        return math.exp(log_weight)
    
    def select(self, texts: list[str], selection_rate: float = 0.1) -> list[str]:
        """鏍规嵁DSIR閫夋嫨鏂囨湰"""
        selected = []
        for text in texts:
            weight = self.compute_importance_weight(text)
            prob = min(1.0, selection_rate * weight)
            if random.random() < prob:
                selected.append(text)
        return selected
```

#### DSIR鏁堟灉楠岃瘉

![DSIR瀹為獙缁撴灉](./images/l14-dsir-results.png)

DSIR鍦ㄤ笅娓镐换鍔′笂鐨勮〃鐜版樉钁椾紭浜庨殢鏈洪噰鏍凤紝璇佹槑浜嗛噸瑕佹€ч噸閲囨牱鐨勬湁鏁堟€с€?

---

## 浜屻€佽繃婊ゅ簲鐢ㄥ満鏅?(Filtering Applications)

### 2.1 璇█璇嗗埆杩囨护 (Language Identification)

璇█璇嗗埆鏄渶鍩虹鐨勮繃婊ゆ楠わ紝鐢ㄤ簬绛涢€夌壒瀹氳瑷€鐨勬枃鏈€?

```python
import fasttext

class LanguageFilter:
    """璇█杩囨护鍣?""
    
    def __init__(self, model_path: str = "lid.176.bin"):
        self.model = fasttext.load_model(model_path)
    
    def filter(self, texts: list[str], 
               target_lang: str = "en", 
               threshold: float = 0.8) -> list[str]:
        """绛涢€夋寚瀹氳瑷€鐨勬枃鏈?""
        filtered = []
        for text in texts:
            predictions = self.model.predict(text)
            lang = predictions[0][0].replace("__label__", "")
            confidence = predictions[1][0]
            
            if lang == target_lang and confidence >= threshold:
                filtered.append(text)
        
        return filtered
```

### 2.2 璐ㄩ噺杩囨护 (Quality Filtering)

璐ㄩ噺杩囨护浣跨敤澶氱淇″彿璇勪及鏂囨湰璐ㄩ噺锛?

#### 2.2.1 鍩轰簬瑙勫垯鐨勮繃婊?

```python
def is_high_quality_rule_based(text: str) -> bool:
    """鍩轰簬瑙勫垯鐨勮川閲忚繃婊?""
    # 妫€鏌ラ暱搴?
    word_count = len(text.split())
    if word_count < 50 or word_count > 100000:
        return False
    
    # 妫€鏌ョ壒娈婂瓧绗︽瘮渚?
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.7:
        return False
    
    # 妫€鏌ラ噸澶嶈姣斾緥
    lines = text.split('\n')
    unique_lines = set(lines)
    if len(unique_lines) / len(lines) < 0.5:
        return False
    
    # 妫€鏌?the"/"be"/"and"绛夊父瑙佽瘝姣斾緥锛堣嫳鏂囪川閲忎俊鍙凤級
    words = text.lower().split()
    common_words = set(["the", "be", "to", "of", "and", "a", "in"])
    common_ratio = sum(1 for w in words if w in common_words) / len(words)
    if common_ratio < 0.02 or common_ratio > 0.3:
        return False
    
    return True
```

#### 2.2.2 鍩轰簬妯″瀷鐨勮繃婊わ紙濡侱CLM鏂规硶锛?

DCLM浣跨敤fastText鍒嗙被鍣ㄥ尯鍒嗛珮璐ㄩ噺锛堝OpenAI鐨刉ebText绛涢€夌粨鏋滐級鍜屼綆璐ㄩ噺鏂囨湰锛?

```python
class QualityClassifier:
    """鍩轰簬fastText鐨勮川閲忓垎绫诲櫒"""
    
    def __init__(self, model_path: str):
        self.model = fasttext.load_model(model_path)
    
    def is_high_quality(self, text: str, threshold: float = 0.5) -> bool:
        predictions = self.model.predict(text)
        label = predictions[0][0]
        confidence = predictions[1][0]
        
        return label == "__label__hq" and confidence >= threshold
```

### 2.3 姣掓€ц繃婊?(Toxicity Filtering)

姣掓€ц繃婊ょЩ闄ゆ湁瀹炽€佹敾鍑绘€ф垨涓嶅綋鍐呭銆?

```python
class ToxicityFilter:
    """姣掓€у唴瀹硅繃婊ゅ櫒"""
    
    def __init__(self, model_path: str, blocklist_path: str = None):
        self.model = fasttext.load_model(model_path)
        self.blocklist = set()
        if blocklist_path:
            with open(blocklist_path, 'r') as f:
                self.blocklist = set(line.strip().lower() for line in f)
    
    def contains_blocklist(self, text: str) -> bool:
        """妫€鏌ユ槸鍚﹀寘鍚粦鍚嶅崟璇嶆眹"""
        words = set(text.lower().split())
        return bool(words & self.blocklist)
    
    def is_toxic(self, text: str, threshold: float = 0.5) -> bool:
        """浣跨敤妯″瀷鍒ゆ柇鏄惁鏈夋瘨"""
        predictions = self.model.predict(text)
        label = predictions[0][0]
        confidence = predictions[1][0]
        return label == "__label__toxic" and confidence >= threshold
    
    def filter(self, texts: list[str]) -> list[str]:
        """杩囨护姣掓€у唴瀹?""
        return [t for t in texts 
                if not self.contains_blocklist(t) and not self.is_toxic(t)]
```

---

## 涓夈€佸幓閲嶆妧鏈?(Deduplication Techniques)

### 3.1 涓轰粈涔堥渶瑕佸幓閲嶏紵

鐮旂┒琛ㄦ槑锛?
- **閲嶅鏁版嵁鎹熷妯″瀷鎬ц兘**: 妯″瀷鍙兘杩囨嫙鍚堝埌閲嶅鍐呭
- **璁粌鏁堢巼涓嬮檷**: 娴垂璁＄畻璧勬簮鍦ㄩ噸澶嶆牱鏈笂
- **闅愮椋庨櫓**: 閲嶅鍐呭鏇村鏄撹璁板繂鍜屾硠闇?

### 3.2 绮剧‘鍘婚噸 (Exact Deduplication)

绮剧‘鍘婚噸绉婚櫎瀹屽叏鐩稿悓鐨勬枃妗ｆ垨娈佃惤銆?

#### 3.2.1 鍩轰簬鍝堝笇鐨勭簿纭幓閲?

```python
import hashlib

class ExactDeduplicator:
    """鍩轰簬鍝堝笇鐨勭簿纭幓閲?""
    
    def __init__(self):
        self.seen_hashes = set()
    
    def get_hash(self, text: str) -> str:
        """璁＄畻鏂囨湰鐨凷HA-256鍝堝笇"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def deduplicate(self, texts: list[str]) -> list[str]:
        """鍘婚櫎閲嶅鏂囨湰"""
        unique_texts = []
        for text in texts:
            h = self.get_hash(text)
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                unique_texts.append(text)
        return unique_texts
```

#### 3.2.2 琛岀骇鍘婚噸

```python
def deduplicate_lines(texts: list[str]) -> list[str]:
    """瀵规瘡绡囨枃妗ｇ殑琛岃繘琛屽幓閲?""
    result = []
    for text in texts:
        lines = text.split('\n')
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        result.append('\n'.join(unique_lines))
    return result
```

### 3.3 甯冮殕杩囨护鍣?(Bloom Filter)

甯冮殕杩囨护鍣ㄦ槸涓€绉嶇┖闂撮珮鏁堢殑姒傜巼鏁版嵁缁撴瀯锛岀敤浜庢祴璇曞厓绱犳槸鍚﹀湪闆嗗悎涓€?

#### 鐗规€?

- **鍙兘鏈夊亣闃虫€?(False Positive)**: 鍙兘閿欒鍦拌涓轰笉鍦ㄩ泦鍚堜腑鐨勫厓绱犲湪闆嗗悎涓?
- **缁濇棤鍋囬槾鎬?(No False Negative)**: 濡傛灉璇翠笉鍦紝鍒欑‘瀹炰笉鍦?
- **绌洪棿鏁堢巼鏋侀珮**: 杩滃皬浜庡瓨鍌ㄥ疄闄呭厓绱?

#### Python瀹炵幇

```python
import mmh3  # MurmurHash3
from bitarray import bitarray
import math

class BloomFilter:
    """甯冮殕杩囨护鍣ㄥ疄鐜?""
    
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        # 璁＄畻鏈€浼樹綅鏁扮粍澶у皬
        self.size = self._optimal_size(expected_items, false_positive_rate)
        # 璁＄畻鏈€浼樺搱甯屽嚱鏁版暟閲?
        self.num_hashes = self._optimal_num_hashes(self.size, expected_items)
        # 鍒濆鍖栦綅鏁扮粍
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)
    
    def _optimal_size(self, n: int, p: float) -> int:
        """璁＄畻鏈€浼樹綅鏁扮粍澶у皬: m = -n*ln(p) / (ln(2)^2)"""
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(m)
    
    def _optimal_num_hashes(self, m: int, n: int) -> int:
        """璁＄畻鏈€浼樺搱甯屾暟閲? k = (m/n) * ln(2)"""
        k = (m / n) * math.log(2)
        return int(k)
    
    def _get_hash_positions(self, item: str) -> list[int]:
        """鑾峰彇鍏冪礌鐨勬墍鏈夊搱甯屼綅缃?""
        positions = []
        for seed in range(self.num_hashes):
            hash_value = mmh3.hash(item, seed) % self.size
            positions.append(hash_value)
        return positions
    
    def add(self, item: str):
        """娣诲姞鍏冪礌"""
        for pos in self._get_hash_positions(item):
            self.bit_array[pos] = 1
    
    def contains(self, item: str) -> bool:
        """妫€鏌ュ厓绱犳槸鍚﹀彲鑳藉瓨鍦?""
        return all(self.bit_array[pos] for pos in self._get_hash_positions(item))
    
    def add_and_check(self, item: str) -> bool:
        """娣诲姞鍏冪礌骞惰繑鍥炴槸鍚﹀凡瀛樺湪锛堢敤浜庡幓閲嶏級"""
        positions = self._get_hash_positions(item)
        was_present = all(self.bit_array[pos] for pos in positions)
        for pos in positions:
            self.bit_array[pos] = 1
        return was_present
```

#### 浣跨敤甯冮殕杩囨护鍣ㄨ繘琛孨-gram鍘婚噸

```python
def bloom_ngram_dedup(texts: list[str], n: int = 5, 
                       bloom_size: int = 10_000_000) -> list[str]:
    """浣跨敤甯冮殕杩囨护鍣ㄨ繘琛孨-gram绾у埆鍘婚噸"""
    bloom = BloomFilter(expected_items=bloom_size)
    result = []
    
    for text in texts:
        words = text.split()
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        # 璁＄畻閲嶅n-gram姣斾緥
        duplicate_count = 0
        for ngram in ngrams:
            if bloom.add_and_check(ngram):
                duplicate_count += 1
        
        duplicate_ratio = duplicate_count / len(ngrams) if ngrams else 0
        
        # 濡傛灉閲嶅鐜囦綆浜庨槇鍊硷紝淇濈暀鏂囨。
        if duplicate_ratio < 0.5:
            result.append(text)
    
    return result
```

### 3.4 杩戜技鍘婚噸锛歁inHash

**MinHash** 鐢ㄤ簬浼拌涓や釜闆嗗悎鐨凧accard鐩镐技搴︼紝鏄繎浼煎幓閲嶇殑鏍稿績绠楁硶銆?

#### Jaccard鐩镐技搴?

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

#### MinHash鍘熺悊

1. 灏嗘枃妗ｈ〃绀轰负n-gram闆嗗悎
2. 瀵归泦鍚堝簲鐢ㄥ涓殢鏈哄搱甯屽嚱鏁?
3. 鍙栨瘡涓搱甯屽嚱鏁扮殑鏈€灏忓€间綔涓虹鍚?
4. 涓や釜鏂囨。鐨勭鍚嶇浉鍚岀殑姒傜巼绛変簬鍏禞accard鐩镐技搴?

```python
import mmh3
import numpy as np

class MinHash:
    """MinHash绛惧悕鐢熸垚鍣?""
    
    def __init__(self, num_hashes: int = 128):
        self.num_hashes = num_hashes
        # 浣跨敤涓嶅悓seed鐢熸垚澶氫釜鍝堝笇鍑芥暟
        self.seeds = list(range(num_hashes))
    
    def get_signature(self, document: str, ngram_size: int = 5) -> np.ndarray:
        """璁＄畻鏂囨。鐨凪inHash绛惧悕"""
        # 鐢熸垚n-gram闆嗗悎
        words = document.split()
        ngrams = set(' '.join(words[i:i+ngram_size]) 
                     for i in range(len(words) - ngram_size + 1))
        
        # 鍒濆鍖栫鍚嶄负鏈€澶у€?
        signature = np.full(self.num_hashes, np.iinfo(np.int32).max, dtype=np.int32)
        
        # 瀵规瘡涓猲-gram璁＄畻鎵€鏈夊搱甯屽€硷紝鍙栨渶灏?
        for ngram in ngrams:
            for i, seed in enumerate(self.seeds):
                hash_value = mmh3.hash(ngram, seed)
                if hash_value < signature[i]:
                    signature[i] = hash_value
        
        return signature
    
    def estimate_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """浼拌涓や釜绛惧悕鐨凧accard鐩镐技搴?""
        return np.mean(sig1 == sig2)
```

### 3.5 灞€閮ㄦ晱鎰熷搱甯?(LSH - Locality Sensitive Hashing)

**LSH** 鐢ㄤ簬楂樻晥鏌ユ壘鐩镐技鏂囨。锛岄伩鍏嶄袱涓ゆ瘮杈冦€?

#### 鍒嗘《绛栫暐 (Banding)

灏哅inHash绛惧悕鍒嗘垚 $b$ 涓甫 (bands)锛屾瘡甯?$r$ 琛岋細
- 濡傛灉浠绘剰涓€涓甫鐨勭鍚嶅畬鍏ㄧ浉鍚岋紝鍒欒涓烘槸鍊欓€夊
- 璋冩暣 $b$ 鍜?$r$ 鍙互鎺у埗鐩镐技搴﹂槇鍊?

```python
class LSH:
    """灞€閮ㄦ晱鎰熷搱甯屽疄鐜?""
    
    def __init__(self, num_hashes: int = 128, num_bands: int = 32):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        # 姣忎釜甯︿竴涓搱甯岃〃
        self.hash_tables = [dict() for _ in range(num_bands)]
    
    def add(self, doc_id: str, signature: np.ndarray):
        """娣诲姞鏂囨。绛惧悕鍒癓SH绱㈠紩"""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(signature[start:end])
            
            if band_signature not in self.hash_tables[band_idx]:
                self.hash_tables[band_idx][band_signature] = []
            self.hash_tables[band_idx][band_signature].append(doc_id)
    
    def find_candidates(self, signature: np.ndarray) -> set:
        """鎵惧埌鍊欓€夌浉浼兼枃妗?""
        candidates = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(signature[start:end])
            
            if band_signature in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][band_signature])
        
        return candidates
```

### 3.6 瀹屾暣鐨勮繎浼煎幓閲嶆祦姘寸嚎

```python
class FuzzyDeduplicator:
    """鍩轰簬MinHash + LSH鐨勮繎浼煎幓閲?""
    
    def __init__(self, num_hashes: int = 128, num_bands: int = 32,
                 similarity_threshold: float = 0.8):
        self.minhash = MinHash(num_hashes)
        self.lsh = LSH(num_hashes, num_bands)
        self.threshold = similarity_threshold
        self.signatures = {}
    
    def deduplicate(self, texts: list[str]) -> list[str]:
        """鎵ц杩戜技鍘婚噸"""
        unique_indices = []
        
        for idx, text in enumerate(texts):
            sig = self.minhash.get_signature(text)
            
            # 鎵惧埌鍊欓€夌浉浼兼枃妗?
            candidates = self.lsh.find_candidates(sig)
            
            is_duplicate = False
            for cand_id in candidates:
                if cand_id in self.signatures:
                    similarity = self.minhash.estimate_similarity(
                        sig, self.signatures[cand_id])
                    if similarity >= self.threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                self.signatures[idx] = sig
                self.lsh.add(idx, sig)
                unique_indices.append(idx)
        
        return [texts[i] for i in unique_indices]
```

---

## 鍥涖€佸畬鏁存祦姘寸嚎 (Complete Pipeline)

### 4.1 绔埌绔暟鎹鐞嗘祦姘寸嚎

```python
class DataProcessingPipeline:
    """瀹屾暣鐨勬暟鎹繃婊ゅ拰鍘婚噸娴佹按绾?""
    
    def __init__(self):
        self.lang_filter = LanguageFilter()
        self.quality_classifier = QualityClassifier("quality_model.bin")
        self.toxicity_filter = ToxicityFilter("toxicity_model.bin")
        self.exact_dedup = ExactDeduplicator()
        self.fuzzy_dedup = FuzzyDeduplicator()
    
    def process(self, texts: list[str], target_lang: str = "en") -> list[str]:
        """鎵ц瀹屾暣鐨勫鐞嗘祦姘寸嚎"""
        # Step 1: 璇█杩囨护
        texts = self.lang_filter.filter(texts, target_lang)
        print(f"After language filter: {len(texts)} documents")
        
        # Step 2: 绮剧‘鍘婚噸
        texts = self.exact_dedup.deduplicate(texts)
        print(f"After exact dedup: {len(texts)} documents")
        
        # Step 3: 璐ㄩ噺杩囨护
        texts = [t for t in texts if self.quality_classifier.is_high_quality(t)]
        print(f"After quality filter: {len(texts)} documents")
        
        # Step 4: 姣掓€ц繃婊?
        texts = self.toxicity_filter.filter(texts)
        print(f"After toxicity filter: {len(texts)} documents")
        
        # Step 5: 杩戜技鍘婚噸
        texts = self.fuzzy_dedup.deduplicate(texts)
        print(f"After fuzzy dedup: {len(texts)} documents")
        
        return texts
```

---

## 浜斻€佸叧閿鐐规€荤粨 (Key Takeaways)

### 杩囨护鎶€鏈€荤粨

| 鎶€鏈?| 鐢ㄩ€?| 浼樼偣 | 缂虹偣 |
|------|------|------|------|
| KenLM | 璐ㄩ噺璇勫垎/DSIR | 蹇€熴€佸噯纭?| 闇€瑕佽缁冭鏂?|
| fastText | 璇█/璐ㄩ噺/姣掓€у垎绫?| 楂樻晥銆佹槗鐢?| 闇€瑕佹爣娉ㄦ暟鎹?|
| DSIR | 鍒嗗竷鍖归厤閲囨牱 | 鐞嗚淇濊瘉 | 闇€瑕佷袱涓瑷€妯″瀷 |

### 鍘婚噸鎶€鏈€荤粨

| 鎶€鏈?| 澶嶆潅搴?| 绌洪棿 | 鐗圭偣 |
|------|--------|------|------|
| 鍝堝笇绮剧‘鍘婚噸 | O(n) | O(n) | 鏃犳崯锛屽彧澶勭悊瀹屽叏鐩稿悓 |
| 甯冮殕杩囨护鍣?| O(n) | O(1)* | 鏈夊亣闃虫€э紝鏋佺渷绌洪棿 |
| MinHash | O(n) | O(n路k) | 杩戜技锛屽鐞嗙浉浼兼枃妗?|
| MinHash + LSH | O(n) 鏈熸湜 | O(n路k) | 閬垮厤涓や袱姣旇緝 |

### 鏈€浣冲疄璺?

1. **娴佹按绾块『搴?*: 鍏堜究瀹滅殑杩囨护锛堣鍒欍€佽瑷€锛夆啋 鍐嶆槀璐电殑锛堟ā鍨嬭川閲忥級鈫?鏈€鍚庡幓閲?
2. **闃堝€艰皟浼?*: 鏍规嵁涓嬫父浠诲姟琛ㄧ幇璋冩暣鍚勮繃婊ら槇鍊?
3. **鍒嗗竷寮忓鐞?*: 瀵逛簬TB绾ф暟鎹紝闇€瑕丮apReduce/Spark鍒嗗竷寮忓疄鐜?
4. **澶氭杩唬**: 鍙兘闇€瑕佸杞幓閲嶈揪鍒版渶浣虫晥鏋?

---

## 鍙傝€冭祫鏂?

1. **KenLM**: Heafield, K. (2011). KenLM: Faster and Smaller Language Model Queries
2. **fastText**: Joulin, A. et al. (2016). Bag of Tricks for Efficient Text Classification
3. **DSIR**: Xie et al. (2023). Data Selection for Language Models via Importance Resampling
4. **MinHash**: Broder, A. (1997). On the Resemblance and Containment of Documents
5. **DCLM**: Li et al. (2024). DataComp-LM: In Search of the next Generation of Training Data

