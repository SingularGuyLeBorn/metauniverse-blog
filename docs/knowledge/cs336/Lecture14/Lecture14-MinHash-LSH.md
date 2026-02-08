# 娣卞叆鎺㈣: MinHash涓嶭SH绠楁硶

鏈枃鏄疞ecture 14鐨勭簿鑻辫ˉ鍏呯瑪璁帮紝娣卞叆璁茶В杩戜技鍘婚噸鐨勬暟瀛﹀師鐞嗭紝鍖呮嫭Jaccard鐩镐技搴︺€丮inHash绛惧悕銆佸眬閮ㄦ晱鎰熷搱甯?LSH)鍙婂叾鐞嗚淇濊瘉銆?

---

## 涓€銆侀棶棰樺畾涔夛細澶ц妯¤繎浼煎幓閲?

### 1.1 绮剧‘鍘婚噸鐨勫眬闄?

绮剧‘鍖归厤鍙兘鎵惧埌**瀹屽叏鐩稿悓**鐨勬枃妗ｃ€傜劧鑰岋細
- 澶嶅埗绮樿创甯稿甫鏈夊井灏忎慨鏀癸紙鏃ユ湡銆佹爣鐐癸級
- 鍚屼竴妯℃澘鐢熸垚鐨勫唴瀹归珮搴︾浉浼间絾涓嶇浉鍚?
- 缈昏瘧銆佹敼鍐欎粛鍖呭惈澶ч噺閲嶅淇℃伅

### 1.2 杩戜技鍘婚噸鐨勭洰鏍?

鎵惧埌**楂樺害鐩镐技**锛堝>80%鐩镐技锛夌殑鏂囨。瀵癸紝骞剁Щ闄ゅ叾涓箣涓€銆?

**鎸戞垬**: N涓枃妗ｏ紝涓や袱姣旇緝闇€瑕?$O(N^2)$ 鏃堕棿銆傚浜庢暟鍗佷嚎鏂囨。锛屼笉鍙锛?

---

## 浜屻€丣accard鐩镐技搴?

### 2.1 瀹氫箟

涓や釜闆嗗悎 $A$ 鍜?$B$ 鐨凧accard鐩镐技搴?

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**鎬ц川**:
- $J(A, B) \in [0, 1]$
- $J(A, A) = 1$
- $J(A, B) = J(B, A)$

### 2.2 鏂囨。琛ㄧず涓洪泦鍚?

灏嗘枃妗ｈ浆鍖栦负**shingles (k-gram)闆嗗悎**:

```python
def shingling(document: str, k: int = 5) -> set:
    """灏嗘枃妗ｈ浆涓簁-shingle闆嗗悎"""
    words = document.split()
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i+k])
        shingles.add(shingle)
    return shingles

# 绀轰緥
doc1 = "the quick brown fox jumps over the lazy dog"
doc2 = "the quick brown fox leaps over the lazy dog"

s1 = shingling(doc1, k=3)  # {"the quick brown", "quick brown fox", ...}
s2 = shingling(doc2, k=3)

jaccard = len(s1 & s2) / len(s1 | s2)  # 璁＄畻鐩镐技搴?
```

### 2.3 k鍊肩殑閫夋嫨

| k鍊?| 鐗圭偣 |
|-----|------|
| k=2-3 | 澶皬锛屽亣闃虫€ч珮锛堝父瑙佺煭璇尮閰嶏級 |
| k=5 | 甯哥敤榛樿鍊?|
| k=9-10 | 鏇翠弗鏍硷紝鍋囬槾鎬у彲鑳藉鍔?|

---

## 涓夈€丮inHash锛氱浉浼煎害鐨勫帇缂╀及璁?

### 3.1 鏍稿績鎬濇兂

**鐩爣**: 灏嗗ぇ闆嗗悎鍘嬬缉鎴愬浐瀹氶暱搴︾殑**绛惧悕**锛屼娇寰楃鍚嶇殑鐩镐技搴﹁繎浼煎師闆嗗悎鐨凧accard鐩镐技搴︺€?

**鍏抽敭瀹氱悊**: 
$$P[h_{min}(A) = h_{min}(B)] = J(A, B)$$

鍏朵腑 $h_{min}(S) = \arg\min_{x \in S} h(x)$ 鏄泦鍚?$S$ 涓搱甯屽€兼渶灏忕殑鍏冪礌銆?

### 3.2 瀹氱悊璇佹槑

鑰冭檻 $A \cup B$ 涓殑鎵€鏈夊厓绱狅紝姣忎釜閮芥湁鍞竴鐨勫搱甯屽€硷紙鍋囪鏃犵鎾烇級銆?

浠?$x^* = \arg\min_{x \in A \cup B} h(x)$ 鏄搱甯屽€兼渶灏忕殑鍏冪礌銆?

- 濡傛灉 $x^* \in A \cap B$锛屽垯 $h_{min}(A) = h_{min}(B) = h(x^*)$
- 濡傛灉 $x^* \in A \setminus B$锛屽垯 $h_{min}(A) = h(x^*) \neq h_{min}(B)$
- 濡傛灉 $x^* \in B \setminus A$锛屽垯 $h_{min}(B) = h(x^*) \neq h_{min}(A)$

鐢变簬鍝堝笇鍑芥暟鏄殢鏈虹殑锛?x^*$ 钀藉湪 $A \cap B$ 涓殑姒傜巼涓?
$$P[x^* \in A \cap B] = \frac{|A \cap B|}{|A \cup B|} = J(A, B)$$

### 3.3 澶氫釜鍝堝笇鍑芥暟

浣跨敤 $k$ 涓嫭绔嬬殑鍝堝笇鍑芥暟 $h_1, ..., h_k$锛岀敓鎴愰暱搴︿负 $k$ 鐨勭鍚?

$$\text{sig}(A) = [h_{1,min}(A), h_{2,min}(A), ..., h_{k,min}(A)]$$

**浼拌鐩镐技搴?*:
$$\hat{J}(A, B) = \frac{1}{k} \sum_{i=1}^{k} \mathbf{1}[h_{i,min}(A) = h_{i,min}(B)]$$

杩欐槸Jaccard鐩镐技搴︾殑**鏃犲亸浼拌**銆?

### 3.4 浼拌鐨勬柟宸?

$$\text{Var}[\hat{J}] = \frac{J(1-J)}{k}$$

**鍚箟**: 绛惧悕瓒婇暱 ($k$ 瓒婂ぇ)锛屼及璁¤秺鍑嗙‘銆?

**瀹炶返涓?*: $k = 128$ 鎴?$k = 256$ 鏄父瑙侀€夋嫨銆?

### 3.5 瀹炵幇浼樺寲锛氬崟鍝堝笇澶氱瀛?

瀹為檯涓婁笉闇€瑕?$k$ 涓畬鍏ㄧ嫭绔嬬殑鍝堝笇鍑芥暟:

```python
import mmh3

def minhash_signature(shingles: set, num_hashes: int = 128) -> list:
    """浣跨敤涓嶅悓绉嶅瓙鐨凪urmurHash鐢熸垚绛惧悕"""
    signature = []
    for seed in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            h = mmh3.hash(shingle, seed)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature
```

---

## 鍥涖€佸眬閮ㄦ晱鎰熷搱甯?(LSH)

### 4.1 闂锛歂涓鍚嶇殑涓や袱姣旇緝

鍗充娇鏈変簡闀垮害 $k$ 鐨勭鍚嶏紝N涓枃妗ｄ粛闇€ $O(N^2 \cdot k)$ 姣旇緝銆?

**LSH鐨勭洰鏍?*: 鍙瘮杈?*鍙兘鐩镐技**鐨勬枃妗ｅ銆?

### 4.2 鍒嗗甫鎶€鏈?(Banding)

灏嗙鍚嶅垎鎴?$b$ 涓甫 (bands)锛屾瘡甯?$r$ 琛岋紝婊¤冻 $b \cdot r = k$銆?

**鍊欓€夊瑙勫垯**: 濡傛灉涓や釜鏂囨。鍦?*浠绘剰涓€涓甫**瀹屽叏鐩稿悓锛屽垯鎴愪负鍊欓€夊銆?

```python
def lsh_buckets(signature: list, num_bands: int = 32) -> list:
    """灏嗙鍚嶅垎甯﹀苟鍝堝笇鍒版《涓?""
    rows_per_band = len(signature) // num_bands
    buckets = []
    
    for band_idx in range(num_bands):
        start = band_idx * rows_per_band
        end = start + rows_per_band
        band = tuple(signature[start:end])
        bucket_id = hash(band)  # 姣忎釜甯︾嫭绔嬪搱甯?
        buckets.append((band_idx, bucket_id))
    
    return buckets
```

### 4.3 姒傜巼鍒嗘瀽

璁剧湡瀹濲accard鐩镐技搴︿负 $s$銆?

**鍗曚釜MinHash鐩哥瓑鐨勬鐜?*: $s$

**涓€涓甫锛?r$琛岋級瀹屽叏鐩稿悓鐨勬鐜?*: $s^r$

**涓€涓甫涓嶅畬鍏ㄧ浉鍚岀殑姒傜巼**: $1 - s^r$

**鎵€鏈?b$涓甫閮戒笉瀹屽叏鐩稿悓鐨勬鐜?*: $(1 - s^r)^b$

**鑷冲皯涓€涓甫鐩稿悓锛堟垚涓哄€欓€夊锛夌殑姒傜巼**:
$$P(\text{candidate}) = 1 - (1 - s^r)^b$$

### 4.4 S鏇茬嚎鐗规€?

杩欎釜姒傜巼鍑芥暟鍛堢幇**S鏇茬嚎**鐗规€?

```
鍊欓€夋鐜?
    1 |              ___________
      |           __/
      |         _/
      |        /
      |      _/
    0 |_____/
      鈹溾攢鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈻?鐩镐技搴
      0   0.2  0.4  0.6  0.8  1.0
```

**闃堝€肩偣**锛堟洸绾块櫋宄锛夌害鍦?
$$t \approx (1/b)^{1/r}$$

### 4.5 鍙傛暟閫夋嫨

| b | r | k=b脳r | 闃堝€尖増 | 鐢ㄩ€?|
|---|---|-------|-------|------|
| 20 | 5 | 100 | 0.55 | 瀹芥澗鍖归厤 |
| 32 | 4 | 128 | 0.60 | 甯哥敤璁剧疆 |
| 50 | 4 | 200 | 0.50 | 鏇村鍊欓€夊 |
| 25 | 10 | 250 | 0.85 | 涓ユ牸鍖归厤 |

**閫夋嫨鍘熷垯**:
- 鎯宠鏇撮珮闃堝€?鈫?澧炲姞 $r$
- 鎯宠鏇村鍊欓€夊锛堜笉婕忔帀鐩镐技鏂囨。锛夆啋 澧炲姞 $b$

### 4.6 瀹屾暣LSH娴佺▼

```python
class LSH:
    def __init__(self, num_hashes=128, num_bands=32):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.hash_tables = [{} for _ in range(num_bands)]
    
    def add(self, doc_id, signature):
        """娣诲姞鏂囨。鍒扮储寮?""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_sig = tuple(signature[start:end])
            
            if band_sig not in self.hash_tables[band_idx]:
                self.hash_tables[band_idx][band_sig] = set()
            self.hash_tables[band_idx][band_sig].add(doc_id)
    
    def query(self, signature):
        """鏌ヨ鍊欓€夌浉浼兼枃妗?""
        candidates = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_sig = tuple(signature[start:end])
            
            if band_sig in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][band_sig])
        
        return candidates
```

---

## 浜斻€佽宸垎鏋?

### 5.1 鍋囬槼鎬т笌鍋囬槾鎬?

| 閿欒绫诲瀷 | 瀹氫箟 | 鍚庢灉 |
|----------|------|------|
| 鍋囬槼鎬?(FP) | 涓嶇浉浼间絾鎴愪负鍊欓€夊 | 澶氬仛涓€娆＄簿纭瘮杈冿紙娴垂鏃堕棿锛?|
| 鍋囬槾鎬?(FN) | 鐩镐技浣嗕笉鎴愪负鍊欓€夊 | 婕忔帀鐪熸鐨勯噸澶嶏紙璐ㄩ噺鎹熷け锛?|

### 5.2 鏉冭　

璁剧洰鏍囬槇鍊间负 $t$:

**鍋囬槼鎬х巼**: $P(\text{candidate} | s < t)$ - S鏇茬嚎鍦?$s < t$ 鍖哄煙鐨勭Н鍒?

**鍋囬槾鎬х巼**: $P(\text{not candidate} | s \geq t)$ - S鏇茬嚎鍦?$s \geq t$ 鍖哄煙鐨勶紙1-绉垎锛?

**浼樺寲鐩爣**:
- 闄嶄綆FN 鈫?澧炲ぇ $b$锛堟洿澶氬甫锛屾洿瀹规槗鍖归厤锛?
- 闄嶄綆FP 鈫?澧炲ぇ $r$锛堟瘡甯︽洿闀匡紝鏇撮毦鍖归厤锛?

### 5.3 涓ら樁娈佃繃婊?

瀹炶返涓€氬父閲囩敤涓ら樁娈垫柟娉?

1. **LSH闃舵**: 蹇€熸壘鍒板€欓€夊锛堝厑璁镐竴浜汧P锛屽敖閲忓噺灏慒N锛?
2. **楠岃瘉闃舵**: 瀵瑰€欓€夊璁＄畻绮剧‘Jaccard鐩镐技搴?

```python
def deduplicate(documents, threshold=0.8):
    # 1. 鐢熸垚绛惧悕
    signatures = {doc_id: minhash(doc) for doc_id, doc in documents}
    
    # 2. LSH鎵惧€欓€夊
    lsh = LSH(num_hashes=128, num_bands=32)
    for doc_id, sig in signatures.items():
        lsh.add(doc_id, sig)
    
    # 3. 楠岃瘉鍊欓€夊
    duplicates = set()
    for doc_id, sig in signatures.items():
        candidates = lsh.query(sig)
        for cand_id in candidates:
            if cand_id >= doc_id:  # 閬垮厤閲嶅姣旇緝
                continue
            # 绮剧‘璁＄畻鐩镐技搴?
            sim = exact_jaccard(documents[doc_id], documents[cand_id])
            if sim >= threshold:
                duplicates.add(doc_id)  # 鏍囪涓洪噸澶?
                break
    
    # 4. 杩斿洖闈為噸澶嶆枃妗?
    return [d for doc_id, d in documents if doc_id not in duplicates]
```

---

## 鍏€佸ぇ瑙勬ā瀹炵幇鑰冭檻

### 6.1 鍒嗗竷寮廙inHash

```
Map闃舵:
  - 姣忎釜Mapper澶勭悊涓€鎵规枃妗?
  - 璁＄畻MinHash绛惧悕
  - 鎸塨and杈撳嚭 (band_id, band_sig) 鈫?doc_id

Reduce闃舵:
  - 姣忎釜Reducer澶勭悊涓€涓猙and
  - 鐩稿悓band_sig鐨勬枃妗ｆ槸鍊欓€夊
  - 杈撳嚭鍊欓€夊鎴栫洿鎺ユ爣璁伴噸澶?
```

### 6.2 鍐呭瓨浼樺寲

- **鍦ㄧ嚎娣诲姞**: 绛惧悕璁＄畻鍚庣珛鍗冲姞鍏SH锛屼笉闇€瀛樺偍鎵€鏈夌鍚?
- **甯冮殕杩囨护鍣ㄨ緟鍔?*: 鐢ㄥ竷闅嗚繃婊ゅ櫒蹇€熷垽鏂璪and鏄惁瑙佽繃
- **鍘嬬缉绛惧悕**: 浣跨敤鏇村皯鐨刡its瀛樺偍MinHash鍊?

### 6.3 澶氳疆鍘婚噸

鍗曡疆LSH鍙兘婕忔帀涓€浜涢噸澶嶏紙鍋囬槾鎬э級銆傚彲浠?
1. 澶氳疆浣跨敤涓嶅悓鐨勯殢鏈虹瀛?
2. 閫愭闄嶄綆闃堝€艰繘琛屽杞?
3. 浣跨敤涓嶅悓鐨剆hingle澶у皬

---

## 鍙傝€冭祫鏂?

1. Broder, A. Z. (1997). On the resemblance and containment of documents
2. Leskovec, J., Rajaraman, A., & Ullman, J. (2020). Mining of Massive Datasets, Chapter 3
3. Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: towards removing the curse of dimensionality
4. Gionis, A., Indyk, P., & Motwani, R. (1999). Similarity search in high dimensions via hashing

