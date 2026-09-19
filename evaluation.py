"""賽評生成的評估框架。三個不依賴訓練資料的客觀指標。

為什麼要客觀指標：驗收標準的人工盲測是 3 到 5 場 × 3 系統 = 9 到 15 篇，
樣本太小，而且評分者是作者本人，認得自己的句法習慣，盲測不夠盲。
人工 rubric 留著，但不能單獨承重。

三個指標：
  hallucination_rate  生成文裡出現輸入事實以外的數字，即記一次幻覺。
                      沿用 F1_Race_Analysis 的零容忍捏造原則。
  simplified_ratio    簡體字元比例。基底模型會漂到簡體，Qwen 尤其明顯，
                      這是微調後第一個該檢查的硬指標。
  style_distance      功能詞分布距離。文風相似度的客觀代理，
                      數值越小越接近參考文本。
"""

import math
import re
import unicodedata

import zhconv

# 數字型態：秒數、圈數、名次、百分比、時間格式。事實查核只比這些。
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 中文高頻功能詞。文風的指紋主要落在這些詞的使用比例上，
# 而不是在內容詞，所以拿它們當代理指標。
FUNCTION_WORDS = (
    "的", "了", "在", "是", "有", "和", "就", "不", "也", "都",
    "而", "但", "卻", "又", "還", "才", "很", "更", "最", "太",
    "把", "被", "讓", "給", "從", "到", "對", "與", "以", "為",
    "這", "那", "其", "之", "所", "者", "如", "若", "則", "並",
)


def extract_numbers(text):
    """抓出所有數字。事實查核只認數字，因為那是社群會抓錯的東西。"""
    return set(NUMBER_RE.findall(text))


def hallucination_rate(input_facts, generated):
    """生成文裡有幾成的數字是輸入事實裡沒有的。

    回傳 (rate, 幻覺數字清單)。rate 是幻覺數字數 / 生成文數字總數。
    生成文沒有任何數字時回傳 (0.0, [])，因為沒有可查核的斷言。
    """
    allowed = extract_numbers(input_facts)
    produced = NUMBER_RE.findall(generated)
    if not produced:
        return 0.0, []
    invented = [n for n in produced if n not in allowed]
    return len(invented) / len(produced), sorted(set(invented))


def simplified_ratio(text):
    """簡體字元佔所有漢字的比例。微調後應該接近 0。

    判定方式是逐字做簡轉繁，字形有變的才算簡體。手打「簡體專用字清單」
    不可靠，繁簡同形字（圈、胎、乎）會被誤判，第一版就是這樣錯的。
    逐字轉換而非整句，是因為整句轉換在少數情況會改變長度而失去對齊。
    """
    han = [c for c in text if unicodedata.category(c) == "Lo" and "一" <= c <= "鿿"]
    if not han:
        return 0.0
    return sum(zhconv.convert(c, "zh-tw") != c for c in han) / len(han)


def function_word_profile(text):
    """功能詞的相對分布。長度已歸一化，所以不同長度的文本可以比。"""
    counts = {w: text.count(w) for w in FUNCTION_WORDS}
    total = sum(counts.values())
    if total == 0:
        return {w: 0.0 for w in FUNCTION_WORDS}
    return {w: counts[w] / total for w in FUNCTION_WORDS}


def style_distance(reference, candidate):
    """兩份文本的功能詞分布距離，用 Jensen-Shannon。

    範圍 0 到 1，0 代表分布完全相同。用 JS 而不是 KL 是因為它對稱、
    有界，而且其中一邊出現 0 機率時不會爆掉。
    """
    p = function_word_profile(reference)
    q = function_word_profile(candidate)

    def kl(a, b):
        return sum(
            a[w] * math.log2(a[w] / b[w])
            for w in FUNCTION_WORDS
            if a[w] > 0 and b[w] > 0
        )

    m = {w: (p[w] + q[w]) / 2 for w in FUNCTION_WORDS}
    js = (kl(p, m) + kl(q, m)) / 2
    return round(max(0.0, min(1.0, js)), 4)


def evaluate(input_facts, generated, reference_style):
    """三個指標一次算完，回傳可直接進報告表格的 dict。"""
    rate, invented = hallucination_rate(input_facts, generated)
    return {
        "hallucination_rate": round(rate, 4),
        "invented_numbers": invented,
        "simplified_ratio": round(simplified_ratio(generated), 4),
        "style_distance": style_distance(reference_style, generated),
        "char_count": len(generated),
    }
