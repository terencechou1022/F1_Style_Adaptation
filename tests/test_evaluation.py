"""評估框架的測試。每個指標測「該抓到」與「不該誤報」兩個方向。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import (
    evaluate,
    function_word_profile,
    hallucination_rate,
    simplified_ratio,
    style_distance,
)

FACTS = """
分站：2026 日本大獎賽
桿位：Verstappen 1:28.197
冠軍：Verstappen
第二名：Norris 落後 3.242 秒
最快單圈：Piastri 第 42 圈 1:30.115
Verstappen 進站 2 次
"""


class TestHallucination:
    def test_忠實引用不算幻覺(self):
        text = "Verstappen 以 1:28.197 拿下桿位，正賽再進站 2 次奪冠。"
        rate, invented = hallucination_rate(FACTS, text)
        assert rate == 0.0
        assert invented == []

    def test_編造的數字要被抓到(self):
        text = "Verstappen 領先 Norris 足足 9.876 秒。"
        rate, invented = hallucination_rate(FACTS, text)
        assert rate > 0
        assert "9.876" in invented

    def test_沒有數字時不算幻覺(self):
        rate, invented = hallucination_rate(FACTS, "這場比賽精彩絕倫。")
        assert rate == 0.0
        assert invented == []

    def test_比例是編造數佔生成數(self):
        # 三個數字，其中 5566 是編造的
        text = "Norris 落後 3.242 秒，Piastri 跑出第 42 圈，還有 5566。"
        rate, invented = hallucination_rate(FACTS, text)
        assert invented == ["5566"]
        assert abs(rate - 1 / 3) < 1e-9


class TestSimplified:
    def test_純繁體應接近零(self):
        assert simplified_ratio("這場比賽的輪胎策略很關鍵，車隊選擇提前進站。") == 0.0

    def test_簡體要被抓到(self):
        assert simplified_ratio("这场比赛的轮胎策略很关键") > 0

    def test_沒有漢字時回零(self):
        assert simplified_ratio("Verstappen 1:28.197") == 0.0


class TestStyle:
    def test_同一份文本距離為零(self):
        t = "他的進站時機把握得很好，但是輪胎的衰退還是讓他掉了名次。"
        assert style_distance(t, t) == 0.0

    def test_風格不同要有距離(self):
        a = "他的策略很好，而且輪胎的管理也不錯，所以就贏了。"
        b = "冠軍 Verstappen。桿位 Verstappen。最快單圈 Piastri。"
        assert style_distance(a, b) > 0

    def test_沒有功能詞時分布全零(self):
        assert all(v == 0.0 for v in function_word_profile("ABC 123").values())

    def test_距離有界(self):
        d = style_distance("的的的的的", "了了了了了")
        assert 0.0 <= d <= 1.0


class TestEvaluate:
    def test_回傳完整欄位(self):
        got = evaluate(FACTS, "Verstappen 以 1:28.197 奪桿。", "他的表現很好。")
        assert set(got) == {
            "hallucination_rate",
            "invented_numbers",
            "simplified_ratio",
            "style_distance",
            "char_count",
        }
        assert got["hallucination_rate"] == 0.0

    def test_三個系統可以並排比較(self):
        reference = "他的進站時機把握得很好，但是輪胎的衰退還是讓他掉了名次。"
        candidates = {
            "忠實繁中": "Verstappen 的表現很好，他在第 42 圈之後就守住了。",
            "有幻覺": "Verstappen 的表現很好，領先了 99.999 秒。",
            "簡體漂移": "他的表现很好，轮胎的管理也不错。",
        }
        rows = {k: evaluate(FACTS, v, reference) for k, v in candidates.items()}
        assert rows["忠實繁中"]["hallucination_rate"] == 0.0
        assert rows["有幻覺"]["hallucination_rate"] > 0
        assert rows["簡體漂移"]["simplified_ratio"] > 0
        assert rows["忠實繁中"]["simplified_ratio"] == 0.0
