"""fuzzy.core 的行为测试：gram 集合、相似系数、前缀与长度修正、过滤、截断与并列。"""

import unittest

from fuzzy.core import DEFAULT_NGRAM
from fuzzy.core import PREFIX_CAP
from fuzzy.core import FuzzyMatcher
from fuzzy.core import common_prefix_length
from fuzzy.core import dice
from fuzzy.core import jaccard
from fuzzy.core import ngrams
from fuzzy.core import normalize
from fuzzy.core import score


def keys(hits):
    """命中列表里的候选原文，用来比较名次。"""
    return [hit.key for hit in hits]


class GramSetTest(unittest.TestCase):
    """n-gram 集合：滑窗、短串与空串。"""

    def test_01_gram_sets_cover_every_window_and_short_texts(self):
        """集合要覆盖全部滑窗，末尾那段也算；短串退化成整串，空串给空集。"""
        self.assertEqual(ngrams("abcd", 2), frozenset(["ab", "bc", "cd"]))
        self.assertEqual(ngrams("abcde", 3), frozenset(["abc", "bcd", "cde"]))
        self.assertEqual(ngrams("abab", 2), frozenset(["ab", "ba"]))
        self.assertEqual(ngrams("abc", DEFAULT_NGRAM), frozenset(["ab", "bc"]))
        self.assertNotEqual(ngrams("cats", 2), ngrams("catz", 2))
        self.assertNotEqual(ngrams("cats", 2), frozenset())
        self.assertEqual(ngrams("ab", 2), frozenset(["ab"]))
        self.assertEqual(ngrams("a", 2), frozenset(["a"]))
        self.assertEqual(ngrams("", 2), frozenset())
        self.assertEqual(ngrams("Cat", 2), frozenset(["ca", "at"]))
        self.assertEqual(ngrams("A B", 2), frozenset(["a ", " b"]))
        self.assertEqual(ngrams("ab", 2), ngrams("  AB  ", 2))


class CoefficientTest(unittest.TestCase):
    """两个相似系数的取值口径。"""

    def test_02_coefficients_of_two_gram_sets(self):
        """交集、并集与空集三种情形都要按定义给值。"""
        first = frozenset(["ab", "bc", "cd"])
        second = frozenset(["ab", "bc", "de"])
        self.assertAlmostEqual(jaccard(first, first), 1.0, places=9)
        self.assertAlmostEqual(jaccard(first, second), 2.0 / 4.0, places=9)
        self.assertAlmostEqual(jaccard(first, frozenset(["xy"])), 0.0, places=9)
        self.assertAlmostEqual(jaccard(frozenset(), frozenset()), 1.0, places=9)
        self.assertAlmostEqual(jaccard(frozenset(), first), 0.0, places=9)
        self.assertAlmostEqual(dice(first, first), 1.0, places=9)
        self.assertAlmostEqual(dice(first, second), 4.0 / 6.0, places=9)
        self.assertAlmostEqual(dice(first, frozenset(["xy"])), 0.0, places=9)
        self.assertAlmostEqual(dice(frozenset(), frozenset()), 1.0, places=9)
        self.assertAlmostEqual(dice(frozenset(), first), 0.0, places=9)
        self.assertGreaterEqual(dice(first, second), jaccard(first, second))


class ScoreTest(unittest.TestCase):
    """综合分：满分、区间、前缀奖励与长度惩罚。"""

    def test_03_a_text_matches_itself_and_only_itself(self):
        """完全相同的文本得满分，任何不同的文本都必须低于满分。"""
        for text in ("cat", "cats", "ab", "fuzzy", "x", ""):
            self.assertAlmostEqual(score(text, text), 1.0, places=9)
            self.assertAlmostEqual(score(text, " " + text.upper() + "  "), 1.0, places=9)
        self.assertLess(score("cat", "car"), 1.0)
        self.assertLess(score("fuzzy", "fuzzi"), 1.0)
        self.assertLess(score("cats", "catsanddogs"), 1.0)
        self.assertAlmostEqual(score("", "abc"), 0.0, places=9)
        self.assertAlmostEqual(score("abc", ""), 0.0, places=9)

    def test_04_prefix_and_length_corrections(self):
        """共同前缀按上限封顶，长度差按占比扣分，都没把分数顶出区间。"""
        self.assertEqual(common_prefix_length("abcdef", "abcxyz"), 3)
        self.assertEqual(common_prefix_length("abcdefgh", "abcdefgh"), PREFIX_CAP)
        self.assertEqual(common_prefix_length("", "abc"), 0)
        self.assertEqual(common_prefix_length("Cat", "cat"), 3)
        self.assertAlmostEqual(score("cat", "car"), 0.566666667, places=6)
        self.assertAlmostEqual(score("cat", "cats"), 0.908333333, places=6)
        self.assertGreater(score("cats", "catsanddogs"), 0.4)
        self.assertLess(score("cats", "catsanddogs"), 0.9)


class MatcherTest(unittest.TestCase):
    """候选表：收录口径、过滤、排序、截断与并列。"""

    def test_05_vocabulary_bookkeeping(self):
        """规范化后相同的候选只占一份，移除与列出按原文走。"""
        matcher = FuzzyMatcher(limit=2)
        self.assertEqual(len(matcher), 0)
        self.assertEqual(matcher.candidates(), [])
        self.assertEqual(matcher.rank("cat"), [])
        self.assertIsNone(matcher.best("cat"))
        self.assertEqual(matcher.limit, 2)
        self.assertEqual(matcher.n, DEFAULT_NGRAM)
        self.assertTrue(matcher.add("Cat"))
        self.assertFalse(matcher.add("cat"))
        self.assertFalse(matcher.add("  CAT "))
        self.assertEqual(len(matcher), 1)
        self.assertIn("cat", matcher)
        self.assertNotIn("cats", matcher)
        self.assertTrue(matcher.add("cats"))
        self.assertTrue(matcher.add("car"))
        self.assertEqual(matcher.candidates(), ["Cat", "car", "cats"])
        self.assertTrue(matcher.remove("CAT"))
        self.assertFalse(matcher.remove("cat"))
        self.assertEqual(matcher.candidates(), ["car", "cats"])

    def test_06_truncation_keeps_the_best_hits(self):
        """截断必须发生在排序之后：留下的是分数最高的那几条。"""
        matcher = FuzzyMatcher(limit=2)
        for word in ("catsanddogs", "cat", "car", "dog", "cats"):
            self.assertTrue(matcher.add(word))
        self.assertEqual(keys(matcher.rank("cat")), ["cat", "cats"])
        self.assertEqual(keys(matcher.rank("cat", limit=1)), ["cat"])
        self.assertEqual(keys(matcher.rank("cat", limit=3)), ["cat", "cats", "car"])
        self.assertEqual(keys(matcher.rank("cat", limit=9)),
                         ["cat", "cats", "car", "catsanddogs", "dog"])
        hits = matcher.rank("cat", limit=2)
        self.assertAlmostEqual(hits[0].score, 1.0, places=9)
        self.assertLess(hits[1].score, hits[0].score)

    def test_07_ties_keep_a_fixed_order(self):
        """同分候选按规范化文本的字典序定序，与加入顺序无关。"""
        orders = (("hat", "bat", "dog"), ("bat", "hat", "dog"), ("dog", "hat", "bat"))
        for order in orders:
            matcher = FuzzyMatcher(limit=2)
            for word in order:
                self.assertTrue(matcher.add(word))
            hits = matcher.rank("cat")
            self.assertEqual(keys(hits), ["bat", "hat"])
            self.assertAlmostEqual(hits[0].score, hits[1].score, places=12)
            self.assertEqual(keys(matcher.rank("cat", limit=3)), ["bat", "hat", "dog"])

    def test_08_threshold_is_a_closed_lower_bound(self):
        """阈值以下的候选必须被过滤，分数正好等于阈值的候选必须留下。"""
        matcher = FuzzyMatcher(limit=5)
        for word in ("cat", "cats", "abab", "zzzz"):
            self.assertTrue(matcher.add(word))
        self.assertEqual(keys(matcher.rank("cat", threshold=0.5)), ["cat", "cats"])
        self.assertEqual(keys(matcher.rank("cat", threshold=0.0)),
                         ["cat", "cats", "abab", "zzzz"])
        self.assertEqual(keys(matcher.rank("cat", threshold=1.0)), ["cat"])
        self.assertEqual(keys(matcher.rank("zzzz", threshold=0.0)),
                         ["zzzz", "abab", "cat", "cats"])

    def test_09_ranking_is_sorted_and_best_is_the_top_hit(self):
        """命中按分数降序，分数落在区间内，best 取第一条。"""
        matcher = FuzzyMatcher(limit=3)
        for word in ("dog", "cat", "cats", "car"):
            matcher.add(word)
        hits = matcher.rank("cat")
        self.assertEqual(keys(hits), ["cat", "cats", "car"])
        for hit in hits:
            self.assertLess(hit.score, 1.0 + 1e-9)
            self.assertGreater(hit.score, 0.0 - 1e-9)
        self.assertEqual([round(hit.score, 6) for hit in hits], [1.0, 0.908333, 0.566667])
        self.assertEqual(keys(matcher.rank("cat", limit=2)), ["cat", "cats"])
        best = matcher.best("cat")
        self.assertEqual(best.key, "cat")
        self.assertAlmostEqual(best.score, 1.0, places=9)
        self.assertEqual(keys(matcher.rank("cat", threshold=0.6)), ["cat", "cats"])

    def test_10_rejected_arguments_and_empty_query(self):
        """非法入参报错；空查询按正常查询走，只有空候选自己满分。"""
        with self.assertRaises(TypeError):
            normalize(7)
        with self.assertRaises(TypeError):
            ngrams("abc", "2")
        with self.assertRaises(ValueError):
            ngrams("abc", 0)
        with self.assertRaises(TypeError):
            score(7, "cat")
        with self.assertRaises(ValueError):
            FuzzyMatcher(limit=0)
        with self.assertRaises(TypeError):
            FuzzyMatcher(limit=1.5)
        with self.assertRaises(ValueError):
            FuzzyMatcher(threshold=1.5)
        with self.assertRaises(TypeError):
            FuzzyMatcher(n=True)
        matcher = FuzzyMatcher(limit=3)
        for word in ("abc", "a", ""):
            self.assertTrue(matcher.add(word))
        hits = matcher.rank("")
        self.assertEqual(keys(hits), ["", "a", "abc"])
        self.assertAlmostEqual(hits[0].score, 1.0, places=9)
        self.assertAlmostEqual(hits[1].score, 0.0, places=9)
        self.assertAlmostEqual(hits[2].score, 0.0, places=9)
        self.assertEqual(keys(matcher.rank("", threshold=1.0)), [""])
        self.assertEqual(keys(matcher.rank("zzz", threshold=0.0)), ["", "a", "abc"])


if __name__ == "__main__":
    unittest.main()
