"""fuzzy：确定性的模糊匹配打分内核。

组件：
    normalize             文本规整：去首尾空白、内部空白折叠成一个空格、统一小写
    ngrams                文本的 n-gram 集合，长度不足 n 的文本退化成整串一个 gram
    jaccard / dice        两个 gram 集合的相似系数
    common_prefix_length  两串从头连续相同的字符数，最多 PREFIX_CAP 个
    score                 综合分：系数均值加前缀奖励减长度惩罚，恒在 0 到 1 之间
    Match                 一条命中：候选原文与分数
    FuzzyMatcher          候选表的收录、过滤、排序与 Top-K 截断

约定：
    * 打分只取决于两个文本本身：不读时钟、不用随机数、不做任何 I/O，同样的输入
      永远给同样的输出；
    * 分数恒在 0.0 到 1.0 之间，两个文本规范化后完全相同时得 1.0，不同文本拿不满；
    * 命中先按阈值过滤再截断：阈值是闭区间下界，分数正好等于阈值的候选必须留下，
      低于阈值的一律丢掉；
    * 截断发生在排序之后，最多 limit 条；顺序是分数降序，同分按规范化文本的
      字典序升序，与候选的加入顺序无关。
"""

__all__ = [
    "DEFAULT_LIMIT",
    "DEFAULT_NGRAM",
    "LENGTH_PENALTY",
    "PREFIX_CAP",
    "PREFIX_WEIGHT",
    "TOLERANCE",
    "FuzzyError",
    "FuzzyMatcher",
    "Match",
    "common_prefix_length",
    "dice",
    "jaccard",
    "ngrams",
    "normalize",
    "score",
]

DEFAULT_NGRAM = 2
PREFIX_CAP = 4
PREFIX_WEIGHT = 0.30
LENGTH_PENALTY = 0.20
DEFAULT_LIMIT = 3
TOLERANCE = 1e-9


class FuzzyError(Exception):
    """调用方式或取值不合法时抛出。"""


def normalize(text):
    """文本的规整形状：去首尾空白、内部连续空白折成一个空格、统一小写。

    规整只影响比较，不影响调用方手里的原文。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是 str")
    return " ".join(text.split()).lower()


def _check_n(n):
    """n-gram 长度：大于零的整数。"""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n 必须是 int")
    if n < 1:
        raise ValueError("n 必须是正整数")
    return n


def _check_limit(limit):
    """条数上界：大于零的整数。"""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit 必须是 int")
    if limit < 1:
        raise ValueError("limit 必须是正整数")
    return limit


def _check_threshold(threshold):
    """最低分：0 到 1 之间的实数。"""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold 必须是实数")
    value = float(threshold)
    if value != value:
        raise ValueError("threshold 不能是 NaN")
    if value < 0.0 or value > 1.0:
        raise ValueError("threshold 必须落在 0 到 1 之间")
    return value


def ngrams(text, n=DEFAULT_NGRAM):
    """文本的 n-gram 集合，重复出现的 gram 只留一份。

    长度不足 n 的文本退化成整串一个 gram：空文本给空集，非空短串给自己。
    这样短串之间仍然可以比较，也不必给窗口补位；够长的文本按滑窗切开，
    从头一路切到尾，末尾那个窗口同样算一份。
    """
    text = normalize(text)
    n = _check_n(n)
    if len(text) < n:
        return frozenset() if not text else frozenset([text])
    return frozenset(text[i:i + n] for i in range(len(text) - n))


def _as_gram_set(values):
    """把调用方给的 gram 序列收成集合。"""
    if isinstance(values, str):
        raise TypeError("gram 集合不接受裸字符串")
    return frozenset(values)


def jaccard(left, right):
    """Jaccard 系数 |A 交 B| 除以 |A 并 B|。

    两边都空视为两个文本完全一致，给 1.0；只有一边空给 0.0。
    """
    left = _as_gram_set(left)
    right = _as_gram_set(right)
    if not left or not right:
        return 1.0
    return len(left & right) / len(left | right)


def dice(left, right):
    """Dice 系数 2 倍交集除以两边大小之和。

    两边都空给 1.0；只有一边空给 0.0。同样的两个集合，Dice 不小于 Jaccard。
    """
    left = _as_gram_set(left)
    right = _as_gram_set(right)
    total = len(left) + len(right)
    if total == 0:
        return 1.0
    return float(len(left & right)) / total


def common_prefix_length(left, right):
    """两串从头连续相同的字符数，最多 PREFIX_CAP 个。

    上限用来挡住长串的开头把分数整体抬满：超过上限的部分不再加奖励。
    """
    left = normalize(left)
    right = normalize(right)
    limit = len(left) if len(left) < len(right) else len(right)
    if limit > PREFIX_CAP:
        limit = PREFIX_CAP
    count = 0
    while count < limit and left[count] == right[count]:
        count += 1
    return count


def _length_gap(left, right):
    """长度差相对较长串的占比，两个空串给 0.0。

    占比让长度惩罚与串的绝对长度无关，长串与短串一视同仁。
    """
    return abs(len(left) - len(right))


def _prefix_bonus(left, right):
    """前缀奖励：共同前缀每满一个字符加一份权重，最多 PREFIX_WEIGHT。"""
    return PREFIX_WEIGHT * common_prefix_length(left, right)


def _clamp(value):
    """把分数夹回 0.0 到 1.0。"""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _not_below(value, floor):
    """value 是否达到 floor 这条下界。"""
    return value > floor + TOLERANCE


def score(query, candidate, n=DEFAULT_NGRAM):
    """两个文本的综合相似度，恒在 0.0 到 1.0 之间。

    两个系数各占一半打底，共同前缀加奖励，长度差按占比扣分，最后夹回区间。
    """
    left = normalize(query)
    right = normalize(candidate)
    left_grams = ngrams(left, n)
    right_grams = ngrams(right, n)
    base = 0.5 * (jaccard(left_grams, right_grams) + dice(left_grams, right_grams))
    value = base + _prefix_bonus(left, right) - LENGTH_PENALTY * _length_gap(left, right)
    return _clamp(value)


class Match:
    """一条命中：候选原文与它的分数。"""

    __slots__ = ("key", "score")

    def __init__(self, key, score):
        self.key = key
        self.score = score

    def __repr__(self):
        return "Match(key=%r, score=%r)" % (self.key, self.score)


class FuzzyMatcher:
    """一份候选表，加上按查询打分的过滤、排序与截断。

    参数：
        limit      默认最多返回几条命中，大于零的整数
        threshold  默认最低分，0 到 1 之间的实数
        n          n-gram 长度，大于零的整数

    规范化后相同的候选只占一份，保留首次加入时的原文写法。
    """

    def __init__(self, limit=DEFAULT_LIMIT, threshold=0.0, n=DEFAULT_NGRAM):
        self._limit = _check_limit(limit)
        self._threshold = _check_threshold(threshold)
        self._n = _check_n(n)
        self._entries = {}

    @property
    def limit(self):
        """默认的条数上界。"""
        return self._limit

    @property
    def threshold(self):
        """默认的最低分。"""
        return self._threshold

    @property
    def n(self):
        """n-gram 长度。"""
        return self._n

    def __len__(self):
        """表里有多少个不同的候选。"""
        return len(self._entries)

    def __contains__(self, candidate):
        """某个候选是否在表里，比较按规范化后的形状。"""
        return normalize(candidate) in self._entries

    def add(self, candidate):
        """收录一个候选；规范化后已经收过就返回 False，不再占第二份。"""
        if not isinstance(candidate, str):
            raise TypeError("candidate 必须是 str")
        norm = normalize(candidate)
        if norm in self._entries:
            return False
        self._entries[norm] = candidate
        return True

    def remove(self, candidate):
        """移除一个候选；返回它此前是否在表里。"""
        if not isinstance(candidate, str):
            raise TypeError("candidate 必须是 str")
        norm = normalize(candidate)
        if norm in self._entries:
            del self._entries[norm]
            return True
        return False

    def candidates(self):
        """表里所有候选原文，按字典序。"""
        return sorted(self._entries.values())

    def rank(self, query, limit=None, threshold=None):
        """查询的命中列表：先按阈值过滤，再按分数降序截断。

        limit 省略时用构造时的默认值，threshold 同理。命中最多 limit 条，
        留下的必须是分数最高的那几条；顺序是分数降序，同分按规范化文本的
        字典序升序，与候选的加入顺序无关。
        """
        if not isinstance(query, str):
            raise TypeError("query 必须是 str")
        limit = self._limit if limit is None else _check_limit(limit)
        threshold = self._threshold if threshold is None else _check_threshold(threshold)
        scored = []
        for norm, raw in self._entries.items():
            value = score(query, norm, self._n)
            if _not_below(value, threshold):
                scored.append((value, norm, raw))
        scored = scored[:limit]
        scored.sort(key=lambda row: -row[0])
        return [Match(raw, value) for value, _, raw in scored]

    def best(self, query, threshold=None):
        """分数最高的那条命中；表空或全被过滤时返回 None。"""
        hits = self.rank(query, limit=1, threshold=threshold)
        return hits[0] if hits else None
