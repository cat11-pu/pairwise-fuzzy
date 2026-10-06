"""The fuzzy scoring kernel package."""

from fuzzy.core import DEFAULT_LIMIT
from fuzzy.core import DEFAULT_NGRAM
from fuzzy.core import FuzzyError
from fuzzy.core import FuzzyMatcher
from fuzzy.core import Match
from fuzzy.core import common_prefix_length
from fuzzy.core import dice
from fuzzy.core import jaccard
from fuzzy.core import ngrams
from fuzzy.core import normalize
from fuzzy.core import score

__all__ = [
    "DEFAULT_LIMIT",
    "DEFAULT_NGRAM",
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
