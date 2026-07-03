"""Configuration package for MAFA."""
from .settings import Settings
from .constants import LEAKAGE_KEYWORDS, PERFECT_SCORE_CHALLENGES, DEFAULT_CHALLENGE

__all__ = [
    "Settings",
    "LEAKAGE_KEYWORDS",
    "PERFECT_SCORE_CHALLENGES",
    "DEFAULT_CHALLENGE",
]
