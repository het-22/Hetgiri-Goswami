"""Configuration defaults and schemas for Automated Insight Generation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


REQUIRED_COLUMNS: List[str] = [
    "month",
    "district",
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]

DEFAULT_TREND_THRESHOLD: float = 10.0  # percentage change threshold
DEFAULT_CORRELATION_THRESHOLD: float = 0.70  # Pearson correlation magnitude threshold
DEFAULT_OUTLIER_METHOD: str = "IQR"  # 'IQR' or 'Z-score'
DEFAULT_Z_SCORE_THRESHOLD: float = 3.0  # standard deviations
DEFAULT_IQR_MULTIPLIER: float = 1.5


@dataclass
class EngineConfig:
    """Runtime configuration for the analytics engine."""
    trend_threshold_pct: float = DEFAULT_TREND_THRESHOLD
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    outlier_method: str = DEFAULT_OUTLIER_METHOD
    z_score_threshold: float = DEFAULT_Z_SCORE_THRESHOLD
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER
    custom_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # custom_thresholds format: {indicator_name: {"min": float, "max": float}}
