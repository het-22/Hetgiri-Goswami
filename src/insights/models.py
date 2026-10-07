"""Data models for structured findings and insights."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class Insight:
    """Structured insight representation.
    
    Adheres strictly to the required specification:
    - insight_id
    - type: 'trend' | 'outlier' | 'correlation' | 'threshold_breach'
    - indicator: name of metric or composite pair
    - entity: district name or 'system-wide'
    - period: month or date range
    - value: observed numerical value
    - prev_value: previous observation where applicable
    - change_pct: relative change percentage where applicable
    - severity: 'Low' | 'Medium' | 'High'
    - explanation: dynamically generated human-readable narrative
    """
    insight_id: str
    type: str
    indicator: str
    entity: str
    period: str
    value: float
    prev_value: Optional[float] = None
    change_pct: Optional[float] = None
    severity: str = "Low"
    explanation: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert insight to clean dictionary format."""
        data = asdict(self)
        if self.metadata is None:
            data.pop("metadata", None)
        return data


@dataclass
class TrendFinding:
    """Calculated trend finding before narrative transformation."""
    district: str
    indicator: str
    period: str
    current_value: float
    previous_value: float
    change_pct: float
    direction: str  # 'increase' | 'decrease'
    threshold: float
    is_significant: bool


@dataclass
class OutlierFinding:
    """Calculated outlier finding before narrative transformation."""
    district: str
    indicator: str
    month: str
    value: float
    method: str  # 'IQR' | 'Z-score'
    score: float  # z-score or IQR distance
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


@dataclass
class CorrelationFinding:
    """Calculated correlation finding before narrative transformation."""
    indicator_1: str
    indicator_2: str
    correlation: float
    direction: str  # 'positive' | 'negative'
    threshold: float
    sample_size: int
    is_small_sample: bool


@dataclass
class ThresholdBreachFinding:
    """Calculated threshold breach finding before narrative transformation."""
    district: str
    indicator: str
    period: str
    value: float
    breach_type: str  # 'below_minimum' | 'above_maximum'
    threshold_value: float
