"""Master insight generation pipeline.

Coordinates findings across analytical modules, applies severity engine,
and generates standardized Insight records with unique IDs.
"""

from typing import List, Dict, Optional
import pandas as pd
from src.config import EngineConfig
from src.insights.models import Insight
from src.insights.severity import assess_severity
from src.insights.templates import (
    render_trend_explanation,
    render_outlier_explanation,
    render_correlation_explanation,
    render_breach_explanation,
)
from src.analytics.trends import detect_trends
from src.analytics.outliers import detect_outliers
from src.analytics.correlations import detect_correlations
from src.analytics.thresholds import detect_threshold_breaches


class InsightGenerator:
    """Orchestrates analytics detection and produces standardized Insight objects."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._counter = 0

    def _next_id(self) -> str:
        """Generate consecutive unique insight ID (e.g. INS-0001)."""
        self._counter += 1
        return f"INS-{self._counter:04d}"

    def reset_counter(self) -> None:
        """Reset ID counter for fresh analysis run."""
        self._counter = 0

    def generate_all_insights(
        self,
        df_clean: pd.DataFrame,
        indicator_cols: List[str]
    ) -> List[Insight]:
        """Execute full analytics suite and compile standardized insights list.
        
        Args:
            df_clean: Cleaned DataFrame.
            indicator_cols: List of indicator column names.
            
        Returns:
            List of Insight objects ordered by severity (High -> Medium -> Low).
        """
        self.reset_counter()
        insights: List[Insight] = []

        if df_clean.empty or not indicator_cols:
            return insights

        # 1. Trend Detection
        trend_findings = detect_trends(
            df_clean,
            indicator_cols,
            trend_threshold_pct=self.config.trend_threshold_pct
        )
        for tf in trend_findings:
            sev = assess_severity(tf)
            explanation = render_trend_explanation(tf)
            insights.append(
                Insight(
                    insight_id=self._next_id(),
                    type="trend",
                    indicator=tf.indicator,
                    entity=tf.district,
                    period=tf.period,
                    value=tf.current_value,
                    prev_value=tf.previous_value,
                    change_pct=tf.change_pct,
                    severity=sev,
                    explanation=explanation,
                    metadata={"direction": tf.direction, "threshold": tf.threshold}
                )
            )

        # 2. Outlier Detection
        outlier_findings = detect_outliers(
            df_clean,
            indicator_cols,
            method=self.config.outlier_method,
            z_score_threshold=self.config.z_score_threshold,
            iqr_multiplier=self.config.iqr_multiplier
        )
        for of in outlier_findings:
            sev = assess_severity(of)
            explanation = render_outlier_explanation(of)
            insights.append(
                Insight(
                    insight_id=self._next_id(),
                    type="outlier",
                    indicator=of.indicator,
                    entity=of.district,
                    period=of.month,
                    value=of.value,
                    prev_value=None,
                    change_pct=None,
                    severity=sev,
                    explanation=explanation,
                    metadata={"method": of.method, "score": of.score}
                )
            )

        # 3. Correlation Detection
        corr_findings = detect_correlations(
            df_clean,
            indicator_cols,
            correlation_threshold=self.config.correlation_threshold
        )
        for cf in corr_findings:
            sev = assess_severity(cf)
            explanation = render_correlation_explanation(cf)
            insights.append(
                Insight(
                    insight_id=self._next_id(),
                    type="correlation",
                    indicator=f"{cf.indicator_1} & {cf.indicator_2}",
                    entity="System-wide",
                    period="All Monitored Periods",
                    value=cf.correlation,
                    prev_value=None,
                    change_pct=None,
                    severity=sev,
                    explanation=explanation,
                    metadata={
                        "indicator_1": cf.indicator_1,
                        "indicator_2": cf.indicator_2,
                        "sample_size": cf.sample_size,
                        "is_small_sample": cf.is_small_sample
                    }
                )
            )

        # 4. Threshold Breach Detection
        if self.config.custom_thresholds:
            breach_findings = detect_threshold_breaches(
                df_clean,
                self.config.custom_thresholds
            )
            for bf in breach_findings:
                sev = assess_severity(bf)
                explanation = render_breach_explanation(bf)
                insights.append(
                    Insight(
                        insight_id=self._next_id(),
                        type="threshold_breach",
                        indicator=bf.indicator,
                        entity=bf.district,
                        period=bf.period,
                        value=bf.value,
                        prev_value=None,
                        change_pct=None,
                        severity=sev,
                        explanation=explanation,
                        metadata={
                            "breach_type": bf.breach_type,
                            "threshold_value": bf.threshold_value
                        }
                    )
                )

        # Sort insights deterministically: Severity High -> Medium -> Low, then by ID
        severity_rank = {"High": 0, "Medium": 1, "Low": 2}
        insights.sort(key=lambda x: (severity_rank.get(x.severity, 3), x.insight_id))

        return insights
