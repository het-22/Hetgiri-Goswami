"""Dynamic narrative generation templates.

All strings and explanations are constructed dynamically from analytical outputs.
NO hardcoded district names or hardcoded numerical assertions are permitted.
"""

from src.insights.models import (
    TrendFinding,
    OutlierFinding,
    CorrelationFinding,
    ThresholdBreachFinding,
)
from src.utils.formatting import format_indicator_name, format_number, format_percent


def render_trend_explanation(finding: TrendFinding) -> str:
    """Dynamically generate explanation for a trend finding."""
    indicator_title = format_indicator_name(finding.indicator)
    action = "decreased" if finding.direction == "decrease" else "increased"
    pct_formatted = format_percent(abs(finding.change_pct), decimals=1)
    thresh_formatted = format_percent(finding.threshold, decimals=1)
    
    return (
        f"{finding.district} {indicator_title} {action} by {pct_formatted} "
        f"compared with the previous month ({format_number(finding.previous_value)} -> {format_number(finding.current_value)}), "
        f"exceeding the configured {thresh_formatted} significant-change threshold."
    )


def render_outlier_explanation(finding: OutlierFinding) -> str:
    """Dynamically generate explanation for an outlier finding."""
    indicator_title = format_indicator_name(finding.indicator)
    val_formatted = format_number(finding.value)
    
    if finding.method == "IQR":
        direction_clause = "below lower boundary" if finding.lower_bound is not None and finding.value < finding.lower_bound else "above upper boundary"
        fence_clause = f"[{format_number(finding.lower_bound)}, {format_number(finding.upper_bound)}]"
        return (
            f"Observed value of {val_formatted} for {indicator_title} in {finding.district} ({finding.month}) "
            f"is an statistical outlier ({direction_clause} of expected IQR range {fence_clause}, score={finding.score})."
        )
    else:  # Z-score
        return (
            f"Observed value of {val_formatted} for {indicator_title} in {finding.district} ({finding.month}) "
            f"deviates significantly from dataset mean with a Z-score of {finding.score}."
        )


def render_correlation_explanation(finding: CorrelationFinding) -> str:
    """Dynamically generate explanation for a correlation finding.
    
    Adheres strictly to statistical wording:
    - strong positive/negative correlation
    - explicitly avoids claiming causation
    - includes sample size caveats
    """
    ind1 = format_indicator_name(finding.indicator_1)
    ind2 = format_indicator_name(finding.indicator_2)
    strength = "Very strong" if abs(finding.correlation) >= 0.85 else "Strong"
    r_val = f"{finding.correlation:+.2f}"
    
    warning_clause = ""
    if finding.is_small_sample:
        warning_clause = f" Warning: Small sample size (N={finding.sample_size}); correlation estimates may be unstable."

    return (
        f"{strength} {finding.direction} correlation (r = {r_val}) detected between "
        f"{ind1} and {ind2} across monitored districts (threshold: |r| >= {finding.threshold:.2f})."
        f"{warning_clause} (Note: correlation does not imply causation)."
    )


def render_breach_explanation(finding: ThresholdBreachFinding) -> str:
    """Dynamically generate explanation for a threshold breach."""
    indicator_title = format_indicator_name(finding.indicator)
    val_formatted = format_number(finding.value)
    thresh_formatted = format_number(finding.threshold_value)
    
    if finding.breach_type == "below_minimum":
        return (
            f"{finding.district} reported {indicator_title} of {val_formatted} in {finding.period}, "
            f"which falls below the configured target threshold of {thresh_formatted}."
        )
    else:
        return (
            f"{finding.district} reported {indicator_title} of {val_formatted} in {finding.period}, "
            f"which exceeds the configured ceiling threshold of {thresh_formatted}."
        )
