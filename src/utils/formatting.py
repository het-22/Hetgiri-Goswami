"""Text and number formatting utilities."""

from typing import Union


def format_indicator_name(indicator: str) -> str:
    """Format snake_case or raw column name into Title Case for presentation.
    
    Examples:
        anc_coverage -> ANC Coverage
        high_risk_cases -> High Risk Cases
        institutional_delivery -> Institutional Delivery
    """
    acronyms = {"anc": "ANC", "id": "ID", "iqr": "IQR"}
    parts = indicator.replace("_", " ").split()
    formatted = [acronyms.get(p.lower(), p.capitalize()) for p in parts]
    return " ".join(formatted)


def format_number(val: Union[int, float], decimals: int = 1) -> str:
    """Safely format numbers with fixed decimal places or integer formatting."""
    if val is None:
        return "N/A"
    if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
        return str(int(val))
    return f"{val:.{decimals}f}"


def format_percent(val: Union[int, float], decimals: int = 1, show_sign: bool = False) -> str:
    """Safely format a percentage value."""
    if val is None:
        return "N/A"
    sign = "+" if show_sign and val > 0 else ""
    return f"{sign}{val:.{decimals}f}%"
