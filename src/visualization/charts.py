"""Visualization module using Plotly."""

from typing import List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.insights.models import Insight
from src.utils.formatting import format_indicator_name


SEVERITY_COLORS = {
    "High": "#EF4444",    # Vibrant Red
    "Medium": "#F59E0B",  # Vibrant Amber
    "Low": "#10B981"      # Vibrant Emerald
}


def build_severity_bar_chart(insights: List[Insight]) -> go.Figure:
    """Generate a clean bar chart showing distribution of insights across severity levels."""
    if not insights:
        fig = go.Figure()
        fig.update_layout(title="No insights detected under current thresholds")
        return fig

    counts = {"High": 0, "Medium": 0, "Low": 0}
    for ins in insights:
        if ins.severity in counts:
            counts[ins.severity] += 1

    df_counts = pd.DataFrame({
        "Severity": list(counts.keys()),
        "Count": list(counts.values())
    })

    fig = px.bar(
        df_counts,
        x="Severity",
        y="Count",
        color="Severity",
        color_discrete_map=SEVERITY_COLORS,
        text="Count",
        title="Detected Insights by Severity Level"
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Severity Level",
        yaxis_title="Total Insights",
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def build_insight_type_pie_chart(insights: List[Insight]) -> go.Figure:
    """Generate a donut chart showing breakdown by insight type."""
    if not insights:
        fig = go.Figure()
        return fig

    type_counts = {}
    for ins in insights:
        t = ins.type.replace("_", " ").title()
        type_counts[t] = type_counts.get(t, 0) + 1

    df_types = pd.DataFrame({
        "Type": list(type_counts.keys()),
        "Count": list(type_counts.values())
    })

    fig = px.pie(
        df_types,
        names="Type",
        values="Count",
        hole=0.45,
        title="Breakdown by Insight Category",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def build_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Generate an annotated correlation heatmap from correlation matrix."""
    if corr_matrix.empty or len(corr_matrix) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient numeric indicators or zero variance to construct correlation matrix.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#6B7280")
        )
        fig.update_layout(template="plotly_white", height=380)
        return fig

    # Human-friendly labels
    display_names = [format_indicator_name(c) for c in corr_matrix.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=display_names,
            y=display_names,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(title="Pearson r")
        )
    )
    fig.update_layout(
        title="Indicator Pearson Correlation Matrix",
        template="plotly_white",
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(tickangle=-30)
    )
    return fig


def build_district_trend_chart(
    df_clean: pd.DataFrame,
    district: str,
    indicator: str,
    trend_insights: Optional[List[Insight]] = None
) -> go.Figure:
    """Generate a time-series line chart for a specific district and indicator."""
    subset = df_clean[df_clean["district"] == district].copy()
    if subset.empty or indicator not in subset.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data available for district '{district}' and indicator '{indicator}'",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(template="plotly_white", height=360)
        return fig

    sort_cols = ["month_dt"] if "month_dt" in subset.columns else ["month"]
    subset = subset.sort_values(by=sort_cols).reset_index(drop=True)

    indicator_name = format_indicator_name(indicator)

    fig = go.Figure()
    
    # Base line plot
    fig.add_trace(
        go.Scatter(
            x=subset["month"].astype(str),
            y=subset[indicator],
            mode="lines+markers",
            name=indicator_name,
            line=dict(color="#2563EB", width=3),
            marker=dict(size=8, color="#1D4ED8")
        )
    )

    # Highlight significant trends if provided
    if trend_insights:
        for ins in trend_insights:
            if ins.entity == district and ins.indicator == indicator and ins.type == "trend":
                # Find matching period
                fig.add_trace(
                    go.Scatter(
                        x=[str(ins.period)],
                        y=[ins.value],
                        mode="markers",
                        name=f"Trend Alert ({ins.severity})",
                        marker=dict(
                            size=14,
                            color=SEVERITY_COLORS.get(ins.severity, "#EF4444"),
                            symbol="diamond"
                        ),
                        hoverinfo="text",
                        hovertext=f"{ins.severity} Alert: {ins.change_pct:+.1f}% change"
                    )
                )

    fig.update_layout(
        title=f"{district}: Monthly Trajectory of {indicator_name}",
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title=indicator_name,
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
