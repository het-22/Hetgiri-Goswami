"""Streamlit entrypoint for Automated Insight Generation Engine."""

import io
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import EngineConfig, REQUIRED_COLUMNS
from src.data.loader import load_csv
from src.data.validator import validate_dataset
from src.data.transformer import prepare_dataset
from src.insights.generator import InsightGenerator
from src.analytics.correlations import calculate_correlation_matrix
from src.visualization.charts import (
    build_severity_bar_chart,
    build_insight_type_pie_chart,
    build_correlation_heatmap,
    build_district_trend_chart,
)
from src.utils.formatting import format_indicator_name

# Configure page layout and visual identity
st.set_page_config(
    page_title="Automated Insight Generation Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles for cards and badges
st.markdown("""
<style>
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #B45309;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #047857;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .insight-card {
        border: 1px solid #E5E7EB;
        border-left: 5px solid #3B82F6;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("Automated Insight Generation Engine")
    st.caption("Data-driven healthcare performance monitoring & statistical auto-analytics")

    # ==========================================
    # DATASET INGESTION & VALIDATION
    # ==========================================
    sample_csv_path = Path("data/healthcare_performance.csv")
    if not sample_csv_path.exists():
        st.error(f"❌ Dataset file not found at '{sample_csv_path}'. Please ensure data file is present.")
        return

    try:
        raw_df = load_csv(sample_csv_path)
    except Exception as exc:
        st.error(f"Failed to read dataset: {exc}")
        return

    # Validate dataset
    validation_report = validate_dataset(raw_df)

    if not validation_report.is_valid:
        st.error("❌ Dataset Schema Validation Failed:")
        for err in validation_report.errors:
            st.error(f"• {err}")
        return

    if validation_report.warnings:
        with st.expander("⚠️ Data Quality Advisories", expanded=False):
            for w in validation_report.warnings:
                st.warning(w)

    # Clean & Prepare dataset
    df_clean, indicator_cols = prepare_dataset(raw_df)

    # Analytical Thresholds in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Analytical Parameters")

    trend_threshold = st.sidebar.slider(
        "Trend Threshold (|% change|)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="Minimum percentage change between periods to trigger a trend alert."
    )

    corr_threshold = st.sidebar.slider(
        "Correlation Threshold (|r|)",
        min_value=0.40,
        max_value=0.99,
        value=0.70,
        step=0.05,
        help="Minimum Pearson correlation coefficient to flag indicator relationship."
    )

    outlier_method = st.sidebar.selectbox(
        "Outlier Detection Method",
        options=["IQR", "Z-score"],
        index=0
    )

    z_threshold = 3.0
    iqr_multiplier = 1.5
    if outlier_method == "Z-score":
        z_threshold = st.sidebar.slider(
            "Z-Score Threshold (σ)",
            min_value=1.5,
            max_value=4.0,
            value=3.0,
            step=0.1
        )
    else:
        iqr_multiplier = st.sidebar.slider(
            "IQR Multiplier",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1
        )

    # Optional Custom Thresholds
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Custom Threshold Rules")
    enable_custom_thresh = st.sidebar.checkbox("Configure Target Minimum/Ceiling", value=False)
    custom_thresholds = {}
    if enable_custom_thresh and indicator_cols:
        target_ind = st.sidebar.selectbox("Indicator", options=indicator_cols)
        min_val = st.sidebar.number_input(f"Target Floor for {target_ind}", value=0.0, step=1.0)
        max_val = st.sidebar.number_input(f"Ceiling for {target_ind}", value=100.0, step=1.0)
        custom_thresholds[target_ind] = {}
        if min_val > 0:
            custom_thresholds[target_ind]["min"] = min_val
        if max_val < 1000:
            custom_thresholds[target_ind]["max"] = max_val

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Interactive UI Filters")
    all_districts = ["All Districts"] + sorted(df_clean["district"].unique().tolist())
    selected_district = st.sidebar.selectbox("Filter District", options=all_districts)

    all_months = ["All Months"] + sorted(df_clean["month"].unique().tolist())
    selected_month = st.sidebar.selectbox("Filter Month", options=all_months)

    all_indicators = ["All Indicators"] + indicator_cols
    selected_indicator = st.sidebar.selectbox("Filter Indicator", options=all_indicators)

    # Filter cleaned dataframe for display
    filtered_df = df_clean.copy()
    if selected_district != "All Districts":
        filtered_df = filtered_df[filtered_df["district"] == selected_district]
    if selected_month != "All Months":
        filtered_df = filtered_df[filtered_df["month"] == selected_month]

    # Instantiate Analytics Engine & Generate Insights
    config = EngineConfig(
        trend_threshold_pct=trend_threshold,
        correlation_threshold=corr_threshold,
        outlier_method=outlier_method,
        z_score_threshold=z_threshold,
        iqr_multiplier=iqr_multiplier,
        custom_thresholds=custom_thresholds
    )
    generator = InsightGenerator(config)
    all_insights = generator.generate_all_insights(df_clean, indicator_cols)

    # Apply UI filters to insights
    filtered_insights = all_insights
    if selected_district != "All Districts":
        filtered_insights = [i for i in filtered_insights if i.entity == selected_district or i.entity == "System-wide"]
    if selected_month != "All Months":
        filtered_insights = [i for i in filtered_insights if i.period == selected_month or i.period == "All Monitored Periods"]
    if selected_indicator != "All Indicators":
        filtered_insights = [i for i in filtered_insights if selected_indicator in i.indicator]

    # ==========================================
    # SECTION 1: KPI OVERVIEW
    # ==========================================
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("Total Rows", len(df_clean))
    with col2:
        st.metric("Districts", len(validation_report.districts))
    with col3:
        st.metric("Indicators", len(indicator_cols))
    with col4:
        st.metric("Total Insights", len(filtered_insights))
    with col5:
        high_cnt = sum(1 for i in filtered_insights if i.severity == "High")
        st.metric("High Severity", high_cnt)
    with col6:
        med_cnt = sum(1 for i in filtered_insights if i.severity == "Medium")
        st.metric("Med Severity", med_cnt)
    with col7:
        low_cnt = sum(1 for i in filtered_insights if i.severity == "Low")
        st.metric("Low Severity", low_cnt)

    # ==========================================
    # SECTION 2: DATA INSPECTION TABS
    # ==========================================
    with st.expander("📋 Dataset Health & Structure (head, info, missing values)", expanded=False):
        tab_head, tab_info, tab_missing = st.tabs(["Data Preview (head)", "Schema Details (info)", "Missing Values"])
        with tab_head:
            st.dataframe(filtered_df.drop(columns=["month_dt", "month_str"], errors="ignore"), use_container_width=True)
        with tab_info:
            buffer = io.StringIO()
            raw_df.info(buf=buffer)
            st.text(buffer.getvalue())
        with tab_missing:
            missing_series = raw_df.isnull().sum()
            missing_table = pd.DataFrame({
                "Column": missing_series.index,
                "Missing Count": missing_series.values,
                "Missing %": (missing_series.values / len(raw_df) * 100).round(2)
            })
            st.dataframe(missing_table, use_container_width=True)

    # ==========================================
    # SECTION 3: AUTOMATED INSIGHTS DISPLAY
    # ==========================================
    st.subheader("💡 Automated Insights Feed")
    
    # Insights filtering toolbar
    col_filter_sev, col_filter_type = st.columns([1, 1])
    with col_filter_sev:
        sev_filter = st.multiselect("Filter by Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with col_filter_type:
        available_types = sorted(list(set(i.type for i in filtered_insights)))
        type_filter = st.multiselect("Filter by Insight Type", available_types, default=available_types)

    display_insights = [
        i for i in filtered_insights
        if i.severity in sev_filter and i.type in type_filter
    ]

    if not display_insights:
        st.info("No insights matching the selected severity and filter criteria.")
    else:
        for ins in display_insights:
            badge_class = f"badge-{ins.severity.lower()}"
            prev_info = f" | Previous: {ins.prev_value}" if ins.prev_value is not None else ""
            change_info = f" | Change: {ins.change_pct:+.1f}%" if ins.change_pct is not None else ""
            
            with st.container():
                st.markdown(f"""
                <div class="insight-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span>
                            <span class="{badge_class}">{ins.severity.upper()}</span>
                            <strong style="margin-left: 10px; font-size: 1.05rem;">{ins.entity} — {format_indicator_name(ins.indicator)}</strong>
                        </span>
                        <span style="color: #6B7280; font-size: 0.9rem;">{ins.insight_id} | {ins.type.replace('_', ' ').title()}</span>
                    </div>
                    <div style="color: #4B5563; font-size: 0.92rem; margin-bottom: 8px;">
                        Period: <strong>{ins.period}</strong> | Value: <strong>{ins.value}</strong>{prev_info}{change_info}
                    </div>
                    <div style="font-size: 0.98rem; color: #1F2937;">
                        {ins.explanation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ==========================================
    # SECTION 4: VISUALIZATIONS & CHARTS
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Analytical Visualizations")
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(build_severity_bar_chart(filtered_insights), use_container_width=True)
    with chart_col2:
        st.plotly_chart(build_insight_type_pie_chart(filtered_insights), use_container_width=True)

    # Per-District Trend Deep Dive
    st.markdown("---")
    st.subheader("📈 District Trend Trajectory")
    trend_dist_col, trend_ind_col = st.columns(2)
    with trend_dist_col:
        dist_options = validation_report.districts
        chosen_dist = st.selectbox("Select District for Deep Dive", options=dist_options, index=0)
    with trend_ind_col:
        chosen_ind = st.selectbox("Select Performance Indicator", options=indicator_cols, index=0)

    st.plotly_chart(
        build_district_trend_chart(df_clean, chosen_dist, chosen_ind, filtered_insights),
        use_container_width=True
    )

    # Correlation Matrix & Heatmap
    st.markdown("---")
    st.subheader("🔗 Pearson Correlation Heatmap")
    corr_matrix = calculate_correlation_matrix(df_clean, indicator_cols)
    st.plotly_chart(build_correlation_heatmap(corr_matrix), use_container_width=True)
    st.caption("ℹ️ *Note: Correlation estimates can be unstable when datasets contain a small number of observations. Correlation does not imply causation.*")

    # ==========================================
    # SECTION 5: DATA & INSIGHT EXPORTS
    # ==========================================
    st.markdown("---")
    st.subheader("💾 Export Findings")
    
    # Prepare exports
    insights_dicts = [i.to_dict() for i in all_insights]
    df_insights_export = pd.DataFrame(insights_dicts)
    if not df_insights_export.empty:
        # Standard exported columns
        export_cols = ["insight_id", "type", "indicator", "entity", "period", "value", "prev_value", "change_pct", "severity", "explanation"]
        available_exp_cols = [c for c in export_cols if c in df_insights_export.columns]
        df_insights_export = df_insights_export[available_exp_cols]

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        csv_insights = df_insights_export.to_csv(index=False).encode("utf-8") if not df_insights_export.empty else b""
        st.download_button(
            label="📥 Download Insights (CSV)",
            data=csv_insights,
            file_name="automated_insights.csv",
            mime="text/csv",
            disabled=df_insights_export.empty
        )
    with exp_col2:
        json_insights = json.dumps(insights_dicts, indent=2).encode("utf-8") if insights_dicts else b"[]"
        st.download_button(
            label="📥 Download Insights (JSON)",
            data=json_insights,
            file_name="automated_insights.json",
            mime="application/json",
            disabled=not insights_dicts
        )
    with exp_col3:
        corr_csv = corr_matrix.to_csv().encode("utf-8") if not corr_matrix.empty else b""
        st.download_button(
            label="📥 Download Correlation Matrix (CSV)",
            data=corr_csv,
            file_name="correlation_matrix.csv",
            mime="text/csv",
            disabled=corr_matrix.empty
        )


if __name__ == "__main__":
    main()
