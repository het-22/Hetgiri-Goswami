"""Unit and end-to-end tests for Insight Generation Pipeline."""

import pandas as pd
from src.config import EngineConfig
from src.data.transformer import prepare_dataset
from src.insights.generator import InsightGenerator


def test_insight_generation_end_to_end_on_sample_data():
    df = pd.DataFrame({
        "month": ["2026-07", "2026-08", "2026-07", "2026-08"],
        "district": ["Ahmedabad", "Ahmedabad", "Surat", "Surat"],
        "anc_coverage": [85.0, 69.0, 81.0, 83.0],
        "institutional_delivery": [91.0, 90.0, 87.0, 89.0],
        "immunization": [93.0, 92.0, 91.0, 92.0],
        "high_risk_cases": [10.0, 13.0, 13.0, 12.0]
    })
    
    df_clean, indicators = prepare_dataset(df)
    config = EngineConfig(trend_threshold_pct=10.0, correlation_threshold=0.7)
    generator = InsightGenerator(config)
    insights = generator.generate_all_insights(df_clean, indicators)

    assert len(insights) > 0
    
    # 1. Check insight structure
    for ins in insights:
        assert ins.insight_id.startswith("INS-")
        assert ins.type in ["trend", "outlier", "correlation", "threshold_breach"]
        assert ins.severity in ["Low", "Medium", "High"]
        assert len(ins.explanation) > 0
        
    # 2. Verify Ahmedabad ANC coverage drop was detected dynamically
    ahmedabad_trend = [
        i for i in insights 
        if i.entity == "Ahmedabad" and i.indicator == "anc_coverage" and i.type == "trend"
    ]
    assert len(ahmedabad_trend) == 1
    item = ahmedabad_trend[0]
    assert item.value == 69.0
    assert item.prev_value == 85.0
    assert round(item.change_pct, 1) == -18.8
    assert "decreased by 18.8%" in item.explanation


def test_no_hardcoded_district_dependency():
    # Use synthetic districts and new metric names
    df = pd.DataFrame({
        "month": ["2026-01", "2026-02"],
        "district": ["Mars_Sector_9", "Mars_Sector_9"],
        "oxygen_level": [100.0, 60.0]  # -40% drop
    })
    df_clean, indicators = prepare_dataset(df)
    generator = InsightGenerator(EngineConfig(trend_threshold_pct=10.0))
    insights = generator.generate_all_insights(df_clean, indicators)

    assert len(insights) == 1
    ins = insights[0]
    assert ins.entity == "Mars_Sector_9"
    assert ins.indicator == "oxygen_level"
    assert "Mars_Sector_9 Oxygen Level decreased by 40.0%" in ins.explanation
    assert ins.severity == "High"


def test_insight_ids_are_consecutive_and_unique():
    df = pd.DataFrame({
        "month": ["2026-01", "2026-02", "2026-03"],
        "district": ["DistA", "DistA", "DistA"],
        "metric_x": [100.0, 50.0, 10.0]
    })
    df_clean, indicators = prepare_dataset(df)
    generator = InsightGenerator(EngineConfig(trend_threshold_pct=10.0))
    insights = generator.generate_all_insights(df_clean, indicators)

    ids = [i.insight_id for i in insights]
    assert len(ids) == len(set(ids))
    assert ids[0] == "INS-0001"
