# Automated Insight Generation — AI/ML Auto-Analytics Engine

Production-grade, general-purpose Python statistical analytics engine and interactive dashboard for automated insight discovery across district-level healthcare performance indicators.

---

## 1. Project Overview

Public health administrators and clinical directors regularly review district-level indicator datasets. Manually inspecting rows and time-series to detect shifts, anomalies, or systemic trade-offs is slow, error-prone, and reactive.

The **Automated Insight Generation Engine** provides a generalized, production-ready pipeline that ingests tabular performance data, mathematically isolates anomalies and shifts, evaluates empirical evidence strength, and synthesizes clear, structured human-readable insights with Low/Medium/High severity ratings.

---

## 2. Problem Statement

Traditional reporting systems rely on hardcoded rules or district-specific static report generators (e.g., hardcoding scripts that only look for "Ahmedabad" or specific predefined indices). This architecture is fragile, non-transferable, and costly to maintain.

This engine solves that challenge by maintaining a strict decoupling between:
1. **The Ingested Schema** (dynamic column discovery and type verification)
2. **Analytical Algorithms** (IQR, Z-score, relative chronological delta, Pearson correlation matrix)
3. **Evidence-based Severity Scoring** (signal strength ratios vs. arbitrary static cutoffs)
4. **Dynamic Natural Language Rendering** (synthesized purely from computed statistics)

---

## 3. Core Features

- **Automated Trend Detection**: Evaluates period-over-period percentage shifts per district-indicator pair, accounting for division-by-zero edge cases.
- **Dual Outlier Detection**: Offers both non-parametric **Interquartile Range (IQR)** and parametric **Z-Score** anomaly detection with configurable sensitivities.
- **Correlation Discovery**: Computes a full Pearson correlation matrix, eliminates duplicate pairs ($A-B$ vs. $B-A$), and flags low-sample volatility warnings.
- **Threshold Breach Engine**: Allows user-defined floors and ceilings for any indicator without altering code.
- **Explainable Severity Engine**: Assigns `Low`, `Medium`, or `High` severity purely based on signal strength ratios ($|\Delta| / \text{threshold}$).
- **Dynamic Natural Language Synthesis**: Zero hardcoded strings or entities; narratives are constructed on the fly from analytical output objects.
- **Interactive Streamlit Dashboard**: Full KPI cards, severity distribution charts, correlation heatmaps, district trajectories, and interactive filters.
- **Multi-Format Export**: One-click export for insights in CSV and JSON formats, plus correlation matrices in CSV format.

---

## 4. Architecture

```
DATA SOURCE (CSV Upload or Default)
        ↓
SCHEMA VALIDATION (Missing data, column checks, duplicate detection)
        ↓
DATA TRANSFORMATION & SORTING (Chronological sorting, numeric coercion)
        ↓
ANALYTICAL ENGINE (Parallel statistical modules)
  ├── Trend Detection (Percentage shift against threshold)
  ├── Outlier Detection (IQR fences or Z-score deviations)
  ├── Correlation Engine (Pearson r matrix)
  └── Threshold Breach (Floor / Ceiling limits)
        ↓
STRUCTURED FINDINGS (Typed Dataclass Objects)
        ↓
SEVERITY ENGINE (Signal-strength ratio assessment)
        ↓
DYNAMIC NARRATIVE GENERATOR (Template synthesis)
        ↓
STREAMLIT UI / PLOTLY VISUALIZATIONS / EXPORTS (CSV & JSON)
```

### Decoupled Directory Structure

```
AIML2/
├── app.py                      # Interactive Streamlit application
├── data/
│   └── healthcare_performance.csv # Sample benchmark dataset
├── src/
│   ├── config.py               # Engine configuration & schema constants
│   ├── data/
│   │   ├── loader.py           # Robust CSV ingestion
│   │   ├── validator.py        # Schema & data quality verification
│   │   └── transformer.py      # Reshaping, datetime parsing, sorting
│   ├── analytics/
│   │   ├── trends.py           # Trend percentage & direction detection
│   │   ├── outliers.py         # IQR & Z-score outlier detection
│   │   ├── correlations.py     # Pearson correlation calculation
│   │   └── thresholds.py       # Configurable target breach detection
│   ├── insights/
│   │   ├── models.py           # Dataclass definitions for findings & insights
│   │   ├── generator.py        # Master pipeline orchestrator & unique ID assignment
│   │   ├── severity.py         # Explainable signal-strength severity engine
│   │   └── templates.py        # Dynamic string narrative synthesis
│   ├── visualization/
│   │   └── charts.py           # Plotly charts (heatmaps, time series, bars)
│   └── utils/
│       └── formatting.py       # Title formatting, number & percentage helpers
├── tests/
│   ├── test_loader.py          # Ingestion tests
│   ├── test_validator.py       # Schema validation tests
│   ├── test_trends.py          # Trend calculation & zero-division tests
│   ├── test_outliers.py        # IQR and Z-score outlier tests
│   ├── test_correlations.py    # Correlation matrix & deduplication tests
│   ├── test_severity.py        # Severity ratio scoring tests
│   └── test_insights.py        # End-to-end pipeline & dynamic narrative tests
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 5. Technology Stack

- **Python 3.11+** (Tested on Python 3.13)
- **Pandas**: Vectorized tabular transformations and correlation computation
- **NumPy & SciPy**: Percentiles, statistical summaries, and variance analysis
- **Streamlit**: Interactive user dashboard
- **Plotly**: Interactive charts and heatmaps
- **Pytest**: Automated test suite

---

## 6. Installation & Execution

### 1. Clone or Navigate to Directory
```bash
cd AIML2
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Pytest Test Suite
```bash
pytest tests/
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 7. Dataset Schema

The platform expects a CSV containing at least:
- `month`: Chronological period (e.g. `2026-07` or `YYYY-MM`)
- `district`: Name of geographic entity or district
- Two or more numerical indicators (e.g. `anc_coverage`, `institutional_delivery`, `immunization`, `high_risk_cases`)

### Example:
```csv
month,district,anc_coverage,institutional_delivery,immunization,high_risk_cases
2026-07,Ahmedabad,85,91,93,10
2026-08,Ahmedabad,69,90,92,13
2026-07,Mehsana,84,89,92,11
2026-08,Mehsana,42,88,91,28
```

---

## 8. Analytics Methodology

### A. Trend Detection Formula
For every district and numerical indicator ordered chronologically:
$$\text{pct\_change} = \left(\frac{\text{current} - \text{previous}}{\text{previous}}\right) \times 100$$
A trend finding is triggered when:
$$|\text{pct\_change}| \ge \text{trend\_threshold}$$
*(Default: $10\%$, configurable in UI).*
Zero division is safeguarded against: transitions from $0 \to 0$ produce $0\%$, while transitions from $0 \to x$ ($x \neq 0$) trigger an alert indicating baseline initiation.

### B. Outlier Detection
#### 1. Interquartile Range (IQR) — Non-parametric:
- $Q_1 = 25\text{th percentile}$, $Q_3 = 75\text{th percentile}$, $\text{IQR} = Q_3 - Q_1$
- $\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$
- $\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$
- An observation is flagged if $x < \text{Lower Bound}$ or $x > \text{Upper Bound}$.

#### 2. Z-Score — Parametric:
$$z = \frac{x - \mu}{\sigma}$$
An observation is flagged if $|z| \ge \text{threshold}$ *(Default: $3.0\sigma$, configurable in UI)*.

### C. Pearson Correlation Methodology
For indicator pairs $(X, Y)$:
$$r_{xy} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
Pairs where $|r| \ge \text{correlation\_threshold}$ are flagged.

> [!WARNING]
> **Sample Size & Causation Notice:**
> 1. Correlation estimates can be unstable when the dataset contains only a small number of observations ($N < 10$).
> 2. **Correlation does not imply causation.** Strong co-movement between indices does not mean changes in one drive changes in the other.

---

## 9. Explainable Severity Scoring Engine

Rather than relying on arbitrary hardcoded cutoffs, severity reflects **signal strength**:

| Insight Type | Signal Strength Metric | Low Severity | Medium Severity | High Severity |
| :--- | :--- | :--- | :--- | :--- |
| **Trend** | $\frac{|\text{pct\_change}|}{\text{threshold}}$ | $< 1.4$ | $1.4 \le \text{ratio} < 2.0$ | $\ge 2.0$ |
| **Outlier (IQR)** | $\frac{\text{distance beyond fence}}{\text{IQR}}$ | $< 0.3$ | $0.3 \le \text{ratio} < 1.0$ | $\ge 1.0$ |
| **Outlier (Z-score)** | $|z|$ | $< 3.0$ | $3.0 \le |z| < 4.0$ | $\ge 4.0$ |
| **Correlation** | $|r|$ | $< 0.75$ | $0.75 \le |r| < 0.90$ | $\ge 0.90$ |
| **Threshold Breach** | Overshoot ratio | $< 15\%$ | $15\% \le \text{ratio} < 30\%$ | $\ge 30\%$ |

---

## 10. Sample Output Structure

```json
{
  "insight_id": "INS-0004",
  "type": "trend",
  "indicator": "anc_coverage",
  "entity": "Mehsana",
  "period": "2026-08",
  "value": 42.0,
  "prev_value": 84.0,
  "change_pct": -50.0,
  "severity": "High",
  "explanation": "Mehsana ANC Coverage decreased by 50.0% compared with the previous month (84 -> 42), exceeding the configured 10.0% significant-change threshold."
}
```

---

## 11. Testing & Verification

The project includes unit and integration tests across all modules:
```
tests/test_correlations.py .....
tests/test_insights.py .....
tests/test_loader.py .....
tests/test_outliers.py .....
tests/test_severity.py .....
tests/test_trends.py .....
tests/test_validator.py .....
```
Run all tests via:
```bash
pytest -v
```

---

## 12. Limitations & Future Roadmap

### Current Limitations:
1. **Bivariate Correlation**: Currently detects pairwise linear relationships; non-linear or multi-way interactions require polynomial/copula modeling.
2. **Short-horizon Trends**: Evaluates consecutive period deltas; seasonal decomposition (e.g. SARIMA/Prophet) requires multi-year monthly data.

### Future Roadmap:
- Optional local LLM summarizer plug-in using structured facts.
- Automatic seasonal decomposition and anomaly forecasting.
- Automated PDF report generator.
