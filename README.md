# Equipment Failure Risk Ranking (Probability × Consequence)

**Author:** Rashmita Perumanla

**Project Domain:** Predictive Maintenance | Data Analytics | Reliability Engineering

**Tools:** Python, Pandas, Matplotlib, Excel, Power BI

---

## 1. Project Overview

This project analyzes equipment failure risk across 15 equipment segments using the publicly available, synthetic **AI4I 2020 Predictive Maintenance Dataset**, containing 10,000 machine records.

The objective is to identify which equipment segments have the highest failure risk by combining:

- **Failure Probability:** Estimated from historical failure frequencies in the dataset.
- **Failure Consequence:** Assigned engineering consequence scores based on project assumptions.
- **Risk Score:** Calculated by combining failure probability and consequence scores.

The analysis includes data validation, failure-mode analysis, risk ranking, a risk matrix, an Excel-based risk assessment model, and recommendations framed as hypotheses for further investigation.

The project demonstrates how data analytics can support reliability engineering and predictive maintenance decision-making.

---

## 2. Problem Statement

Which equipment segments carry the highest failure risk, and which reliability actions are worth testing first?

The analysis aims to:

1. Identify equipment segments with higher failure probabilities.
2. Understand the relationship between tool wear and equipment failures.
3. Compare failure patterns across equipment types.
4. Rank equipment segments based on probability and consequence.
5. Identify major failure modes.
6. Develop data-driven recommendations for reliability improvement.

---

## 3. Dataset Description

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository

**Dataset size:** 10,000 records

**Dataset type:** Synthetic industrial predictive maintenance data.

The dataset contains equipment operating conditions, including:

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine failure indicator
- Individual failure-mode indicators

The analysis divides equipment into segments based on:

- Equipment type: L, M, H
- Tool-wear bands: Below 50, 50–100, 100–150, 150–200, and 200+ minutes

This produces 15 equipment segments for comparative risk assessment.

---

## 4. Key Results and Findings

### 4.1 Overall Equipment Failure Rate

Out of 10,000 machine records:

- Total machine failures: 339
- Overall failure rate: 3.39%

Failure rates by equipment type:

| Equipment Type | Failure Rate |
|---|---:|
| L | 3.92% |
| M | 2.77% |
| H | 2.09% |

Equipment Type L has the highest observed failure rate.

### 4.2 Tool Wear and Failure Probability

Tool wear shows a substantial association with equipment failure probability.

| Tool Wear | Failure Probability |
|---|---:|
| Below 200 minutes | 2.21%–2.66% |
| 200 minutes or more | 15.36% |

The results indicate that equipment operating at 200 minutes or more of tool wear has a considerably higher observed failure probability.

### 4.3 Failure Mode Analysis

The three major failure modes are:

- Heat Dissipation Failure (HDF)
- Overstrain Failure (OSF)
- Power Failure (PWF)

Together, these account for 82.57% of the 373 failure-mode flags.

Note: Failure-mode flags are not equivalent to unique machine failures because multiple failure-mode indicators can occur in the same record.

### 4.4 Equipment Risk Ranking

The project evaluates 15 equipment segments using probability and consequence scores.

Four segments are classified as Critical:

| Equipment Segment | Risk Score | Risk Category |
|---|---:|---|
| H × 200+ | 20 | Critical |
| L × 200+ | 15 | Critical |
| M × 200+ | 15 | Critical |
| H × 100–150 | 15 | Critical |

The L × 200+ segment has the highest observed failure probability at 18.29%.

It also accounts for 92 of the 339 machine failures, representing approximately 27.1% of all failures.

**Important:** Risk rankings depend on the assumed consequence scores. These scores are engineering assumptions rather than measured financial, operational, or safety consequences.

---

## 5. Data Quality and Validation

Data validation was performed to identify inconsistencies in the dataset.

Key findings:

- 18 records had RNF = 1 while machine_failure = 0.
- 9 machine failures had no corresponding failure-mode flag.
- These records were retained without alteration.

The analysis distinguishes between machine failures and failure-mode flags to avoid treating them as equivalent measurements.

### Statistical Uncertainty

Some equipment segments have relatively few observed failures.

For example, H × 100–150 has only 6 failures.

Therefore, its risk ranking may be sensitive to a small number of observations.

95% Wilson confidence intervals are provided to communicate uncertainty in estimated failure probabilities.

---

## 6. Risk Assessment Methodology

The project uses a Probability × Consequence approach.

### Step 1: Data Preparation

- Load the AI4I 2020 dataset.
- Inspect records and columns.
- Validate machine failure indicators.
- Check failure-mode inconsistencies.
- Prepare the cleaned dataset.

### Step 2: Equipment Segmentation

Divide the equipment into 15 segments using:

- Three equipment types: L, M, H.
- Five tool-wear bands.

### Step 3: Failure Probability Calculation

Calculate observed failure probability for each segment:

Failure Probability = Number of Failures / Total Records in Segment

### Step 4: Consequence Assessment

Assign consequence scores based on engineering assumptions.

These scores are used to represent the relative consequence of failure across equipment segments.

They are not measured downtime costs, repair costs, production losses, or safety impacts.

### Step 5: Risk Scoring

Combine probability and consequence scores to calculate the segment-level risk ranking.

The resulting scores are used to classify equipment segments into risk categories.

### Step 6: Visualization and Reporting

Generate:

- Risk ranking tables.
- Failure-mode frequency analysis.
- Equipment condition comparisons.
- Pareto charts.
- Risk matrix.
- Excel-based risk assessment workbook.

---

## 7. Project Folder Structure

```text
failure-risk-ranking/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   └── processed/
│       └── clean.csv
│
├── src/
│   ├── risk_analysis.py
│   └── make_powerbi_tables.py
│
├── outputs/
│   ├── risk_ranking.csv
│   ├── segment_summary.csv
│   ├── failure_mode_counts.csv
│   ├── condition_comparison.csv
│   │
│   ├── charts/
│   │   ├── pareto_failure_modes.png
│   │   └── risk_matrix.png
│   │
│   └── excel/
│       └── risk_analysis.xlsx
│
├── docs/
│   ├── methodology.md
│   ├── assumptions.md
│   ├── validation_report.md
│   ├── data_dictionary.md
│   └── recommendations.md
│
├── powerbi/
│   └── BUILD_GUIDE.md
│
├── requirements.txt
└── README.md
```

---

## 8. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data processing and analysis |
| Pandas | Data cleaning and manipulation |
| Matplotlib | Data visualization |
| Excel | Formula-driven risk assessment |
| Openpyxl | Excel workbook generation |
| LibreOffice | Formula recalculation |
| Power BI | Data visualization and reporting |

---

## 9. How to Run the Project

### Prerequisites

Install Python and the required libraries.

### Step 1: Clone the Repository

```bash
git clone https://github.com/rashmita-21/failure-risk-ranking.git
```

### Step 2: Navigate to the Project

```bash
cd failure-risk-ranking
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Main Analysis

```bash
python src/risk_analysis.py
```

This script performs:

- Data cleaning
- Validation
- Equipment segmentation
- Failure probability calculations
- Risk scoring
- CSV generation
- Chart generation

### Step 5: Generate Power BI Tables

```bash
python src/make_powerbi_tables.py
```

This generates Power BI-ready data, including wear bands and segment labels.

### Step 6: Activate Excel Reconciliation

Paste the values from:

`data/processed/clean.csv`

into:

`Clean_Data!A2:O10001`

in the Excel workbook, without the CSV header, to activate the reconciliation columns.

---

## 10. Project Outputs

The project generates the following deliverables:

| Output | Description |
|---|---|
| risk_ranking.csv | Equipment segment risk rankings |
| segment_summary.csv | Segment-level failure statistics |
| failure_mode_counts.csv | Failure-mode frequency analysis |
| condition_comparison.csv | Equipment condition comparisons |
| pareto_failure_modes.png | Pareto analysis of failure modes |
| risk_matrix.png | Probability versus consequence risk matrix |
| risk_analysis.xlsx | Formula-driven Excel risk model |

Power BI-ready CSVs are also generated for dashboard development.

---

## 11. Reliability Recommendations

Based on the observed patterns, the following hypotheses are worth testing:

**1. Prioritize High Tool-Wear Equipment**

Investigate whether additional inspections or maintenance interventions before tool wear reaches 200 minutes reduce failures.

**2. Investigate Type L Equipment**

Type L has the highest observed overall failure rate. Investigate whether operating conditions, maintenance practices, or equipment characteristics explain this difference.

**3. Review Critical-Risk Segments**

Prioritize investigation of the four Critical segments identified in the risk assessment.

**4. Investigate Major Failure Modes**

Focus diagnostic investigations on HDF, OSF, and PWF because they account for a substantial proportion of failure-mode flags.

**5. Validate Consequence Assumptions**

Replace assumed consequence scores with actual downtime, repair cost, production loss, and safety information when available.

These are proposed reliability actions to investigate, not proven maintenance interventions.

---

## 12. Limitations

The project has several limitations:

- The AI4I 2020 dataset is synthetic and does not represent actual industrial operating records.
- No actual repair costs, downtime, production losses, or safety consequences are available.
- Consequence scores are engineering assumptions.
- The analysis is based on equipment segments rather than individual machine-level predictive models.
- Some segments have small failure counts, making their rankings less stable.
- Rankings depend on the selected consequence scores.
- Observed associations do not establish causation.

The results should be interpreted as a comparative risk-screening exercise rather than a validated operational failure prediction system.

For more details, refer to the documentation in the `docs/` folder.

---

## 13. Future Improvements

Potential future extensions include:

- Developing machine-learning models for failure prediction.
- Comparing Logistic Regression, Random Forest, and XGBoost.
- Incorporating actual maintenance and downtime data.
- Performing consequence-score sensitivity analysis.
- Building an interactive Power BI risk dashboard.
- Developing a maintenance prioritization framework.
- Evaluating model performance using precision, recall, F1-score, and ROC-AUC.

---

## 14. Data Source and Citation

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Repository:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset is publicly available and synthetic.

Please follow the citation and attribution requirements specified on the UCI dataset page.

---

## 15. Author

**Rashmita Perumanla**

GitHub: https://github.com/rashmita-21

Email: rashmitaperumanla2112@gmail.com

---
