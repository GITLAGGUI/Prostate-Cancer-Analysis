
# Prostate Cancer Analytics Dashboard

![Dashboard Overview](Dashboard_Overview.png)

---

> **Comprehensive Interactive Analysis for Prostate Cancer Patient Data**
> Built with **Streamlit** and **Plotly** for data-driven decision-making and quality improvement in prostate cancer care.

---

## Project Overview

> This dashboard delivers an interactive, data-driven analysis of prostate cancer patient data, including clinical indicators, treatment outcomes, and actionable insights for clinicians and researchers.
>
> **Key Benefits:**
>
> - Advanced data visualization
> - Real-time risk stratification
> - Professional, responsive UI
> - Actionable clinical recommendations

---

## Dashboard Preview

**Dashboard Filters:**

- Age Range (slider)
- Cancer Stage (dropdown)
- Treatment Type (dropdown)
- Diagnosis Year (dropdown)

**Executive Summary:**

- **Total Patients:** 993 `<span style="color:green;">`(+43)
- **Average Age:** 64.5 years `<span style="color:red;">`(-0.5)
- **High-Risk Cases:** 406 `<span style="color:green;">`(+40.9%)
- **Excellent Outcomes:** 479 `<span style="color:green;">`(+48.2%)

**Clinical Overview:**

- Cancer Stage Distribution (Pie Chart)
- Key Findings:
  - **Early Detection:** 68.2% of cases are Stage I-II
  - **Treatment Success:** 48.2% show excellent outcomes
  - **Risk Assessment:** 40.9% are high-risk cases
  - **Follow-up:** Average 28.6 months monitoring

---

## Key Features

| Feature                       | Description                                                              |
| ----------------------------- | ------------------------------------------------------------------------ |
| **Interactive Visuals** | Demographics, clinical indicators, and outcomes with real-time filtering |
| **Risk Stratification** | Advanced risk assessment and treatment effectiveness analysis            |
| **Professional UI**     | Responsive design, smooth transitions, and customizable filters          |
| **Clinical Insights**   | Actionable recommendations and research directions                       |

---

## Dashboard Sections

- **Executive Summary**: High-level metrics and trends
- **Patient Demographics**: Age, stage, and treatment breakdowns
- **Clinical Indicators**: PSA, Gleason, and stage analysis
- **Treatment Analysis**: Outcomes by treatment type
- **Outcomes & Survival**: Survival rates and follow-up
- **Insights & Recommendations**: Key findings and next steps

---

## Quick Start

```sh
# 1. Clone the repository
git clone <repo-url>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run dashboard.py
```

Open your browser at [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
Prostate_Cancer/
├── dashboard.py         # Main Streamlit dashboard app
├── run_dashboard.py     # Alternate run script
├── requirements.txt     # Python dependencies
├── Notebook.ipynb       # Data exploration notebook
├── README.md            # Project documentation
├── QUICKSTART.md        # Quick start guide
├── Dashboard_Overview.png # Dashboard screenshot
└── data/
    └── prostate_cancer_data.csv
```

---

## Data Requirements

**Required Columns:**

| Column        | Description       |
| ------------- | ----------------- |
| Patient_ID    | Unique identifier |
| Age           | Patient age       |
| PSA_Level     | PSA level (ng/mL) |
| Gleason_Score | Score (6-10)      |
| Stage         | Cancer stage      |
| Treatment     | Treatment type    |

---

## Live Demo & Research

- [Kaggle Notebook: Prostate Cancer Risk Analysis – Data-Driven Insight](https://www.kaggle.com/code/joellaggui/prostate-cancer-risk-analysis-data-driven-insight)

---

## Collaboration & Contact

- [Portfolio](https://joellaggui.vercel.app)
- [Kaggle Profile](https://www.kaggle.com/joellaggui)

For project discussions or research partnerships, please use the contact form on my portfolio.

---

## Technical Specifications

| Frontend      | Visualization      | Data Processing   | Performance         |
| ------------- | ------------------ | ----------------- | ------------------- |
| Streamlit     | Plotly, Matplotlib | Pandas, NumPy     | Cached data loading |
| Custom CSS    | Seaborn            | Efficient caching | Optimized queries   |
| Responsive UI | Real-time updates  | Dynamic filtering | Mobile compatible   |

---

## Future Enhancements

- **Integration:** Electronic Health Records (EHR), API connectivity
- **Advanced Analytics:** Machine learning models, predictive risk scoring
- **User Experience:** Exportable reports, custom dashboard themes

---

## License

This project is released under the **MIT License**.

---

> **Note:**
> This dashboard is designed for clinical research and educational purposes. Always consult with healthcare professionals for medical decisions.
>
