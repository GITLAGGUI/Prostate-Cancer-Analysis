# Quick Start Guide

## Running the Dashboard

### Option 1: Direct Command
```bash
streamlit run dashboard.py
```

### Option 2: Using the Launcher
```bash
python run_dashboard.py
```

## What You'll See

The dashboard will open in your browser with 6 main tabs:

1. **Executive Summary** - Key metrics and overview
2. **Patient Demographics** - Age distribution and population analysis  
3. **Clinical Indicators** - PSA levels, Gleason scores, risk assessment
4. **Treatment Analysis** - Treatment effectiveness and outcomes
5. **Outcomes & Survival** - Patient outcomes and follow-up analysis
6. **Insights & Recommendations** - Clinical insights and actionable recommendations

## Simplified Filters

**Easy-to-use sidebar filters:**
- **Age Range** - Slider to select age range
- **Cancer Stage** - Dropdown (All Stages, Stage I, II, III, IV)
- **Treatment Type** - Dropdown (All Treatments, Surgery, Radiation, etc.)
- **Diagnosis Year** - Dropdown (All Years, 2020, 2021, 2022, etc.)

*Default: All options selected for comprehensive view*

## Features

✅ **Simplified Interface** - Less buttons, cleaner design
✅ **Professional Design** - Clean, medical-grade interface
✅ **Responsive Layout** - Works on desktop and mobile
✅ **No Overflow** - Content organized in tabs for optimal viewing
✅ **Clear Visualizations** - High-quality charts and graphs
✅ **Quality Analytics** - Professional medical analytics

## Using Your Own Data

Replace the sample data by creating a CSV file named `prostate_cancer_data.csv` with these columns:
- Patient_ID, Age, PSA_Level, Gleason_Score, Stage, Treatment, Outcome, Follow_up_Months, Diagnosis_Date

The dashboard will automatically detect and use your data file.

## Browser Access

Once running, access the dashboard at: **http://localhost:8501**

## Stopping the Dashboard

Press `Ctrl+C` in the terminal to stop the dashboard.

---

**Ready to explore your prostate cancer data with simplified, professional analytics!**