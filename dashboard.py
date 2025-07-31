import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Prostate Cancer Analytics Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2e8b57;
    }
    
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .stTab {
        background-color: #f8f9fa;
    }
    
    .tab-content {
        padding: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sample data generator for demonstration
@st.cache_data
def generate_sample_data():
    """Generate sample prostate cancer data for demonstration"""
    np.random.seed(42)
    n_patients = 1000
    
    # Patient demographics
    ages = np.random.normal(65, 10, n_patients).astype(int)
    ages = np.clip(ages, 40, 90)
    
    # PSA levels (normal: <4, elevated: 4-10, high: >10)
    psa_levels = np.random.lognormal(1.2, 0.8, n_patients)
    
    # Gleason scores (6-10, with 6-7 being lower grade, 8-10 being higher grade)
    gleason_scores = np.random.choice([6, 7, 8, 9, 10], n_patients, 
                                    p=[0.2, 0.4, 0.2, 0.15, 0.05])
    
    # Stage (I, II, III, IV)
    stages = np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], 
                            n_patients, p=[0.3, 0.35, 0.25, 0.1])
    
    # Treatment types
    treatments = np.random.choice(['Surgery', 'Radiation', 'Hormone Therapy', 'Chemotherapy', 'Active Surveillance'], 
                                n_patients, p=[0.35, 0.25, 0.2, 0.1, 0.1])
    
    # Outcomes (based on stage and treatment)
    outcomes = []
    for i in range(n_patients):
        if stages[i] in ['Stage I', 'Stage II']:
            outcome = np.random.choice(['Excellent', 'Good', 'Fair'], p=[0.6, 0.3, 0.1])
        elif stages[i] == 'Stage III':
            outcome = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], p=[0.3, 0.4, 0.2, 0.1])
        else:  # Stage IV
            outcome = np.random.choice(['Good', 'Fair', 'Poor'], p=[0.2, 0.4, 0.4])
        outcomes.append(outcome)
    
    # Follow-up months
    follow_up_months = np.random.randint(1, 61, n_patients)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Patient_ID': range(1, n_patients + 1),
        'Age': ages,
        'PSA_Level': psa_levels,
        'Gleason_Score': gleason_scores,
        'Stage': stages,
        'Treatment': treatments,
        'Outcome': outcomes,
        'Follow_up_Months': follow_up_months,
        'Diagnosis_Date': pd.date_range(start='2020-01-01', periods=n_patients, freq='D')
    })
    
    return data

# Load data
@st.cache_data
def load_data():
    """Load prostate cancer data"""
    try:
        # Try to load actual data files
        # You can modify this to load your specific data files
        data = pd.read_csv('data/synthetic_prostate_cancer_risk.csv')
    except FileNotFoundError:
        # Use sample data if no data file exists
        data = generate_sample_data()
    
    return data

def main():
    # Header
    st.markdown('<div class="main-header">Prostate Cancer Analytics Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Dashboard Filters")
    
    # Age filter
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )
    
    # Stage filter - simplified with selectbox
    stage_options = ['All Stages'] + list(df['Stage'].unique())
    selected_stage = st.sidebar.selectbox(
        "Cancer Stage",
        options=stage_options,
        index=0
    )
    
    # Treatment filter - simplified with selectbox
    treatment_options = ['All Treatments'] + list(df['Treatment'].unique())
    selected_treatment = st.sidebar.selectbox(
        "Treatment Type",
        options=treatment_options,
        index=0
    )
    
    # Year filter instead of date range
    df['Year'] = df['Diagnosis_Date'].dt.year
    year_options = ['All Years'] + sorted(df['Year'].unique().tolist())
    selected_year = st.sidebar.selectbox(
        "Diagnosis Year",
        options=year_options,
        index=0
    )

    # Apply filters
    filtered_df = df[
        (df['Age'] >= age_range[0]) & 
        (df['Age'] <= age_range[1])
    ]
    
    # Apply stage filter
    if selected_stage != 'All Stages':
        filtered_df = filtered_df[filtered_df['Stage'] == selected_stage]
    
    # Apply treatment filter
    if selected_treatment != 'All Treatments':
        filtered_df = filtered_df[filtered_df['Treatment'] == selected_treatment]
    
    # Apply year filter
    if selected_year != 'All Years':
        filtered_df = filtered_df[filtered_df['Year'] == selected_year]
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Summary", 
        "Patient Demographics", 
        "Clinical Indicators", 
        "Treatment Analysis", 
        "Outcomes & Survival", 
        "Insights & Recommendations"
    ])
    
    with tab1:
        executive_summary_tab(filtered_df)
    
    with tab2:
        demographics_tab(filtered_df)
    
    with tab3:
        clinical_indicators_tab(filtered_df)
    
    with tab4:
        treatment_analysis_tab(filtered_df)
    
    with tab5:
        outcomes_tab(filtered_df)
    
    with tab6:
        insights_tab(filtered_df)

def executive_summary_tab(df):
    """Executive Summary Tab with Key Metrics and Overview"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Total Patients",
            value=f"{len(df):,}",
            delta=f"+{len(df) - 950}" if len(df) > 950 else f"{len(df) - 950}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_age = df['Age'].mean()
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Average Age",
            value=f"{avg_age:.1f} years",
            delta=f"{avg_age - 65:.1f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        high_risk = len(df[df['Gleason_Score'] >= 8])
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="High-Risk Cases",
            value=f"{high_risk:,}",
            delta=f"{(high_risk/len(df)*100):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        excellent_outcomes = len(df[df['Outcome'] == 'Excellent'])
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Excellent Outcomes",
            value=f"{excellent_outcomes:,}",
            delta=f"{(excellent_outcomes/len(df)*100):.1f}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Story telling section
    st.subheader("Clinical Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stage distribution pie chart
        stage_counts = df['Stage'].value_counts()
        fig_pie = px.pie(
            values=stage_counts.values,
            names=stage_counts.index,
            title="Cancer Stage Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(
            height=400,
            showlegend=True,
            title_font_size=16,
            font=dict(size=12)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>Key Findings</h4>
        <ul>
        <li><strong>Early Detection:</strong> {:.1f}% of cases are Stage I-II</li>
        <li><strong>Treatment Success:</strong> {:.1f}% show excellent outcomes</li>
        <li><strong>Risk Assessment:</strong> {:.1f}% are high-risk cases</li>
        <li><strong>Follow-up:</strong> Average {:.1f} months monitoring</li>
        </ul>
        </div>
        """.format(
            (len(df[df['Stage'].isin(['Stage I', 'Stage II'])]) / len(df)) * 100,
            (len(df[df['Outcome'] == 'Excellent']) / len(df)) * 100,
            (len(df[df['Gleason_Score'] >= 8]) / len(df)) * 100,
            df['Follow_up_Months'].mean()
        ), unsafe_allow_html=True)
    
    # Treatment effectiveness overview
    st.subheader("Treatment Effectiveness Overview")
    
    treatment_outcome = pd.crosstab(df['Treatment'], df['Outcome'], normalize='index') * 100
    
    fig_heatmap = px.imshow(
        treatment_outcome.values,
        x=treatment_outcome.columns,
        y=treatment_outcome.index,
        color_continuous_scale='RdYlGn',
        title="Treatment Success Rates by Outcome (%)"
    )
    fig_heatmap.update_layout(height=400, title_font_size=16)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def demographics_tab(df):
    """Patient Demographics Analysis"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.subheader("Patient Demographics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        fig_age = px.histogram(
            df, 
            x='Age', 
            nbins=20,
            title="Age Distribution of Patients",
            color_discrete_sequence=['#2E8B57']
        )
        fig_age.add_vline(
            x=df['Age'].mean(), 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {df['Age'].mean():.1f}"
        )
        fig_age.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Age by stage box plot
        fig_age_stage = px.box(
            df, 
            x='Stage', 
            y='Age',
            title="Age Distribution by Cancer Stage",
            color='Stage',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_age_stage.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_age_stage, use_container_width=True)
    
    # Diagnosis trends over time
    st.subheader("Diagnosis Trends Over Time")
    
    df['Year_Month'] = df['Diagnosis_Date'].dt.to_period('M')
    monthly_diagnoses = df.groupby('Year_Month').size().reset_index(name='Count')
    monthly_diagnoses['Year_Month'] = monthly_diagnoses['Year_Month'].astype(str)
    
    fig_trend = px.line(
        monthly_diagnoses,
        x='Year_Month',
        y='Count',
        title="Monthly Diagnosis Trends",
        markers=True
    )
    fig_trend.update_layout(
        height=400,
        title_font_size=16,
        xaxis_title="Month",
        yaxis_title="Number of Diagnoses"
    )
    fig_trend.update_xaxes(tickangle=45)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Demographics summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>Age Statistics</h4>
        <ul>
        <li>Mean Age: {:.1f} years</li>
        <li>Median Age: {:.1f} years</li>
        <li>Age Range: {}-{} years</li>
        <li>Std Deviation: {:.1f} years</li>
        </ul>
        </div>
        """.format(
            df['Age'].mean(),
            df['Age'].median(),
            df['Age'].min(),
            df['Age'].max(),
            df['Age'].std()
        ), unsafe_allow_html=True)
    
    with col2:
        stage_stats = df['Stage'].value_counts()
        st.markdown("""
        <div class="insight-box">
        <h4>Stage Distribution</h4>
        <ul>
        <li>Stage I: {} ({:.1f}%)</li>
        <li>Stage II: {} ({:.1f}%)</li>
        <li>Stage III: {} ({:.1f}%)</li>
        <li>Stage IV: {} ({:.1f}%)</li>
        </ul>
        </div>
        """.format(
            stage_stats.get('Stage I', 0), (stage_stats.get('Stage I', 0)/len(df))*100,
            stage_stats.get('Stage II', 0), (stage_stats.get('Stage II', 0)/len(df))*100,
            stage_stats.get('Stage III', 0), (stage_stats.get('Stage III', 0)/len(df))*100,
            stage_stats.get('Stage IV', 0), (stage_stats.get('Stage IV', 0)/len(df))*100
        ), unsafe_allow_html=True)
    
    with col3:
        early_stage = len(df[df['Stage'].isin(['Stage I', 'Stage II'])])
        advanced_stage = len(df[df['Stage'].isin(['Stage III', 'Stage IV'])])
        
        st.markdown("""
        <div class="insight-box">
        <h4>Early vs Advanced</h4>
        <ul>
        <li>Early Stage: {} ({:.1f}%)</li>
        <li>Advanced Stage: {} ({:.1f}%)</li>
        <li>Detection Ratio: {:.1f}:1</li>
        <li>Total Cases: {}</li>
        </ul>
        </div>
        """.format(
            early_stage, (early_stage/len(df))*100,
            advanced_stage, (advanced_stage/len(df))*100,
            early_stage/advanced_stage if advanced_stage > 0 else 0,
            len(df)
        ), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def clinical_indicators_tab(df):
    """Clinical Indicators Analysis"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.subheader("Clinical Indicators Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PSA levels distribution
        fig_psa = px.histogram(
            df,
            x='PSA_Level',
            nbins=30,
            title="PSA Level Distribution",
            color_discrete_sequence=['#FF6B6B']
        )
        
        # Add reference lines for PSA levels
        fig_psa.add_vline(x=4, line_dash="dash", line_color="orange", 
                         annotation_text="Normal Threshold (4 ng/mL)")
        fig_psa.add_vline(x=10, line_dash="dash", line_color="red", 
                         annotation_text="High Risk (10 ng/mL)")
        
        fig_psa.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_psa, use_container_width=True)
    
    with col2:
        # Gleason Score distribution
        gleason_counts = df['Gleason_Score'].value_counts().sort_index()
        fig_gleason = px.bar(
            x=gleason_counts.index,
            y=gleason_counts.values,
            title="Gleason Score Distribution",
            color=gleason_counts.values,
            color_continuous_scale='Reds'
        )
        fig_gleason.update_layout(
            height=400,
            title_font_size=16,
            xaxis_title="Gleason Score",
            yaxis_title="Number of Patients",
            showlegend=False
        )
        st.plotly_chart(fig_gleason, use_container_width=True)
    
    # PSA vs Gleason Score correlation
    st.subheader("PSA Level vs Gleason Score Correlation")
    
    fig_scatter = px.scatter(
        df,
        x='PSA_Level',
        y='Gleason_Score',
        color='Stage',
        size='Age',
        title="PSA Level vs Gleason Score by Stage",
        hover_data=['Age', 'Treatment']
    )
    fig_scatter.update_layout(height=500, title_font_size=16)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Clinical risk stratification
    st.subheader("Clinical Risk Stratification")
    
    # Create risk categories
    def categorize_risk(row):
        if row['PSA_Level'] < 4 and row['Gleason_Score'] <= 6:
            return 'Low Risk'
        elif row['PSA_Level'] < 10 and row['Gleason_Score'] <= 7:
            return 'Intermediate Risk'
        else:
            return 'High Risk'
    
    df['Risk_Category'] = df.apply(categorize_risk, axis=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = df['Risk_Category'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Category Distribution",
            color_discrete_sequence=['#90EE90', '#FFD700', '#FF6347']
        )
        fig_risk.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Risk by stage
        risk_stage = pd.crosstab(df['Risk_Category'], df['Stage'])
        fig_risk_stage = px.bar(
            risk_stage,
            title="Risk Categories by Cancer Stage",
            barmode='group'
        )
        fig_risk_stage.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_risk_stage, use_container_width=True)
    
    # Clinical insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        normal_psa = len(df[df['PSA_Level'] < 4])
        elevated_psa = len(df[(df['PSA_Level'] >= 4) & (df['PSA_Level'] < 10)])
        high_psa = len(df[df['PSA_Level'] >= 10])
        
        st.markdown("""
        <div class="insight-box">
        <h4>PSA Level Categories</h4>
        <ul>
        <li>Normal (<4): {} ({:.1f}%)</li>
        <li>Elevated (4-10): {} ({:.1f}%)</li>
        <li>High (>10): {} ({:.1f}%)</li>
        <li>Mean PSA: {:.2f} ng/mL</li>
        </ul>
        </div>
        """.format(
            normal_psa, (normal_psa/len(df))*100,
            elevated_psa, (elevated_psa/len(df))*100,
            high_psa, (high_psa/len(df))*100,
            df['PSA_Level'].mean()
        ), unsafe_allow_html=True)
    
    with col2:
        low_gleason = len(df[df['Gleason_Score'] <= 6])
        mid_gleason = len(df[df['Gleason_Score'] == 7])
        high_gleason = len(df[df['Gleason_Score'] >= 8])
        
        st.markdown("""
        <div class="insight-box">
        <h4>Gleason Score Categories</h4>
        <ul>
        <li>Low Grade (≤6): {} ({:.1f}%)</li>
        <li>Intermediate (7): {} ({:.1f}%)</li>
        <li>High Grade (≥8): {} ({:.1f}%)</li>
        <li>Mean Score: {:.1f}</li>
        </ul>
        </div>
        """.format(
            low_gleason, (low_gleason/len(df))*100,
            mid_gleason, (mid_gleason/len(df))*100,
            high_gleason, (high_gleason/len(df))*100,
            df['Gleason_Score'].mean()
        ), unsafe_allow_html=True)
    
    with col3:
        risk_counts = df['Risk_Category'].value_counts()
        
        st.markdown("""
        <div class="insight-box">
        <h4>Overall Risk Assessment</h4>
        <ul>
        <li>Low Risk: {} ({:.1f}%)</li>
        <li>Intermediate: {} ({:.1f}%)</li>
        <li>High Risk: {} ({:.1f}%)</li>
        <li>Risk Ratio: {:.1f}:1</li>
        </ul>
        </div>
        """.format(
            risk_counts.get('Low Risk', 0), (risk_counts.get('Low Risk', 0)/len(df))*100,
            risk_counts.get('Intermediate Risk', 0), (risk_counts.get('Intermediate Risk', 0)/len(df))*100,
            risk_counts.get('High Risk', 0), (risk_counts.get('High Risk', 0)/len(df))*100,
            (risk_counts.get('Low Risk', 0) + risk_counts.get('Intermediate Risk', 0)) / risk_counts.get('High Risk', 1)
        ), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def treatment_analysis_tab(df):
    """Treatment Analysis"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.subheader("Treatment Analysis and Effectiveness")
    
    # Treatment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        treatment_counts = df['Treatment'].value_counts()
        fig_treatment = px.bar(
            x=treatment_counts.values,
            y=treatment_counts.index,
            orientation='h',
            title="Treatment Type Distribution",
            color=treatment_counts.values,
            color_continuous_scale='Blues'
        )
        fig_treatment.update_layout(height=400, title_font_size=16, showlegend=False)
        st.plotly_chart(fig_treatment, use_container_width=True)
    
    with col2:
        # Treatment by stage
        treatment_stage = pd.crosstab(df['Stage'], df['Treatment'])
        fig_treatment_stage = px.bar(
            treatment_stage,
            title="Treatment Distribution by Cancer Stage",
            barmode='stack'
        )
        fig_treatment_stage.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_treatment_stage, use_container_width=True)
    
    # Treatment outcomes analysis
    st.subheader("Treatment Outcomes Analysis")
    
    # Create treatment outcome matrix
    outcome_treatment = pd.crosstab(df['Treatment'], df['Outcome'], normalize='index') * 100
    
    fig_outcome = px.bar(
        outcome_treatment,
        title="Treatment Success Rates by Outcome",
        barmode='stack',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_outcome.update_layout(height=500, title_font_size=16)
    st.plotly_chart(fig_outcome, use_container_width=True)
    
    # Treatment effectiveness by risk category
    if 'Risk_Category' in df.columns:
        st.subheader("Treatment Effectiveness by Risk Category")
        
        risk_treatment_outcome = df.groupby(['Risk_Category', 'Treatment', 'Outcome']).size().unstack(fill_value=0)
        risk_treatment_pct = risk_treatment_outcome.div(risk_treatment_outcome.sum(axis=1), axis=0) * 100
        
        for risk in df['Risk_Category'].unique():
            if risk in risk_treatment_pct.index:
                st.write(f"**{risk} Patients**")
                
                risk_data = risk_treatment_pct.loc[risk]
                fig_risk_treatment = px.bar(
                    risk_data,
                    title=f"Treatment Outcomes for {risk} Patients",
                    barmode='stack'
                )
                fig_risk_treatment.update_layout(height=300, title_font_size=14)
                st.plotly_chart(fig_risk_treatment, use_container_width=True)
    
    # Treatment duration and follow-up
    st.subheader("Follow-up Duration by Treatment")
    
    fig_followup = px.box(
        df,
        x='Treatment',
        y='Follow_up_Months',
        title="Follow-up Duration by Treatment Type",
        color='Treatment'
    )
    fig_followup.update_layout(height=400, title_font_size=16)
    fig_followup.update_xaxes(tickangle=45)
    st.plotly_chart(fig_followup, use_container_width=True)
    
    # Treatment insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        surgery_success = len(df[(df['Treatment'] == 'Surgery') & (df['Outcome'] == 'Excellent')])
        total_surgery = len(df[df['Treatment'] == 'Surgery'])
        
        st.markdown("""
        <div class="insight-box">
        <h4>Surgery Outcomes</h4>
        <ul>
        <li>Total Cases: {}</li>
        <li>Excellent Outcomes: {} ({:.1f}%)</li>
        <li>Average Follow-up: {:.1f} months</li>
        <li>Success Rate: {:.1f}%</li>
        </ul>
        </div>
        """.format(
            total_surgery,
            surgery_success,
            (surgery_success/total_surgery)*100 if total_surgery > 0 else 0,
            df[df['Treatment'] == 'Surgery']['Follow_up_Months'].mean() if total_surgery > 0 else 0,
            (surgery_success/total_surgery)*100 if total_surgery > 0 else 0
        ), unsafe_allow_html=True)
    
    with col2:
        radiation_success = len(df[(df['Treatment'] == 'Radiation') & (df['Outcome'] == 'Excellent')])
        total_radiation = len(df[df['Treatment'] == 'Radiation'])
        
        st.markdown("""
        <div class="insight-box">
        <h4>Radiation Therapy</h4>
        <ul>
        <li>Total Cases: {}</li>
        <li>Excellent Outcomes: {} ({:.1f}%)</li>
        <li>Average Follow-up: {:.1f} months</li>
        <li>Success Rate: {:.1f}%</li>
        </ul>
        </div>
        """.format(
            total_radiation,
            radiation_success,
            (radiation_success/total_radiation)*100 if total_radiation > 0 else 0,
            df[df['Treatment'] == 'Radiation']['Follow_up_Months'].mean() if total_radiation > 0 else 0,
            (radiation_success/total_radiation)*100 if total_radiation > 0 else 0
        ), unsafe_allow_html=True)
    
    with col3:
        hormone_success = len(df[(df['Treatment'] == 'Hormone Therapy') & (df['Outcome'] == 'Excellent')])
        total_hormone = len(df[df['Treatment'] == 'Hormone Therapy'])
        
        st.markdown("""
        <div class="insight-box">
        <h4>Hormone Therapy</h4>
        <ul>
        <li>Total Cases: {}</li>
        <li>Excellent Outcomes: {} ({:.1f}%)</li>
        <li>Average Follow-up: {:.1f} months</li>
        <li>Success Rate: {:.1f}%</li>
        </ul>
        </div>
        """.format(
            total_hormone,
            hormone_success,
            (hormone_success/total_hormone)*100 if total_hormone > 0 else 0,
            df[df['Treatment'] == 'Hormone Therapy']['Follow_up_Months'].mean() if total_hormone > 0 else 0,
            (hormone_success/total_hormone)*100 if total_hormone > 0 else 0
        ), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def outcomes_tab(df):
    """Outcomes and Survival Analysis"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.subheader("Patient Outcomes and Survival Analysis")
    
    # Overall outcomes distribution
    col1, col2 = st.columns(2)
    
    with col1:
        outcome_counts = df['Outcome'].value_counts()
        fig_outcomes = px.pie(
            values=outcome_counts.values,
            names=outcome_counts.index,
            title="Overall Patient Outcomes",
            color_discrete_sequence=['#28a745', '#17a2b8', '#ffc107', '#dc3545']
        )
        fig_outcomes.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_outcomes, use_container_width=True)
    
    with col2:
        # Outcomes by stage
        stage_outcome = pd.crosstab(df['Stage'], df['Outcome'], normalize='index') * 100
        fig_stage_outcome = px.bar(
            stage_outcome,
            title="Outcome Distribution by Cancer Stage (%)",
            barmode='stack'
        )
        fig_stage_outcome.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_stage_outcome, use_container_width=True)
    
    # Survival analysis simulation
    st.subheader("Follow-up Duration Analysis")
    
    # Create survival-like analysis based on follow-up duration and outcomes
    col1, col2 = st.columns(2)
    
    with col1:
        # Follow-up duration by outcome
        fig_followup_outcome = px.box(
            df,
            x='Outcome',
            y='Follow_up_Months',
            title="Follow-up Duration by Outcome",
            color='Outcome'
        )
        fig_followup_outcome.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_followup_outcome, use_container_width=True)
    
    with col2:
        # Age vs outcome
        fig_age_outcome = px.violin(
            df,
            x='Outcome',
            y='Age',
            title="Age Distribution by Outcome",
            color='Outcome'
        )
        fig_age_outcome.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_age_outcome, use_container_width=True)
    
    # Outcome prediction factors
    st.subheader("Factors Influencing Outcomes")
    
    # Create a correlation matrix for numerical factors
    numerical_cols = ['Age', 'PSA_Level', 'Gleason_Score', 'Follow_up_Months']
    correlation_data = df[numerical_cols].corr()
    
    fig_corr = px.imshow(
        correlation_data,
        title="Correlation Matrix of Clinical Factors",
        color_continuous_scale='RdBu'
    )
    fig_corr.update_layout(height=500, title_font_size=16)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Outcome statistics by different factors
    st.subheader("Outcome Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        excellent_outcomes = len(df[df['Outcome'] == 'Excellent'])
        good_outcomes = len(df[df['Outcome'] == 'Good'])
        
        st.markdown("""
        <div class="success-box">
        <h4>Positive Outcomes</h4>
        <ul>
        <li>Excellent: {} ({:.1f}%)</li>
        <li>Good: {} ({:.1f}%)</li>
        <li>Combined Success: {:.1f}%</li>
        <li>Average Age: {:.1f} years</li>
        </ul>
        </div>
        """.format(
            excellent_outcomes, (excellent_outcomes/len(df))*100,
            good_outcomes, (good_outcomes/len(df))*100,
            ((excellent_outcomes + good_outcomes)/len(df))*100,
            df[df['Outcome'].isin(['Excellent', 'Good'])]['Age'].mean()
        ), unsafe_allow_html=True)
    
    with col2:
        fair_outcomes = len(df[df['Outcome'] == 'Fair'])
        poor_outcomes = len(df[df['Outcome'] == 'Poor'])
        
        st.markdown("""
        <div class="warning-box">
        <h4>Challenging Outcomes</h4>
        <ul>
        <li>Fair: {} ({:.1f}%)</li>
        <li>Poor: {} ({:.1f}%)</li>
        <li>Needs Attention: {:.1f}%</li>
        <li>Average Age: {:.1f} years</li>
        </ul>
        </div>
        """.format(
            fair_outcomes, (fair_outcomes/len(df))*100,
            poor_outcomes, (poor_outcomes/len(df))*100,
            ((fair_outcomes + poor_outcomes)/len(df))*100,
            df[df['Outcome'].isin(['Fair', 'Poor'])]['Age'].mean() if len(df[df['Outcome'].isin(['Fair', 'Poor'])]) > 0 else 0
        ), unsafe_allow_html=True)
    
    with col3:
        avg_followup = df['Follow_up_Months'].mean()
        long_followup = len(df[df['Follow_up_Months'] > 36])
        
        st.markdown("""
        <div class="insight-box">
        <h4>Follow-up Analysis</h4>
        <ul>
        <li>Average Duration: {:.1f} months</li>
        <li>Long-term (>3 years): {}</li>
        <li>Percentage: {:.1f}%</li>
        <li>Max Follow-up: {} months</li>
        </ul>
        </div>
        """.format(
            avg_followup,
            long_followup,
            (long_followup/len(df))*100,
            df['Follow_up_Months'].max()
        ), unsafe_allow_html=True)
    
    # Risk factors for poor outcomes
    st.subheader("Risk Factors for Poor Outcomes")
    
    poor_outcome_patients = df[df['Outcome'].isin(['Fair', 'Poor'])]
    
    if len(poor_outcome_patients) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Stage distribution in poor outcomes
            poor_stage = poor_outcome_patients['Stage'].value_counts()
            fig_poor_stage = px.bar(
                x=poor_stage.index,
                y=poor_stage.values,
                title="Cancer Stages in Poor Outcomes",
                color=poor_stage.values,
                color_continuous_scale='Reds'
            )
            fig_poor_stage.update_layout(height=400, title_font_size=16, showlegend=False)
            st.plotly_chart(fig_poor_stage, use_container_width=True)
        
        with col2:
            # Treatment distribution in poor outcomes
            poor_treatment = poor_outcome_patients['Treatment'].value_counts()
            fig_poor_treatment = px.bar(
                x=poor_treatment.index,
                y=poor_treatment.values,
                title="Treatments in Poor Outcomes",
                color=poor_treatment.values,
                color_continuous_scale='Oranges'
            )
            fig_poor_treatment.update_layout(height=400, title_font_size=16, showlegend=False)
            fig_poor_treatment.update_xaxes(tickangle=45)
            st.plotly_chart(fig_poor_treatment, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def insights_tab(df):
    """Insights and Recommendations"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.subheader("Clinical Insights and Recommendations")
    
    # Calculate key insights
    total_patients = len(df)
    early_stage_pct = (len(df[df['Stage'].isin(['Stage I', 'Stage II'])]) / total_patients) * 100
    excellent_outcome_pct = (len(df[df['Outcome'] == 'Excellent']) / total_patients) * 100
    high_risk_pct = (len(df[df['Gleason_Score'] >= 8]) / total_patients) * 100
    avg_age = df['Age'].mean()
    avg_psa = df['PSA_Level'].mean()
    
    # Key insights section
    st.markdown("### Key Clinical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>Early Detection Success</h4>
        <p><strong>{:.1f}%</strong> of cases are detected at early stages (Stage I-II), indicating effective screening programs. This early detection rate is crucial for improving patient outcomes and treatment success rates.</p>
        
        <h4>Treatment Effectiveness</h4>
        <p><strong>{:.1f}%</strong> of patients achieve excellent outcomes, demonstrating the effectiveness of current treatment protocols. However, there's room for improvement in treatment selection and personalization.</p>
        </div>
        """.format(early_stage_pct, excellent_outcome_pct), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>Risk Profile</h4>
        <p><strong>{:.1f}%</strong> of patients present with high-risk features (Gleason ≥8), requiring aggressive treatment approaches and closer monitoring.</p>
        
        <h4>Patient Demographics</h4>
        <p>Average patient age is <strong>{:.1f} years</strong> with mean PSA of <strong>{:.2f} ng/mL</strong>, consistent with typical prostate cancer demographics.</p>
        </div>
        """.format(high_risk_pct, avg_age, avg_psa), unsafe_allow_html=True)
    
    # Treatment recommendations
    st.markdown("### Treatment Optimization Recommendations")
    
    # Analyze treatment effectiveness by stage
    treatment_effectiveness = df.groupby(['Stage', 'Treatment'])['Outcome'].apply(
        lambda x: (x == 'Excellent').sum() / len(x) * 100
    ).reset_index()
    treatment_effectiveness.columns = ['Stage', 'Treatment', 'Success_Rate']
    
    # Find best treatments for each stage
    best_treatments = treatment_effectiveness.loc[treatment_effectiveness.groupby('Stage')['Success_Rate'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Optimal Treatment by Stage")
        for _, row in best_treatments.iterrows():
            st.markdown(f"**{row['Stage']}**: {row['Treatment']} ({row['Success_Rate']:.1f}% success rate)")
        
        st.markdown("""
        <div class="success-box">
        <h4>Treatment Protocol Recommendations</h4>
        <ul>
        <li>Continue current early-stage treatment protocols</li>
        <li>Consider combination therapies for advanced stages</li>
        <li>Implement personalized treatment selection</li>
        <li>Enhance follow-up protocols for high-risk patients</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Treatment success rates visualization
        fig_treatment_success = px.bar(
            treatment_effectiveness,
            x='Treatment',
            y='Success_Rate',
            color='Stage',
            title="Treatment Success Rates by Stage",
            barmode='group'
        )
        fig_treatment_success.update_layout(height=400, title_font_size=16)
        fig_treatment_success.update_xaxes(tickangle=45)
        st.plotly_chart(fig_treatment_success, use_container_width=True)
    
    # Risk stratification insights
    st.markdown("### Risk Stratification and Management")
    
    if 'Risk_Category' in df.columns:
        risk_outcomes = df.groupby('Risk_Category')['Outcome'].value_counts(normalize=True) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            low_risk_excellent = risk_outcomes.get(('Low Risk', 'Excellent'), 0)
            st.markdown("""
            <div class="success-box">
            <h4>Low Risk Patients</h4>
            <ul>
            <li>Excellent outcomes: {:.1f}%</li>
            <li>Recommendation: Active surveillance</li>
            <li>Follow-up: Every 6 months</li>
            <li>PSA monitoring essential</li>
            </ul>
            </div>
            """.format(low_risk_excellent), unsafe_allow_html=True)
        
        with col2:
            intermediate_risk_excellent = risk_outcomes.get(('Intermediate Risk', 'Excellent'), 0)
            st.markdown("""
            <div class="warning-box">
            <h4>Intermediate Risk</h4>
            <ul>
            <li>Excellent outcomes: {:.1f}%</li>
            <li>Recommendation: Definitive treatment</li>
            <li>Options: Surgery or radiation</li>
            <li>Close monitoring required</li>
            </ul>
            </div>
            """.format(intermediate_risk_excellent), unsafe_allow_html=True)
        
        with col3:
            high_risk_excellent = risk_outcomes.get(('High Risk', 'Excellent'), 0)
            st.markdown("""
            <div class="warning-box">
            <h4>High Risk Patients</h4>
            <ul>
            <li>Excellent outcomes: {:.1f}%</li>
            <li>Recommendation: Aggressive treatment</li>
            <li>Multimodal therapy consideration</li>
            <li>Frequent follow-up essential</li>
            </ul>
            </div>
            """.format(high_risk_excellent), unsafe_allow_html=True)
    
    # Quality improvement recommendations
    st.markdown("### Quality Improvement Initiatives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
        <h4>Screening and Early Detection</h4>
        <ul>
        <li>Maintain current screening protocols</li>
        <li>Target high-risk populations</li>
        <li>Implement risk-based screening intervals</li>
        <li>Enhance patient education programs</li>
        <li>Consider genetic testing for family history</li>
        </ul>
        
        <h4>Treatment Optimization</h4>
        <ul>
        <li>Develop treatment decision algorithms</li>
        <li>Implement multidisciplinary care teams</li>
        <li>Consider patient preferences in treatment selection</li>
        <li>Monitor treatment response more closely</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
        <h4>Follow-up and Monitoring</h4>
        <ul>
        <li>Standardize follow-up protocols</li>
        <li>Implement risk-stratified monitoring</li>
        <li>Use technology for remote monitoring</li>
        <li>Track quality of life outcomes</li>
        <li>Enhance survivorship care plans</li>
        </ul>
        
        <h4>Data and Analytics</h4>
        <ul>
        <li>Implement real-time outcome tracking</li>
        <li>Develop predictive models</li>
        <li>Enhance data collection protocols</li>
        <li>Regular outcome benchmarking</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Future directions
    st.markdown("### Future Research Directions")
    
    st.markdown("""
    <div class="insight-box">
    <h4>Recommended Research Priorities</h4>
    <ol>
    <li><strong>Personalized Medicine</strong>: Develop genomic profiling for treatment selection</li>
    <li><strong>Biomarker Discovery</strong>: Identify new prognostic and predictive biomarkers</li>
    <li><strong>Treatment Innovation</strong>: Investigate novel therapeutic approaches</li>
    <li><strong>Quality of Life</strong>: Study long-term survivorship outcomes</li>
    <li><strong>Health Economics</strong>: Analyze cost-effectiveness of different strategies</li>
    <li><strong>Technology Integration</strong>: Implement AI-driven decision support systems</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Action items
    st.markdown("### Immediate Action Items")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>Short-term (1-3 months)</h4>
        <ul>
        <li>Review treatment protocols for high-risk patients</li>
        <li>Implement standardized outcome tracking</li>
        <li>Enhance patient education materials</li>
        <li>Establish multidisciplinary care teams</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>Long-term (6-12 months)</h4>
        <ul>
        <li>Develop predictive outcome models</li>
        <li>Implement quality improvement programs</li>
        <li>Establish research collaborations</li>
        <li>Create comprehensive survivorship programs</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()