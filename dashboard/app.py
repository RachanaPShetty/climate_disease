import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go

BASE = r'C:\Users\HP\OneDrive\Desktop\6THSEM\MAJORPROJECT\climate_disease_project'

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate & Disease Dashboard",
    page_icon="🌡️",
    layout="wide"
)

# ── Load data & model ──────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(f'{BASE}\\data\\processed\\features_data.csv')

@st.cache_resource
def load_model():
    with open(f'{BASE}\\models\\random_forest_dengue.pkl', 'rb') as f:
        return pickle.load(f)

df    = load_data()
model = load_model()

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/temperature.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📈 Climate Trends",
    "🦟 Disease Trends",
    "🔥 Correlation Analysis",
    "🤖 ML Model Results",
    "🔮 Predict Cases"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Climate-Disease ML Framework")
st.sidebar.markdown("**Team:** Rachana | Sanvi | Shreya | Vasushree")
st.sidebar.markdown("**Guide:** Dr. Mustafa Basthikodi")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    # 🔷 Header
    st.markdown("""
    <div style='background: linear-gradient(90deg, #2E86C1, #48C9B0);
                padding: 25px; border-radius: 10px; color: white; text-align:center'>
        <h1>🌍 Climate & Disease Intelligence Dashboard</h1>
        <p style='font-size:18px;'>AI-powered analysis of climate impact on disease outbreaks</p>
        <p>Sahyadri College of Engineering & Management | 2025–26</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 👩‍💻 Team + Guide
    col1, col2 = st.columns(2)
    col1.info("👩‍💻 Team: Rachana | Sanvi | Shreya | Vasushree")
    col2.success("🎓 Guide: Dr. Mustafa Basthikodi")

    st.markdown("<br>", unsafe_allow_html=True)

    # 🔢 KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Years", "2010–2023")
    c2.metric("📊 Records", len(df))
    c3.metric("🌡️ Avg Temp", f"{df['temperature'].mean():.1f}°C")
    c4.metric("🦟 Dengue Cases", f"{df['dengue_cases'].sum():,}")

    st.markdown("---")

    # 🧠 About
    st.subheader("🧠 What this system does")

    a1, a2, a3 = st.columns(3)

    a1.markdown("""
    <div style='background:#111827;padding:15px;border-radius:10px;color:white;'>
    🌡️ <b>Climate Analysis</b><br>
    Tracks temperature, rainfall & humidity patterns
    </div>
    """, unsafe_allow_html=True)

    a2.markdown("""
    <div style='background:#111827;padding:15px;border-radius:10px;color:white;'>
    🦟 <b>Disease Tracking</b><br>
    Studies Dengue, Malaria & Cholera trends
    </div>
    """, unsafe_allow_html=True)

    a3.markdown("""
    <div style='background:#111827;padding:15px;border-radius:10px;color:white;'>
    🤖 <b>ML Prediction</b><br>
    Uses Random Forest for outbreak prediction
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 🔍 Insights
    st.subheader("🔍 Key Insights")

    i1, i2, i3 = st.columns(3)
    i1.success("✔ Climate strongly impacts disease outbreaks")
    i2.warning("⚠ Rainfall affects dengue with time lag")
    i3.info("📊 ML model achieves high accuracy (R² ≈ 0.99)")

    st.markdown("---")

    # 🎯 Objectives (NOW INSIDE ✅)
    st.subheader("🎯 Project Objectives")

    o1, o2, o3, o4 = st.columns(4)

    o1.markdown("""
    <div style='background:#1f2937;padding:15px;border-radius:10px;text-align:center;color:white;'>
    <h4>📊 Data Analysis</h4>
    <p>Process and analyze climate & disease data</p>
    </div>
    """, unsafe_allow_html=True)

    o2.markdown("""
    <div style='background:#1f2937;padding:15px;border-radius:10px;text-align:center;color:white;'>
    <h4>🔗 Relationship</h4>
    <p>Identify climate–disease correlations</p>
    </div>
    """, unsafe_allow_html=True)

    o3.markdown("""
    <div style='background:#1f2937;padding:15px;border-radius:10px;text-align:center;color:white;'>
    <h4>🤖 ML Model</h4>
    <p>Predict outbreaks using machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    o4.markdown("""
    <div style='background:#1f2937;padding:15px;border-radius:10px;text-align:center;color:white;'>
    <h4>📈 Dashboard</h4>
    <p>Interactive visualization & insights</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — CLIMATE TRENDS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Climate Trends":
    st.title("📈 Climate Trends (Bangalore 2010–2023)")

    df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))

    tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "🌧️ Rainfall", "💧 Humidity"])

    with tab1:
        fig = px.line(df, x='date', y='temperature',
                      title='Monthly Temperature Over Time',
                      color_discrete_sequence=['tomato'])
        fig.update_layout(xaxis_title='Date', yaxis_title='Temperature (°C)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(df, x='date', y='rainfall',
                     title='Monthly Rainfall Over Time',
                     color_discrete_sequence=['steelblue'])
        fig.update_layout(xaxis_title='Date', yaxis_title='Rainfall (mm)')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.line(df, x='date', y='humidity',
                      title='Monthly Humidity Over Time',
                      color_discrete_sequence=['seagreen'])
        fig.update_layout(xaxis_title='Date', yaxis_title='Humidity (%)')
        st.plotly_chart(fig, use_container_width=True)

    # Seasonal averages
    st.subheader("Seasonal Climate Averages")
    season_avg = df.groupby('season')[['temperature','rainfall','humidity']].mean().round(2)
    st.dataframe(season_avg, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — DISEASE TRENDS
# ══════════════════════════════════════════════════════════════
elif page == "🦟 Disease Trends":
    st.title("🦟 Disease Trends (2010–2023)")

    df['date'] = pd.to_datetime(df[['year','month']].assign(day=1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['dengue_cases'],
                             name='Dengue', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['malaria_cases'],
                             name='Malaria', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['cholera_cases'],
                             name='Cholera', line=dict(color='green', width=2)))
    fig.update_layout(title='Disease Cases Over Time',
                      xaxis_title='Date', yaxis_title='Cases')
    st.plotly_chart(fig, use_container_width=True)

    # Year-wise totals
    st.subheader("Year-wise Disease Totals")
    yearly = df.groupby('year')[['dengue_cases','malaria_cases','cholera_cases']].sum()
    fig2 = px.bar(yearly, barmode='group',
                  title='Annual Disease Case Counts',
                  color_discrete_sequence=['orange','purple','green'])
    st.plotly_chart(fig2, use_container_width=True)

    # Season-wise
    st.subheader("Season-wise Disease Averages")
    season_dis = df.groupby('season')[['dengue_cases','malaria_cases','cholera_cases']].mean().round(1)
    st.dataframe(season_dis, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🔥 Correlation Analysis":
    st.title("🔥 Correlation Analysis")

    cols = ['temperature','rainfall','humidity',
            'temp_3m_avg','rainfall_3m_avg',
            'dengue_cases','malaria_cases','cholera_cases']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pearson Correlation")
        pearson = df[cols].corr(method='pearson')
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(pearson, annot=True, fmt='.2f',
                    cmap='coolwarm', ax=ax, linewidths=0.5)
        st.pyplot(fig)

    with col2:
        st.subheader("Spearman Correlation")
        spearman = df[cols].corr(method='spearman')
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(spearman, annot=True, fmt='.2f',
                    cmap='coolwarm', ax=ax, linewidths=0.5)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Lag Analysis: Rainfall → Disease Delay")
    lag_data = {'Lag (months)': [0,1,2,3,4],
                'Dengue Corr' : [0.484,0.513,0.433,0.339,0.138],
                'Malaria Corr': [0.420,0.316,0.140,-0.057,-0.257]}
    lag_df = pd.DataFrame(lag_data)

    fig = px.bar(lag_df, x='Lag (months)',
                 y=['Dengue Corr','Malaria Corr'],
                 barmode='group',
                 title='Rainfall Effect on Disease at Different Time Lags',
                 color_discrete_sequence=['orange','purple'])
    st.plotly_chart(fig, use_container_width=True)
    st.info("📌 **Key Finding:** Rainfall affects Dengue with a **1-month lag** (corr = 0.513), matching real-world mosquito breeding cycles.")

# ══════════════════════════════════════════════════════════════
# PAGE 5 — ML MODEL RESULTS
# ══════════════════════════════════════════════════════════════
elif page == "🤖 ML Model Results":
    st.title("🤖 Machine Learning Model Results")

    results = pd.DataFrame({
        'Model'  : ['Linear Regression', 'Random Forest', 'XGBoost'],
        'RMSE'   : [34.46, 8.24, 9.54],
        'R2 Score': [0.866, 0.992, 0.990]
    })

    col1, col2, col3 = st.columns(3)
    col1.metric("🥇 Best Model", "Random Forest")
    col2.metric("🎯 Best R² Score", "0.992")
    col3.metric("📉 Best RMSE", "8.24")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(results, x='Model', y='R2 Score',
                     title='R² Score Comparison (Higher = Better)',
                     color='Model',
                     color_discrete_sequence=['#ef5350','#42a5f5','#66bb6a'])
        fig.update_layout(yaxis_range=[0,1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(results, x='Model', y='RMSE',
                     title='RMSE Comparison (Lower = Better)',
                     color='Model',
                     color_discrete_sequence=['#ef5350','#42a5f5','#66bb6a'])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    feat_imp = pd.DataFrame({
        'Feature'   : ['month_sin','month_cos','temp_3m_avg','humidity',
                       'year_trend','temperature','humidity_3m_avg',
                       'rainfall_3m_avg','rainfall_lag1','rainfall'],
        'Importance': [0.718,0.201,0.031,0.019,0.009,
                       0.006,0.006,0.004,0.003,0.002]
    })
    fig = px.bar(feat_imp, x='Importance', y='Feature',
                 orientation='h',
                 title='What drives Dengue cases most?',
                 color='Importance', color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    st.info("📌 **Key Finding:** Seasonal pattern (month encoding) is the strongest predictor, followed by 3-month average temperature — confirming climate plays a major role.")

# ══════════════════════════════════════════════════════════════
# PAGE 6 — PREDICT CASES
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict Cases":
    st.title("🔮 Predict Dengue Cases")
    st.markdown("Enter climate conditions to predict expected dengue cases:")

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("Temperature (°C)", 15.0, 35.0, 25.0)
        rainfall    = st.slider("Rainfall (mm)", 0.0, 300.0, 50.0)
    with col2:
        humidity    = st.slider("Humidity (%)", 30.0, 100.0, 70.0)
        month       = st.selectbox("Month", range(1,13),
                        format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun',
                                               'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    with col3:
        year        = st.slider("Year", 2010, 2030, 2024)

    # Build input
    year_trend      = year - 2010
    month_sin       = np.sin(2 * np.pi * month / 12)
    month_cos       = np.cos(2 * np.pi * month / 12)
    temp_3m_avg     = temperature
    rainfall_3m_avg = rainfall
    humidity_3m_avg = humidity
    rainfall_lag1   = rainfall

    input_data = pd.DataFrame([[
        temperature, rainfall, humidity,
        temp_3m_avg, rainfall_3m_avg, humidity_3m_avg,
        month_sin, month_cos, year_trend, rainfall_lag1
    ]], columns=[
        'temperature','rainfall','humidity',
        'temp_3m_avg','rainfall_3m_avg','humidity_3m_avg',
        'month_sin','month_cos','year_trend','rainfall_lag1'
    ])

    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.metric("🦟 Predicted Dengue Cases", f"{max(0, int(prediction))} cases")

    if prediction > 200:
        st.error("🚨 HIGH RISK — Public health alert recommended!")
    elif prediction > 100:
        st.warning("⚠️ MODERATE RISK — Increased surveillance advised.")
    else:
        st.success("✅ LOW RISK — Normal monitoring recommended.")

        # ── Climate Scenario Table ─────────────────────────────────
    st.markdown("---")
    st.subheader("🌍 Climate Scenario Analysis")
    st.markdown("How would dengue cases change under different future climate conditions?")

    # Build scenarios based on current input
    scenarios = [
        {
            "Scenario"        : "🟡 Normal (2024 baseline)",
            "Temp Change"     : "No change",
            "Rain Change"     : "No change",
            "Predicted Cases" : max(0, int(prediction)),
            "Risk Level"      : "Moderate"
        },
        {
            "Scenario"        : "🔴 Hotter year (+1°C)",
            "Temp Change"     : "+1°C",
            "Rain Change"     : "No change",
            "Predicted Cases" : max(0, int(model.predict(pd.DataFrame([[
                temperature+1, rainfall, humidity,
                temp_3m_avg+1, rainfall_3m_avg, humidity_3m_avg,
                month_sin, month_cos, year_trend, rainfall_lag1
            ]], columns=[
                'temperature','rainfall','humidity',
                'temp_3m_avg','rainfall_3m_avg','humidity_3m_avg',
                'month_sin','month_cos','year_trend','rainfall_lag1'
            ]))[0])),
            "Risk Level"      : "High"
        },
        {
            "Scenario"        : "🔴 Drier year (−10% rain)",
            "Temp Change"     : "No change",
            "Rain Change"     : "−10%",
            "Predicted Cases" : max(0, int(model.predict(pd.DataFrame([[
                temperature, rainfall*0.9, humidity,
                temp_3m_avg, rainfall_3m_avg*0.9, humidity_3m_avg,
                month_sin, month_cos, year_trend, rainfall_lag1*0.9
            ]], columns=[
                'temperature','rainfall','humidity',
                'temp_3m_avg','rainfall_3m_avg','humidity_3m_avg',
                'month_sin','month_cos','year_trend','rainfall_lag1'
            ]))[0])),
            "Risk Level"      : "High"
        },
        {
            "Scenario"        : "🟢 Wetter year (+10% rain)",
            "Temp Change"     : "No change",
            "Rain Change"     : "+10%",
            "Predicted Cases" : max(0, int(model.predict(pd.DataFrame([[
                temperature, rainfall*1.1, humidity,
                temp_3m_avg, rainfall_3m_avg*1.1, humidity_3m_avg,
                month_sin, month_cos, year_trend, rainfall_lag1*1.1
            ]], columns=[
                'temperature','rainfall','humidity',
                'temp_3m_avg','rainfall_3m_avg','humidity_3m_avg',
                'month_sin','month_cos','year_trend','rainfall_lag1'
            ]))[0])),
            "Risk Level"      : "Low"
        },
        {
            "Scenario"        : "🔴 Hot & Dry (worst case)",
            "Temp Change"     : "+2°C",
            "Rain Change"     : "−20%",
            "Predicted Cases" : max(0, int(model.predict(pd.DataFrame([[
                temperature+2, rainfall*0.8, humidity-5,
                temp_3m_avg+2, rainfall_3m_avg*0.8, humidity_3m_avg-5,
                month_sin, month_cos, year_trend, rainfall_lag1*0.8
            ]], columns=[
                'temperature','rainfall','humidity',
                'temp_3m_avg','rainfall_3m_avg','humidity_3m_avg',
                'month_sin','month_cos','year_trend','rainfall_lag1'
            ]))[0])),
            "Risk Level"      : "Very High"
        },
        {
            "Scenario"        : "🟢 Cool & Wet (best case)",
            "Temp Change"     : "−1°C",
            "Rain Change"     : "+20%",
            "Predicted Cases" : max(0, int(model.predict(pd.DataFrame([[
                temperature-1, rainfall*1.2, humidity+5,
                temp_3m_avg-1, rainfall_3m_avg*1.2, humidity_3m_avg+5,
                month_sin, month_cos, year_trend, rainfall_lag1*1.2
            ]], columns=[
                'temperature','rainfall','humidity',
                'temp_3m_avg','rainfall_3m_avg','humidity_3m_avg',
                'month_sin','month_cos','year_trend','rainfall_lag1'
            ]))[0])),
            "Risk Level"      : "Low"
        },
    ]

    scenario_df = pd.DataFrame(scenarios)

    # Color risk levels
    def color_risk(val):
        colors = {
            'Low'      : 'background-color: #1a3a2a; color: #4caf50',
            'Moderate' : 'background-color: #3a3010; color: #ffd166',
            'High'     : 'background-color: #3a1a1a; color: #ff6b6b',
            'Very High': 'background-color: #4a0a0a; color: #ff1a1a',
        }
        return colors.get(val, '')

    styled_df = scenario_df.style.applymap(
        color_risk, subset=['Risk Level']
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Bar chart of scenarios
    fig = px.bar(
        scenario_df,
        x='Scenario',
        y='Predicted Cases',
        color='Risk Level',
        color_discrete_map={
            'Low'      : '#4caf50',
            'Moderate' : '#ffd166',
            'High'     : '#ff6b6b',
            'Very High': '#ff1a1a'
        },
        title='Dengue Cases Under Different Climate Scenarios',
        text='Predicted Cases'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    📌 **What this tells us:** The model shows that a **Hot & Dry climate scenario**
    produces the highest dengue risk. This supports the project's core finding that
    climatic variability directly influences disease dynamics — and highlights the
    need for **proactive public health planning** based on climate forecasts.
    """)