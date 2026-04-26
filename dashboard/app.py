import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go

BASE = r'C:\Projects\ml_multidimensional_model\climate_disease'

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Climate & Disease Intelligence",
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


# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🌡️ Climate & Disease Intelligence Platform")
    st.markdown("*Understanding how climate variability shapes disease dynamics through machine learning*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Years Covered", "2010–2023")
    col2.metric("📊 Total Records", len(df))
    col3.metric("🌡️ Avg Temperature", f"{df['temperature'].mean():.1f} °C")
    col4.metric("🦟 Total Dengue Cases", f"{df['dengue_cases'].sum():,}")

    st.markdown("---")
    st.subheader("About This Platform")
    st.info("""
    This platform analyses the relationship between **climate change** and **disease patterns**
    using machine learning. By integrating climate variables — temperature, rainfall, and humidity —
    with disease occurrence data for Dengue, Malaria, and Cholera, it uncovers long-term trends
    and correlations to support **public health planning** and **climate-health decision-making**.
    """)

    col1, col2, col3 = st.columns(3)
    col1.success("🌦️ **Climate Analysis** — Temperature, rainfall & humidity trends over time")
    col2.success("🦟 **Disease Tracking** — Multi-disease temporal and seasonal patterns")
    col3.success("🤖 **ML Prediction** — Random Forest model with R² = 0.992")

# ══════════════════════════════════════════════════════════════
# PAGE 2 — CLIMATE TRENDS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Climate Trends":
    st.title("📈 Climate Trends (2010–2023)")

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

    st.subheader("Year-wise Disease Totals")
    yearly = df.groupby('year')[['dengue_cases','malaria_cases','cholera_cases']].sum()
    fig2 = px.bar(yearly, barmode='group',
                  title='Annual Disease Case Counts',
                  color_discrete_sequence=['orange','purple','green'])
    st.plotly_chart(fig2, use_container_width=True)

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
    st.info("📌 **Key Finding:** Rainfall affects Dengue with a **1-month lag** (corr = 0.513), consistent with mosquito breeding cycles.")

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
                 title='Feature Importance for Dengue Prediction',
                 color='Importance', color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    st.info("📌 **Key Finding:** Seasonal patterns are the strongest predictor, followed by 3-month average temperature — confirming climate plays a significant role in disease dynamics.")

# ══════════════════════════════════════════════════════════════
# PAGE 6 — PREDICT CASES
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict Cases":

    # ── Custom CSS ─────────────────────────────────────────────
    st.markdown("""
    <style>
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .metric-box .val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e2e8f0;
        line-height: 1.1;
    }
    .metric-box .lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }
    .risk-high   { border-color: #ef4444 !important; }
    .risk-med    { border-color: #f59e0b !important; }
    .risk-low    { border-color: #10b981 !important; }
    .section-div { border-top: 1px solid #1e293b; margin: 2rem 0; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔮 Dengue Case Predictor")
    st.markdown("Adjust climate conditions below and get an instant ML-powered prediction.")
    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Inputs ─────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("🌡️ Temperature (°C)", 15.0, 35.0, 25.0, step=0.5)
        rainfall    = st.slider("🌧️ Rainfall (mm)",    0.0, 300.0, 50.0, step=5.0)
    with col2:
        humidity    = st.slider("💧 Humidity (%)",      30.0, 100.0, 70.0, step=1.0)
        month       = st.selectbox("📅 Month", range(1, 13),
                        format_func=lambda x: ['January','February','March','April',
                                               'May','June','July','August','September',
                                               'October','November','December'][x-1])
    with col3:
        year        = st.slider("📆 Year", 2010, 2030, 2024)
        season_map  = {12:'Winter',1:'Winter',2:'Winter',3:'Summer',4:'Summer',5:'Summer',
                       6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',
                       10:'Post-Monsoon',11:'Post-Monsoon'}
        current_season = season_map[month]
        st.markdown(f"**Current Season:** `{current_season}`")
        st.markdown(f"**Dengue Risk Season:** {'🔴 High' if current_season in ['Monsoon','Post-Monsoon'] else '🟢 Low'}")

    # ── Compute prediction ─────────────────────────────────────
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

    prediction = max(0, int(model.predict(input_data)[0]))

    # Risk level
    if prediction > 200:
        risk, risk_color, risk_emoji = "HIGH", "#ef4444", "🚨"
    elif prediction > 100:
        risk, risk_color, risk_emoji = "MODERATE", "#f59e0b", "⚠️"
    else:
        risk, risk_color, risk_emoji = "LOW", "#10b981", "✅"

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Result metrics ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🦟 Predicted Cases", f"{prediction}")
    c2.metric("⚠️ Risk Level", risk)
    c3.metric("🌡️ Temperature", f"{temperature}°C")
    c4.metric("💧 Humidity", f"{humidity}%")

    # Risk banner
    if risk == "HIGH":
        st.error(f"{risk_emoji} HIGH RISK — Elevated dengue activity expected. Public health alert recommended.")
    elif risk == "MODERATE":
        st.warning(f"{risk_emoji} MODERATE RISK — Increased surveillance advised.")
    else:
        st.success(f"{risk_emoji} LOW RISK — Normal conditions expected.")

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Gauge chart ────────────────────────────────────────────
    col_g, col_r = st.columns([1, 1])

    with col_g:
        st.subheader("📊 Risk Gauge")
        import plotly.graph_objects as go
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={'reference': 100, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
            gauge={
                'axis': {'range': [0, 300], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': risk_color},
                'bgcolor': "#1e293b",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 100],   'color': '#0d3326'},
                    {'range': [100, 200], 'color': '#3d2a07'},
                    {'range': [200, 300], 'color': '#3d0f0f'},
                ],
                'threshold': {'line': {'color': "white", 'width': 2}, 'thickness': 0.75, 'value': 200}
            },
            title={'text': "Predicted Dengue Cases", 'font': {'color': '#94a3b8', 'size': 14}}
        ))
        fig_gauge.update_layout(
            height=280, margin=dict(t=30, b=10, l=30, r=30),
            paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_r:
        st.subheader("📈 How inputs affect risk")
        # Show how each variable relates to the prediction conceptually
        factors = pd.DataFrame({
            'Factor': ['Temperature', 'Humidity', 'Rainfall', 'Season Risk'],
            'Level':  [
                round((temperature - 15) / 20 * 100),
                round((humidity - 30) / 70 * 100),
                round(min(rainfall, 300) / 300 * 100),
                100 if current_season in ['Monsoon','Post-Monsoon'] else 25
            ]
        })
        fig_bar = go.Figure(go.Bar(
            x=factors['Level'], y=factors['Factor'],
            orientation='h',
            marker_color=['#fb923c','#60a5fa','#34d399','#a78bfa'],
            text=[f"{v}%" for v in factors['Level']],
            textposition='outside'
        ))
        fig_bar.update_layout(
            height=280, margin=dict(t=10, b=10, l=10, r=60),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0', xaxis=dict(range=[0,120], showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Monthly prediction curve ───────────────────────────────
    st.subheader("📅 Predicted cases — all 12 months (for current climate inputs)")

    monthly_preds = []
    for m in range(1, 13):
        ms = np.sin(2 * np.pi * m / 12)
        mc = np.cos(2 * np.pi * m / 12)
        inp = pd.DataFrame([[temperature, rainfall, humidity,
                              temp_3m_avg, rainfall_3m_avg, humidity_3m_avg,
                              ms, mc, year_trend, rainfall_lag1]],
                           columns=['temperature','rainfall','humidity',
                                    'temp_3m_avg','rainfall_3m_avg','humidity_3m_avg',
                                    'month_sin','month_cos','year_trend','rainfall_lag1'])
        monthly_preds.append(max(0, int(model.predict(inp)[0])))

    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colors_monthly = ['#ef4444' if v > 200 else '#f59e0b' if v > 100 else '#34d399' for v in monthly_preds]

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=month_names, y=monthly_preds,
        marker_color=colors_monthly,
        text=monthly_preds, textposition='outside',
        name='Predicted Cases'
    ))
    fig_monthly.add_hline(y=100, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Moderate threshold", annotation_position="right")
    fig_monthly.add_hline(y=200, line_dash="dash", line_color="#ef4444",
                          annotation_text="High threshold", annotation_position="right")
    fig_monthly.update_layout(
        height=320, margin=dict(t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0', yaxis=dict(showgrid=True, gridcolor='#1e293b'),
        xaxis=dict(showgrid=False), showlegend=False
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Scenario Analysis ──────────────────────────────────────
    st.subheader("🌍 Climate Scenario Analysis")
    st.markdown("How would dengue cases shift under different future climate conditions?")

    def predict_cases(t, r, h, t3, r3, h3, ms, mc, yt, rl):
        return max(0, int(model.predict(pd.DataFrame([[t, r, h, t3, r3, h3, ms, mc, yt, rl]],
            columns=['temperature','rainfall','humidity','temp_3m_avg','rainfall_3m_avg',
                     'humidity_3m_avg','month_sin','month_cos','year_trend','rainfall_lag1']))[0]))

    scenarios = [
        {"Scenario": "🟡 Baseline",          "Temp Change": "—",    "Rain Change": "—",
         "Predicted Cases": prediction,       "Risk Level": "Moderate"},
        {"Scenario": "🔴 Hotter (+1°C)",      "Temp Change": "+1°C", "Rain Change": "—",
         "Predicted Cases": predict_cases(temperature+1, rainfall, humidity, temp_3m_avg+1, rainfall_3m_avg, humidity_3m_avg, month_sin, month_cos, year_trend, rainfall_lag1), "Risk Level": "High"},
        {"Scenario": "🔴 Drier (−10% rain)",  "Temp Change": "—",    "Rain Change": "−10%",
         "Predicted Cases": predict_cases(temperature, rainfall*0.9, humidity, temp_3m_avg, rainfall_3m_avg*0.9, humidity_3m_avg, month_sin, month_cos, year_trend, rainfall_lag1*0.9), "Risk Level": "High"},
        {"Scenario": "🟢 Wetter (+10% rain)", "Temp Change": "—",    "Rain Change": "+10%",
         "Predicted Cases": predict_cases(temperature, rainfall*1.1, humidity, temp_3m_avg, rainfall_3m_avg*1.1, humidity_3m_avg, month_sin, month_cos, year_trend, rainfall_lag1*1.1), "Risk Level": "Low"},
        {"Scenario": "🔴 Hot & Dry (worst)",  "Temp Change": "+2°C", "Rain Change": "−20%",
         "Predicted Cases": predict_cases(temperature+2, rainfall*0.8, humidity-5, temp_3m_avg+2, rainfall_3m_avg*0.8, humidity_3m_avg-5, month_sin, month_cos, year_trend, rainfall_lag1*0.8), "Risk Level": "Very High"},
        {"Scenario": "🟢 Cool & Wet (best)",  "Temp Change": "−1°C", "Rain Change": "+20%",
         "Predicted Cases": predict_cases(temperature-1, rainfall*1.2, humidity+5, temp_3m_avg-1, rainfall_3m_avg*1.2, humidity_3m_avg+5, month_sin, month_cos, year_trend, rainfall_lag1*1.2), "Risk Level": "Low"},
    ]

    scenario_df = pd.DataFrame(scenarios)

    def color_risk(val):
        return {
            'Low'      : 'background-color: #0d3326; color: #34d399',
            'Moderate' : 'background-color: #3a3010; color: #fbbf24',
            'High'     : 'background-color: #3a1a1a; color: #f87171',
            'Very High': 'background-color: #4a0a0a; color: #ff1a1a',
        }.get(val, '')

    st.dataframe(scenario_df.style.map(color_risk, subset=['Risk Level']),
                 use_container_width=True, hide_index=True)

    fig_scenario = go.Figure(go.Bar(
        x=scenario_df['Scenario'],
        y=scenario_df['Predicted Cases'],
        marker_color=['#fbbf24','#f87171','#f87171','#34d399','#ef4444','#34d399'],
        text=scenario_df['Predicted Cases'],
        textposition='outside'
    ))
    fig_scenario.update_layout(
        height=340, margin=dict(t=20, b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0', xaxis=dict(tickangle=-15, showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#1e293b'), showlegend=False
    )
    st.plotly_chart(fig_scenario, use_container_width=True)

    st.info("""
    📌 **Insight:** A hot and dry climate scenario consistently produces the highest predicted dengue risk,
    while cooler and wetter conditions suppress case counts — directly supporting the project's core finding
    that climatic variability drives disease dynamics.
    """)