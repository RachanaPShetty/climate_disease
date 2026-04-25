import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

BASE = r'C:\Users\HP\OneDrive\Desktop\6THSEM\MAJORPROJECT\climate_disease_project'

# ── Load data ──────────────────────────────────────────────────
df = pd.read_csv(f'{BASE}\\data\\processed\\features_data.csv')
print("Data loaded:", df.shape)

# ══════════════════════════════════════════════════════════════
# PART 1 — PCA (Principal Component Analysis)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("PART 1: PCA ANALYSIS")
print("="*50)

pca_cols = ['temperature','rainfall','humidity',
            'temp_3m_avg','rainfall_3m_avg','humidity_3m_avg',
            'dengue_cases','malaria_cases','cholera_cases']

X_pca    = df[pca_cols].dropna()
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# Fit full PCA
pca           = PCA()
pca.fit(X_scaled)
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print("\nExplained Variance per Component:")
for i, (ev, cv) in enumerate(zip(explained_var, cumulative_var)):
    print(f"  PC{i+1}: {ev*100:.2f}%  |  Cumulative: {cv*100:.2f}%")

n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"\n✅ {n_components_95} components explain 95%+ of variance")

# ── Plot 1: Scree Plot + Biplot ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PCA Analysis — Climate & Disease Variables', fontsize=14)

# Scree plot
axes[0].bar(range(1, len(explained_var)+1), explained_var*100,
            color='steelblue', alpha=0.8, label='Individual')
axes[0].plot(range(1, len(explained_var)+1), cumulative_var*100,
             color='red', marker='o', linewidth=2, label='Cumulative')
axes[0].axhline(95, color='green', linestyle='--',
                linewidth=1.5, label='95% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('Scree Plot')
axes[0].legend()
axes[0].set_xticks(range(1, len(explained_var)+1))

# Biplot PC1 vs PC2
pca2   = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

season_colors = {
    'Winter'     : 'blue',
    'Summer'     : 'orange',
    'Monsoon'    : 'green',
    'Post-Monsoon': 'red'
}
colors = df['season'].map(season_colors)

axes[1].scatter(X_pca2[:,0], X_pca2[:,1],
                c=colors, alpha=0.6, s=40)
axes[1].set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('Biplot: PC1 vs PC2 (coloured by season)')

# Feature arrows
loadings = pca2.components_.T
for i, col in enumerate(pca_cols):
    axes[1].arrow(0, 0,
                  loadings[i,0]*3, loadings[i,1]*3,
                  head_width=0.1, head_length=0.05,
                  fc='black', ec='black', alpha=0.7)
    axes[1].text(loadings[i,0]*3.2, loadings[i,1]*3.2,
                 col, fontsize=7, ha='center')

legend_elements = [Patch(facecolor=v, label=k)
                   for k,v in season_colors.items()]
axes[1].legend(handles=legend_elements, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{BASE}\\reports\\pca_scree_biplot.png')
plt.show()
print("Scree plot + Biplot saved!")

# ── Plot 2: PCA Loadings Heatmap ──────────────────────────────
pca3 = PCA(n_components=4)
pca3.fit(X_scaled)
loadings_df = pd.DataFrame(
    pca3.components_.T,
    columns=[f'PC{i+1}' for i in range(4)],
    index=pca_cols
)

plt.figure(figsize=(8, 6))
sns.heatmap(loadings_df, annot=True, fmt='.2f',
            cmap='RdBu_r', linewidths=0.5, center=0)
plt.title('PCA Loadings Heatmap (Top 4 Components)')
plt.tight_layout()
plt.savefig(f'{BASE}\\reports\\pca_loadings.png')
plt.show()
print("PCA loadings heatmap saved!")

print("\n── PCA Key Findings ──")
print(f"PC1 explains {pca2.explained_variance_ratio_[0]*100:.1f}%"
      " — mainly driven by seasonal climate")
print(f"PC2 explains {pca2.explained_variance_ratio_[1]*100:.1f}%"
      " — mainly driven by disease variation")


# ══════════════════════════════════════════════════════════════
# PART 2 — ARIMA (Time Series Forecasting)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("PART 2: ARIMA FORECASTING")
print("="*50)

df_sorted     = df.sort_values(['year','month']).reset_index(drop=True)
dengue_series = df_sorted['dengue_cases']

# ── ADF Test on original series ────────────────────────────────
adf_orig = adfuller(dengue_series)
print(f"\nADF on original series:")
print(f"  ADF Statistic : {adf_orig[0]:.4f}")
print(f"  p-value       : {adf_orig[1]:.4f}")
if adf_orig[1] < 0.05:
    print("✅ Series is STATIONARY")
else:
    print("⚠️ NON-STATIONARY — checking differenced series...")

# ── ADF Test on differenced series ────────────────────────────
adf_diff = adfuller(dengue_series.diff().dropna())
print(f"\nADF on differenced series:")
print(f"  ADF Statistic : {adf_diff[0]:.4f}")
print(f"  p-value       : {adf_diff[1]:.4f}")
if adf_diff[1] < 0.05:
    print("✅ Differenced series is STATIONARY — ARIMA(1,1,1) ready!")

# ── Train / Test split ─────────────────────────────────────────
train_size = int(len(dengue_series) * 0.8)
train      = dengue_series[:train_size]
test       = dengue_series[train_size:]

print(f"\nARIMA Training on {len(train)} months")
print(f"Forecasting    {len(test)} months ahead")

# ── Fit ARIMA(1,1,1) ──────────────────────────────────────────
model_arima   = ARIMA(train, order=(1,1,1))
fitted        = model_arima.fit()
forecast_vals = np.array(fitted.forecast(steps=len(test)))

# ── Future 12-month forecast ───────────────────────────────────
future_model    = ARIMA(dengue_series, order=(1,1,1))
future_fitted   = future_model.fit()
future_forecast = future_fitted.forecast(steps=12)

print("\n📅 Forecast for next 12 months (2024):")
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']
for i, val in enumerate(future_forecast):
    bar = '█' * min(int(max(0, val)/10), 25)
    print(f"  2024-{months[i]:3s}: {max(0,int(val)):4d} cases  {bar}")

# ── Accuracy ───────────────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(test.values, forecast_vals))
mae  = mean_absolute_error(test.values, forecast_vals)
print(f"\nARIMA Test Performance:")
print(f"  RMSE : {rmse:.2f}")
print(f"  MAE  : {mae:.2f}")

# ── Plot 1: Test forecast vs actual ───────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('ARIMA(1,1,1) Forecasting — Dengue Cases', fontsize=14)

axes[0].plot(range(len(train)), train.values,
             label='Training Data', color='blue', linewidth=1.5)
axes[0].plot(range(len(train), len(dengue_series)),
             test.values, label='Actual', color='green', linewidth=2)
axes[0].plot(range(len(train), len(dengue_series)),
             forecast_vals, label='ARIMA(1,1,1) Forecast',
             color='red', linewidth=2, linestyle='--')
axes[0].set_title('ARIMA: Test Period — Actual vs Forecast')
axes[0].set_xlabel('Month Index')
axes[0].set_ylabel('Dengue Cases')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ── Plot 2: Future 12-month forecast ──────────────────────────
future_x = list(range(len(dengue_series), len(dengue_series)+12))

axes[1].plot(range(len(dengue_series)), dengue_series.values,
             label='Historical (2010-2023)', color='blue', linewidth=1.5)
axes[1].plot(future_x, future_forecast.values,
             label='2024 Forecast', color='red',
             linewidth=2.5, linestyle='--', marker='o', markersize=6)
axes[1].fill_between(future_x,
                     future_forecast.values * 0.75,
                     future_forecast.values * 1.25,
                     alpha=0.2, color='red', label='±25% uncertainty band')
axes[1].axvline(x=len(dengue_series)-1, color='gray',
                linestyle=':', linewidth=1.5, label='Forecast start')
axes[1].set_title('ARIMA: 12-Month Future Forecast (2024)')
axes[1].set_xlabel('Month Index')
axes[1].set_ylabel('Dengue Cases')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{BASE}\\reports\\arima_forecast.png')
plt.show()
print("\nARIMA forecast chart saved! ✅")

print("\n" + "="*50)
print("✅ PCA + ARIMA COMPLETE!")
print("="*50)
print("\nAll charts saved in reports/ folder:")
print("  ✅ pca_scree_biplot.png")
print("  ✅ pca_loadings.png")
print("  ✅ arima_forecast.png")