import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Load merged data ───────────────────────────────────────────
# NEW (correct)
df = pd.read_csv(r'C:\Projects\ml_multidimensional_model\climate_disease\data\processed\merged_data.csv')

print("Loaded data shape:", df.shape)
print(df.head())

# ══════════════════════════════════════════════════════════════
# STEP 1 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

# 1a. Rolling averages (climate conditions of past 3 months)
df['temp_3m_avg']     = df['temperature'].rolling(window=3, min_periods=1).mean()
df['rainfall_3m_avg'] = df['rainfall'].rolling(window=3, min_periods=1).mean()
df['humidity_3m_avg'] = df['humidity'].rolling(window=3, min_periods=1).mean()

# 1b. Season encoding
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Summer'
    elif month in [6, 7, 8, 9]: return 'Monsoon'
    else: return 'Post-Monsoon'

df['season'] = df['month'].apply(get_season)

# 1c. Month sine/cosine encoding (captures cyclical nature)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# 1d. Year trend (captures long-term climate effect)
df['year_trend'] = df['year'] - df['year'].min()

print("\nNew features added:")
print(df[['year','month','season','temp_3m_avg','rainfall_3m_avg','month_sin']].head(12))

# ══════════════════════════════════════════════════════════════
# STEP 2 — LAG ANALYSIS
# (Disease peaks AFTER climate change — find the delay)
# ══════════════════════════════════════════════════════════════

print("\n── Lag Analysis: Rainfall vs Dengue ──")
print("Lag 0 = same month, Lag 1 = 1 month delay, etc.\n")

lag_results = []
for lag in range(0, 5):
    corr_dengue  = df['rainfall'].corr(df['dengue_cases'].shift(-lag))
    corr_malaria = df['rainfall'].corr(df['malaria_cases'].shift(-lag))
    lag_results.append({
        'lag_months'    : lag,
        'rainfall_dengue_corr' : round(corr_dengue, 3),
        'rainfall_malaria_corr': round(corr_malaria, 3)
    })
    print(f"Lag {lag} month(s): Dengue={corr_dengue:.3f}  |  Malaria={corr_malaria:.3f}")

lag_df = pd.DataFrame(lag_results)

# Plot lag analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Lag Analysis: Rainfall Effect on Disease Cases', fontsize=14)

axes[0].bar(lag_df['lag_months'], lag_df['rainfall_dengue_corr'], color='orange')
axes[0].set_title('Rainfall → Dengue')
axes[0].set_xlabel('Lag (months)')
axes[0].set_ylabel('Correlation')
axes[0].axhline(0, color='black', linewidth=0.5)

axes[1].bar(lag_df['lag_months'], lag_df['rainfall_malaria_corr'], color='purple')
axes[1].set_title('Rainfall → Malaria')
axes[1].set_xlabel('Lag (months)')
axes[1].axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(r'C:\Projects\ml_multidimensional_model\climate_disease\reports\lag_analysis.png')
plt.show()
print("\nLag analysis chart saved!")

# ══════════════════════════════════════════════════════════════
# STEP 3 — PEARSON + SPEARMAN CORRELATION
# ══════════════════════════════════════════════════════════════

print("\n── Pearson Correlation ──")
cols = ['temperature','rainfall','humidity','temp_3m_avg',
        'rainfall_3m_avg','dengue_cases','malaria_cases','cholera_cases']

pearson_corr = df[cols].corr(method='pearson')
spearman_corr = df[cols].corr(method='spearman')

print("\nPearson - Dengue correlations:")
print(pearson_corr['dengue_cases'].drop('dengue_cases').sort_values(ascending=False))

print("\nSpearman - Dengue correlations:")
print(spearman_corr['dengue_cases'].drop('dengue_cases').sort_values(ascending=False))

# Plot both heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[0], linewidths=0.5)
axes[0].set_title('Pearson Correlation Matrix')

sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[1], linewidths=0.5)
axes[1].set_title('Spearman Correlation Matrix')

plt.tight_layout()
plt.savefig(r'C:\Projects\ml_multidimensional_model\climate_disease\reports\pearson_spearman.png')
plt.show()
print("Correlation heatmaps saved!")

# ══════════════════════════════════════════════════════════════
# STEP 4 — SAVE ENRICHED DATASET
# ══════════════════════════════════════════════════════════════

df.to_csv(r'C:\Projects\ml_multidimensional_model\climate_disease\data\processed\features_data.csv', index=False)
print("\nEnriched dataset saved to data/processed/features_data.csv ✅")
print("Final dataset shape:", df.shape)
print("Columns:", list(df.columns))