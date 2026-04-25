
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load both datasets ─────────────────────────────────────────
climate_df = pd.read_csv('../data/raw/climate_data.csv')
disease_df = pd.read_csv('../data/raw/disease_data.csv')

# ── Merge on year + month ──────────────────────────────────────
df = pd.merge(climate_df, disease_df, on=['year', 'month'])
df.to_csv('../data/processed/merged_data.csv', index=False)

print("Merged dataset shape:", df.shape)
print(df.head())

# ── Plot 1: Climate variables over time ────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Bangalore Climate (2010–2023)', fontsize=16)

axes[0].plot(range(len(df)), df['temperature'], color='tomato')
axes[0].set_ylabel('Temperature (°C)')

axes[1].plot(range(len(df)), df['rainfall'], color='steelblue')
axes[1].set_ylabel('Rainfall (mm)')

axes[2].plot(range(len(df)), df['humidity'], color='seagreen')
axes[2].set_ylabel('Humidity (%)')

plt.tight_layout()
plt.savefig('../reports/climate_trends.png')
plt.show()
print("Chart 1 saved!")

# ── Plot 2: Disease cases over time ───────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7))
fig.suptitle('Disease Cases Over Time', fontsize=16)

axes[0].plot(range(len(df)), df['dengue_cases'], color='orange')
axes[0].set_ylabel('Dengue Cases')

axes[1].plot(range(len(df)), df['malaria_cases'], color='purple')
axes[1].set_ylabel('Malaria Cases')

plt.tight_layout()
plt.savefig('../reports/disease_trends.png')
plt.show()
print("Chart 2 saved!")

# ── Plot 3: Correlation heatmap ────────────────────────────────
plt.figure(figsize=(8, 6))
sns.heatmap(df[['temperature','rainfall','humidity',
                'dengue_cases','malaria_cases']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation: Climate vs Disease')
plt.tight_layout()
plt.savefig('../reports/correlation_heatmap.png')
plt.show()
print("Chart 3 saved!")

print("\nCorrelation with Dengue:")
print(df[['temperature','rainfall','humidity','dengue_cases']].corr()['dengue_cases'])