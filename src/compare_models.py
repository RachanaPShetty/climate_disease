import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── Load data ──────────────────────────────────────────────────
df = pd.read_csv('../data/processed/merged_data.csv')

features = ['temperature', 'rainfall', 'humidity', 'month']
X        = df[features]
y_dengue  = df['dengue_cases']
y_malaria = df['malaria_cases']

# ── Train/Test Split ───────────────────────────────────────────
X_train, X_test, yd_train, yd_test = train_test_split(X, y_dengue,  test_size=0.2, random_state=42)
_,       _,      ym_train, ym_test = train_test_split(X, y_malaria, test_size=0.2, random_state=42)

# ── Define both models ─────────────────────────────────────────
models = {
    'Random Forest': {
        'dengue' : RandomForestRegressor(n_estimators=100, random_state=42),
        'malaria': RandomForestRegressor(n_estimators=100, random_state=42)
    },
    'XGBoost': {
        'dengue' : XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'malaria': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
}

# ── Train & Evaluate ───────────────────────────────────────────
results = []

for model_name, model_dict in models.items():
    print(f"\nTraining {model_name}...")

    # Dengue
    model_dict['dengue'].fit(X_train, yd_train)
    yd_pred = model_dict['dengue'].predict(X_test)
    d_r2  = r2_score(yd_test, yd_pred)
    d_mae = mean_absolute_error(yd_test, yd_pred)

    # Malaria
    model_dict['malaria'].fit(X_train, ym_train)
    ym_pred = model_dict['malaria'].predict(X_test)
    m_r2  = r2_score(ym_test, ym_pred)
    m_mae = mean_absolute_error(ym_test, ym_pred)

    print(f"  Dengue  — R²: {d_r2:.4f} | MAE: {d_mae:.2f}")
    print(f"  Malaria — R²: {m_r2:.4f} | MAE: {m_mae:.2f}")

    results.append({
        'Model'      : model_name,
        'Dengue_R2'  : round(d_r2,  4),
        'Dengue_MAE' : round(d_mae, 2),
        'Malaria_R2' : round(m_r2,  4),
        'Malaria_MAE': round(m_mae, 2)
    })

# ── Results Table ──────────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n── COMPARISON TABLE ──────────────────────────────────")
print(results_df.to_string(index=False))

# ── Plot Comparison ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Random Forest vs XGBoost', fontsize=16)

x = np.arange(2)
width = 0.35
model_names = [r['Model'] for r in results]

# R² comparison
r2_dengue  = [r['Dengue_R2']  for r in results]
r2_malaria = [r['Malaria_R2'] for r in results]

axes[0].bar(x - width/2, r2_dengue,  width, label='Dengue',  color='orange')
axes[0].bar(x + width/2, r2_malaria, width, label='Malaria', color='purple')
axes[0].set_title('R² Score (higher = better)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names)
axes[0].set_ylim(0.8, 1.0)
axes[0].legend()
axes[0].set_ylabel('R² Score')

# MAE comparison
mae_dengue  = [r['Dengue_MAE']  for r in results]
mae_malaria = [r['Malaria_MAE'] for r in results]

axes[1].bar(x - width/2, mae_dengue,  width, label='Dengue',  color='orange')
axes[1].bar(x + width/2, mae_malaria, width, label='Malaria', color='purple')
axes[1].set_title('MAE (lower = better)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names)
axes[1].legend()
axes[1].set_ylabel('Mean Absolute Error')

plt.tight_layout()
plt.savefig('../reports/model_comparison.png')
plt.show()

print("\nComparison chart saved to reports/model_comparison.png!")