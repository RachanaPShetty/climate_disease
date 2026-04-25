import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pickle

BASE = r'C:\Users\HP\OneDrive\Desktop\6THSEM\MAJORPROJECT\climate_disease_project'

# ── Load enriched data ─────────────────────────────────────────
df = pd.read_csv(f'{BASE}\\data\\processed\\features_data.csv')
print("Data loaded:", df.shape)

# ══════════════════════════════════════════════════════════════
# STEP 1 — PREPARE FEATURES
# Best predictors found in lag/correlation analysis
# ══════════════════════════════════════════════════════════════

features = [
    'temperature', 'rainfall', 'humidity',
    'temp_3m_avg', 'rainfall_3m_avg', 'humidity_3m_avg',
    'month_sin', 'month_cos', 'year_trend'
]

# Add 1-month rainfall lag (best lag for dengue)
df['rainfall_lag1'] = df['rainfall'].shift(1).fillna(0)
features.append('rainfall_lag1')

X = df[features]
y_dengue  = df['dengue_cases']
y_malaria = df['malaria_cases']

# Train/test split (80/20)
X_train, X_test, yd_train, yd_test = train_test_split(
    X, y_dengue, test_size=0.2, random_state=42)
_, _, ym_train, ym_test = train_test_split(
    X, y_malaria, test_size=0.2, random_state=42)

print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ══════════════════════════════════════════════════════════════
# STEP 2 — TRAIN 3 MODELS ON DENGUE
# ══════════════════════════════════════════════════════════════

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest'    : RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost'          : xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

results = []
print("\n── Model Performance on Dengue Cases ──\n")

for name, model in models.items():
    model.fit(X_train, yd_train)
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(yd_test, preds))
    r2    = r2_score(yd_test, preds)
    results.append({'Model': name, 'RMSE': round(rmse,2), 'R2': round(r2,3)})
    print(f"{name:20s} → RMSE: {rmse:.2f}  |  R²: {r2:.3f}")

results_df = pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════
# STEP 3 — FEATURE IMPORTANCE (Random Forest)
# ══════════════════════════════════════════════════════════════

rf_model = models['Random Forest']
importance_df = pd.DataFrame({
    'Feature'   : features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n── Feature Importance (Random Forest) ──")
print(importance_df.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance for Dengue Prediction (Random Forest)')
plt.tight_layout()
plt.savefig(f'{BASE}\\reports\\feature_importance.png')
plt.show()
print("\nFeature importance chart saved!")

# ══════════════════════════════════════════════════════════════
# STEP 4 — ACTUAL vs PREDICTED PLOT
# ══════════════════════════════════════════════════════════════

rf_preds = rf_model.predict(X_test)

plt.figure(figsize=(12, 5))
plt.plot(range(len(yd_test)), yd_test.values,
         label='Actual', color='red', linewidth=2)
plt.plot(range(len(yd_test)), rf_preds,
         label='Predicted', color='blue', linewidth=2, linestyle='--')
plt.title('Random Forest: Actual vs Predicted Dengue Cases')
plt.xlabel('Test Samples')
plt.ylabel('Dengue Cases')
plt.legend()
plt.tight_layout()
plt.savefig(f'{BASE}\\reports\\actual_vs_predicted.png')
plt.show()
print("Actual vs Predicted chart saved!")

# ══════════════════════════════════════════════════════════════
# STEP 5 — MODEL COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Model Comparison', fontsize=14)

sns.barplot(data=results_df, x='Model', y='R2',
            palette='Blues_d', ax=axes[0])
axes[0].set_title('R² Score (higher = better)')
axes[0].set_ylim(0, 1)

sns.barplot(data=results_df, x='Model', y='RMSE',
            palette='Reds_d', ax=axes[1])
axes[1].set_title('RMSE (lower = better)')

plt.tight_layout()
plt.savefig(f'{BASE}\\reports\\model_comparison.png')
plt.show()
print("Model comparison chart saved!")

# ══════════════════════════════════════════════════════════════
# STEP 6 — SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════

with open(f'{BASE}\\models\\random_forest_dengue.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\n✅ Best model saved to models/random_forest_dengue.pkl")
print("\n── Final Results Summary ──")
print(results_df.to_string(index=False)) 