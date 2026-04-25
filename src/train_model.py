import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# ── Load merged data ───────────────────────────────────────────
df = pd.read_csv('../data/processed/merged_data.csv')

# ── Features & Target ──────────────────────────────────────────
features = ['temperature', 'rainfall', 'humidity', 'month']
X = df[features]
y_dengue  = df['dengue_cases']
y_malaria = df['malaria_cases']

# ── Train/Test Split ───────────────────────────────────────────
X_train, X_test, yd_train, yd_test = train_test_split(X, y_dengue,  test_size=0.2, random_state=42)
_,       _,      ym_train, ym_test = train_test_split(X, y_malaria, test_size=0.2, random_state=42)

# ── Train Random Forest ────────────────────────────────────────
print("Training Dengue model...")
dengue_model = RandomForestRegressor(n_estimators=100, random_state=42)
dengue_model.fit(X_train, yd_train)

print("Training Malaria model...")
malaria_model = RandomForestRegressor(n_estimators=100, random_state=42)
malaria_model.fit(X_train, ym_train)

# ── Evaluate ───────────────────────────────────────────────────
yd_pred = dengue_model.predict(X_test)
ym_pred = malaria_model.predict(X_test)

print("\n── DENGUE MODEL ──────────────────────────")
print(f"R² Score : {r2_score(yd_test, yd_pred):.4f}")
print(f"MAE      : {mean_absolute_error(yd_test, yd_pred):.2f} cases")

print("\n── MALARIA MODEL ─────────────────────────")
print(f"R² Score : {r2_score(ym_test, ym_pred):.4f}")
print(f"MAE      : {mean_absolute_error(ym_test, ym_pred):.2f} cases")

# ── Feature Importance ─────────────────────────────────────────
print("\n── FEATURE IMPORTANCE (Dengue) ───────────")
for feat, imp in zip(features, dengue_model.feature_importances_):
    print(f"  {feat:15s}: {imp:.4f}")

# ── Plot Predicted vs Actual ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(yd_test, yd_pred, color='orange', alpha=0.7)
axes[0].plot([yd_test.min(), yd_test.max()],
             [yd_test.min(), yd_test.max()], 'r--')
axes[0].set_xlabel('Actual Cases')
axes[0].set_ylabel('Predicted Cases')
axes[0].set_title('Dengue: Predicted vs Actual')

axes[1].scatter(ym_test, ym_pred, color='purple', alpha=0.7)
axes[1].plot([ym_test.min(), ym_test.max()],
             [ym_test.min(), ym_test.max()], 'r--')
axes[1].set_xlabel('Actual Cases')
axes[1].set_ylabel('Predicted Cases')
axes[1].set_title('Malaria: Predicted vs Actual')

plt.tight_layout()
plt.savefig('../reports/model_performance.png')
plt.show()

# ── Save models ────────────────────────────────────────────────
pickle.dump(dengue_model,  open('../models/dengue_model.pkl',  'wb'))
pickle.dump(malaria_model, open('../models/malaria_model.pkl', 'wb'))
print("\nModels saved to models/ folder!")