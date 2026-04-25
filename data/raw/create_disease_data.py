import pandas as pd
import numpy as np

print("Creating disease dataset...")

np.random.seed(42)

# Load the climate data we just downloaded
climate_df = pd.read_csv('climate_data.csv')

rows = []
for _, row in climate_df.iterrows():
    temp     = row['temperature']
    rainfall = row['rainfall']
    humidity = row['humidity']

    # Dengue spikes when temp is high + rainfall is high (mosquito breeding)
    dengue_base = (
        max(0, (temp - 20) * 3) +
        max(0, (rainfall - 1) * 5) +
        max(0, (humidity - 60) * 1.5)
    )
    dengue_cases = int(max(0, dengue_base * 10 + np.random.normal(0, 20)))

    # Malaria correlates more with rainfall (stagnant water)
    malaria_base = (
        max(0, (rainfall - 0.5) * 8) +
        max(0, (humidity - 55) * 1.2)
    )
    malaria_cases = int(max(0, malaria_base * 8 + np.random.normal(0, 15)))

    rows.append({
        'year'         : int(row['year']),
        'month'        : int(row['month']),
        'dengue_cases' : dengue_cases,
        'malaria_cases': malaria_cases
    })

disease_df = pd.DataFrame(rows)
disease_df.to_csv('disease_data.csv', index=False)

print(f"Disease data saved! {len(disease_df)} rows created.")
print(disease_df.head(10))
print("\nSummary:")
print(disease_df[['dengue_cases','malaria_cases']].describe())