import pandas as pd
import numpy as np

print("Generating synthetic disease data...")

np.random.seed(42)

years  = range(2010, 2024)
months = range(1, 13)

rows = []
for year in years:
    for month in months:

        # Dengue peaks in post-monsoon (Aug-Nov)
        dengue_season   = 1 if month in [8, 9, 10, 11] else 0

        # Malaria peaks in monsoon (Jun-Sep)
        malaria_season  = 1 if month in [6, 7, 8, 9] else 0

        # Cholera peaks in summer (Mar-Jun)
        cholera_season  = 1 if month in [3, 4, 5, 6] else 0

        # Gradually increasing trend over years (climate effect)
        year_factor = (year - 2010) * 0.05

        dengue_cases  = int(np.random.poisson(
            max(10, 200 * dengue_season  + 20 + year_factor * 30)))

        malaria_cases = int(np.random.poisson(
            max(10, 150 * malaria_season + 15 + year_factor * 20)))

        cholera_cases = int(np.random.poisson(
            max(5,  80  * cholera_season + 10 + year_factor * 10)))

        rows.append({
            'year'         : year,
            'month'        : month,
            'dengue_cases' : dengue_cases,
            'malaria_cases': malaria_cases,
            'cholera_cases': cholera_cases,
            'region'       : 'Bangalore'
        })

disease_df = pd.DataFrame(rows)
disease_df.to_csv('disease_data.csv', index=False)

print(f"Disease data saved! {len(disease_df)} rows generated.")
print(disease_df.head(12))