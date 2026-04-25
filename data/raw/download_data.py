import requests
import pandas as pd

print("Downloading climate data from NASA POWER...")

url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

params = {
    "parameters": "T2M,PRECTOTCORR,RH2M",  # Temperature, Rainfall, Humidity
    "community": "AG",
    "longitude": "77.5946",
    "latitude": "12.9716",  # Bangalore coordinates
    "start": "2010",
    "end": "2023",
    "format": "JSON"
}

response = requests.get(url, params=params)
data = response.json()

temp     = data['properties']['parameter']['T2M']
rainfall = data['properties']['parameter']['PRECTOTCORR']
humidity = data['properties']['parameter']['RH2M']

rows = []
for key in temp:
    if key.endswith('13'):
        continue
    year  = int(key[:4])
    month = int(key[4:])
    rows.append({
        'year'        : year,
        'month'       : month,
        'temperature' : temp[key],
        'rainfall'    : rainfall[key],
        'humidity'    : humidity[key]
    })

climate_df = pd.DataFrame(rows).sort_values(['year','month']).reset_index(drop=True)
climate_df.to_csv('climate_data.csv', index=False)

print(f"Climate data saved! {len(climate_df)} rows downloaded.")
print(climate_df.head())