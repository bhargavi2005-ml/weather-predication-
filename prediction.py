import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Dataset
data = {
 'Humidity': [78, 80, 75, 72, 85, 90],
 'WindSpeed': [10, 12, 8, 15, 9, 11],
 'TodayTemp': [30, 31, 29, 32, 28, 27],
 'TomorrowTemp': [31, 32, 30, 33, 29, 28]
}
df = pd.DataFrame(data)

# Model
model = RandomForestRegressor()
model.fit(df[['Humidity','WindSpeed','TodayTemp']], df['TomorrowTemp'])

# Prediction
result = model.predict([[78, 10, 30]])
print(f"Tomorrow Temperature: {result[0]:.1f} C")
print("Done by Bhargavi")
