import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("winequality.csv")
df = df.dropna(subset=["quality"])
X = df.drop(columns=["quality"])
y = df["quality"]

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, "wine_rf_model.joblib")
print("Model saved as wine_rf_model.joblib")
