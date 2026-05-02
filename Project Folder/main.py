# USA HOUSE PRICE PREDICTION

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# 1. LOAD DATASET
df = pd.read_csv(r"D:\house_price_prediction_project\USA_Housing.csv")
df.columns = df.columns.str.strip()

# 2. FEATURES & TARGET

X = df[[
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population"
]]

y = df["Price"]

# 3. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. TRAIN MODEL

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# 5. SAVE MODEL

model_folder = r"D:\house_price_prediction_project\Model"
os.makedirs(model_folder, exist_ok=True)

joblib.dump(model, os.path.join(model_folder, "usa_house_model.pkl"))

# 6. USER INPUT (WITH CLEAR RANGES)

print("\nENTER HOUSE DETAILS (Follow ranges carefully)\n")

# 1. INCOME
print("1. Avg Area Income(20,000 to 120,000 USD)")
income = float(input("Enter Income: "))

# 2. HOUSE AGE
print("\n2. Avg Area House Age(1 to 50 years (e.g., 20 is valid))")
house_age = float(input("Enter House Age: "))

# 3. ROOMS
print("\n3. Avg Number of Rooms(1 to 10)")
rooms = float(input("Enter Rooms: "))

# 4. BEDROOMS
print("\n4. Avg Number of Bedrooms(1 to 6)")
bedrooms = float(input("Enter Bedrooms: "))

# 5. POPULATION
print("\n5. Area Population(1,000 to 20,000)")
population = float(input("Enter Population: "))


# 7. PREDICTION

input_data = np.array([[
    income,
    house_age,
    rooms,
    bedrooms,
    population
]])

price = model.predict(input_data)

print("\n================ RESULT ================")
print("Estimated House Price:", round(price[0], 2))
print("========================================")

# 8. MODEL EVALUATION

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nMODEL PERFORMANCE")
print("R2 Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)


# 9. VISUALIZATION

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, alpha=0.6)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.show()