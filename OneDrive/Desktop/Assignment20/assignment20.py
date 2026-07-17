import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("ford_car_dataset.csv")

# Q1
X = df.drop("price", axis=1)
Y = df["price"]

# Q2
X = pd.get_dummies(X)
X = X.astype(int)

# Save column names
columns = X.columns.tolist()

# Q3
scaler = StandardScaler()
numerical_columns = ["year", "mileage", "tax", "mpg", "engineSize"]
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Q4
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42
)

# Q5
model = LinearRegression()
model.fit(X_train, y_train)

# Q6
y_pred = model.predict(X_test)

print("First 10 Predicted Values:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test.iloc[:10].values)

# Q7
r2 = r2_score(y_test, y_pred)
print("\nR² Score:", r2)

# Q8 - Save Model
joblib.dump(model, "LR_ford_car.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(columns, "columns.pkl")

print("\nModel and preprocessing objects saved successfully.")