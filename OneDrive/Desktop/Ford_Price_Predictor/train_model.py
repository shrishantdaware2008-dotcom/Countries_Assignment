import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("ford_car_dataset.csv")

# One-hot encoding
df = pd.get_dummies(df, columns=["model", "transmission", "fuelType"], drop_first=True)

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Save column names
joblib.dump(X.columns.tolist(), "columns.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

joblib.dump(scaler, "scaler.pkl")
# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "LR_model.pkl")

print("Model saved successfully!")