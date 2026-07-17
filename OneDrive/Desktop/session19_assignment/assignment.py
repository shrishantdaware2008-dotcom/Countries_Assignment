#Q2
import pandas as pd

df = pd.read_csv("House Price Prediction Dataset.csv")

print(df.head(10))

#Q3
# Missing values
print(df.isnull().sum())

# Missing value percentage
print((df.isnull().sum() / len(df)) * 100)

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Check duplicates
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

#Q4
print(df.describe())

# Replace 'Price' with your target column name if different
print("Minimum:", df["Price"].min())
print("Maximum:", df["Price"].max())
print("Mean:", df["Price"].mean())
print("Median:", df["Price"].median())
#Q5
import matplotlib.pyplot as plt

df.hist(figsize=(12, 10))
plt.show()
#Q6
import seaborn as sns
import matplotlib.pyplot as plt

categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col])
    plt.xticks(rotation=45)
    plt.show()

    #Q7

    import matplotlib.pyplot as plt
import seaborn as sns

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()
#Q8
# Independent Features (X)
X = df.drop("Price", axis=1)

# Dependent Feature (y)
y = df["Price"]

print("Independent Features (X):")
print(X.head())

print("\nDependent Feature (y):")
print(y.head())

#Q9
# Display categorical columns before encoding
print("Categorical Columns:")
print(df.select_dtypes(include=["object"]).head())

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Display dataset after encoding
print("\nDataset After One-Hot Encoding:")
print(df_encoded.head())

#Q10
from sklearn.preprocessing import StandardScaler

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Independent Features (X)
X = df_encoded.drop("Price", axis=1)

# Dependent Feature (y)
y = df_encoded["Price"]

# Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data into DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Display first 5 rows
print("First 5 Rows of Scaled Features:")
print(X_scaled.head())