import pandas as pd

df = pd.read_excel("heart.xlsx")

print("First 10 Rows:")
print(df.head(10))

print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

#Q2
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

#Q3
print("\nDuplicate Rows:")
duplicates = df.duplicated().sum()
print("Number of duplicate rows:", duplicates)

df = df.drop_duplicates()

print("New Shape of Dataset:")
print(df.shape)

#Q4 
print("\nInvalid Values:")

cholesterol_zero = (df["Cholesterol"] == 0).sum()
restingbp_zero = (df["RestingBP"] == 0).sum()

print("Rows with Cholesterol = 0:", cholesterol_zero)
print("Rows with RestingBP = 0:", restingbp_zero)

#Q5
print("\nBefore Cleaning:")
print(df[["Cholesterol", "RestingBP"]].describe())

# Calculate mean excluding 0 values
chol_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
bp_mean = df.loc[df["RestingBP"] != 0, "RestingBP"].mean()

# Replace 0 with mean
df["Cholesterol"] = df["Cholesterol"].replace(0, chol_mean)
df["RestingBP"] = df["RestingBP"].replace(0, bp_mean)

# Round to 2 decimal places
df["Cholesterol"] = df["Cholesterol"].round(2)
df["RestingBP"] = df["RestingBP"].round(2)

print("\nAfter Cleaning:")
print(df[["Cholesterol", "RestingBP"]].describe())

#Q6
import matplotlib.pyplot as plt

def plot_histograms(data):
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.hist(data["Age"], bins=10)
    plt.title("Age")

    plt.subplot(2, 2, 2)
    plt.hist(data["RestingBP"], bins=10)
    plt.title("RestingBP")

    plt.subplot(2, 2, 3)
    plt.hist(data["Cholesterol"], bins=10)
    plt.title("Cholesterol")

    plt.subplot(2, 2, 4)
    plt.hist(data["MaxHR"], bins=10)
    plt.title("MaxHR")

    plt.tight_layout()
    plt.show()
plot_histograms(df)

#Q7
print("\nNumerical Columns:")
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print(numerical_columns)

print("\nCategorical Columns:")
categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)

#Q8
print("\nOne-Hot Encoding:")

df_encoded = pd.get_dummies(df)

print("Shape of Encoded Dataset:")
print(df_encoded.shape)

print("\nFirst 5 Rows:")
print(df_encoded.head())

# Q9
print("\nFinal Shape of Encoded Dataset:")
print(df_encoded.shape)

print("\nColumn Names:")
print(df_encoded.columns)