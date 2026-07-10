import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("insurance.xlsx")

print(df.head())

#Q2
import pandas as pd

df = pd.read_excel("insurance.xlsx")

# Question 1
print("First 10 Rows")
print(df.head(10))

# Question 2
print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# Question 3

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nNumerical Columns:")
print(df.select_dtypes(include='number').columns)

print("\nCategorical Columns:")
print(df.select_dtypes(include='object').columns)


# Q5
print("\nAge Distribution")

sns.histplot(df["age"])
plt.title("Age Distribution")
plt.show()

# Q6
# Count Plot for Sex
sns.countplot(x="sex", data=df)
plt.title("Count of Sex")
plt.show()

# Count Plot for Smoker
sns.countplot(x="smoker", data=df)
plt.title("Count of Smoker")
plt.show()

# Count Plot for Region
sns.countplot(x="region", data=df)
plt.title("Count of Region")
plt.show()

#Q7
import matplotlib.pyplot as plt

print("BMI Distribution")

plt.figure(figsize=(8,5))
plt.hist(df["bmi"], bins=15)
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Count")
plt.show()

# Question 7 - Correlation Heatmap

correlation = df.select_dtypes(include=['int64', 'float64']).corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Question 8 - Count Plot for Smoker

plt.figure(figsize=(6,4))
sns.countplot(x="smoker", data=df)
plt.title("Count of Smokers")
plt.xlabel("Smoker")
plt.ylabel("Count")
plt.show()

# Question 8 - Count Plot for Smoker

plt.figure(figsize=(6,4))
sns.countplot(x="smoker", data=df)
plt.title("Count of Smokers")
plt.xlabel("Smoker")
plt.ylabel("Count")
plt.show()

# ===============================
# Question 10 - Analysis Summary
# ===============================

print("\nAverage Age:", df["age"].mean())
print("Average BMI:", df["bmi"].mean())

print("\nAverage Insurance Charges:", df["expenses"].mean())
print("Maximum Insurance Charges:", df["expenses"].max())
print("Minimum Insurance Charges:", df["expenses"].min())

print("\nAverage Charges by Smoker:")
print(df.groupby("smoker")["expenses"].mean())

print("\nCustomers by Region:")
print(df["region"].value_counts())
