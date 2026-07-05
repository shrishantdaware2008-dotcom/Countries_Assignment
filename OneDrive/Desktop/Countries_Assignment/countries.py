import pandas as pd

df = pd.read_csv("countries.csv")

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())
print(df.nunique())
print(df.sample(5))