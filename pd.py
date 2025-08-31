import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)

print(df.head())
print("-------------------------------------------------")
print(df["Name"])
print("-------------------------------------------------")
print(df[df["Age"] >28])
print("-------------------------------------------------")
print(df.describe())
print("-------------------------------------------------")
print(df.groupby("Name")["Age"].mean())