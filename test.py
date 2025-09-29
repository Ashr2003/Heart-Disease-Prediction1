import pandas as pd
df = pd.read_csv('data/heart.csv')
print(df.shape)
print(df.columns.tolist())
df.head()

