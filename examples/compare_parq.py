import pandas as pd

df1 = pd.read_parquet("cache_table.parquet")
df2 = pd.read_parquet("new_table.parquet")

df1_sorted = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
df2_sorted = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)

if df1_sorted.equals(df2_sorted):
    print("The two files have the same rows with respect to the last 3 columns.")
else:
    print("The two files have differences in the last 3 columns.")

differences = pd.concat([df1_sorted, df2_sorted]).drop_duplicates(keep=False)
print("Differences:")
print(differences)

