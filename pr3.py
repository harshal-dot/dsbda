import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/harsh/Downloads/DSBDAL/PRAC3/evSales.csv")

# Display the DataFrame
print(df)

# MEAN

# Mean of full Battery_Capacity_kWh column
print("Mean (full):", df['Battery_Capacity_kWh'].mean())

# Alternate syntax using loc
print("Mean using loc:", df.loc[:, 'Battery_Capacity_kWh'].mean())

# Mean of first 4 entries
print("Mean (first 4 rows):", df['Battery_Capacity_kWh'][0:4].mean())

# MEDIAN

print("Median (full):", df['Battery_Capacity_kWh'].median())
print("Median using loc:", df.loc[:, 'Battery_Capacity_kWh'].median())
print("Median (first 4 rows):", df['Battery_Capacity_kWh'][0:4].median())

# MODE

# Full mode of DataFrame (may return multiple columns if multiple modes exist)
print("Full DataFrame mode:\n", df.mode())

# Mode of specific column
print("Mode of Battery_Capacity_kWh:", df['Battery_Capacity_kWh'].mode())

# MINIMUM


print("Min Battery_Capacity_kWh:", df['Battery_Capacity_kWh'].min(skipna=True))
print("Min using loc:", df.loc[:, 'Battery_Capacity_kWh'].min(skipna=True))

# MAXIMUM


print("Max of entire DataFrame:\n", df.max(numeric_only=True))  # avoid issues with non-numeric


print("Max Battery_Capacity_kWh:", df.loc[:, 'Battery_Capacity_kWh'].max(skipna=True))

# STANDARD DEVIATION

print("Standard Deviation:", df['Battery_Capacity_kWh'].std())
print("Standard Deviation using loc:", df.loc[:, 'Battery_Capacity_kWh'].std())
