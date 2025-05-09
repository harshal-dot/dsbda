import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Read CSV file
data = pd.read_csv("C:/Users/harsh/Downloads/DSBDAL/PRAC2/StudentPerformance.csv")
print(data)

# Checking for null values
print(data.isnull().count())
print(data.notnull().count())

# Label Encoding
le = LabelEncoder()
if 'placement_offer_count' in data.columns:
    data['placement_offer_count'] = le.fit_transform(data['placement_offer_count'])

# Handle missing value symbols
missing_values = ["Na", "na"]
df = pd.read_csv("C:/Users/harsh/Downloads/DSBDAL/PRAC2/StudentPerformance.csv", na_values=missing_values)
print(df)

# Filling all nulls with 0
df = df.fillna(0)
print(df)

# Filling math_score missing values using various strategies
if 'math_score' in df.columns:
    df['math_score'] = df['math_score'].fillna(df['math_score'].mean())
    df['math_score'] = df['math_score'].fillna(df['math_score'].median())
    df['math_score'] = df['math_score'].fillna(df['math_score'].mode()[0])
    df['math_score'] = df['math_score'].fillna(df['math_score'].min())
    df['math_score'] = df['math_score'].fillna(df['math_score'].max())

# Replace NaNs with -99 using np.nan (not np.NaN)
df = df.replace(to_replace=np.nan, value=-99)

# Deleting null values
df = pd.read_csv("C:/Users/harsh/Downloads/DSBDAL/PRAC2/StudentPerformance.csv")
df.dropna()
df.dropna(how='all')
df.dropna(axis=1)
df.dropna(axis=0, how='any')
print(df)

# Outlier detection using boxplot
df = pd.read_csv("C:/Users/harsh/Downloads/DSBDAL/PRAC2/StudentPerformance.csv")
col = ['math_score', 'reading_score', 'writing_score', 'placement_offer_count']
df.boxplot(column=col)

# Print specific outliers
if 'math_score' in df.columns:
    print(np.where(df['math_score'] > 90))
if 'reading_score' in df.columns:
    print(np.where(df['reading_score'] < 25))
if 'writing_score' in df.columns:
    print(np.where(df['writing_score'] < 30))

# Scatterplot for outliers
if 'placement_score' in df.columns and 'placement_offer_count' in df.columns:
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.scatter(df['placement_score'], df['placement_offer_count'])
    plt.show()

# Z-score outlier detection
if 'math_score' in df.columns:
    z = np.abs(stats.zscore(df['math_score']))
    print(z)
    threshold = 0.18
    sample_outliers = np.where(z < threshold)
    print(sample_outliers)

    df['math_score'].plot(kind='hist')
    
    # Prevent log(0) issue
    df['log_math'] = np.log10(df['math_score'].replace(0, np.nan).dropna())
    df['log_math'].plot(kind='hist')
