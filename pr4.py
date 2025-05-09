from sklearn.datasets import fetch_california_housing
import pandas as pd

california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data)
df.columns = california_housing.feature_names
df['Price'] = california_housing.target

print(df.head())

print("\nMissing values in the dataset:")
print(df.isnull().sum())

X = df.drop(['Price'], axis=1)
y = df['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train, y_train)

y_test_pred = lm.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_test_pred)
print("\nMean Squared Error (MSE):", mse)

r2 = r2_score(y_test, y_test_pred)
print("\nR-squared:", r2)
