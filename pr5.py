import pandas as pd

# Loading the dataset
df = pd.read_csv("C:/Users/harsh/Downloads/DSBDAL/PRAC5/Social_Network_Ads.csv")
print(df)

# Preprocessing
# Checking for missing values
print(df.isnull().sum())

# Checking the data types of columns
print(df.dtypes)

# Encoding the 'Gender' column (Male -> 1, Female -> 0)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
print(df['Gender'].head())

# Train-test split
X = df.drop(['Purchased', 'User ID'], axis=1)
y = df['Purchased']

# Checking the features and target
print(X.head())
print(y.head())

# Splitting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scaling the features using StandardScaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Logistic Regression model training
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Model evaluation - Accuracy
accuracy = classifier.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Evaluating the performance using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
