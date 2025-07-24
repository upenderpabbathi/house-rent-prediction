## House Rent Prediction System using Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('Dataset/House_Rent_Dataset.csv')

# Basic info
print(df.shape)
print(df.info())
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Encode categorical features
object_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in object_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Define features and label
X = df.drop(['Rent'], axis=1)
y = df['Rent']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Initialize evaluation lists
mae_list, mse_list, rmse_list, r2_list = [], [], [], []

def calculateMetrics(algorithm, predict, testY):
    mae = mean_absolute_error(testY, predict)
    mse = mean_squared_error(testY, predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(testY, predict)

    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

    print(f"\n[{algorithm}]")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²   : {r2:.2f}")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=testY, y=predict, alpha=0.6)
    plt.plot([min(testY), max(testY)], [min(testY), max(testY)], 'r--')
    plt.xlabel('Actual Rent')
    plt.ylabel('Predicted Rent')
    plt.title(f"{algorithm} - Actual vs Predicted")
    plt.grid(True)
    plt.show()

# Make sure model directory exists
os.makedirs("model", exist_ok=True)

# ---- Linear Regression ----
linear_model_path = 'model/Linear.pkl'
if os.path.exists(linear_model_path):
    Linear = joblib.load(linear_model_path)
    print("Linear Regression model loaded.")
else:
    Linear = LinearRegression()
    Linear.fit(X_train, y_train)
    joblib.dump(Linear, linear_model_path)
    print("Linear Regression model trained and saved.")

linear_pred = Linear.predict(X_test)
calculateMetrics("Linear Regression", linear_pred, y_test)

# ---- Decision Tree Regressor ----
tree_model_path = 'model/DecisionTreeRegressor.pkl'
if os.path.exists(tree_model_path):
    tree_model = joblib.load(tree_model_path)
    print("Decision Tree Regressor loaded.")
else:
    tree_model = DecisionTreeRegressor(random_state=42)
    tree_model.fit(X_train, y_train)
    joblib.dump(tree_model, tree_model_path)
    print("Decision Tree Regressor trained and saved.")

tree_pred = tree_model.predict(X_test)
calculateMetrics("Decision Tree Regressor", tree_pred, y_test)

# ---- Predict on New Test Data ----
if os.path.exists('testdata.csv'):
    testdata = pd.read_csv('testdata.csv')

    # Encode object columns to match training data
    for col in object_cols:
        if col in testdata.columns:
            testdata[col] = le.fit_transform(testdata[col].astype(str))

    pred = tree_model.predict(testdata)
    testdata['Predicted Rent'] = pred
    print("\nPredicted Test Data:")
    print(testdata)
else:
    print("No 'testdata.csv' found for prediction.")
