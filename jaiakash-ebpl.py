# 🚦 AI-Driven Traffic Accident Prediction - Sample Code (No CSV Upload Required)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Download sample dataset (road safety-related)
url = "https://raw.githubusercontent.com/selva86/datasets/master/Carseats.csv"
df = pd.read_csv(url)

# Step 2: Show data sample
print("Sample Data:")
print(df.head())

# Step 3: Convert 'High' sales into binary target (1: High Sales, 0: Low Sales)
df['High'] = df['Sales'].apply(lambda x: 1 if x > 8 else 0)

# Step 4: Drop 'Sales' and prepare features
X = df.drop(['Sales', 'High'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # encode categorical variables
y = df['High']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Predict & Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Visualize feature importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()
