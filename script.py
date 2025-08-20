import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Automatically find CSV in Downloads
downloads = r"C:\Users\hp\Downloads"
csv_files = [f for f in os.listdir(downloads) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV files found in Downloads folder.")
file_path = os.path.join(downloads, csv_files[0])

# Load CSV
df = pd.read_csv(file_path)
df.fillna(0, inplace=True)

print(df.head(5))

# Prepare features and target
X = df.drop(['Name', 'Surname', 'Result'], axis=1)
y = df['Result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred = dt_model.predict(X_test_scaled)

print("Baseline Model Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0)}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0)}")
print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature Importance:")
for i, feature in enumerate(X.columns):
    print(f"{feature}: {dt_model.feature_importances_[i]}")

# PCA for dimensionality reduction
pca = PCA()
pca.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance >= 0.95) + 1

print(f"\nPCA Components for 95% variance: {n_components}")

pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Decision Tree on PCA data
dt_pca = DecisionTreeClassifier(random_state=42)
dt_pca.fit(X_train_pca, y_train)
y_pred_pca = dt_pca.predict(X_test_pca)

print("\nPCA Model Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_pca)}")
print(f"Precision: {precision_score(y_test, y_pred_pca, zero_division=0)}")
print(f"Recall: {recall_score(y_test, y_pred_pca, zero_division=0)}")
print(f"F1 Score: {f1_score(y_test, y_pred_pca, zero_division=0)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_pca))

print("\nComparison:")
print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"PCA Accuracy: {accuracy_score(y_test, y_pred_pca)}")

