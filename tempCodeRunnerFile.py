import rust_svm
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset from scikit-learn
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Flatten the data for Rust input format
X_train_flat = X_train.flatten().tolist()
X_test_flat = X_test.flatten().tolist()
y_train_flat = y_train.tolist()

# Call the Rust SVM implementation (from the compiled library)
predictions = rust_svm.svm_predict(X_train_flat, y_train_flat, X_test_flat)

# Convert predictions back to numpy array for easier handling
predictions = np.array(predictions)

# Print predictions for the test set
print(f"Predictions: {predictions}")

# Compare with the ground truth
print(f"Ground Truth: {y_test}")
