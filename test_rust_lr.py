import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# Add the directory containing the compiled Rust module to Python's path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import rust_svm
except ImportError as e:
    print(f"Error importing rust_svm: {e}")
    print("Make sure you've built the Rust library using 'maturin develop'")
    sys.exit(1)


# Load the diabetes dataset from scikit-learn
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Flatten the data for Rust input format
X_train_flat = X_train.flatten().tolist()
X_test_flat = X_test.flatten().tolist()
y_train_flat = y_train.tolist()

try:
    # Call the Rust linear regression implementation
    predictions = rust_svm.linear_regression(X_train_flat, y_train_flat, X_test_flat)

    # Convert predictions back to numpy array for easier handling
    predictions = np.array(predictions)

    # Print predictions for the test set
    print(f"Predictions (first 5): {predictions[:5]}")
    print(f"Ground Truth (first 5): {y_test[:5]}")

    # Calculate and print metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

except Exception as e:
    print(f"Error during linear regression: {e}")