# Rust_Enhanced_Linear_Regression
Rust-Enhanced Linear Regression is a project that integrates a Rust-implemented linear regression model with Python using PyO3. This project demonstrates how to leverage Rust's performance and safety features in a Python environment for machine learning tasks.
# Project Name: Rust-Enhanced LR

## Installation

### Prerequisites
- Rust (latest stable version)
- Python (3.6 or higher)
- [Maturin](https://github.com/PyO3/maturin) for building the Rust extension

### Steps

1. **Build the Rust library:**
    ```sh
    maturin develop
    ```

2. **Install Python dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Example

1. **Run the Python script:**
    ```sh
    python test_rust_lr.py
    ```

### Example Output
The script will load the diabetes dataset, normalize the features, split the data into training and testing sets, and then use the Rust-implemented SVM model to make predictions. The script will print the predictions and the ground truth for the test set, along with the Mean Squared Error and R-squared Score.

## Results Comparison
## This Rust Enhanced Linear Regression giving results
```sh
Predictions (first 5): [140.5868988  181.7411499  140.36080933 294.54711914 120.96458435]
Ground Truth (first 5): [219.  70. 202. 230. 111.]
Mean Squared Error: 2885.15
R-squared Score: 0.46
```

## The traditional python implementation of linear regression giving results
```sh
Predictions (first 5): [139.5475584  179.51720835 134.03875572 291.41702925 123.78965872]
Ground Truth (first 5): [219.  70. 202. 230. 111.]
Mean Squared Error: 2900.19
Root Mean Squared Error: 53.85
```

## Performance Comparison

| Metric                     | Rust-Enhanced           | Python (scikit-learn)    |
|----------------------------|-------------------------|--------------------------|
| **Predictions (1st 5)**     | [140.59, 181.74, 140.36, 294.55, 120.96] | [139.55, 179.52, 134.04, 291.42, 123.79] |
| **Mean Squared Error (MSE)**| 2885.15                 | 2900.19                  |
| **R-squared Score**         | 0.46                    | N/A                      |
| **Root Mean Squared Error** | N/A                     | 53.85                    |

## Conclusion

- The **Rust-enhanced linear regression** demonstrated slightly better performance, with a lower Mean Squared Error (MSE) of `2885.15` compared to the Python model's `2900.19`.
- The **R-squared score** for the Rust model was `0.46`, indicating moderate predictive accuracy.
- The **Python implementation** produced comparable results with only a marginally higher MSE and a root mean squared error of `53.85`.

The Rust-enhanced implementation is a viable alternative to Python, potentially offering better scalability and performance on larger datasets due to Rust's memory safety and optimization benefits.
```
