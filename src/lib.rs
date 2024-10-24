use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ndarray::{Array1, Array2};

struct LinearRegression {
    weights: Array1<f32>,
    bias: f32,
}

impl LinearRegression {
    fn new(n_features: usize) -> Self {
        LinearRegression {
            weights: Array1::zeros(n_features),
            bias: 0.0,
        }
    }

    fn fit(&mut self, x: &Array2<f32>, y: &Array1<f32>, learning_rate: f32, n_iterations: usize) {
        let n_samples = x.nrows();

        for _ in 0..n_iterations {
            let predictions = x.dot(&self.weights) + self.bias;
            let errors = &predictions - y;

            let gradient = x.t().dot(&errors) / n_samples as f32;
            self.weights -= &(gradient * learning_rate);
            self.bias -= errors.mean().unwrap_or(0.0) * learning_rate;
        }
    }

    fn predict(&self, x: &Array2<f32>) -> Array1<f32> {
        x.dot(&self.weights) + self.bias
    }
}

#[pyfunction]
fn linear_regression(features: Vec<f32>, labels: Vec<f32>, test_features: Vec<f32>) -> PyResult<Vec<f32>> {
    let n_samples = labels.len();
    let n_features = features.len() / n_samples;

    // Convert inputs to ndarray format
    let x = Array2::from_shape_vec((n_samples, n_features), features)
        .map_err(|e| PyValueError::new_err(format!("Invalid features shape: {}", e)))?;
    let y = Array1::from_vec(labels);

    // Create and train the linear regression model
    let mut model = LinearRegression::new(n_features);
    model.fit(&x, &y, 0.01, 1000);

    // Prepare test data
    let n_test_samples = test_features.len() / n_features;
    let x_test = Array2::from_shape_vec((n_test_samples, n_features), test_features)
        .map_err(|e| PyValueError::new_err(format!("Invalid test features shape: {}", e)))?;

    // Perform predictions
    let predictions = model.predict(&x_test);

    Ok(predictions.to_vec())
}

#[pymodule]
fn rust_lr(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(linear_regression, m)?)?;
    Ok(())
}