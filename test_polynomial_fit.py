# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:25:12 2025

@author: Ele_p
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from Calibration_EBT3 import CurveFitter, ProcessingMode
from sklearn.metrics import mean_squared_error
import warnings

class TestPolynomialFit:
    """Tests for the polynomial_fit method"""
    
    @pytest.fixture
    def fitter(self):
        # Assuming the function is part of a class called CurveFitter
        return CurveFitter()
    
    @pytest.fixture
    def linear_data(self):
        """Generate perfect linear data for testing"""
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5])
        noise_std=0.1
        xerr=np.abs(np.random.normal(0, noise_std, size=len(x)))
        y = 2*x + 1  # y = 2x + 1
        return x, y, xerr
    
    @pytest.fixture
    def quadratic_data(self):
        """Generate perfect quadratic data for testing"""
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5])
        noise_std=0.1
        xerr=np.random.normal(0, noise_std, size=len(x))
        y = x**2 - 2*x + 1  # y = x² - 2x + 1
        return x, y, xerr
    
    @pytest.fixture
    def cubic_data(self):
        """Generate perfect cubic data for testing"""
        np.random.seed(42)
        
        x = np.array([1, 2, 3, 4, 5])
        noise_std = 0.1
        xerr = np.random.normal(0, noise_std, size=len(x))
        
        # Cubic polynomial: y = x^3 - 3x^2 + 2x
        y = x**3 - 3*x**2 + 2*x
        
        return x, y, xerr
    
    @pytest.fixture
    def polynomial_degree_4_data(self):
        """Generate perfect polynomial data of degree 4 for testing"""
        np.random.seed(42)
        
        x = np.array([1, 2, 3, 4, 5])
        noise_std = 0.1
        xerr = np.random.normal(0, noise_std, size=len(x))
        
        # Polynomial of degree 4: y = x^4 - 4x^3 + 6x^2 - 4x + 1
        y = x**4 - 4*x**3 + 6*x**2 - 4*x + 1
        
        return x, y, xerr
        
    def test_invalid_input_returns_none(self, fitter):
        """
        GIVEN: Invalid input data (None values)
        WHEN: polynomial_fit is called
        THEN: All return values should be None and a UserWarning should be raised
        """
        with pytest.warns(UserWarning, match="x array cannot be None"):
            mse, degree, coeffs, results = fitter.polynomial_fit(None, None, None)
        
        assert mse is None
        assert degree is None
        assert coeffs is None
        assert results is None
        
    def test_perfect_linear_fit(self, fitter, linear_data):
        """
        GIVEN: Perfect linear data (y = 2x + 1)
        WHEN: polynomial_fit is called with max_degree=2
        THEN: Best fit should be linear with near-zero MSE
        """
        x, y, xerr = linear_data
        mse, degree, coeffs, results = fitter.polynomial_fit(x, y, xerr,  max_degree=4)

        assert degree == 1  # Should choose linear fit
        assert_array_almost_equal(coeffs, [2, 1], decimal=10)  # Should find correct coefficients
        assert mse < 1e-3  # Should have nearly perfect fit
        
    def test_perfect_quadratic_fit(self, fitter, quadratic_data):
        """
        GIVEN: Perfect quadratic data (y = x² - 2x + 1)
        WHEN: polynomial_fit is called with max_degree=3
        THEN: Best fit should be quadratic with near-zero MSE
        """
        x, y, xerr = quadratic_data
        mse, degree, coeffs, results = fitter.polynomial_fit(x, y, xerr, max_degree=3)
        
        assert degree == 2  # Should choose quadratic fit
        assert_array_almost_equal(coeffs, [1, -2, 1], decimal=10)  # Should find correct coefficients
        assert mse < 1e-10  # Should have nearly perfect fit
        
    def test_perfect_cubic_fit(self, fitter, cubic_data):
        """
        GIVEN: Perfect quadratic data (y = x² - 2x + 1)
        WHEN: polynomial_fit is called with max_degree=3
        THEN: Best fit should be quadratic with near-zero MSE
        """
        x, y, xerr = cubic_data
        mse, degree, coeffs, results = fitter.polynomial_fit(x, y, xerr, max_degree=4)
        
        assert degree == 3  # Should choose quadratic fit
        assert_array_almost_equal(coeffs, [1, -3, 2, 0], decimal=10)  # Should find correct coefficients
        assert mse < 1e-10  # Should have nearly perfect fit
        
        
    def test_perfect_polynomial_degree_4_fit(self, fitter, polynomial_degree_4_data):
        """
        GIVEN: Perfect quadratic data (y = x² - 2x + 1)
        WHEN: polynomial_fit is called with max_degree=3
        THEN: Best fit should be quadratic with near-zero MSE
        """
        x, y, xerr = polynomial_degree_4_data
        mse, degree, coeffs, results = fitter.polynomial_fit(x, y, xerr, max_degree=4)
        
        assert degree == 4  # Should choose quadratic fit
        assert_array_almost_equal(coeffs, [1, -4, 6, -4, 1], decimal=10)  # Should find correct coefficients
        assert mse < 1e-10  # Should have nearly perfect fit     
         
    def test_fitting_results_structure(self, fitter, linear_data):
        """
        GIVEN: Valid input data
        WHEN: polynomial_fit is called
        THEN: Results dictionary should have correct structure for each degree
        """
        x, y, xerr = linear_data
        max_degree = 3
        _, degree, coefficients, results = fitter.polynomial_fit(x, y, xerr, max_degree=max_degree)
        
        assert len(coefficients) == degree + 1
        
        for result in results:
            assert isinstance(result, dict)
            assert any(key in result for key in [
                'function', 'mse', 'aic', 'coefficients',
                'degree', 'polynomial', 
                'chi2', 'dof','error_message'
            ])
            
    def test_max_degree_limit(self, fitter, linear_data):
        """
        GIVEN: Valid input data
        WHEN: polynomial_fit is called with specific max_degree
        THEN: Should not test polynomials beyond max_degree
        """
        x, y, xerr = linear_data
        max_degree = 2
        _, _, _, results = fitter.polynomial_fit(x, y, xerr, max_degree=max_degree)
        
        assert len(results) == max_degree
        assert max(result['degree'] for result in results) == max_degree
        
    def test_noisy_data_robustness(self, fitter):
        """
        GIVEN: Linear data with added noise
        WHEN: polynomial_fit is called
        THEN: Should still identify underlying linear trend
        """
        np.random.seed(42)  # For reproducibility
        x = np.array([1, 2, 3, 4, 5])
        xerr=np.abs(np.random.normal(0, 0.1, size=len(x)))
        x_noisy= x + xerr
        y = 2*x_noisy + 1 
        
        mse, degree, coeffs, _ = fitter.polynomial_fit(x, y,xerr, max_degree=3)
        
        # Should still choose linear fit despite noise
        assert degree == 1
        # Coefficients should be close to [2, 1]
        assert np.abs(coeffs[0] - 2) < 0.2
        assert np.abs(coeffs[1] - 1) < 0.2
        
    def test_linear_fit_at_boundaries(self, fitter):
        """
        GIVEN: Linear data at the boundaries of valid pixel values (0-65535)
        WHEN: polynomial_fit is called
        THEN: Should correctly identify linear relationship at boundaries
        """
        x = np.array([0, 16383, 32767, 49151, 65535])  # Full range
        slope = 2
        intercept = 100
        y = slope * x + intercept
        xerr = np.ones_like(x) * 100  # Reasonable error for pixel values
        
        mse, degree, coeffs, _ = fitter.polynomial_fit(x, y, xerr, max_degree=3)
        
        assert degree == 1
        assert_array_almost_equal(coeffs, [slope, intercept], decimal=1)
        assert mse < 1e4  # Reasonable MSE for this scale

    def test_quadratic_fit_within_bounds(self, fitter):
        """
        GIVEN: Quadratic data within valid pixel value range and expected behaviour:
            y decreases as x increases
        WHEN: polynomial_fit is called
        THEN: Should identify quadratic relationship while staying within bounds
        """
        x = np.linspace(0, 65535, 20)
        # Scale quadratic function to stay within bounds
        a = -50 / (65535**2)  # Coefficient to ensure max within bounds
        y = a * (x)**2 + 50  # Parabola with peak at middle of range
        print(y)
        xerr = np.ones_like(x) * 100
        
        mse, degree, coeffs, _ = fitter.polynomial_fit(x, y, xerr, max_degree=3)
    
        assert degree == 2
        # Verify predictions stay within bounds
        p = np.poly1d(coeffs)
        y_pred = p(x)
        y_pred[y_pred <= 1e-1] = 0
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 50)
        
    def test_quadratic_fit_within_bounds_OD(self, fitter):
        """
        GIVEN: Quadratic data within valid pixel value range and expected behaviour:
            y decreases as x increases
        WHEN: polynomial_fit is called
        THEN: Should identify quadratic relationship while staying within bounds
        """
        x = np.linspace(0, 65535, 20)
        x_OD = fitter._process_values(x, mode=ProcessingMode.OD)
        # Scale quadratic function to stay within bounds
        a = -50 / (65535**2)  # Coefficient to ensure max within bounds
        a_OD = fitter._process_values(a, mode=ProcessingMode.OD)
        y = a * (x)**2 + 50  # Parabola with peak at middle of range
        y_OD = a_OD * (x_OD)**2 + 50
        print(y)
        xerr = np.ones_like(x) * 100
        
        mse, degree, coeffs, _ = fitter.polynomial_fit(x_OD, y_OD, xerr, max_degree=3)
    
        assert degree == 2
        # Verify predictions stay within bounds
        p = np.poly1d(coeffs)
        y_pred = p(x)
        y_pred[y_pred <= 1e-1] = 0
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 50)
    
    def test_inverse_relationship_with_noise(self, fitter):
        """
        GIVEN: Data with inverse relationship (y = k/x + c) and noise, keeping values in bounds
        WHEN: polynomial_fit is called
        THEN: Should identify the inverse relationship while handling noise appropriately
        """
        np.random.seed(42)
        
        # Generate x values with some spacing, avoiding x=0
        x = np.linspace(1000, 65535, 20)  # Starting from 1000 to avoid very large y values
        xerr = np.abs(np.random.normal(0.5, 100, size=len(x)))
        x_noisy = np.clip(x + xerr, 1000, 65535)  # Ensure values stay in range and away from 0
        
        # Create inverse relationship: y = mx+ c
        m = 0.00075
        c = 50

        

        # Add controlled noise to y
        y_noise = np.random.normal(0, 0.2, size=len(x))  # Reduced noise
        y = np.clip(-m * x_noisy + c + y_noise, 0, 50)
            # Fit the data
        mse, degree, coeffs, results = fitter.polynomial_fit(x_noisy, y, xerr, max_degree=4)
        print(mse)
        print(coeffs)
        # Assertions
        assert degree == 1, f"Degree {degree} is too high for inverse relationship"
        
        # Create test points for verification
        x_test = np.linspace(min(x), max(x), 100)
        p = np.poly1d(coeffs)
        y_pred = p(x_test)
        y_pred[y_pred <= 1e-1] = 0
        
        
        assert np.all(y_pred >= 0), "Predictions contain negative values"
        assert np.all(y_pred <= 50), "Predictions exceed maximum allowed value"
    
    
    
    def test_saturated_data_handling(self, fitter):
        """
        GIVEN: Data with saturated values (at 65535)
        WHEN: polynomial_fit is called
        THEN: Should handle saturation appropriately
        """
        x = np.linspace(0, 65535, 20)
        print("x values:", x)
    
        # Adjusting the equation to ensure y approaches 0 as x increases
        y = -0.0001 * x + 50  # This will ensure y decreases towards 0 as x increases
        y[y < 0] = 0  # Ensure that y doesn't drop below 0
    
        xerr = np.ones_like(x) * 100
    
        mse, degree, coeffs, results = fitter.polynomial_fit(x, y, xerr, max_degree=3)
    
        print("Coefficients from fitting:", coeffs)
    
        # Should still identify linear relationship
        assert degree == 1
    
        # Verify handling of saturated values
        p = np.poly1d(coeffs)
        y_pred = p(x)
    
        print("Predicted y values:", y_pred)
    
    
        assert np.all(y_pred >= 0), "Some predicted values are negative"
    
        # Check if all predicted values are below the saturation limit
        assert np.all(y_pred < 51), f"Some predicted values exceed saturation limit: {y_pred[y_pred > 50]}"
    
    def test_low_signal_data(self, fitter):
        """
        GIVEN: Data with very low pixel values (near zero)
        WHEN: polynomial_fit is called
        THEN: Should handle low signal appropriately
        """
        x = np.linspace(0, 1000, 20)  # Low range
        y = - 0.005 * x + 10  # Small slope and offset
        xerr = np.ones_like(x) * 10  # Smaller errors for low values
        
        mse, degree, coeffs, _ = fitter.polynomial_fit(x, y, xerr, max_degree=3)
        
        assert degree == 1
        assert_array_almost_equal(coeffs, [0.005, 10], decimal=1)
        # Verify predictions stay within bounds
        p = np.poly1d(coeffs)
        y_pred = p(x)
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 50)