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
from enum import Enum
import logging
from hypotesis import settings, st, given, assume


class TestPolynomialBestFit:
    """Tests best fit for the polynomial_fit method"""
    
    @pytest.fixture
    def fitter(self):
        # Assuming the function is part of a class called CurveFitter
        return CurveFitter()
    
    @staticmethod
    def _get_best_fit(fitter, x, y, mode=ProcessingMode.PV):
        """
        Utility per ottenere il miglior fit.
        """
        fitting_results = fitter.polynomial_fit(x, y, mode=mode)
        return fitter.select_best_fit(fitting_results)
    
    
    @pytest.fixture
    def linear_data(self):
        """Generate perfect linear data for testing"""
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5])
        y = 2*x + 1  # y = 2x + 1
        return x, y
    
    @pytest.fixture
    def quadratic_data(self):
        """Generate perfect quadratic data for testing"""
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5])
        y = x**2 - 2*x + 1  # y = x² - 2x + 1
        return x, y
    
    @pytest.fixture
    def cubic_data(self):
        """Generate perfect cubic data for testing"""
        np.random.seed(42)
        
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                
        # Cubic polynomial: y = x^3 - 3x^2 + 2x
        y = x**3 - 3*x**2 + 2*x
        
        return x, y
    
    @pytest.fixture
    def polynomial_degree_4_data(self):
        """Generate perfect polynomial data of degree 4 for testing"""
        np.random.seed(42)
        
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
                
        # Polynomial of degree 4: y = x^4 - 4x^3 + 6x^2 - 4x + 1
        y = x**4 - 4*x**3 + 6*x**2 - 4*x + 1
        
        return x, y
    
        
    def test_perfect_linear_fit(self, fitter, linear_data):
        """
        Base case test for linear fitting.
        GIVEN: Perfect linear data (y = 2x + 1)
        WHEN: _get_best_fit is called
        THEN: Best fit should be linear with near-zero MSE
        """
        x, y = linear_data
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        assert degree == 1  # Should choose linear fit
        assert_array_almost_equal(coeffs, [2, 1], decimal=10)  # Should find correct coefficients
        assert score < 1e-3  # Should have nearly perfect fit
        
    def test_perfect_quadratic_fit(self, fitter, quadratic_data):
        """
        Base case test for quadratic fitting.
        GIVEN: Perfect quadratic data (y = x² - 2x + 1)
        WHEN: _get_best_fit is called with max_degree=3
        THEN: Best fit should be quadratic with near-zero MSE
        """
        x, y = quadratic_data
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        assert degree == 2  # Should choose quadratic fit
        assert_array_almost_equal(coeffs, [1, -2, 1], decimal=10)  # Should find correct coefficients
        assert score < 1e-10  # Should have nearly perfect fit
        
    def test_perfect_cubic_fit(self, fitter, cubic_data):
        """
        Base case test for cubic fitting.
        GIVEN: Perfect quadratic data (y = x² - 2x + 1)
        WHEN: _get_best_fit is called with max_degree=3
        THEN: Best fit should be quadratic with near-zero MSE
        """
        x, y= cubic_data
        with pytest.warns(UserWarning):
            best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
            degree = len(coeffs) - 1 
        
        assert degree == 3  # Should choose quadratic fit
        assert_array_almost_equal(coeffs, [1, -3, 2, 0], decimal=10)  # Should find correct coefficients
        assert score < 1e-10  # Should have nearly perfect fit
        
        
    def test_perfect_polynomial_degree_4_fit(self, fitter, polynomial_degree_4_data):
        """
        GIVEN: Perfect quadratic data (y = x² - 2x + 1)
        WHEN: _get_best_fit is called with max_degree=4
        THEN: Best fit should be quadratic with near-zero MSE
        """
        x, y = polynomial_degree_4_data
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        
        assert degree == 4  # Should choose quadratic fit
        assert_array_almost_equal(coeffs, [1, -4, 6, -4, 1], decimal=6)  # Should find correct coefficients
        assert score < 1e-10  # Should have nearly perfect fit     
              

        
    def test_noisy_data_robustness_linear_data(self, fitter):
        """
        GIVEN: Linear data with added noise
        WHEN: _get_best_fit is called
        THEN: Should still identify underlying linear trend
        """
        np.random.seed(42)  # For reproducibility
        x = np.array([1, 2, 3, 4, 5])
        xerr=np.abs(np.random.normal(0, 0.1, size=len(x)))
        x_noisy= x + xerr
        y = 2*x_noisy + 1 
        
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        
        # Should still choose linear fit despite noise
        assert degree == 1
        # Coefficients should be close to [2, 1]
        assert np.abs(coeffs[0] - 2) < 0.2
        assert np.abs(coeffs[1] - 1) < 0.2
        
    def test_noisy_data_robustness_quadratic_data(self, fitter):
        """
        GIVEN: quadratic data with added noise
        WHEN: _get_best_fit is called
        THEN: Should still identify underlying linear trend
        """
        np.random.seed(42)  # For reproducibility
        x = np.linspace(1000, 5000, 20)
        xerr=np.abs(np.random.normal(0, 0.5, size=len(x)))
        x_noisy= x + xerr
        y = 2*x_noisy**2 + 1*x_noisy
        
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        
        # Should still choose linear fit despite noise
        assert degree == 2
        # Coefficients should be close to [2, 1]
        assert np.abs(coeffs[0] - 2) < 0.5
        assert np.abs(coeffs[1] - 1) < 0.5
        
    def test_noisy_data_robustness_cubic_data(self, fitter):
        """
        GIVEN: quadratic data with added noise
        WHEN: _get_best_fit is called
        THEN: Should still identify underlying linear trend
        """
        np.random.seed(42)  # For reproducibility
        x = np.linspace(0, 10, 20)
        xerr=np.abs(np.random.normal(0, 0.01, size=len(x)))
        x_noisy= x + xerr
        y = -(1/200)*x_noisy**3 + (3/20)*x_noisy**2 - 2 *x_noisy + 50
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        p = np.poly1d(coeffs)
        y_pred = p(x)
        # Should still choose cubic fit despite noise
        assert degree == 3
        assert np.all(np.abs(y_pred - y) <= 1e-1), "y_pred and y are not equal"
    
    def test_noisy_data_robustness_quartic_data(self, fitter):
        """
        GIVEN: quadratic data with added noise
        WHEN: _get_best_fit is called
        THEN: Should still identify underlying linear trend
        """
        np.random.seed(42)  # For reproducibility
        x = np.linspace(0, 10, 20)
        xerr=np.abs(np.random.normal(0, 0.001, size=len(x)))
        x_noisy= x + xerr
        y = -(1/1000)*x_noisy**4 + (1/50)*x_noisy**3 - (1/10) *x_noisy**2 - 1 * x_noisy + 50
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        
        p = np.poly1d(coeffs)
        y_pred = p(x)
        # Should still choose cubic fit despite noise
        assert degree == 4
        assert np.all(np.abs(y_pred - y) <= 1e-1), "y_pred and y are not equal"
        

    def test_linear_extremely_large_values(self, fitter):
        """
        GIVEN: Data with extremely large values
        WHEN: _get_best_fit is called
        THEN: Should handle numerical stability issues
        """
        x = np.array([1e4, 2e4, 3e4, 4e4, 5e4])
        y = 2 * x + 1
        with pytest.warns(UserWarning):
            best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
            assert len(coeffs) == 2
        
    def test_linear_fit_at_boundaries(self, fitter):
        """
        GIVEN: Linear data at the boundaries of valid pixel values (0-65535)
        WHEN: _get_best_fit is called
        THEN: Should correctly identify linear relationship at boundaries
        """
        x = np.array([0, 16383, 32767, 49151, 65535])  # Full range
        slope = 2
        intercept = 100
        y = slope * x + intercept
               
        with pytest.warns(UserWarning):
            best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
            degree = len(coeffs) - 1 
        
        assert degree == 1
        assert_array_almost_equal(coeffs, [slope, intercept], decimal=1)
        assert score < 1e4  # Reasonable MSE for this scale

    def test_quadratic_fit_within_bounds(self, fitter):
        """
        GIVEN: Quadratic data within valid pixel value range and expected behaviour:
            y decreases as x increases
        WHEN: _get_best_fit is called
        THEN: Should identify quadratic relationship while staying within bounds
        """
        x = np.linspace(0, 65535, 20)
        # Scale quadratic function to stay within bounds
        a = -50 / (65535**2)  # Coefficient to ensure max within bounds
        y = a * (x)**2 + 50  # Parabola with peak at middle of range

        #xerr = np.ones_like(x) * 100
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
    
        assert degree == 2
        # Verify predictions stay within bounds
        p = np.poly1d(coeffs)
        y_pred = p(x)
        y_pred[y_pred <= 1e-1] = 0
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 50)
        
    def test_within_bounds_OD(self, fitter):
        """
        GIVEN: A dataset and the processed mode OD, 
            where y is obtained through a quadratic relationship.
        WHEN: The _get_best_fit method is called using x_OD derived from the dataset.
        THEN: The predicted y values should remain within the specified bounds.
        """

        x = np.linspace(10000, 40000, 20)
        assert isinstance(ProcessingMode.OD, ProcessingMode) #Check this
        x_OD = fitter._process_values(x, mode=ProcessingMode.OD)
        # Scale quadratic function to stay within bounds
        a = -50 / (65535**2)  # Coefficient to ensure max within bounds
        y = a * (x)**2 + 50  # Parabola with peak at middle of range
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        
    
        # Verify predictions stay within bounds
        p = np.poly1d(coeffs)
        y_pred = p(x_OD)
        y_pred[y_pred <= 1e-1] = 0
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 50)
        
    def test_within_bounds_netOD(self, fitter):
        """
        GIVEN: A dataset and the processed mode net_OD, 
            where y is obtained through a quadratic relationship.
        WHEN: The _get_best_fit method is called using x_net_OD derived from the dataset.
        THEN: The predicted y values should remain within the specified bounds.
        """
        x = np.linspace(10000, 65535, 20)
        
        # Scale quadratic function to stay within bounds
        a = -50 / (65535**2)  # Coefficient to ensure max within bounds
        y = a * (x)**2 + 50  # Parabola with peak at middle of range
        x_net_OD = fitter._process_values(x, y, mode=ProcessingMode.NET_OD)
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
    
        # Verify predictions stay within bounds
        p = np.poly1d(coeffs)
        y_pred = p(x_net_OD)
        y_pred[y_pred <= 1e-1] = 0
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 50)
        
    def test_processed_mode(self, fitter):
        """
        GIVEN: A dataset and the processed modes OD and net_OD, 
            where y is obtained through a quadratic relationship.
        WHEN: The _get_best_fit method is called using PV, x_OD and x_net_OD derived from the dataset.
        THEN: The predicted y values should be almost equal for each mode.
        """
        x_PV = np.linspace(10000, 65535, 20)
        # Scale quadratic function to stay within bounds
        a = -50 / (65535**2)  # Coefficient to ensure max within bounds
        y = a * (x_PV)**2 + 50  # Parabola with peak at middle of range
        x_net_OD = fitter._process_values(x_PV, y, mode=ProcessingMode.NET_OD)
        x_OD = fitter._process_values(x_PV, y, mode= ProcessingMode.OD)
        best_funct_PV, coeffs_PV, score_PV, fitting_results_PV = self._get_best_fit(fitter, x_PV, y)
        best_funct_OD, coeffs_OD, score_OD, fitting_results_OD = self._get_best_fit(fitter, x_OD, y)
        best_funct_net_OD, coeffs_net_OD, score, fitting_results_net_OD = self._get_best_fit(fitter, x_net_OD, y)
        # Verify predictions are the same for all x processing mode
        p = np.poly1d(coeffs_PV)
        y_pred_PV = p(x_PV)
        p = np.poly1d(coeffs_OD)
        y_pred_OD = p(x_OD)
        p = np.poly1d(coeffs_net_OD)
        y_pred_net_OD = p(x_net_OD)
        assert np.all(np.abs(y_pred_PV - y) <= 0.3), "y_pred_PV and y are not equal"
        assert np.all(np.abs(y_pred_OD - y)<= 0.3), "y_pred_OD and y are not equal"
        assert np.all(np.abs(y_pred_net_OD - y) <= 0.3), "y_pred_net_OD and y are not equal"
        
        
        
    def test_inverse_relationship_with_noise(self, fitter):
        """
        GIVEN: Data with inverse relationship (y = k/x + c) and noise, keeping values in bounds
        WHEN: _get_best_fit is called
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
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x_noisy, y)
        
        degree = len(coeffs) -1

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
        WHEN: _get_best_fit is called
        THEN: Should handle saturation appropriately
        """
        x = np.linspace(0, 65535, 20)
    
        # Adjusting the equation to ensure y approaches 0 as x increases
        y = -0.0001 * x + 50  # This will ensure y decreases towards 0 as x increases
        y[y < 0] = 0  # Ensure that y doesn't drop below 0
    
    
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        # Should still identify linear relationship
        assert degree == 1
        # Verify handling of saturated values
        p = np.poly1d(coeffs)
        y_pred = p(x)
        assert np.all(y_pred >= 0), "Some predicted values are negative"
        # Check if all predicted values are below the saturation limit
        assert np.all(y_pred < 51), f"Some predicted values exceed saturation limit: {y_pred[y_pred > 50]}"
    
    def test_low_signal_data(self, fitter):
        """
        GIVEN: Data with very low pixel values (near zero)
        WHEN: _get_best_fit is called
        THEN: Should handle low signal appropriately
        """
        x = np.linspace(0, 1, 10)  # Low range
        y = - 0.005 * x + 10  # Small slope and offset
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1         
        assert degree == 1
        assert_array_almost_equal(coeffs, [0.005, 10], decimal=1)
        # Verify predictions stay within bounds
        p = np.poly1d(coeffs)
        y_pred = p(x)
        assert np.all(y_pred >= 0)
        assert np.all(y_pred <= 50)
        
    def test_single_point(self, fitter):
        """ 
        GIVEN: A single data point (x=1, y=2)
        WHEN: _get_best_fit is called
        THEN: Best fit should return None, as fitting is impossible with one point.
        """
        x = np.array([1])
        y = np.array([2])
        with pytest.raises(ValueError, match="Number of points is not sufficients to fit data"):
            best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
           
        
    def test_two_points(self, fitter):
        """ 
        GIVEN: Two data points forming a line (x=[1, 2], y=[3, 5])
        WHEN: _get_best_fit is called
        THEN: Best fit should return a linear fit with zero MSE.
        """
        x = np.array([1, 2])
        y = np.array([3, 5])
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        
        assert degree == 1  # Should choose linear fit
        assert_array_almost_equal(coeffs, [2, 1], decimal=10)  # y = 2x + 1
        assert score < 1e-3  # Should have nearly perfect fit


    def test_normalization(self, fitter):
        """ 
        GIVEN: Data with large x values
        WHEN: _get_best_fit is called
        THEN: Should handle normalization without errors.
        """
        x = np.array([1000, 2000, 3000, 4000, 5000])
        y = np.array([2, 3, 5, 7, 11])
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        
        assert degree > 0  # Should find a polynomial fit
        
        
       
    def test_unequal_length_arrays(self, fitter):
        """
        GIVEN: x and y arrays of different lengths
        WHEN: _get_best_fit is called
        THEN: Should raise ValueError
        """
        x = np.array([1, 2, 3])
        y = np.array([1, 2])
        with pytest.warns(UserWarning):
            self._get_best_fit(fitter, x, y)
    


class TestPolynomialFitStructure:
    """Tests best fit for the polynomial_fit method"""
    
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
    
    def test_invalid_input_returns_none(self, fitter):
        """
        GIVEN: Invalid input data (None values)
        WHEN: _get_best_fit is called
        THEN: All return values should be None and a UserWarning should be raised
        """
        with pytest.warns(UserWarning, match="x array cannot be None"):
            mse, degree, coeffs, results = fitter.polynomial_fit(None, None)
        
        assert mse is None
        assert degree is None
        assert coeffs is None
        assert results is None

    def test_fitting_results_structure(self, fitter, linear_data):
        """
        GIVEN: Valid input data
        WHEN: polynomial_fit is called
        THEN: Results dictionary should have correct structure for each degree
        """
        x, y, xerr = linear_data
        max_degree = 3
        fitting_results = fitter.polynomial_fit(x, y, mode=ProcessingMode.PV, max_degree=max_degree)
        
        for result in fitting_results:

            coefficients = result.get('coefficients')
            degree = result.get('degree')
        
            # Assicura che i coefficienti e il grado siano coerenti
            assert len(coefficients) == degree + 1, f"Mismatch: coefficients {coefficients}, degree {degree}"
            assert isinstance(result, dict)
            assert any(key in result for key in [
                'function', 'metrics', 'polynomial', 'coefficients',
                'degree', 'success'
            ])
            
    def test_max_degree_limit(self, fitter, linear_data):
        """
        GIVEN: Valid input data
        WHEN: polynomial_fit is called with specific max_degree
        THEN: Should not test polynomials beyond max_degree
        """
        x, y, xerr = linear_data
        max_degree = 2
        results = fitter.polynomial_fit(x, y, mode=ProcessingMode.PV, max_degree=max_degree)
        
        assert len(results) == max_degree
        assert max(result['degree'] for result in results) == max_degree 
     
    def test_constant_y_values(self, fitter):
        """ 
        GIVEN: Multiple x values with constant y (y=5)
        WHEN: polynomial_fit is called
        THEN: Best fit should return a constant function with zero MSE.
        """
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 5, 5, 5, 5])
        fitting_results = fitter.polynomial_fit(x, y)
        coeffs = fitting_results[0]['coefficients']
        degree = len(coeffs) - 1 
        assert degree == 0  # Should choose constant fit
        assert_array_almost_equal(coeffs, [5], decimal=10)  # y = 5

        
    def test_constant_x_values(self, fitter):
        """ 
        GIVEN: Multiple x values with constant y (y=5)
        WHEN: polynomial_fit is called
        THEN: Best fit should return a constant function with zero MSE.
        """
        y = np.array([1, 2, 3, 4, 5])
        x = np.array([5, 5, 5, 5, 5])
        with pytest.raises(ValueError, match="Cannot fit polynomial when all x values are constant"):
            mse, degree, coeffs, results = fitter.polynomial_fit(x, y)
        
    
    def test_max_degree_validation(self, fitter):
        """ 
        GIVEN: A valid dataset
        WHEN: polynomial_fit is called with max_degree > 4
        THEN: Should issue a warning and return the best fit for degree <= 4.
        """
        x = np.array([1, 2, 3, 4, 5, 6, 7 ])
        y = np.array([2, 3, 5, 7, 11, 13, 15])
        
        with warnings.catch_warnings(record=True) as w:
            fitter.polynomial_fit(x, y, max_degree=5)
            assert len(w) > 0  # Warning should have been triggered
            assert any(
                "Polynomial degrees higher than 4 might lead to overfitting and numerical instability." in str(warning.message)
                for warning in w
            )

    def test_normalization(self, fitter):
        
        """ 
        Test polynomial fitting with data requiring normalization.
        GIVEN: Data with large x values
        WHEN: polynomial_fit is called
        THEN: Should handle normalization without errors.
        """
        # Test with large x values to verify normalization
        x_large = np.linspace(1e6, 1e6 + 100, 50)
        y_large = 2 * x_large + 1
        
        fitting_results = self._get_best_fit(x_large, y_large)
        
        # Verify we got results
        assert fitting_results is not None
        assert len(fitting_results) > 0
        
        # Get the linear fit result
        linear_fit = next(result for result in fitting_results 
                         if result['function'] == 'polynomial_degree_1')
        
        assert linear_fit['success']
        assert linear_fit['degree'] == 1
        
    def test_unequal_length_arrays(self, fitter):
        """
        GIVEN: x and y arrays of different lengths
        WHEN: polynomial_fit is called
        THEN: Should raise ValueError
        """
        x = np.array([1, 2, 3])
        y = np.array([1, 2])
        with pytest.raises(ValueError):
            fitter.polynomial_fit(x, y)
    
    def test_extremely_large_values(self, fitter):
        """
        GIVEN: Data with extremely large values
        WHEN: polynomial_fit is called
        THEN: Should handle numerical stability issues
        """
        x = np.array([1e4, 2e4, 3e4, 4e4, 5e4])
        y = 2 * x + 1
        with pytest.warns(UserWarning):
            best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y, max_degree=4)
            assert len(coeffs) == 2
        
    def test_fitting_error_handling(self, fitter):
        """
        GIVEN: A scenario that will cause a fitting error
        WHEN: polynomial_fit is called
        THEN: Should capture error information in fitting_results
        """
        # Create a deliberately problematic dataset
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([float('inf'), 2, 3, 4, 5])  # Introduce an inf value
        
        with pytest.warns(UserWarning):
             results = fitter.polynomial_fit(x, y, max_degree=4)
        
        # Check that results contain error information for some degrees
        error_results = [r for r in results if 'error_message' in r]
        assert len(error_results) > 0, "No error results captured"
        
        # Verify error result structure
        for error_result in error_results:
            assert error_result['function'].startswith('polynomial_degree_')
            assert error_result['mse'] is None
            assert error_result['success'] == False
            assert isinstance(error_result['error_message'], str)
            assert len(error_result['error_message']) > 0

class TestPolynomialPropertyBased:
    """Tests best fit for the polynomial_fit method with propriety-based test"""
    
    @pytest.fixture(scope="class")
    def fitter(self):
        # Assuming the function is part of a class called CurveFitter
        return CurveFitter()
    
    @classmethod
    def setup_class(cls):
        # Suppress all warnings
        warnings.filterwarnings("ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
        
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)  # Only show CRITICAL logs
        
        # Remove any existing handlers to prevent double logging
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Optional: Add a null handler if you want to prevent warning about no handlers
        logger.addHandler(logging.NullHandler())
          
    @staticmethod
    def _get_best_fit(fitter, x, y):
        """
        Utility per ottenere il miglior fit.
        """
        fitting_results = fitter.polynomial_fit(x, y)
        return fitter.select_best_fit(fitting_results)
    

    @settings(max_examples=25)
    @given(
        x_min=st.integers(min_value=0, max_value=65000),
        x_max=st.integers(min_value=2, max_value=65535),
        n_points=st.integers(min_value=15, max_value=100)
    )
    def test_linear_fit_hypotesis(self, fitter, x_min, x_max, n_points):
        """
        Verify that a linear function is correctly identified in a randomized dataset.

        GIVEN: A randomly generated dataset following the equation y = 2x + 1.
        WHEN: The function attempts to fit a model to the dataset.
        THEN: The best-fit function should be linear, with a degree of 1.
        """
        assume(x_max > x_min)
        assume(x_max - x_min > 1)
        x = np.linspace(x_min, x_max, n_points)
        y = 2*x + 1
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        
        assert degree > 0  # Should find a polynomial fit
        assert score < 1e-3  # Should have a reasonable MSE  
        
    @settings(max_examples=25)
    @given(
        x_min=st.integers(min_value=0, max_value=65000),
        x_max=st.integers(min_value=2, max_value=65535),
        n_points=st.integers(min_value=10, max_value=100)
    )
    def test_quadratic_fit_hypothesis(self, fitter, x_min, x_max, n_points):
        """
        Verify that a quadratic function is correctly identified in a randomized dataset.

        GIVEN: A randomly generated dataset following the equation y = 3*x**2 + 2*x + 1.
        WHEN: The function attempts to fit a model to the dataset.
        THEN: The best-fit function should be quadratic, with a degree of 2.
        """
        assume(x_max > x_min)
        assume(x_max - x_min > 1)
        x = np.linspace(x_min, x_max, n_points)
        y = 3*x**2 + 2*x + 1
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1

        assert degree == 2  # Should choose quadratic fit

    @settings(max_examples=25)
    @given(
        x_min=st.integers(min_value=0, max_value=65000),
        x_max=st.integers(min_value=2, max_value=65535),
        n_points=st.integers(min_value=10, max_value=100)
    )
    def test_cubic_fit_hypothesis(self, fitter, x_min, x_max, n_points):
        """
        Verify that a cubic function is correctly identified in a randomized dataset.

        GIVEN: A randomly generated dataset following the equation y = 4*x**3 + 3*x**2 + 2*x + 1.
        WHEN: The function attempts to fit a model to the dataset.
        THEN: The best-fit function should be cubic, with a degree of 3.
        """
        assume(x_max > x_min)
        assume(x_max - x_min > 1)
        x = np.linspace(x_min, x_max, n_points)
        y = 4*x**3 + 3*x**2 + 2*x + 1
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1

        assert degree == 3  # Should choose cubic fit


    @settings(max_examples=25)
    @given(
        x_min=st.integers(min_value=0, max_value=65000),
        x_max=st.integers(min_value=2, max_value=65535),
        n_points=st.integers(min_value=20, max_value=100)
    )
    def test_quartic_fit_hypothesis(self, fitter, x_min, x_max, n_points):
        """
        Verify that a quartic function is correctly identified in a randomized dataset.

        GIVEN: A randomly generated dataset following the equation y = 5*x**4 + 4*x**3 + 3*x**2 + 2*x + 1.
        WHEN: The function attempts to fit a model to the dataset.
        THEN: The best-fit function should be quartic, with a degree of 4.
        """
        assume(x_max > x_min)
        assume(x_max - x_min > 1)
        x = np.linspace(x_min, x_max, n_points)
        y = 5*x**4 + 4*x**3 + 3*x**2 + 2*x + 1
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1

        assert degree == 4  # Should choose quartic fit
    
    
    @settings(deadline=None, max_examples = 50, derandomize=True)
    @given(
        x_min=st.integers(min_value=0, max_value=65000),
        x_max=st.integers(min_value=2, max_value=65535),
        n_points=st.integers(min_value=20, max_value=100)
    )
    def test_polynomial_fits_hypothesis(self, fitter, x_min, x_max, n_points):
        """Property-based tests for polynomial fitting across multiple degrees.
        
        Tests if the fitter correctly handles different polynomial degrees
        with randomly generated input ranges and point counts.
        GIVEN a range of x values and a number of data points.
        WHEN polynomial fitting is applied to different polynomial degrees.
        THEN the fitter should correctly identify the expected polynomial degree.
        """
        assume(x_max > x_min)
        assume(x_max - x_min > 1)
        
        x = np.linspace(x_min, x_max, n_points)
        
        # Test different polynomial degrees
        test_cases = [
            (1, [2, 1], "linear"),
            (2, [3, 2, 1], "quadratic"),
            (3, [4, 3, 2, 1], "cubic"),
            (4, [5, 4, 3, 2, 1], "quartic")
        ]
        
        for degree, expected_coeffs, name in test_cases:
            # Generate polynomial of specified degree
            y = sum(coef * x**(deg) for deg, coef in enumerate(expected_coeffs[::-1]))
            
            best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
            fitted_degree = len(coeffs) - 1
            
            assert fitted_degree == degree, \
                f"Failed to identify {name} function (expected degree {degree}, got {fitted_degree})"
                
        @settings(
            deadline=None, derandomize=True, max_examples = 50)  
        
        @given(
            x_min=st.integers(min_value=0, max_value=50000),
            x_max=st.integers(min_value=10, max_value=65535),
            n_points=st.integers(min_value=16, max_value=100)
        )
        def test_polynomial_fits_with_noise(self, fitter, x_min, x_max, n_points):
            """Property-based tests for polynomial fitting across multiple degrees with noise.
            
            Tests if the fitter correctly identifies polynomial degrees in the presence
            of Gaussian noise. Uses a standard noise level of 0.1 * std(y) to ensure
            the noise scale is appropriate for the data range.
            GIVEN polynomial functions with low-level noise.
            WHEN polynomial fitting is performed.
            THEN the fitter should correctly estimate the polynomial degree within an acceptable range.
            """
            assume(x_max > x_min)
            assume(x_max - x_min > 1)
            
            x = np.linspace(x_min, x_max, n_points)
                    
            # Test cases for different polynomial degrees
            test_cases = [
                (1, [2, 1], "linear"),
                (2, [3, 2, 1], "quadratic"),
                (3, [4, 3, 2, 1], "cubic"),
                (4, [5, 4, 3, 2, 1], "quartic")
            ]
            
            np.random.seed(42)  # For reproducibility
            for degree, expected_coeffs, name in test_cases:
                noise_std = 0.1
                noise = np.random.normal(0, noise_std, size=len(x))
                x_noisy = x * (1 + noise)
                y_noisy = np.polyval(expected_coeffs[::-1], x_noisy)
                if np.max(x_noisy) <= 65535: 
                    result = self._get_best_fit(fitter, x_noisy, y_noisy)
                    best_funct, coeffs, score, fitting_results = result
                    fitted_degree = len(coeffs) - 1
                # Assertions
                    assert degree == fitted_degree, \
                        f"Failed to identify {name} function with noise (expected degree {degree}, got {fitted_degree})"
                


# Define test cases with descriptive IDs
EDGE_TEST_CASES = [
    pytest.param(0, 65530, 100, id="boundary-full-range"),
    pytest.param(50000, 65530, 20, id="large-values-sparse"),
    pytest.param(0, 1, 20, id="small-values-sparse"),
    pytest.param(0, 1000, 16, id="medium-range-sparse"),
    pytest.param(0, 1000, 1000, id="medium-range-dense")
]

class TestPolynomialFitsEdgeCases:
    
    @pytest.fixture(scope="class")
    def fitter(self):
        # Assuming the function is part of a class called CurveFitter
        return CurveFitter()
    
    @classmethod
    def setup_class(cls):
        # Suppress all warnings
        warnings.filterwarnings("ignore")
        
        
        logger = logging.getLogger()
        logger.setLevel(logging.CRITICAL)  # Only show CRITICAL logs
        
        # Remove any existing handlers to prevent double logging
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Add a null handler to prevent warning about no handlers
        logger.addHandler(logging.NullHandler())
    
    @staticmethod
    def _get_best_fit(fitter, x, y, mode=ProcessingMode.PV):
        """
        Utility per ottenere il miglior fit.
        """
        fitting_results = fitter.polynomial_fit(x, y)
        return fitter.select_best_fit(fitting_results)
    
    @pytest.mark.parametrize(
        "x_min, x_max, n_points",
        EDGE_TEST_CASES,
    )
    def test_linear_fit_edge_cases(self, fitter, x_min, x_max, n_points):
        """
        Verify that the polynomial fitter correctly identifies linear functions.
        
        GIVEN: A set of x values and corresponding y values defined by a linear function y = 2x + 1.
        WHEN: The polynomial fitting function is applied to the dataset.
        THEN: The fitter should correctly identify the polynomial as a linear function (degree 1),
              recover the coefficients [2, 1] with high precision, and achieve a nearly perfect fit score.
        """
        x = np.linspace(x_min, x_max, n_points)
        y = 2*x + 1
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1 
        assert degree == 1  # Should choose linear fit
        assert_array_almost_equal(coeffs, [2, 1], decimal=6)  # Should find correct coefficients
        assert score < 1e-3  # Should have nearly perfect fit
        
    @pytest.mark.parametrize(
        "x_min, x_max, n_points",
        EDGE_TEST_CASES,
    )
    def test_quadratic_fit_edge_cases(self, fitter, x_min, x_max, n_points):
        """
        Verify that the polynomial fitter correctly identifies quadratic functions.
        
        GIVEN: A set of x values and corresponding y values defined by a quadratic function y = 3x² + 2x + 1.
        WHEN: The polynomial fitting function is applied to the dataset.
        THEN: The fitter should correctly identify the polynomial as quadratic (degree 2).
        """
        x = np.linspace(x_min, x_max, n_points)
        y = 3*x**2 + 2*x + 1
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1

        assert degree == 2, f"Expected quadratic fit (degree 2), got degree {degree}"
    
    @pytest.mark.parametrize(
        "x_min, x_max, n_points",
        EDGE_TEST_CASES,
    )

    def test_quadratic_fit_edge_cases_with_noise(self, fitter, x_min, x_max, n_points):
        """
        Verify that the polynomial fitter correctly identifies quadratic functions in the presence of noise.
        
        GIVEN: A set of x values with small multiplicative noise and corresponding y values following y = 3x² + 2x + 1.
        WHEN: The polynomial fitting function is applied to the noisy dataset.
        THEN: The fitter should still correctly identify the polynomial as quadratic (degree 2).
        """
        x = np.linspace(x_min, x_max, n_points)

        # Generate noise proportional to x values
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, size=len(x))
        x_noisy = x * (1 + noise)  # Multiplicative noise 
        coefficients = [3, 2, 1]
        y_noisy = np.polyval(coefficients[::-1], x_noisy)
        if np.max(x_noisy) <= 65535:
            best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x_noisy, y_noisy)
            degree = len(coeffs) - 1
    
            assert degree == 2, f"Expected quadratic fit (degree 2), got degree {degree}"


    @pytest.mark.parametrize(
        "x_min, x_max, n_points",
        EDGE_TEST_CASES,
    )
    def test_cubic_fit_edge_cases(self, fitter, x_min, x_max, n_points):
        """
        Verify that the polynomial fitter correctly identifies cubic functions.
        
        GIVEN: A set of x values and corresponding y values defined by a cubic function y = 4*x**3 + 3*x**2 + 2*x + 1.
        WHEN: The polynomial fitting function is applied to the dataset.
        THEN: The fitter should correctly identify the polynomial as cubic (degree 3).
        """
        x = np.linspace(x_min, x_max, n_points)
        y = 4*x**3 + 3*x**2 + 2*x + 1
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1

        assert degree == 3, f"Expected cubic fit (degree 3), got degree {degree}"

    @pytest.mark.parametrize(
        "x_min, x_max, n_points",
        EDGE_TEST_CASES,
    )
    def test_cubic_fit_edge_cases_with_noise(self, fitter, x_min, x_max, n_points):
        """
        Verify that the polynomial fitter correctly identifies cubic functions in the presence of noise.
        
        GIVEN: A set of x values with small multiplicative noise and corresponding y values following y = 4*x**3 + 3*x**2 + 2*x + 1.
        WHEN: The polynomial fitting function is applied to the noisy dataset.
        THEN: The fitter should still correctly identify the polynomial as cubic (degree 3).
        """
        x = np.linspace(x_min, x_max, n_points)
        # Generate noise proportional to x values
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, size=len(x))
        x_noisy = x * (1 + noise)  # Multiplicative noise 
        coefficients = [4, 3, 2, 1]
        y_noisy = np.polyval(coefficients[::-1], x_noisy)
        if np.max(x_noisy) <= 65535:
            best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x_noisy, y_noisy)
    
            degree = len(coeffs) - 1
    
            assert degree == 3, f"Expected cubic fit (degree 3), got degree {degree}"

    @pytest.mark.parametrize(
        "x_min, x_max, n_points",
        EDGE_TEST_CASES,
    )
    def test_quartic_fit_edge_cases(self, fitter, x_min, x_max, n_points):
        """
        Verify that the polynomial fitter correctly identifies quartic functions.
        
        GIVEN: A set of x values and corresponding y values defined by a quartic function y =5*x**4+ 4*x**3 + 3*x**2 + 2*x + 1.
        WHEN: The polynomial fitting function is applied to the dataset.
        THEN: The fitter should correctly identify the polynomial as quartic (degree 4).
        """
        x = np.linspace(x_min, x_max, n_points)
        y = 5*x**4 + 4*x**3 + 3*x**2 + 2*x + 1
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree = len(coeffs) - 1

        assert degree == 4, f"Expected quartic fit (degree 4), got degree {degree}"


    @pytest.mark.parametrize(
        "x_min, x_max, n_points",
        EDGE_TEST_CASES,
    )
    def test_quartic_fit_edge_cases_with_noise(self, fitter, x_min, x_max, n_points):
        """
        Verify that the polynomial fitter correctly identifies quartic functions in the presence of noise.
        
        GIVEN: A set of x values with small multiplicative noise and corresponding y values following y = 5*x**4+4*x**3 + 3*x**2 + 2*x + 1.
        WHEN: The polynomial fitting function is applied to the noisy dataset.
        THEN: The fitter should still correctly identify the polynomial as quartic (degree 4).
        """
        x = np.linspace(x_min, x_max, n_points)
        # Generate noise proportional to x values
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, size=len(x))
        x_noisy = x * (1 + noise)  # Multiplicative noise 
        coefficients = [5, 4, 3, 2, 1]
        y_noisy = np.polyval(coefficients[::-1], x_noisy)
        if np.max(x_noisy) <= 65535:
            best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x_noisy, y_noisy)
    
            degree = len(coeffs) - 1
    
            assert degree == 4, f"Expected cubic fit (degree 3), got degree {degree}"  
    
    def test_polynomial_fits_standard_cases(self, fitter):
        """
        Standard test cases for polynomial fitting with predefined scenarios
        to cover basic functionality without randomness
        GIVEN specific polynomial functions without noise.
        WHEN polynomial fitting is performed.
        THEN the fitter should return the correct polynomial degree
        """
        # Test cases with clean, noise-free data
        test_cases = [
            # (x_min, x_max, n_points, coefficients, expected_degree)
            (0, 100, 50, [2, 1], 1),              # Linear
            (0, 100, 50, [3, 2, 1], 2),            # Quadratic
            (0, 100, 50, [4, 3, 2, 1], 3),         # Cubic
            (0, 100, 50, [5, 4, 3, 2, 1], 4)       # Quartic
        ]
        
        for x_min, x_max, n_points, coefficients, expected_degree in test_cases:
            # Generate x values
            x = np.linspace(x_min, x_max, n_points)
            
            # Generate y values using coefficients
            y = np.polyval(coefficients[::-1], x)
            
            # Perform curve fitting
            best_funct, fitted_coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
            
            # Check fitted degree
            fitted_degree = len(fitted_coeffs) - 1
            
            # Assertions
            assert fitted_degree == expected_degree, (
                f"Incorrect polynomial degree for coefficients {coefficients}. "
                f"Expected {expected_degree}, got {fitted_degree}"
            )
            


    @pytest.mark.parametrize("x_min,x_max,n_points,coefficients,expected_degree,name", [
        (0, 1000, 50, [2, 1], 1, "linear"),              # Linear
        (0, 1000, 50, [3, 2, 1], 2, "quadratic"),        # Quadratic
        (0, 1000, 50, [4, 3, 2, 1], 3, "cubic"),         # Cubic
        (0, 1000, 50, [5, 4, 3, 2, 1], 4, "quartic")     # Quartic
    ])
    def test_polynomial_fits_standard_cases_with_noise(self, fitter, x_min, x_max, n_points, 
                                            coefficients, expected_degree, name):
        """
        Parametrized test for polynomial fitting with low-level noise
        GIVEN polynomial functions with low-level noise.
        WHEN polynomial fitting is performed.
        THEN the fitter should correctly estimate the polynomial degree within an acceptable range.
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate x values
        x = np.linspace(x_min, x_max, n_points)

        # Generate noise proportional to x values
        noise_std = 0.1
        noise = np.random.normal(0, noise_std, size=len(x))
        x_noisy = x * (1 + noise)  # Multiplicative noise 
        y_noisy = np.polyval(coefficients[::-1], x_noisy)
       
        # Perform curve fitting
        best_funct, fitted_coeffs, score, fitting_results = self._get_best_fit(fitter, x_noisy, y_noisy)
        
        # Check fitted degree
        fitted_degree = len(fitted_coeffs) - 1
        
        # Assertions
        assert fitted_degree == expected_degree, (
            f"Incorrect polynomial degree for {name} function. "
            f"Expected {expected_degree}, got {fitted_degree}"
        )

    @pytest.mark.parametrize("true_degree,coeffs,polynomial_type", [
        (1, [2, 1], "linear"),
        (2, [3, 2, 1], "quadratic"),
        (3, [4, 3, 2, 1], "cubic"),
        (4, [5, 4, 3, 2, 1], "quartic")
    ])
    @pytest.mark.parametrize("noise_factor", [0.01, 0.05, 0.1, 0.2, 0.5])
    def test_noise_impact_on_degree_selection(self, fitter, true_degree, coeffs, polynomial_type, noise_factor):
        """
        Parametrized test to examine noise impact on degree selection
        
        GIVEN a polynomial function with varying levels of noise.
        WHEN polynomial fitting is applied.
        THEN the fitter should correctly estimate the polynomial degree within a reasonable threshold.

        """
        # Generate x values
        x = np.linspace(0, 1000, 50)
        
        # Set consistent random seed for reproducibility
        np.random.seed(42)
        
        # Add noise
        noise = np.random.normal(0, noise_factor, size=len(x))
        x_noisy = x * (1 + noise)  # Multiplicative noise 
        
        if np.max(x_noisy) <= 65535: 
            # Generate clean polynomial data
            y_noisy = sum(coef * x_noisy**(deg) for deg, coef in enumerate(coeffs[::-1]))
    
            
            # Perform curve fitting
            best_funct, fitted_coeffs, score, _ = self._get_best_fit(fitter, x_noisy, y_noisy)
            fitted_degree = len(fitted_coeffs) - 1
            
            # Assertions
            if noise_factor <= 0.1:  # Check for moderate noise levels
                assert fitted_degree == true_degree, (
                    f"Degree selection incorrect for {polynomial_type} function. "
                    f"Noise level: {noise_factor}, "
                    f"Expected degree: {true_degree}, "
                    f"Fitted degree: {fitted_degree}"
                )
            
            # Additional optional checks
            assert fitted_degree <= true_degree + 1, (
                f"Fitted degree significantly higher than expected for {polynomial_type}. "
                f"Noise level: {noise_factor}"
            )
    
    @pytest.mark.parametrize("x_min, x_max, n_points",
            EDGE_TEST_CASES,
        )
    @settings(
        deadline=None, derandomize=True, max_examples = 50)  
    @given(
        coefficients=st.lists(
            st.floats(min_value=-100, max_value=100).filter(lambda x: abs(x) >= 1e-3), 
            min_size=2, 
            max_size=5
        )
    )
    def test_polynomial_fit(self, fitter, x_min, x_max, n_points, coefficients):
        """
        Test polynomial fitting with various coefficient sets.
        
        GIVEN a range of x values and a list of polynomial coefficients.
        WHEN generating a polynomial using these coefficients and fitting a curve to it.
        THEN the identified polynomial degree should match the expected degree.
        """
        # Additional explicit filtering
        meaningful_coeffs = [coeff for coeff in coefficients if abs(coeff) >= 1e-7]
        
        # Skip if no meaningful coefficients remain
        assume(len(meaningful_coeffs) >= 2)
        x = np.linspace(x_min, x_max, n_points)
        # Create polynomial using coefficients
        y = np.polyval(coefficients[::-1], x)  # [::-1] because np.polyval expects coefficients in ascending order
        degree_expected = len(coefficients) - 1 
        best_funct, fitted_coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        degree_fit = len(fitted_coeffs) - 1
        
        assert degree_expected == degree_fit   # Should identify correct degree

    
