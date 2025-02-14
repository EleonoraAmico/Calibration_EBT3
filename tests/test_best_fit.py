# -*- coding: utf-8 -*-
"""Created on Thu Jan 30 16:41:18 2025.

@author: Ele_p
"""

import numpy as np
import pytest
from Calibration_EBT3 import CurveFitter
from Calibration_EBT3 import ProcessingMode
from numpy.testing import assert_array_almost_equal
from hypothesis import given, assume, settings, seed
from hypothesis import strategies as st

import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)

class TestNonLinearFunctions: 
    @pytest.fixture
    def fitter(self):
        # Assuming the function is part of a class called CurveFitter
        return CurveFitter()
    
    @staticmethod
    def _get_best_fit(fitter, x, y, mode=ProcessingMode.PV, print_results=False):
        """Utility per ottenere il miglior fit."""
        fitting_results = fitter.calculate_non_linear_fit(x, y, mode=mode, print_results=print_results)
        return fitter.select_best_fit(fitting_results)
    
    @pytest.fixture
    def exponential_data(self, fitter):
        """Generate exponential data using CurveFitter's _exponential
        function."""
        x = np.linspace(40000, 50000, 15)
        params = [2.0, 1.5, 3]  # A=2, B=1.5, C=3 
        y = fitter._exponential(x, *params)
        return x, y
    
    @pytest.fixture
    def combination_exp_data(self, fitter):
        """Generate combination of exponentials data using CurveFitter's
        _combination_of_exponential function."""
        
        x = np.linspace(40000, 50000, 15)
        params = [2.0, 1.5, 0.5, -0.5]  # A=2, B=1.5, C=0.5, D=-0.5
        y = fitter._combination_of_exponential(x, *params)
        return x, y
    
    @pytest.fixture
    def log_data(self, fitter):
        """Generate logarithmic data using CurveFitter's _log_function."""
       
        x = np.linspace(0, 1000, 15)  # Start from 0.1 to avoid log(0)
        params = [2.0, 5.0]  # A=2, B=3, C=5
        
        y = fitter._log_function(x, *params)
        return x, y
    
    @pytest.fixture
    def rational_data(self, fitter):
        """Generate rational function data using CurveFitter's
        _generalized_rational."""
        
        x = np.linspace(0, 1000, 15)
        params = [2.0, 3.0, 1.0, 2.0, 3.0]  # a=2, b=1, c=1, d=2, e=3
        y = fitter._generalized_rational(x, *params)
        
        return x, y
    
    def test_perfect_exponential_fit(self, fitter, exponential_data):
        """
        GIVEN: Perfect exponential data (y = A*exp(B*x))
        WHEN: get_best_fit is called
        THEN: Best fit should be exponential with near-zero MSE
        """
        x, y = exponential_data
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        assert best_funct == 'exponential'
        assert_array_almost_equal(coeffs, [2.0, 1.5, 3.0], decimal=3)  # Should find A=2, B=1.5
        assert score < 1e-3  # Should have nearly perfect fit

    def test_perfect_combination_exp_fit(self, fitter, combination_exp_data):
        """
        GIVEN: Perfect combination of exponentials data (y = A*exp(B*x) + C*exp(D*x))
        WHEN: get_best_fit is called
        THEN: Best fit should be combination of exponentials with near-zero MSE
        """
        x, y = combination_exp_data
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        assert best_funct == 'combination_of_exponential', f"the best function is {best_funct} not combination of exponential, {fitting_results}"
        y_fit = fitter._combination_of_exponential(x, *coeffs)
        assert_array_almost_equal(y, y_fit, decimal=3)  # Should find A=2, B=1.5, C=0.5, D=-0.5
        assert score < 1e-3  # Should have nearly perfect fit

    def test_perfect_log_fit(self, fitter, log_data):
        """
        GIVEN: Perfect logarithmic data (y = log(B*x/B)/A)
        WHEN: get_best_fit is called
        THEN: Best fit should be logarithmic with near-zero score
        """
        x, y = log_data
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y, mode=ProcessingMode.PV, print_results = True)
        assert best_funct == 'log_function'
        assert_array_almost_equal(coeffs, [2.0, 5.0], decimal=3)  # Should find A=2, B=3
        assert score < 1e-3  # Should have nearly perfect fit

    def test_perfect_rational_fit(self, fitter, rational_data):
        """
        GIVEN: Perfect rational data (y = (ax + b)/(cx + d))
        WHEN: get_best_fit is called
        THEN: Best fit should be rational with near-zero score
        """
        x, y = rational_data
        
        best_funct, coeffs, score, fitting_results = self._get_best_fit(fitter, x, y)
        assert best_funct == 'generalized_rational', f"don't match {fitting_results}"
        y_fit = fitter._generalized_rational(x, *coeffs)
        assert_array_almost_equal(y, y_fit, decimal=3)  # Should find a=2, b=1, c=1, d=2
        assert score < 1e-3  # Should have nearly perfect fit
        
    def test_best_fit_selection_exception(self, fitter):
        """Handle exception during best fit selection.

        GIVEN: A set of fitting results where at least one fit is successful
        WHEN: An unexpected error occurs while selecting the best fit
        THEN: The function should log an error and return (None, None, None, fitting_results)
        """
        fitting_results = [
        {"success": True, "metrics": {"score": None}},  # This causes a TypeError
        {"success": True, "metrics": {"score": 0.2}}
        ]
    
        # Act: Call select_best_fit with faulty data to trigger an exception
        result = fitter.select_best_fit(fitting_results)
    
        # Assert: Function should return (None, None, None, fitting_results)
        assert result == (None, None, None, fitting_results)
    


# Define test cases with descriptive IDs
EDGE_TEST_CASES = [
    pytest.param(0, 65530, 100, id="boundary-full-range"),
    pytest.param(50000, 65530, 20, id="large-values-sparse"),
    pytest.param(0, 1, 16, id="small-values-sparse"),
    pytest.param(0, 1000, 16, id="medium-range-sparse"),
    pytest.param(0, 1000, 1000, id="medium-range-dense")
]

class TestBestFitEdgeCasesNotPolynomialFunctions:
    """Tests best fit for the polynomial_fit method."""
    
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
            
        # To prevent warning about no handlers
        logger.addHandler(logging.NullHandler())
        
         
    
    # Single exponential test
    @pytest.mark.parametrize(
        "x_min, x_max, n_points",
        EDGE_TEST_CASES,
    )
    
    def test_single_exp_fit_edge_cases_fixed_coefficients(self, fitter, x_min, x_max, n_points, a = 5, b = -7, c = 3):
        """Test the robustness and accuracy of the single exponential fitting
        function.

        GIVEN:
            - A range of `x` values defined by `x_min`, `x_max`, and `n_points`.
            - A single exponential function with fixed coefficients: a = 5, b = -7, c = 3.
            - The coefficient `b` is negative, indicating an exponentially decreasing function,
                which is a typical case in the studied context
            - A non-linear fitting function that can fit multiple models.

        WHEN:
            - The function `_exponential(x, a, b, c)` is used to generate `y`.
            - `calculate_non_linear_fit(x, y)` is called to fit the data.

        THEN:
            - The fitting results should contain an entry for the 'exponential' function.
            - The fit should be marked as successful (`success == True`).
            - The fitted parameters should be close to the expected values (within a 10% relative tolerance):
                - `a ≈ 5`
                - `b ≈ -7`
                - `c ≈ 3`
        """
        x = np.linspace(x_min, x_max, n_points)
        y = fitter._exponential(x,a,b,c)
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for exponential
        exponential = None
        for result in fitting_results:
            if result['function'] == 'exponential':
                exponential = result
                break
        
        # Assert that we found the exponential results
        assert exponential is not None, (
            "exponential results not found in fitting results"
        )
        
        # Check success status
        assert exponential['success'] == True, (
            "Fitting with exponential was not successful"
        )
        assert np.isclose(exponential['coefficients'][0], a, rtol=0.1), "Fitted parameter 'a' deviates too much from expected"
        assert np.isclose(exponential['coefficients'][1], b, rtol=0.1), "Fitted parameter 'b' deviates too much from expected"
        assert np.isclose(exponential['coefficients'][2], c, rtol=0.1), "Fitted parameter 'c' deviates too much from expected"
        
        
    @pytest.mark.parametrize(
         "x_min, x_max, n_points",
         EDGE_TEST_CASES,
     )
    def test_exp_function_fitting_accuracy_on_edge_cases(self, fitter, x_min, x_max, n_points, a = 5, b = -7, c = 3):
        """Test the fitting accuracy and robustness of the selection of best
        fit for exponential function under edge cases.

        GIVEN:
            - A range of `x` values defined by `x_min`, `x_max`, and `n_points`.
            - A exponential function `_exponential(x, a, b, c)` with fixed coefficients `[10.0, 5.0]`.
            - A non-linear fitting function and the selection of the best model.

        WHEN:
            - The `_exponential` is used to generate synthetic `y` values.
            - The `calculate_non_linear_fit(x, y)` method is called to perform curve fitting.
            - The `select_best_fit(fitting_results)` function is used to determine the most suitable model.

        THEN:
            - The `exponential` should be identified as the best-fitting model.
        """        
        # Given: Generate uniform x distribution
        x = np.linspace(x_min, x_max, n_points)
        params = [a, b, c]
        y=fitter._exponential(x, *params)
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for combination_of_exponential
        exponential = None
        for result in fitting_results:
            if result['function'] == 'exponential':
                exponential = result
                break
        
        best_funct, _ , _ , _ = fitter.select_best_fit(fitting_results)
        assert best_funct == exponential['function'], \
            "The best funct is different from exponential, it is {best_funct}"
 
    @pytest.mark.parametrize(
        "x_min, x_max, n_points",
        EDGE_TEST_CASES,
    )
    def test_combination_of_exp_fit_edge_cases(self, fitter, x_min, x_max, n_points):
        """Test the robustness and accuracy of the fitting function for the
        combination of exponential.

        GIVEN:
            - A range of `x` values defined by `x_min`, `x_max`, and `n_points`.
            - A sum of exponential function with fixed coefficients = [ 2.5, -0.003, 1.8, -0.0001]
            - The coefficients `b` and `d` are negative, indicating an exponentially decreasing function,
                which is a typical case in the studied context
            - A non-linear fitting function that can fit multiple models.

        WHEN:
            - The function `_combination_of_exponential(x, a, b, c, d)` is used to generate `y`.
            - `calculate_non_linear_fit(x, y)` is called to fit the data.

        THEN:
            - The fitting results should contain an entry for the 'combination_of_exponential' function.
            - The fit should be marked as successful (`success == True`).
        """
        params = [ 25, 0.003, 10, -0.0001]
        # Given: Generate uniform x distribution
        x = np.linspace(x_min, x_max, n_points)
        y= fitter._combination_of_exponential(x, *params)
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for combination_of_exponential
        combination_exp_result = None
        for result in fitting_results:
            if result['function'] == 'combination_of_exponential':
                combination_exp_result = result
                break
        
        # Assert that we found the combination_of_exponential results
        assert combination_exp_result is not None, (
            "combination_of_exponential results not found in fitting results"
        )
        
        # Check success status
        assert combination_exp_result['success'] == True, (
            "Fitting with combination_of_exponential was not successful"
        )
    #Rational edge cases
    @pytest.mark.parametrize(
         "x_min, x_max, n_points",
         EDGE_TEST_CASES,
     )
    def test_rational_fit_edge_cases(self, fitter, x_min, x_max, n_points):
        """Test the robustness and accuracy of the fitting function for the
        generalized rational.

        GIVEN:
            - A range of `x` values defined by `x_min`, `x_max`, and `n_points`.
            - A rational function with fixed coefficients = [10.0, -5.0, 0.001, 1.0, 0.0]
            - A non-linear fitting function that can fit multiple models.

        WHEN:
            - The function `_generalized_rational(x, a, b, c, d)` is used to generate `y`.
            - `calculate_non_linear_fit(x, y)` is called to fit the data.

        THEN:
            - The fitting results should contain an entry for the '_generalized_rational' function.
            - The fit should be marked as successful (`success == True`).
        """
        # Given: Generate uniform x distribution
        x = np.linspace(x_min, x_max, n_points)
        params = [10.0, -5.0, 0.001, 1.0, 0.0]
        y=fitter._generalized_rational(x, *params)
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for generalized_rational
        generalized_rational = None
        for result in fitting_results:
            if result['function'] == 'generalized_rational':
                generalized_rational = result
                break
        
        # Assert that we found the combination_of_exponential results
        assert generalized_rational is not None, (
            "combination_of_exponential results not found in fitting results"
        )
        
        # Check success status
        assert generalized_rational['success'] == True, (
            "Fitting with generalized_rational was not successful"
        )
    
    @pytest.mark.parametrize(
         "x_min, x_max, n_points",
         EDGE_TEST_CASES,
     )
    def test_log_fit_edge_cases(self, fitter, x_min, x_max, n_points):
        """Test the robustness and accuracy of the fitting function for the log
        function.

        GIVEN:
            - A range of `x` values defined by `x_min`, `x_max`, and `n_points`.
            - A log function with fixed coefficients = [10.0, 5.0]
            - A non-linear fitting function that can fit multiple models.

        WHEN:
            - The function `_log_function(x, a, b)` is used to generate `y`.
            - `calculate_non_linear_fit(x, y)` is called to fit the data.

        THEN:
            - The fitting results should contain an entry for the '_log_function' function.
            - The fit should be marked as successful (`success == True`).
            -The fitted parameters should be close to the expected values (within a 10% relative tolerance)
        """
        # Given: Generate uniform x distribution
        x = np.linspace(x_min, x_max, n_points)
        params = [10.0, 5.0]
        y=fitter._log_function(x, *params)
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for combination_of_exponential
        log_function = None
        for result in fitting_results:
            if result['function'] == 'log_function':
                log_function = result
                break
        
        # Assert that we found the combination_of_exponential results
        assert log_function is not None, (
            "log_function results not found in fitting results"
        )
        
        # Check success status
        assert log_function['success'] == True, (
            "Fitting with log_function was not successful"
        )
        assert np.isclose(log_function['coefficients'][0], params[0], rtol=0.1), "Fitted parameter 'a' deviates too much from expected"
        assert np.isclose(log_function['coefficients'][1], params[1], rtol=0.1), "Fitted parameter 'b' deviates too much from expected"
        
    @pytest.mark.parametrize(
         "x_min, x_max, n_points",
         EDGE_TEST_CASES,
     )
    def test_log_function_fitting_accuracy_on_edge_cases(self, fitter, x_min, x_max, n_points):
        """Test the fitting accuracy and robustness of the selection of best
        fit for logarithmic function under edge cases.

        GIVEN:
            - A range of `x` values defined by `x_min`, `x_max`, and `n_points`.
            - A logarithmic function `_log_function(x, a, b)` with fixed coefficients `[10.0, 5.0]`.
            - A non-linear fitting function and the selection of the best model.

        WHEN:
            - The `_log_function` is used to generate synthetic `y` values.
            - The `calculate_non_linear_fit(x, y)` method is called to perform curve fitting.
            - The `select_best_fit(fitting_results)` function is used to determine the most suitable model.

        THEN:
            - The `log_function` should be identified as the best-fitting model.
        """        
        # Given: Generate uniform x distribution
        x = np.linspace(x_min, x_max, n_points)
        params = [10.0, 5.0]
        y=fitter._log_function(x, *params)
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for combination_of_exponential
        log_function = None
        for result in fitting_results:
            if result['function'] == 'log_function':
                log_function = result
                break
        
        best_funct, _ , _ , _ = fitter.select_best_fit(fitting_results)
        assert best_funct == log_function['function'], \
            "The best funct is different from log_function, it is {best_funct}"

class TestBestPropertyBasedNonLinearFunctions:
    """Tests best fit for the polynomial_fit method."""
    
    @pytest.fixture(scope="class")
    def fitter(self):
        # Assuming the function is part of a class called CurveFitter
        return CurveFitter()

    # Fix the seed for reproducibility
    @seed(42)
    @settings(deadline = None, max_examples = 50)
    @given(
        a=st.floats(min_value=0, max_value=50).filter(lambda a: abs(a) >= 1e-4),
        b=st.floats(min_value=-5, max_value=0).filter(lambda b: abs(b) >= 1e-4),
        c=st.floats(min_value=0, max_value=50).filter(lambda c: abs(c) >= 1e-4),
    )
    def test_single_exp_fit_different_coefficients(self, fitter, a, b, c):
        """Test the fitting of a decreasing exponential function works
        correctly with randomly selected coefficients a, b and c and it behaves
        as expected for values of x ranging from 0 to 65535, with y constrained
        between 0 and 50.

        GIVEN: A decreasing exponential y = a*e^(b*x) + x , where:
                - the range of values for the coefficients is:
                  - a in [0, 50] (positive scaling factor),
                  - b in [-5, 0] (negative decay rate),
                  - c in [0, 50] (offset).
                - The independent variable x is a sequence of 100 values evenly spaced from 0 to 65535.
                - The dependent variable y is generated using the exponential function with the
                    assumption that the values of y will be within the range of [0, 50].

        WHEN:- The test performs a non-linear fitting of the generated data
            - The fitting algorithm attempts to find the values of a,b and c that best match the given data.

        THEN:
            - The fitting result should contain an "exponential" model.
            - The fitting process should succeed.
            -The fitted parameters should be close to the expected values (within a 10% relative tolerance)
        """
        x = np.linspace(0, 1000, 100)
        y = fitter._exponential(x,a,b,c)
        assume(np.all(y <= 50))
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for combination_of_exponential
        exp_result = None
        for result in fitting_results:
            if result['function'] == 'exponential':
                exp_result = result
                break
        
        # Assert that we found the combination_of_exponential results
        assert exp_result is not None, (
            "exp_result not found in fitting results"
        )
        
        # Check success status
        assert exp_result['success'] == True, (
            "Fitting with exp_result was not successful"
        )
        assert np.isclose(exp_result['coefficients'][0], a, rtol=0.1), "Fitted parameter 'a' deviates too much from expected"
        assert np.isclose(exp_result['coefficients'][1], b, rtol=0.1), "Fitted parameter 'b' deviates too much from expected"
        assert np.isclose(exp_result['coefficients'][2], c, rtol=0.1), "Fitted parameter 'c' deviates too much from expected"
        best_funct, _ , _ , _ = fitter.select_best_fit(fitting_results)
        assert best_funct == exp_result['function'], "The best funct is different from exponential, it is {best_funct}"
        
        #assert score < 5e-3
    @seed(42)
    @settings(deadline=None, max_examples = 50)
    @given(
         a=st.floats(min_value=0, max_value=50).filter(lambda a: abs(a) >= 1e-4),
         b=st.floats(min_value=-5, max_value=-1e-4),
         c=st.floats(min_value=0, max_value=30).filter(lambda c: abs(c) >= 1e-4),
     )
    def test_exp_fit_different_coefficients_with_noise(self, fitter, a, b, c):
        """Test of an exponential function to non-uniform noisy data and
        randomly selected coefficients a, b and c.

        GIVEN: -A set of non-uniformly distributed x values between 20,000 and 50,000
               -And Gaussian noise with standard deviation of 100
               -And exponential function parameters a, b, and c
        WHEN:  -y values are generated using the exponential function
                and y values are within bounds (0 < y ≤ 50) and we apply the fitting algorithm
        THEN: -The fitting results should contain exponential function parameters
                and the fitting should successfully converge
        """
        # Fix the seed for reproducibility
        np.random.seed(42)  
    
        # Given: Generate non-uniform x distribution
        x1 = np.random.uniform(20000, 40000, 15)# Dense region
        x2 = np.random.uniform(40000, 50000, 5)# Sparse region
        x = np.sort(np.concatenate([x1, x2]))
    
        #  And: Add Gaussian noise to x values
        noise = np.random.normal(0, 100, x.shape)  # Standard deviation of 100
        x_noisy = x + noise
    
        #  When: Calculate y values using the exponential function
        y = fitter._exponential(x_noisy, a, b, c)
        # And: Verify y values are within expected bounds
        assume(np.all(y <= 50) and np.all(y > 0))
        # And: Perform the fitting
    
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Then: Extract exponential fitting results
        exp_result = None
        for result in fitting_results:
            if result['function'] == 'exponential':
                exp_result = result
                break
        
        # Assert that we found the combination_of_exponential results
        assert exp_result is not None, (
            "exp_result not found in fitting results"
        )
        
        # Check success status
        assert exp_result['success'] == True, (
            "Fitting with exp_result was not successful"
        )
          
    # Double exponential test
    @seed(42)
    @settings(deadline=None, max_examples = 15)
    @given(
        a=st.floats(min_value=-10, max_value=10).filter(lambda a: abs(a) >= 1e-4),
        b=st.floats(min_value=0, max_value=5).filter(lambda a: abs(a) >= 1e-4),
        c=st.floats(min_value=0, max_value=10).filter(lambda a: abs(a) >= 1e-4),
        d=st.floats(min_value=-5, max_value=0).filter(lambda a: abs(a) >= 1e-4),
    )
    def test_combination_of_exp_fit_unfixed_coefficients(self, fitter, a, b, c, d):
        """Test the fitting of a combination of exponential function (sum and
        difference of exponential) works correctly with randomly selected
        coefficients a, b, c and d and it behaves as expected for values of x
        ranging from 0 to 65535, with y constrained between 0 and 50.

        GIVEN: A combination of exponential y = a*e^(b*x) + c*e^(d*x) , where:
                - the range of values for the coefficients is:
                  - a in [-10, 10],
                  - b in [-5, 5].
                  - c in [0, 10],
                  - d in [-5, 5].
                - The independent variable x is a sequence of 100 values evenly spaced from 0 to 65535.
                - The dependent variable y is generated using the exponential function with the
                    assumption that the values of y will be within the range of [0, 50].

        WHEN:- The test performs a non-linear fitting of the generated data
            - The fitting algorithm attempts to find the values of a,b and c that best match the given data.

        THEN:
            - The fitting result should contain an "combination of exponential" model.
            - The fitting process should succeed.
            - The coefficients should be in the expected range:
        """
        # Fix the seed for reproducibility
        np.random.seed(42)  
        # Given: Generate non-uniform x distribution
        x1 = np.random.uniform(20000, 40000, 5)
        x2 = np.random.uniform(40000, 50000, 15)
        x = np.sort(np.concatenate([x1, x2]))
        y= fitter._combination_of_exponential(x,a,b,c,d)
        assume(np.all(y <= 50) & np.all(y > 0))
        # Perform the fitting
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for combination_of_exponential
        combination_exp_result = None
        for result in fitting_results:
            if result['function'] == 'combination_of_exponential':
                combination_exp_result = result
                break
        
        # Assert that we found the combination_of_exponential results
        assert combination_exp_result is not None, (
            "combination_of_exponential results not found in fitting results"
        )
        
        # Check success status
        assert combination_exp_result['success'] == True, (
            "Fitting with combination_of_exponential was not successful"
        )
        
    @seed(42)
    @settings(deadline=None, max_examples = 50)
    @given(
        a=st.floats(min_value=0, max_value=10).filter(lambda a: abs(a) >= 1e-4),
        b=st.floats(min_value=0, max_value=5).filter(lambda b: abs(b) >= 1e-7),
        c=st.floats(min_value=0, max_value=10).filter(lambda c: abs(c) >= 1e-4),
        d=st.floats(min_value=-5, max_value=0).filter(lambda d: abs(d) >= 1e-7),
    )
    def test_double_exp_fit_unfixed_coefficients_noise(self, fitter, a, b, c, d):
        """Test the fitting of a combination of exponential function (sum and
        difference of exponential) works correctly with randomly selected
        coefficients a, b and c and it behaves as expected for values of x
        ranging from 0 to 65535, with y constrained between 0 and 50.

        GIVEN: A combination of exponential y = a*e^(b*x) + c*e^(d*x) , where:
                - the range of values for the coefficients is:
                  - a in [-10, 10],
                  - b in [-5, 5].
                  - c in [0, 10],
                  - d in [-5, 5].
                - The independent variable x is a sequence of 100 values evenly spaced from 0 to 65535.
                - The dependent variable y is generated using the exponential function with the
                    assumption that the values of y will be within the range of [0, 50].

        WHEN:- The test performs a non-linear fitting of the generated data
            - The fitting algorithm attempts to find the values of a,b and c that best match the given data.

        THEN:
            - The fitting result should contain an "combination of exponential" model.
            - The fitting process should succeed.
        """

        # Fix the seed for reproducibility
        np.random.seed(42)
        # Given: Generate non-uniform x distribution
        x1 = np.random.uniform(20000, 40000, 15)
        x2 = np.random.uniform(40000, 50000, 5)
        x = np.sort(np.concatenate([x1, x2]))
    
        # Add noise
        noise = np.random.normal(0, 100, x.shape)  
        x_noisy = x + noise
    
        # Compute y as combination of exponential
        y = fitter._combination_of_exponential(x_noisy, a, b, c, d)
        assume(np.all(y <= 50) and np.all(y > 0))
        
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for combination_of_exponential
        combination_exp_result = None
        for result in fitting_results:
            if result['function'] == 'combination_of_exponential':
                combination_exp_result = result
                break
        
        # Assert that we found the combination_of_exponential results
        assert combination_exp_result is not None, (
            "combination_of_exponential results not found in fitting results"
        )
        
        # Check success status
        assert combination_exp_result['success'] == True, (
            "Fitting with combination_of_exponential was not successful"
        )
    # Rational function test   
    @seed(42)
    @settings(deadline=None, max_examples = 50) 
    @given(
        a=st.floats(min_value=-10, max_value=10).filter(lambda a: abs(a) >= 1e-4),
        b=st.floats(min_value=-10, max_value=10).filter(lambda a: abs(a) >= 1e-4),
        c=st.floats(min_value=0.1, max_value=10),  # Avoid zero in denominator
        d=st.floats(min_value=0.1, max_value=10),  # Avoid zero in denominator
        e=st.floats(min_value=-10, max_value=10)
    )
    def test_rational_fit_different_coefficients(self, fitter, a, b, c, d, e):
        """Test the fitting of a rationa function works correctly with randomly
        selected coefficients a, b and c and it behaves as expected for values
        of x ranging from 0 to 65535, with y constrained between 0 and 50.

        GIVEN: A combination of exponential y = (a+bx)/(cx + d) + e , where:
                - the range of values for the coefficients is:
                  - a in [-10, 10],
                  - b in [-10, 10].
                  - c in [0.1, 10] to avoid zero in denominator,
                  - d in [0.1, 10] to avoid zero in denominator,
                  - e in [-10, 10].
                - The independent variable x is a sequence of 100 values evenly spaced from 0 to 65535.
                - The dependent variable y is generated using the exponential function with the
                    assumption that the values of y will be within the range of [0, 50].

        WHEN:- The test performs a non-linear fitting of the generated data
            - The fitting algorithm attempts to find the values of a,b and c that best match the given data.

        THEN:
            - The fitting result should contain an "combination of exponential" model.
            - The fitting process should succeed.
            - The coefficients should be in the expected range:
        """
        # Given: Generate uniform x distribution
        x = np.linspace(4000, 65535, 100)
        
        k = (a + b*x) / (c*x + d)
        assume (not np.all(np.isclose(k, k[0], atol=1e-6)))
        # Prevent pathological cases
        assume(np.all(c * x + d > 1e-6))  # Non-zero denominator
        assume(np.abs(b/c) > 1e-3)  # Prevent degenerate rational functions
    
        y=fitter._generalized_rational(x,a,b,c,d,e)
        assume(np.all(y <= 50) & np.all(y > 0))
        assume(np.any(y != y[0]))
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for generalized_rational
        generalized_rational = None
        for result in fitting_results:
            if result['function'] == 'generalized_rational':
                generalized_rational = result
                break
        
        # Assert that we found the combination_of_exponential results
        assert generalized_rational is not None, (
            "generalized_rational results not found in fitting results"
        )
        
        # Check success status
        assert generalized_rational['success'] == True, (
            "Fitting with generalized_rational was not successful"
        )
        
    @seed(42)
    @settings(deadline=None, max_examples=50)
    @given(
        a=st.floats(min_value=-10, max_value=10).filter(lambda a: abs(a) >= 1e-4),
        b=st.floats(min_value=-10, max_value=10).filter(lambda a: abs(a) >= 1e-4),
        c=st.floats(min_value=0.1, max_value=10),  # Avoid zero in denominator
        d=st.floats(min_value=0.1, max_value=10),  # Avoid zero in denominator
        e=st.floats(min_value=-10, max_value=10)
    )
    def test_rational_fit_with_noise(self, fitter, a, b, c, d, e):
        """Test of an exponential function to non-uniform noisy data and
        randomly selected coefficients a, b, c, d and e.

        GIVEN: -A set of non-uniformly distributed x values between 20,000 and 50,000
               -And Gaussian noise with standard deviation of 100
               -And exponential function parameters a, b, and c
        WHEN:  -y values are generated using the exponential function
                and y values are within bounds (0 < y ≤ 50) and we apply the fitting algorithm
        THEN: -The fitting results should contain exponential function parameters
                and the fitting should successfully converge
        """
        # Given: Generate non-uniform x distribution
        x1 = np.random.uniform(20000, 40000, 15)
        x2 = np.random.uniform(40000, 50000, 5)
        x = np.sort(np.concatenate([x1, x2]))
        k = (a + b*x) / (c*x + d)
        assume (not np.all(np.isclose(k, k[0], atol=1e-6)))
        # Prevent pathological cases
        assume(np.all(c * x + d > 1e-6))  # Non-zero denominator
        assume(np.abs(b/c) > 1e-3)  # Prevent degenerate rational functions
    
        # Add noise
        noise = np.random.normal(0, 100, x.shape)  # per esempio, std=100
        x_noisy = x + noise
        
        # Compute y with the function generalized_rational
        y = fitter._generalized_rational(x_noisy, a, b, c, d, e)
        assume(np.all(y <= 50) and np.all(y > 0))
        assume(np.any(y != y[0]))
        
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for generalized_rational
        generalized_rational = None
        for result in fitting_results:
            if result['function'] == 'generalized_rational':
                generalized_rational = result
                break
        
        # Assert that we found the generalized_rational results
        assert generalized_rational is not None, (
            "generalized_rational results not found in fitting results"
        )
        
        # Check success status
        assert generalized_rational['success'] == True, (
            "Fitting with generalized_rational was not successful"
        )
    #Property-based test for log function    
    @seed(42)
    @settings(deadline=None, max_examples=50) 
    # Log ratio test
    @given(
        a=st.floats(min_value=-10, max_value=10).filter(lambda a: abs(a) >= 1e-4),
        b=st.floats(min_value=0.1, max_value=10),  # Avoid zero in denominator    
    ) 
    def test_log_ratio_fit(self, fitter, a, b):
        """Test the fitting of a log function works correctly with randomly
        selected coefficients a and b and it behaves as expected for values of
        x ranging from 0 to 65535, with y constrained between 0 and 50.

        GIVEN: A natural logarithmic function y = ln(b+x/b)/a , where:
                - the range of values for the coefficients is:
                  - a in [-10, 10]
                  - b in [0.1, 10] (positive to ensure the logarithm is well-defined)
                - The independent variable x is a sequence of 16 values evenly spaced from 4000 to 65535.
                - The dependent variable y is generated using the exponential function with the
                    assumption that the values of y will be within the range of [0, 50].

        WHEN:- The test performs a non-linear fitting of the generated data
            - The fitting algorithm attempts to find the values of a,b and c that best match the given data.

        THEN:
            - The fitting result should contain an "log_function" model.
            - The fitting process should succeed.
            -The fitted parameters should be close to the expected values (within a 10% relative tolerance)
        """
        # Given: Generate uniform x distribution
        x = np.linspace(4000, 65535, 16)
        y = fitter._log_function(x, a, b)
    
        assume(np.all(y <= 50) & np.all(y > 0))
        assume(np.any(y != y[0]))
        
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for combination_of_exponential
        log_function = None
        for result in fitting_results:
            if result['function'] == 'log_function':
                log_function = result
                break
        
        # Assert that we found the combination_of_exponential results
        assert log_function is not None, (
            "log_function results not found in fitting results"
        )
        
        # Check success status
        assert log_function['success'] == True, (
            "Fitting with log_function was not successful"
        )
        assert np.isclose(log_function['coefficients'][0], a, rtol=0.1), "Fitted parameter 'a' deviates too much from expected"
        assert np.isclose(log_function['coefficients'][1], b, rtol=0.1), "Fitted parameter 'b' deviates too much from expected"
        best_funct, _ , _ , _ = fitter.select_best_fit(fitting_results)
        assert best_funct == log_function['function'], "The best funct is different from log_function, it is {best_funct}"
        
    
    @seed(42)
    @settings(deadline=None, max_examples=50)
    @given(
        a=st.floats(min_value=-10, max_value=10).filter(lambda a: abs(a) >= 1e-4),
        b=st.floats(min_value=0.1, max_value=10).filter(lambda a: abs(a) >= 1e-4),

    )
    def test_log_fit_with_noise(self, fitter, a, b):
        """Test of an exponential function to non-uniform noisy data and
        randomly selected coefficients a, b and c.

        GIVEN: -A set of non-uniformly distributed x values between 20,000 and 50,000
               -And Gaussian noise with standard deviation of 100
               -And exponential function parameters a, b, and c
        WHEN:  -y values are generated using the exponential function
                and y values are within bounds (0 < y ≤ 50) and we apply the fitting algorithm
        THEN: -The fitting results should contain exponential function parameters
                and the fitting should successfully converge
        """
        # Given: Generate non-uniform x distribution
        x1 = np.random.uniform(20000, 40000, 15)
        x2 = np.random.uniform(40000, 50000, 5)
        x = np.sort(np.concatenate([x1, x2]))

        # Add noise
        noise = np.random.normal(0, 100, x.shape)  # per esempio, std=100
        x_noisy = x + noise
        
        # Compute log function and assume that y is between the bounds
        y = fitter._log_function(x_noisy, a, b)
        assume(np.all(y <= 50) and np.all(y > 0))
        assume(np.any(y != y[0]))
        
        fitting_results = fitter.calculate_non_linear_fit(x,y)
        # Find the results for log_function
        log_function = None
        for result in fitting_results:
            if result['function'] == 'log_function':
                log_function = result
                break
        
        # Assert that we found the log_function results
        assert log_function is not None, (
            "generalized_rational results not found in fitting results"
        )
        
        # Check success status
        assert log_function['success'] == True, (
            "Fitting with generalized_rational was not successful"
        )
        assert np.isclose(log_function['coefficients'][0], a, rtol=0.1), "Fitted parameter 'a' deviates too much from expected"
        assert np.isclose(log_function['coefficients'][1], b, rtol=0.1), "Fitted parameter 'b' deviates too much from expected"
        
 
