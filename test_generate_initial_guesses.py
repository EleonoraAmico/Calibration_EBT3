# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 10:01:42 2025

@author: Ele_p
"""
import numpy as np
import pytest
from Calibration_EBT3 import CurveFitter
from Calibration_EBT3 import ProcessingMode
from numpy.testing import assert_array_almost_equal
from hypothesis import given, assume, settings, HealthCheck, Verbosity
from hypothesis import strategies as st

from hypothesis.strategies import one_of
import warnings
import logging

def unknown_func(x, a, b, c):
    return a * x**2 + b * x + c

def one_param_func(x):
    return x**2


class TestInitialGuessGenerator():
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
        return fitter.select_polynomial_best_fit(fitting_results)
    
    # --------------------------
    # Test per _guess_for_exponential
    # --------------------------
    def test_exponential_typical(self, fitter):
        """
        Test exponential function parameter guessing for typical case
        GIVEN: Linear x values and y values defined as 2*exp(0.3*x)+1
        WHEN: Generating initial guess for exponential function
        THEN: Should return correct amplitude, rate, and offset parameters
        """
        
        x = np.linspace(0, 10, 11)
        y = fitter._exponential(x, 2, 0.3, 1)
        expected_amplitude = np.max(y) - np.min(y)
        expected = [expected_amplitude, - 1.0 / np.ptp(x), np.min(y)]
        result = fitter._guess_for_exponential(x, y, fitter._exponential)
        assert_array_almost_equal(result[0], expected[0])
        assert_array_almost_equal(result[1], expected[1])
        assert_array_almost_equal(result[2], expected[2])
    
    def test_exponential_limit(self, fitter):
        """
        Test exponential function parameter guessing for edge case
        GIVEN: Constant x values and y values
        WHEN: Generating initial guess with zero range x values
        THEN: Should handle zero range by using 1.0 as default range
        """
        # Caso limite: x costante (range zero) -> si usa 1.0 come range
        x = np.array([5, 5, 5])
        y = np.array([3, 3, 3])
        expected_amplitude = np.max(y) - np.min(y)  # 0
        expected = [expected_amplitude, -1.0, np.min(y)]
        result = fitter._guess_for_exponential(x, y, fitter._exponential)
        assert_array_almost_equal(result, expected)
    
    # --------------------------
    # Test per _guess_for_combination_of_exponential
    # --------------------------
    def test_combination_of_exponential_typical(self, fitter):
        """
        Test combined exponential function parameter guessing for typical case
        GIVEN: Linear x values and y values as sum of two exponentials
        WHEN: Generating initial guess for combined exponential function
        THEN: Should return correct amplitudes and rates based on peak-to-peak values
        """
        x = np.linspace(0, 10, 11)
        y = fitter._combination_of_exponential(x, 1, 0.2, 2, -0.2)
        amplitude = np.ptp(y)
        expected = [amplitude / 2, 1.0 / np.ptp(x), amplitude / 2, -1.0 / np.ptp(x)]
        result = fitter._guess_for_combination_of_exponential(x, y, fitter._combination_of_exponential)
        for r, e in zip(result, expected):
            assert_array_almost_equal(r, e)
    
    def test_combination_of_exponential_limit(self, fitter):
        """
        Test combined exponential function parameter guessing for edge case
        GIVEN: Constant x values with varying y values
        WHEN: Generating initial guess with zero range x values
        THEN: Should use default value 1.0 for denominators
        """
        x = np.array([3, 3, 3])
        y = np.array([1, 2, 3])
        amplitude = np.ptp(y)
        expected = [amplitude / 2, 1.0, amplitude / 2, -1.0]
        result = fitter._guess_for_combination_of_exponential(x, y, fitter._combination_of_exponential)
        for r, e in zip(result, expected):
            assert_array_almost_equal(r, e)
    
    # --------------------------
    # Test per _guess_for_generalized_rational
    # --------------------------
    def test_generalized_rational_typical(self, fitter):
        """
        Test generalized rational function parameter guessing for typical case
        GIVEN: Linear x and y values (e.g., y = 2x + 1)
        WHEN: Generating initial guess for generalized rational function
        THEN: Should return parameters matching linear fit with appropriate scaling
        """
        # Caso tipico: x e y lineari, ad es. y = 2*x + 1
        x = np.array([0, 1, 2, 3, 4])
        y = 2 * x + 1
        # polyfit dovrebbe restituire slope ~2 e intercept ~1
        x_range = np.ptp(x)
        expected = [1, 2, 1.0 / x_range, 1.0, 0.0]  # dato che min(y) = 1 non è < 0
        result = fitter._guess_for_generalized_rational(x, y, fitter._generalized_rational)
        for r, e in zip(result, expected):
            assert_array_almost_equal(r, e)
    
    def test_generalized_rational_limit(self, fitter):
        """
        Test generalized rational function parameter guessing for edge case
        GIVEN: Linear data with negative y values
        WHEN: Generating initial guess with negative minimum y
        THEN: Should adjust offset parameter to match minimum y value
        """
        # Caso limite: y con minimo negativo per verificare che l'ultimo parametro sia np.min(y)
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([-3, -2, -1, 0, 1])
        # In questo caso, polyfit restituisce slope ~1 e intercept ~-3
        x_range = np.ptp(x)
        expected = [-3, 1, 1.0 / x_range, 1.0, -3]
        result = fitter._guess_for_generalized_rational(x, y, fitter._generalized_rational)
        for r, e in zip(result, expected):
            assert_array_almost_equal(r, e)
    # --------------------------
    # Test per _guess_for_log_function
    # --------------------------
    def test_log_function_typical(self, fitter):
        """
        Test for Logarithmic function parameter guessing for typical case
        GIVEN: Positive x values
        WHEN: Generating initial guess for logarithmic function
        THEN: Should use 1.0 as minimum threshold for parameters
        """
        # Caso tipico: x con valori positivi -> min(x) = 1, quindi max(1.0, -1+1e-5) = 1.0
        x = np.linspace(1, 10, 10)
        y = fitter._log_function(x, 2, 2)  # il valore di y non influenza il guess
        expected_value = max(1.0, -np.min(x) + 1e-5)
        expected = [1.0, expected_value, expected_value]
        result = fitter._guess_for_log_function(x, y, fitter._log_function)
        for r, e in zip(result, expected):
            assert_array_almost_equal(r, e)
    
    def test_log_function_limit(self, fitter):
        """
        Test Logarithmic function parameter guessing for edge case
        GIVEN: X values including negative numbers
        WHEN: Generating initial guess with negative x values
        THEN: Should adjust parameters based on minimum x value plus epsilon
        """
        
        x = np.array([-2, -1, 0, 1, 2])
        y = fitter._log_function(x, 2, 2)
        expected_value = max(1.0, -np.min(x) + 1e-5)  # ~2.00001
        expected = [1.0, expected_value, expected_value]
        result = fitter._guess_for_log_function(x, y, fitter._log_function)
        for r, e in zip(result, expected):
            assert_array_almost_equal(r, e)
    
    # --------------------------
    # Test per _default_initial_guess
    # --------------------------
    def test_default_initial_guess_typical(self, fitter):
        """
        Test default parameter guessing for unknown functions
        GIVEN: Three-parameter unknown function
        WHEN: Generating default initial guess
        THEN: Should return array of ones matching parameter count
        """

        x = np.array([0, 1, 2])
        y = unknown_func(x, 1, 2, 3)
        expected = [1.0, 1.0, 1.0]
        result = fitter._default_initial_guess(x, y, unknown_func)
        assert_array_almost_equal(result, expected)
    
    def test_default_initial_guess_limit(self, fitter):
        """
        Test default parameter guessing for minimal functions
        GIVEN: Function with only x parameter
        WHEN: Generating default initial guess
        THEN: Should return empty array for zero additional parameters
        """
        x = np.array([0, 1, 2])
        y = one_param_func(x)
        expected = []  # num_params = 0 (1-1)
        result = fitter._default_initial_guess(x, y, one_param_func)
        assert_array_almost_equal(result, expected)
    
    # --------------------------
    # Test per generate_initial_guess (dispatch)
    # --------------------------
    def test_generate_initial_guess_dispatch(self, fitter):
        """
        Test Initial guess generator function dispatching
        GIVEN: Different function types (exponential and unknown)
        WHEN: Generating initial guesses
        THEN: Should dispatch to appropriate specific or default guess generators
        """
        x = np.linspace(0, 10, 11)
        # Per la funzione exponential, generate_initial_guess dovrebbe chiamare _guess_for_exponential
        y_exp = fitter._exponential(x, 2, 0.3, 1)
        result_exp = fitter._generate_initial_guess(fitter._exponential, x, y_exp)
        expected_exp = fitter._guess_for_exponential(x, y_exp, fitter._exponential)
        assert_array_almost_equal(result_exp, expected_exp)
        
        # Per una funzione sconosciuta (unknown_func), dovrebbe chiamare _default_initial_guess
        y_unknown = unknown_func(x, 1, 2, 3)
        result_unknown = fitter._generate_initial_guess(unknown_func, x, y_unknown)
        expected_unknown = fitter._default_initial_guess(np.asarray(x), np.asarray(y_unknown), unknown_func)
        assert_array_almost_equal(result_unknown, expected_unknown)