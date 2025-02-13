# -*- coding: utf-8 -*-
"""Created on Sun Feb  9 11:08:09 2025.

@author: Ele_p
"""
import numpy as np
import pytest
from Calibration_EBT3 import CurveFitter
from sklearn.metrics import mean_squared_error

class TestBayesianInitialGuesses():
    @pytest.fixture(scope="class")
    def fitter(self):
        # Assuming the function is part of a class called CurveFitter
        return CurveFitter()
    @pytest.fixture
    def test_data(self):
        """Fixture that provides test data for the exponential function."""
        x = np.linspace(0, 1000, 100)
        
        return x
    
    @pytest.fixture
    def y_data_exponential(self,fitter, test_data):
        """Fixture that provides y values calculated from the exponential
        function."""
        true_params = [2, 0.3, 1]
        x = test_data
        return fitter._exponential(x, *true_params)
    
    def test_bayesian_exponential_parameters_within_range(self, fitter, test_data, y_data_exponential):
        """GIVEN a naive initial guess for exponential function parameters WHEN
        _generate_bayesian_initial_guess is called THEN each parameter in the
        Bayesian guess must lie within the expected range."""
        # GIVEN: Calculate the naive guess
        x  = test_data
        naive_guess = fitter._guess_for_exponential(x, y_data_exponential, fitter._exponential)
        
        # WHEN: Determine the expected ranges
        expected_ranges = []
        for guess in naive_guess:
            if guess == 0:
                lower, upper = -1.0, 1.0
            else:
                scale = max(0.1 * abs(guess), 1e-3)
                lower = guess - 5 * scale
                upper = guess + 5 * scale
            expected_ranges.append(tuple(sorted([lower, upper])))
        
        # Generate the Bayesian initial guess
        bayesian_guess = fitter._generate_bayesian_initial_guess(
            fitter._exponential, 
            x, 
            y_data_exponential, 
            num_samples=1000
        )
        
        # THEN: Verify that each Bayesian parameter is within its expected range
        for i, (param, (lower, upper)) in enumerate(zip(bayesian_guess, expected_ranges)):
            assert lower <= param <= upper, \
                f"Parameter {i} (value: {param}) is outside the expected range [{lower}, {upper}]"
    
    def test_bayesian_exponential_improves_rmse(self, fitter, test_data, y_data_exponential):
        """GIVEN a dataset generated from an exponential function WHEN
        _generate_bayesian_initial_guess is called THEN the Bayesian guess
        should have a lower RMSE compared to the naive guess."""
        # GIVEN: Calculate the naive guess
        x  = test_data
        naive_guess = fitter._guess_for_exponential(x, y_data_exponential, fitter._exponential)
        
        # WHEN: Generate the Bayesian guess
        bayesian_guess = fitter._generate_bayesian_initial_guess(
            fitter._exponential, 
            x, 
            y_data_exponential, 
            num_samples=1000
        )
        
        # THEN: Compute RMSEs and assert that the Bayesian guess yields a lower RMSE
        y_naive = fitter._exponential(x, *naive_guess)
        y_bayes = fitter._exponential(x, *bayesian_guess)
        
        mse_naive = mean_squared_error(y_data_exponential, y_naive)
        mse_bayes = mean_squared_error(y_data_exponential, y_bayes)
        
        assert mse_bayes < mse_naive, \
            f"Bayesian guess RMSE ({mse_bayes:.6f}) should be less than naive guess RMSE ({mse_naive:.6f})"


class TestBayesianInitialGuessLimitCases:
    @pytest.fixture(scope="class")
    def fitter(self):
        return CurveFitter()

    @pytest.fixture
    def test_data(self):
        """Fixture that provides base test data."""
        x = np.linspace(0, 1000, 100)
        return x
            
                
    @pytest.mark.parametrize("bad_input", [
        np.array([np.inf, 1, 1]),
        np.array([np.nan, 1, 1]),
        np.array([-np.inf, 1, 1]),
        np.array([np.inf, np.inf, np.inf]),
        np.array([np.nan, np.nan, np.nan])
    ])
    def test_invalid_initial_guesses(self, fitter, test_data, bad_input):
        """Test behavior with invalid initial guesses (inf, nan)."""
        original_generate = fitter._generate_initial_guess
        fitter._generate_initial_guess = lambda func, x, y: bad_input

        try:
            y = np.ones_like(test_data)
            
            with pytest.raises((RuntimeWarning, ValueError, RuntimeError)):
                fitter._generate_bayesian_initial_guess(
                    fitter._exponential,
                    test_data,
                    y,
                    num_samples=1000
                )

        finally:
            fitter._generate_initial_guess = original_generate
            
    @pytest.mark.parametrize("func_name, num_params, test_case", [
        # Exponential function cases
        ("exponential", 3, {
            "zeros": np.array([0, 0, 0]),
            "small": np.array([1e-6, 1e-6, 1e-6]),
            "large": np.array([1e6, 1e6, 1e6]),
            "mixed": np.array([1e-6, 1.0, 1e6])
        }),
        # Combination of exponential cases
        ("combination_of_exponential", 4, {
            "zeros": np.array([0, 0, 0, 0]),
            "small": np.array([1e-6, 1e-6, 1e-6, 1e-6]),
            "large": np.array([1e6, 1e6, 1e6, 1e6]),
            "mixed": np.array([1e-6, 1.0, 1e3, 1e6])
        }),
        # Generalized rational cases
        ("generalized_rational", 5, {
            "zeros": np.array([0, 0, 0, 0, 0]),
            "small": np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
            "large": np.array([1e6, 1e6, 1e6, 1e6, 1e6]),
            "mixed": np.array([1e-6, 1.0, 1e3, 1e6, 0])
        }),
        # Log function cases
        ("log_function", 2, {
            "zeros": np.array([1e-5, 1e-5]),  # Note: b must be > 0 and a != 0
            "small": np.array([1.0, 1.0]),
            "large": np.array([1e6, 1e6]),
            "mixed": np.array([1e-6, 1.0])
        })
    ])
    def test_bayesian_initial_guess_parameter_count(self, fitter, test_data, func_name, num_params, test_case):
        """Test that the Bayesian initial guess returns the correct number of
        parameters.

        GIVEN a function, number of parameters, and test cases, WHEN generating
        the Bayesian initial guess, THEN the number of parameters should match
        the expected count.
        """
        func = getattr(fitter, f"_{func_name}")
        
        for case_name, initial_values in test_case.items():
            original_generate = fitter._generate_initial_guess
            fitter._generate_initial_guess = lambda f, x, y: initial_values
            
            try:
                # GIVEN: Function and test data
                try:
                    y = func(test_data, *initial_values)
                except:
                    y = np.ones_like(test_data)
                
                # WHEN: Generating Bayesian guess
                bayesian_guess = fitter._generate_bayesian_initial_guess(func, test_data, y, 1000)
                
                # THEN: Correct parameter count
                assert len(bayesian_guess) == num_params, f"Parameter count mismatch for {func_name} {case_name}"
            
            finally:
                fitter._generate_initial_guess = original_generate
        
        @pytest.mark.parametrize("func_name, num_params, test_case", [
            # Exponential function cases
            ("exponential", 3, {
                "zeros": np.array([0, 0, 0]),
                "small": np.array([1e-6, 1e-6, 1e-6]),
                "large": np.array([1e6, 1e6, 1e6]),
                "mixed": np.array([1e-6, 1.0, 1e6])
            }),
            # Combination of exponential cases
            ("combination_of_exponential", 4, {
                "zeros": np.array([0, 0, 0, 0]),
                "small": np.array([1e-6, 1e-6, 1e-6, 1e-6]),
                "large": np.array([1e6, 1e6, 1e6, 1e6]),
                "mixed": np.array([1e-6, 1.0, 1e3, 1e6])
            }),
            # Generalized rational cases
            ("generalized_rational", 5, {
                "zeros": np.array([0, 0, 0, 0, 0]),
                "small": np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                "large": np.array([1e6, 1e6, 1e6, 1e6, 1e6]),
                "mixed": np.array([1e-6, 1.0, 1e3, 1e6, 0])
            }),
            # Log function cases
            ("log_function", 3, {
                "zeros": np.array([1e-5, 1e-5]),  # Note: b > 0 and a != 0
                "small": np.array([1.0, 1.0]),
                "large": np.array([1e6, 1e6]),
                "mixed": np.array([1e-6, 1.0])
            })
        ])
        def test_bayesian_initial_guess_parameter_bounds(self, fitter, test_data, func_name, num_params, test_case):
            """Test that parameters in the Bayesian guess are within valid
            bounds.

            GIVEN initial values (zeros, small, large, mixed), WHEN generating
            the Bayesian initial guess, THEN parameters should stay within
            specified bounds.
            """
            func = getattr(fitter, f"_{func_name}")
            
            for case_name, initial_values in test_case.items():
                original_generate = fitter._generate_initial_guess
                fitter._generate_initial_guess = lambda f, x, y: initial_values
                
                try:
                    # GIVEN: Function and test data
                    try:
                        y = func(test_data, *initial_values)
                    except:
                        y = np.ones_like(test_data)
                    
                    # WHEN: Generating Bayesian guess
                    bayesian_guess = fitter._generate_bayesian_initial_guess(func, test_data, y, 1000)
                    
                    # THEN: Parameter bounds check
                    for i, (guess, bayesian) in enumerate(zip(initial_values, bayesian_guess)):
                        if guess == 0:
                            assert -1.0 <= bayesian <= 1.0, f"Parameter {i} out of bounds (zero case) in {func_name}"
                        else:
                            scale = max(0.1 * abs(guess), 1e-3)
                            assert guess - 5*scale <= bayesian <= guess + 5*scale, \
                                f"Parameter {i} out of bounds ({case_name}) in {func_name}"
                
                finally:
                    fitter._generate_initial_guess = original_generate
            
        @pytest.mark.parametrize("func_name, num_params, test_case", [
            # Exponential function cases
            ("exponential", 3, {
                "zeros": np.array([0, 0, 0]),
                "small": np.array([1e-6, 1e-6, 1e-6]),
                "large": np.array([1e6, 1e6, 1e6]),
                "mixed": np.array([1e-6, 1.0, 1e6])
            }),
            # Combination of exponential cases
            ("combination_of_exponential", 4, {
                "zeros": np.array([0, 0, 0, 0]),
                "small": np.array([1e-6, 1e-6, 1e-6, 1e-6]),
                "large": np.array([1e6, 1e6, 1e6, 1e6]),
                "mixed": np.array([1e-6, 1.0, 1e3, 1e6])
            }),
            # Generalized rational cases
            ("generalized_rational", 5, {
                "zeros": np.array([0, 0, 0, 0, 0]),
                "small": np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                "large": np.array([1e6, 1e6, 1e6, 1e6, 1e6]),
                "mixed": np.array([1e-6, 1.0, 1e3, 1e6, 0])
            }),
            # Log function cases
            ("log_function", 3, {
                "zeros": np.array([0, 1e-5, 1e-5]),  # Note: b and c must be > 0
                "small": np.array([1e-6, 1.0, 1.0]),
                "large": np.array([1e6, 1e6, 1e6]),
                "mixed": np.array([1e-6, 1.0, 1e6])
            })
        ])
        def test_function_output_with_bayesian_guess(self, fitter, test_data, func_name, num_params, test_case):
            """Test that the function produces valid output with Bayesian
            parameters.

            GIVEN: Bayesian parameters from initial guesses,
            WHEN: evaluating the function,
            THEN: outputs should have no infinities or NaNs.
            """
            func = getattr(fitter, f"_{func_name}")
            
            for case_name, initial_values in test_case.items():
                original_generate = fitter._generate_initial_guess
                fitter._generate_initial_guess = lambda f, x, y: initial_values
                
                try:
                    # GIVEN: Function and test data
                    try:
                        y = func(test_data, *initial_values)
                    except:
                        y = np.ones_like(test_data)
                    
                    # WHEN: Generating Bayesian guess
                    bayesian_guess = fitter._generate_bayesian_initial_guess(func, test_data, y, 1000)
                    
                    # THEN: Output validity check
                    try:
                        y_fit = func(test_data, *bayesian_guess)
                        assert not np.any(np.isinf(y_fit)), f"Infinite values in {func_name} {case_name}"
                        assert not np.any(np.isnan(y_fit)), f"NaN values in {func_name} {case_name}"
                    except Exception as e:
                        pytest.fail(f"Function evaluation failed for {func_name} {case_name}: {str(e)}")
                
                finally:
                    fitter._generate_initial_guess = original_generate