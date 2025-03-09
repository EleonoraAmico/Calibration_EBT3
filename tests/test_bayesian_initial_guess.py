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
    
    
    def test_bayesian_exponential_improves_rmse(self, fitter, test_data, y_data_exponential):
        """Test that the Bayesian initial guess returns a lower MSE compared to the naive guess.
        GIVEN a dataset generated from an exponential function
        WHEN _generate_bayesian_initial_guess is called 
        THEN the Bayesian guess should have a lower MSE compared to the naive guess."""

        # GIVEN: Calculate the naive guess
        x  = test_data
        naive_guess = fitter._guess_for_exponential(x, y_data_exponential, fitter._exponential)
        
        # WHEN: Generate the Bayesian guess
        bayesian_guess = fitter._generate_bayesian_initial_guess(
            fitter._exponential, 
            x, 
            y_data_exponential, 
        )
        
        # THEN: Compute RMSEs and assert that the Bayesian guess yields a lower RMSE
        y_naive = fitter._exponential(x, *naive_guess)
        y_bayes = fitter._exponential(x, *bayesian_guess)
        
        mse_naive = mean_squared_error(y_data_exponential, y_naive)
        mse_bayes = mean_squared_error(y_data_exponential, y_bayes)
        
        assert mse_bayes < mse_naive, \
            f"Bayesian guess MSE ({mse_bayes:.6f}) should be less than naive guess MSE ({mse_naive:.6f})"
            
    def test_normal_case(self,fitter):
        """
        What: Test parameter sampling with normal, non-zero initial guesses
        Given: A set of valid non-zero initial guesses and sample count
        When: We call create_parameter_ranges_and_samples
        Then: Returns correctly sized samples
        """
        # Given
        initial_guess = [5.0, -3.0, 10.0]
        num_samples = 100
        
        # When
        samples, param_ranges = fitter._create_parameter_ranges_and_samples(
            initial_guess, num_samples
        )
        
        # Then
        # Check return types
        assert isinstance(samples, list)
        assert isinstance(param_ranges, list)
        
        # Check dimensions
        assert len(samples) == len(initial_guess)
        assert len(param_ranges) == len(initial_guess)
        
        # Each sample array should have num_samples elements
        for sample_array in samples:
            assert len(sample_array) == num_samples
        
        
    
    def test_zero_guesses(self,fitter):
        """
        What: Test parameter sampling with zero values in initial guesses
        Given: Initial guesses including zero values
        When: We call create_parameter_ranges_and_samples
        Then: Zero values get default ranges of [-1.0, 1.0] while other values get normal ranges
        """
        # Given
        initial_guess = [0.0, 5.0, 0.0]
        num_samples = 50
        
        # When
        samples, param_ranges = fitter._create_parameter_ranges_and_samples(
            initial_guess, num_samples
        )
        
        # Then
        # Check zero-value handling
        assert param_ranges[0] == (-1.0, 1.0)
        assert param_ranges[2] == (-1.0, 1.0)
        
        # Non-zero value should have normal range
        assert param_ranges[1] != (-1.0, 1.0)
    
    def test_nan_values(self,fitter):
        """
        What: Test handling of NaN values in initial guesses
        Given: Initial guesses containing NaN values
        When: We call create_parameter_ranges_and_samples
        Then: ValueError is raised
        """
        # Given
        initial_guess = [3.0, np.nan, 7.0]
        num_samples = 30
        
        # When/Then
        with pytest.raises(ValueError):
            fitter._create_parameter_ranges_and_samples(initial_guess, num_samples)
    
    def test_sampling_error(self,fitter):
        """
        What: Test handling of sampling errors
        Given: Valid initial guesses but a mocked sampling function that raises ValueError
        When: We call create_parameter_ranges_and_samples
        Then: ValueError is propagated with additional context about parameters
        """
        # Given
        initial_guess = [0, 0]
        num_samples = -1 
            
        # When/Then
        with pytest.raises(ValueError):
            fitter._create_parameter_ranges_and_samples(initial_guess, num_samples)
        
    
    def test_small_range(self, fitter):
        """
        What: Test handling of very small parameter ranges
        Given: A parameter sampler with extremely small range_width and valid initial guesses
        When: We call create_parameter_ranges_and_samples
        Then: Generated ranges still have positive width despite the small range_width
        """
        # Given
        small_sampler = CurveFitter(range_width=1e-10)
        initial_guess = [1.0, 2.0]
        num_samples = 20
        
        # When
        samples, param_ranges = small_sampler._create_parameter_ranges_and_samples(
            initial_guess, num_samples
        )
        
        # Then
        # Check that minimum range is enforced
        for lower, upper in param_ranges:
            assert upper - lower > 0
    
    def test_negative_samples(self, fitter):
        """
        What: Test with negative values in initial guesses
        Given: Initial guesses with negative values
        When: We call create_parameter_ranges_and_samples
        Then: Ranges are correctly calculated around the negative guesses
        """
        # Given
        initial_guess = [-10.0, -20.0]
        num_samples = 40
        
        # When
        samples, param_ranges = fitter._create_parameter_ranges_and_samples(
            initial_guess, num_samples
        )
        
        # Then
        # Check ranges for negative values
        for i, (lower, upper) in enumerate(param_ranges):
            guess = initial_guess[i]
            assert lower < guess
            assert upper > guess
    



    def test_bayesian_initial_guess_returns_correct_format(self, fitter, test_data, y_data_exponential):
        """
        What: Test that _generate_bayesian_initial_guess returns a properly formatted result
        Given: A CurveFitter instance and sample x, y data
        When: We call _generate_bayesian_initial_guess with a test function
        Then: It returns a list of floats with the expected number of parameters
        """
        # Given
        x = test_data
        # When
        result = fitter._generate_bayesian_initial_guess(fitter._exponential, x, y_data_exponential)
        
        # Then
        assert isinstance(result, list)
        assert len(result) == 3  # test_func has 3 parameters
        assert all(isinstance(param, float) for param in result)
    
    def test_bayesian_initial_guess_with_perfect_data(self, fitter, test_data):
        """
        What: Test that the method finds near-exact parameters with perfect data
        Given: Data generated with known parameters and no noise
        When: We call _generate_bayesian_initial_guess and compute y_result 
        Then: It returns y values close to the original ones
        """
        # Given
        x = test_data
        exact_params = [1, -1, 1.0]
        y = fitter._exponential(x, *exact_params)  # No noise
        
        
        result = fitter._generate_bayesian_initial_guess(fitter._exponential, x, y)
        y_result = fitter._exponential(x, *result) 
        # Then
        # The returned parameters should be close to the exact ones
        # with wider tolerance because we're using a small num_samples
        assert np.allclose(y, y_result, rtol=0.3, atol=0.3)
    
    def test_bayesian_initial_guess_reproducibility(self, fitter, test_data):
        """
        What: Test that the function produces consistent results with fixed seed
        Given: The same input data and function
        When: We call _generate_bayesian_initial_guess twice
        Then: Both calls return identical parameter values
        """
        # Given
        x= test_data
        y= fitter._exponential(x, 2, 3, 4)
        
        # When
        result1 = fitter._generate_bayesian_initial_guess(fitter._exponential, x, y)
        result2 = fitter._generate_bayesian_initial_guess(fitter._exponential, x, y)
        
        # Then
        assert result1 == result2
    
    def test_bayesian_initial_guess_improves_deterministic_guess(self, fitter, test_data):
        """
        What: Test that the Bayesian approach improves on the deterministic initial guess
        Given: Sample data and a deliberately poor initial guess
        When: We call _generate_bayesian_initial_guess
        Then: The resulting parameters produce a better fit than the initial guess
        """
        # Given
        x = test_data
        y = fitter._exponential(x, 2, 3, 4)
        poor_initial_guess = [1.0, -0.1, 0.5]  # Deliberately off from real parameters
        
        
        # When
        improved_guess = fitter._generate_bayesian_initial_guess(fitter._exponential, x, y)
        
        # Then
        initial_y = fitter._exponential(x, *poor_initial_guess)
        improved_y = fitter._exponential(x, *improved_guess)
        
        initial_error = np.mean((y - initial_y)**2)
        improved_error = np.mean((y - improved_y)**2)
        
        assert improved_error < initial_error
    
    
    def test_bayesian_initial_guess_handles_bad_data(self, fitter):
        """
        What: Test that the function handles problematic data appropriately
        Given: Data with different lengths for x and y
        When: We call _generate_bayesian_initial_guess
        Then: It raises a ValueError
        """
        # Given
        x = np.array([1, 2, 3])
        y = np.array([4, 5])  # Different length than x
        
        # When/Then
        with pytest.raises(ValueError):
            fitter._generate_bayesian_initial_guess(fitter._exponential, x, y)
    
    def test_bayesian_initial_guess_with_different_num_samples(self, fitter, test_data):
        """
        What: Test the function's behavior with different num_samples settings
        Given: Same data but CurveFitter instances with different num_samples
        When: We call _generate_bayesian_initial_guess on both instances
        Then: Both succeed and potentially return different results
        """
        # Given
        x = test_data
        y = fitter._exponential(x, 2, 3, 4)
        fitter_small = CurveFitter(num_samples=50)
        fitter_large = CurveFitter(num_samples=200)
        
        
        result_small = fitter_small._generate_bayesian_initial_guess(fitter._exponential, x, y)
        result_large = fitter_large._generate_bayesian_initial_guess(fitter._exponential, x, y)
        
        # Then
        assert len(result_small) == len(result_large)



