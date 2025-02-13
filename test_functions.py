# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:32:33 2025

@author: Ele_p
"""

import numpy as np
from Calibration_EBT3 import CurveFitter, ProcessingMode
import pytest
from hypothesis import given, assume, settings, HealthCheck, strategies as st
import random

# Tests for individual curve functions
class TestExponentialFunctions:
    def setup_method(self):
        """Initialize CurveFitter before each test method"""
        self.fitter = CurveFitter()
        
    def test_exponential_increasing_basic(self):
        """Tests that _exponential function produces expected output for simple input
        
        GIVEN: A CurveFitter instance and simple x values with known parameters
        WHEN: The _exponential function is called
        THEN: The output matches expected values and shape
        """
        
        x = np.array([0, 1, 2])
        a, b, c = 1.0, 1.0, 0.0
        
        result = self.fitter._exponential(x, a, b, c)
        
        # Check shape
        assert result.shape == x.shape
        # Check values are finite
        assert np.all(np.isfinite(result))
        # Check increasing trend for positive b
        assert np.all(np.diff(result) > 0)
    
    def test_exponential_decreasing_basic(self):
        """Tests that _exponential function produces expected output for simple input
        
        GIVEN: A CurveFitter instance and simple x values with known parameters
        WHEN: The _exponential function is called
        THEN: The output matches expected values and shape
        """
        
        x = np.array([0, 1, 2])
        a, b, c = -1.0, 1.0, 0.0
        
        result = self.fitter._exponential(x, a, b, c)
        
        # Check shape
        assert result.shape == x.shape
        # Check values are finite
        assert np.all(np.isfinite(result))
        # Check increasing trend for positive b
        assert np.all(np.diff(result) < 0)
    
    def test_exponential_with_offset_decreasing_trend(self):
        """Tests that _exponential function produces expected output
        
        GIVEN: A CurveFitter instance and simple x values with known parameters
        WHEN: The _exponential function is called
        THEN: The output shows decreasing trend and proper shape
        """
        
        x = np.array([0, 1, 2])
        a, b, c = -1.0, 1.0, 2.0
        
        result = self.fitter._exponential(x, a, b, c)
        
        # Check shape
        assert result.shape == x.shape
        # Check values are finite
        assert np.all(np.isfinite(result))
        # Check decreasing trend
        assert np.all(np.diff(result) < 0)
    
    def test_exponential_with_offset_coefficient_b_equal_to_zero(self):
        """Tests that _exponential function produces expected output
        
        GIVEN: A CurveFitter instance and simple x values with parameter b equal to 0
        WHEN: The _exponential function is called
        THEN: The output is a ValueError becuase b cannot be zero
        """
    
        x = np.array([0, 1, 2])
        a, b, c = -1.0, 0, 2.0
        
        with pytest.raises(ValueError):
            self.fitter._exponential(x, a, b, c)
            
    def test_exponential_with_no_data(self):
        """Tests that _exponential function produces expected output
        
        GIVEN: A CurveFitter instance and simple x values with parameter b equal to 0
        WHEN: The _exponential function is called
        THEN: The output is a ValueError becuase b cannot be zero
        """
        a, b, c = -1.0, 0, 2.0
        x = np.array([])
        with pytest.raises(ValueError):
            self.fitter._exponential(x, a, b, c)
            
    def test_exponential_with_offset_coefficient_a_equal_to_zero(self):
        """Tests that _exponential function produces expected output
        
        GIVEN: A CurveFitter instance and simple x values with known parameters
        WHEN: The _exponential function is called
        THEN: The output shows decreasing trend and proper shape
        """
        x = np.array([0, 1, 2])
        a, b, c = 0.0, 1, 2.0
        
        with pytest.raises(ValueError):
            self.fitter._exponential(x, a, b, c)
        
    def test_exponential_with_offset_increasing_trend(self):
        """Tests that _exponential_with_offset function produces expected output
        
        GIVEN: A CurveFitter instance and simple x values with known parameters
        WHEN: The _exponential_with_offset function is called
        THEN: The output shows decreasing trend and proper shape
        """
        
        x = np.array([0, 1, 2])
        a, b, c = 1.0, 1.0, 2.0
        
        result = self.fitter._exponential(x, a, b, c)
        
        # Check shape
        assert result.shape == x.shape
        # Check values are finite
        assert np.all(np.isfinite(result))
        # Check decreasing trend
        assert np.all(np.diff(result) > 0)
    
    def test_exponential_x_scaling(self):
        """Tests that x values are correctly scaled to prevent overflow
        
        GIVEN: A CurveFitter instance and x values of different magnitudes
        WHEN: The exponential function is called
        THEN: The function handles different scales of x values appropriately
        """
        x1 = np.array([0, 1, 2])
        x2 = x1 * 1000  # Much larger values
        
        # Same parameters for both calls
        a, b, c = 1.0, 1.0, 0.0
        
        result1 = self.fitter._exponential(x1, a, b, c)
        result2 = self.fitter._exponential(x2, a, b, c)
        
        # The results should follow similar patterns despite different x scales
        assert np.all(np.isfinite(result1))
        assert np.all(np.isfinite(result2))
        assert np.all(np.diff(result1) > 0) == np.all(np.diff(result2) > 0)
        
    @given(
        x=st.lists(st.floats(min_value=0, max_value=65535), min_size=1),
        a=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude a=0
        b=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude b=0
        c=st.floats(min_value=-10, max_value=10)
    )
    def test_exponential_bounds(self, x, a, b, c):
        """Tests that the exponential function produces finite values.
        
        GIVEN: Random arrays of x values and parameters
        WHEN: The exponential function is called
        THEN: All output values should be finite
        """
        
        x_arr = np.array(x)
        result = self.fitter._exponential(x_arr, a, b, c)
        
        assert np.all(np.isfinite(result)), "All outputs should be finite"
        
    def test_combination_of_exponential_basic(self):
        """Tests that _double_exponential function produces expected output
        
        GIVEN: A CurveFitter instance and simple x values with known parameters
        WHEN: The _double_exponential function is called
        THEN: The output has correct shape and finite values
        """
        
        x = np.array([0, 1, 2])
        a, b, c, d= 1.0, 1.0, 1.0, 2.0
        
        result = self.fitter._combination_of_exponential(x, a, b, c, d)
        
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
        
    @given(
        x=st.lists(st.floats(min_value=0, max_value=65535), min_size=1),
        a=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude a=0
        b=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude b=0
        c=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude c=0
        d=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude d=0   
    )
    def test_combination_of_exponential_bounds(self,x, a, b, c, d):
        """Tests that the exponential function produces finite values.
        GIVEN: Random arrays of x values and parameters
        WHEN: The exponential function is called
        THEN: All output values should be finite
        """
        x_arr = np.array(x)
        result = self.fitter._combination_of_exponential(x_arr, a, b, c, d)
        
        assert np.all(np.isfinite(result)), "All outputs should be finite"
    
    @pytest.mark.parametrize(
        "a, b, c, d",
        [
            (0, 1.0, 1.0, 1.0),  # a is zero
            (1.0, 0, 1.0, 1.0),  # b is zero
            (1.0, 1.0, 0, 1.0),  # c is zero
            (1.0, 1.0, 1.0, 0),  # d is zero
        ]
    )
    def test_combination_of_exponential_with_zero_parameters(self,a, b, c, d):
        """Tests that _double_exponential raises ValueError for zero parameters.
        
        GIVEN: A CurveFitter instance and simple x values with at least one zero parameter
        WHEN: The _double_exponential function is called
        THEN: A ValueError is raised, as zero parameters are invalid for the function.
        """
        
        x = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="Parameters 'a', 'b', 'c', and 'd' must not be zero"):
            self.fitter._combination_of_exponential(x, a, b, c, d)
    
    def test_combination_of_exponential_edge_case(self):
        """Tests the exponential difference function with extreme parameter values.
        
        GIVEN: An array of x values and large parameter values that could cause overflow
        WHEN: The exponential difference function is called with these extreme values
        THEN: The function should still return finite values due to its clipping mechanism,
              without any NaN or infinite values
        """

        x = np.array([0, 1000, -1000])
        result = self.fitter._combination_of_exponential(x, 1000.0, 1000.0, -1000.0, -1000.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestGeneralizedRational:
    """Test suite for typical cases, edge cases and boundary conditions of generalized rational function"""
    
    
    def setup_method(self):
        """Initialize CurveFitter before each test method"""
        self.fitter = CurveFitter()
    
    # Tests for rational function
    def test_rational_typical(self):
        """Tests the rational function with typical input values.
        
        GIVEN: An array of positive x values and typical parameters
        WHEN: The rational function is called
        THEN: The function should return expected rational function values without any singularities
        """
        x = np.linspace(1, 10, 50)
        result = self.fitter._generalized_rational(x, 1.0, 2.0, 3.0, 0.0, 0.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
            
            
    # Tests for hyperbolic growth
    def test_hyperbolic_growth_typical(self):
        """Tests the hyperbolic growth function with typical input values on the
        generalized rational function
        
        GIVEN: An array of positive x values and positive parameters
        WHEN: The hyperbolic growth function is called
        THEN: The function should return expected saturation curve values
        """

        x = np.linspace(0, 10, 50)
        result = self.fitter._generalized_rational(x, 0.0, 2.0, 0.5, 1.0, 1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.all(np.diff(result) > 0)
        # Test for saturation by checking if consecutive differences decrease
        differences = np.diff(result)
        assert np.all(np.diff(differences) < 0), "Function should show saturating behavior"

    def test_hyperbolic_growth_zero_input(self):
        """Tests the hyperbolic growth function with zero input.
        
        GIVEN: A zero x value input
        WHEN: The hyperbolic growth function is called
        THEN: The function should return zero (as x=0 is a fixed point)
        """

        x = np.array([0.0])
        result = self.fitter._generalized_rational(x, 0.0, 2.0, 5, 1.0, 0.0)
        
        assert result[0] == 0.0

    # Property-based test example using Hypothesis
    @given(x=st.lists(st.floats(min_value=0.1, max_value=65535), min_size=1),
           a=st.just(0),
           b=st.floats(min_value=0.1, max_value=10),
           c=st.floats(min_value=0.1, max_value=10).filter(lambda x: x != 0), #Exclude c=0
           d=st.just(1),
           e=st.floats(min_value=0.1, max_value=10))
    def test_hyperbolic_growth_properties(self,x, a, b, c, d, e):
        """Tests general properties of the hyperbolic growth function using property-based testing.
        
        GIVEN: Random arrays of positive x values and positive parameters
        WHEN: The hyperbolic growth function is called
        THEN: The function should maintain its mathematical properties:
              - Output should be monotonically increasing
              - Output should be bounded for large x
        """

        x_arr = np.array(x)
        
        # Test monotonicity for sorted input
        x_sorted = np.sort(x_arr)
        result_sorted = self.fitter._generalized_rational(x_sorted, a, b, c, d, e)
        assert all(np.diff(result_sorted) >= -1e-10)  # Allow small numerical errors
        
    def test_generalized_rational_error_conditions(self):
        """Tests error handling in generalized rational function
        
        GIVEN: Invalid parameters or inputs that would cause division by zero
        WHEN: The generalized rational function is called
        THEN: Appropriate ValueError exceptions should be raised
        """

        x = np.array([1.0, 2.0, 3.0])
        
        # Test c = 0 condition
        with pytest.raises(ValueError, match="Parameter 'c' must not be zero"):
            self.fitter._generalized_rational(x, 1.0, 1.0, 0.0, 1.0, 0.0)
        
        # Test denominator = 0 condition
        with pytest.raises(ValueError, match="Invalid input: denominator would be zero"):
            # If c = 1 and d = -2, denominator will be zero when x = 2
            self.fitter._generalized_rational(x, 1.0, 1.0, 1.0, -2.0, 0.0)

    def test_generalized_rational_offset_case(self):
        """Tests the special case of rational function with offset
        
        GIVEN: Parameters b=1, e=0 to create (a + x)/(cx + d) form
        WHEN: The generalized rational function is called
        THEN: Function should return expected rational function values
        """

        x = np.array([1.0, 2.0, 3.0])
        
        # Set b=1, e=0, and some arbitrary values for other parameters
        a = 2.0
        c = 0.5
        d = 1.0
        result = self.fitter._generalized_rational(x, a, 1.0, c, d, 0.0)
        
        # Calculate expected values manually
        expected = (a + x) / (c * x + d)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_near_zero_denominator(self):
        """Test behavior near points where denominator approaches zero
            
        GIVEN: An x value that would make the denominator near to zero
        WHEN: The rational function is called
        THEN: The function should not handle the singularity
        """
        # For c=1, d=-2, denominator is zero at x=2
        x = np.array([1.99, 2.01])  # Points very close to x=2 
        self.fitter._generalized_rational(x, a=1.0, b=1.0, c=1.0, d=-2.0, e=0.0)
            
    def test_zero_denominator(self):
        """Test behavior near points where denominator approaches zero
        GIVEN: An x value that would make the denominator zero
        WHEN: The rational function is called
        THEN: The function should handle the singularity appropriately (either by raising
              an exception or returning infinity)
        """
        # For c=1, d=-2, denominator is zero at x=2
        x = np.array([1.50, 2.00, 2.50])  # Point equal to x=2
        with pytest.raises(ValueError, match="Invalid input: denominator would be zero"):
            self.fitter._generalized_rational(x, a=1.0, b=1.0, c=1.0, d=-2.0, e=0.0)
    
    def test_asymptotic_behavior(self):
        """Test the asymptotic behavior of the generalized rational function as x approaches infinity
        GIVEN:
            - A large input array `x = [1e6, 1e7]`
            - The generalized rational function with parameters:a=1.0, b=2.0, c=1.0, d=1.0, e=0.0   
        WHEN: The rational function is called
        THEN:
            - As `x → ∞`, the function should approach the asymptotic limit `b/c + e`
            - The expected asymptote is `2.0 / 1.0 + 0.0 = 2.0`
            - The computed results should be approximately `[2.0, 2.0]` within a precision of 4 decimal places
        """

        x = np.array([1e6, 1e7])
        result = self.fitter._generalized_rational(x, a=1.0, b=2.0, c=1.0, d=1.0, e=0.0)
        
        # As x→∞, f(x) → b/c + e
        expected_asymptote = 2.0/1.0 + 0.0
        np.testing.assert_array_almost_equal(result, [expected_asymptote, expected_asymptote], decimal=4)
    
    def test_very_small_values(self):
        """Test behavior with very small x values

        Given: A small input array `x = [1e-10, 1e-20]` and the generalized rational function with parameters:
                a=2.0, b=1.0, c=1.0, d=1.0, e=0.0
        
        When: The function is evaluated at `x`
        
        Then:
            - As `x → 0`, the function should approach the limit `a/d + e`
            - The expected limit is `2.0 / 1.0 + 0.0 = 1.0`
            - The computed results should be approximately `[2.0, 2.0]` within a precision of 4 decimal places
        """
        x = np.array([1e-10, 1e-20])
        result = self.fitter._generalized_rational(x, a=2.0, b=1.0, c=1.0, d=1.0, e=0.0)
        
        # As x→0, f(x) → a/d + e
        expected_limit = 2.0/1.0 + 0.0
        np.testing.assert_array_almost_equal(result, [expected_limit, expected_limit], decimal=4)
        
    def test_zero_c(self):
        """Test behavior when c is equal to zero
        GIVEN: an input array `x = [1.0, 2.0]` and the generalized rational function with parameters:  
                a=1.0, b=1.0, c=0.0 (invalid_case), d=1.0, e=0.0
       WHEN: the function is evaluated with `c = 0.0`  
    
       THEN: a `ValueError` is raised with the message `"Parameter 'c' must not be zero
       
       """
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Parameter 'c' must not be zero"):
            self.fitter._generalized_rational(x, a=1.0, b=1.0, c=0.0, d=1.0, e=0.0)
    
    def test_almost_zero_c(self):
        """ Test behavior with very small values of parameter 'c'
        GIVEN: the same input array `x = [1.0, 2.0]` and a very small `c = 1e-10`  

        WHEN: the function is evaluated with `c = 1e-10`  
    
        THEN: the function should return finite values  
              and the computed results should not contain `NaN` or `Inf`
        """
        x = np.array([1.0, 2.0])        
        # Test with very small c
        very_small_c = 1e-10
        result = self.fitter._generalized_rational(x, a=1.0, b=1.0, c=very_small_c, d=1.0, e=0.0)
        assert np.all(np.isfinite(result)), "Function should handle very small c values"
    
    def test_extreme_parameters(self):
        """Test behavior with extreme parameter values
  
        GIVEN: an input array `x = [1.0, 2.0]`  
               and the generalized rational function with extreme parameter values  
    
        WHEN: the function is evaluated with very large parameters:  
              a=1e6, b=1e6, c=1e6, d=1e6, e=1e6
              and with very small parameters: 
              a=1e-6, b=1e-6, c=1e-6, d=1e-6, e=1e-6
    
        THEN: the function should return finite values  
              and should not produce `NaN` or `Inf` 
        """
        x = np.array([1.0, 2.0])
        
        # Very large parameters
        result = self.fitter._generalized_rational(x, a=1e6, b=1e6, c=1e6, d=1e6, e=1e6)
        assert np.all(np.isfinite(result)), "Function should handle large parameters"
        
        # Very small (but valid) parameters
        result = self.fitter._generalized_rational(x, a=1e-6, b=1e-6, c=1e-6, d=1e-6, e=1e-6)
        assert np.all(np.isfinite(result)), "Function should handle small parameters"


        
class TestLogFunction:
    
    def setup_method(self):
        self.fitter = CurveFitter()
        
    def test_basic_behavior(self):
        """Test basic function behavior with typical inputs
        GIVEN: an input array `x = [1.0, 2.0, 3.0]` and the logarithmic function with parameters:  
            a=1.0, b=1.0

        WHEN: the function is evaluated with these inputs  
    
        THEN: the result should be a numpy array  
              and its shape should match the shape of `x`  
              and all values should be finite  
              and the computed result should be approximately equal to the expected value:
              `log((x + 1.0 + 1e-10)/(1.0 + x + 1e-10)) - 1.0`
        """
        x = np.array([1.0, 2.0, 3.0])
        result = self.fitter._log_function(x, a=1.0, b=1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
        
        expected = np.log((x + 1.0 + 1e-10)/(1.0 + 1e-10))/1.0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_edge_cases(self):
        """Test behavior with edge cases
        GIVEN: the logarithmic function with parameters: a=1.0, b=1.0 and:
            -an input array `x = [0]` 
            -an input array `x = [1e6]`

        WHEN: the function is evaluated with `x = 0.0`  and `x = 1e6`

        THEN: the result should be finite  
        
        """
        # Test with zero
        x = np.array([0.0])
        result = self.fitter._log_function(x, a=1.0, b=1.0)
        assert np.isfinite(result)  # Should handle zero gracefully
        
        # Test with very large values
        x = np.array([1e6])
        result = self.fitter._log_function(x, a=1.0, b=1.0)
        assert np.isfinite(result)  # Should handle large values
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to parameter changes
        GIVEN: an input array `x = [1.0]`  
           and the logarithmic function with four different initial parameters:  
           1. a=1.0, b=1.0,
           2. a=2.0, b=1.0,
           3. a=1.0, b=2.0
           
        WHEN: the function is evaluated with the different initial parameters

        THEN: the results should not be equal  
              and the function should be sensitive to changes in the parameters   

        """
        x = np.array([1.0])
        
        result1 = self.fitter._log_function(x, a=1.0, b=1.0)
        result2 = self.fitter._log_function(x, a=2.0, b=1.0)
        assert result1 != result2  # Should be sensitive to 'a' parameter
        
        result3 = self.fitter._log_function(x, a=1.0, b=2.0)
        assert result1 != result3  # Should be sensitive to 'b' parameter

    
    def test_log_function_warnings(self):
        """Test that appropriate warnings are raised for invalid values
        GIVEN: an input array `x = [-1.0, -2.0]` and the logarithmic function with parameters:  
                a=1.0, b=1.0

        WHEN: the function is evaluated with negative `x` values  

        THEN: a `UserWarning` should be raised with the message `"Invalid values found at x positions:"`  
              and the result should be `NaN` for the invalid values in `x`
        """
        # Test with negative values that should trigger warning
        x = np.array([-1.0, -2.0])
        
        # Use pytest's warning catching mechanism
        with pytest.warns(UserWarning, match="Invalid values found at x positions:"):
            result = self.fitter._log_function(x, a=1.0, b=1.0)
        
        # Verify the result is NaN for invalid values
        assert np.all(np.isnan(result))
    
    def test_log_function_warnings_a_zero(self):
        """Test that appropriate warnings are raised when parameter `a` is zero
        GIVEN: an input array `x = [1.0, 2.0]` and the logarithmic function with parameters:  
               a = 0.0, b = 1.0
    
        WHEN: the function is evaluated with `a = 0.0`  
    
        THEN: a `UserWarning` should be raised with the message `"Parameter 'a' should not be zero"`  
        """
        # Test with a = 0 that should trigger warning
        x = np.array([1.0, 2.0])
        
        # Use pytest's warning catching mechanism
        with pytest.raises(ValueError, match="Invalid input: denominator would be zero"):
            self.fitter._log_function(x, a=0.0, b=1.0)


        
        
        
        



