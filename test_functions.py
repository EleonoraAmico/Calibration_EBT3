# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:32:33 2025

@author: Ele_p
"""

import numpy as np
from Calibration_EBT3 import CurveFitter
import pytest
from hypothesis import given, strategies as st
from hypothesis import given, assume

# Tests for individual curve functions
def test_exponential_increasing_basic():
    """Tests that _exponential function produces expected output for simple input
    
    GIVEN: A CurveFitter instance and simple x values with known parameters
    WHEN: The _exponential function is called
    THEN: The output matches expected values and shape
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    a, b, c = 1.0, 1.0, 0.0
    
    result = fitter._exponential(x, a, b, c)
    
    # Check shape
    assert result.shape == x.shape
    # Check values are finite
    assert np.all(np.isfinite(result))
    # Check increasing trend for positive b
    assert np.all(np.diff(result) > 0)
    
def test_exponential_decreasing_basic():
    """Tests that _exponential function produces expected output for simple input
    
    GIVEN: A CurveFitter instance and simple x values with known parameters
    WHEN: The _exponential function is called
    THEN: The output matches expected values and shape
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    a, b, c = -1.0, 1.0, 0.0
    
    result = fitter._exponential(x, a, b, c)
    
    # Check shape
    assert result.shape == x.shape
    # Check values are finite
    assert np.all(np.isfinite(result))
    # Check increasing trend for positive b
    assert np.all(np.diff(result) < 0)

def test_exponential_with_offset_decreasing_trend():
    """Tests that _exponential function produces expected output
    
    GIVEN: A CurveFitter instance and simple x values with known parameters
    WHEN: The _exponential function is called
    THEN: The output shows decreasing trend and proper shape
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    a, b, c = -1.0, 1.0, 2.0
    
    result = fitter._exponential(x, a, b, c)
    
    # Check shape
    assert result.shape == x.shape
    # Check values are finite
    assert np.all(np.isfinite(result))
    # Check decreasing trend
    assert np.all(np.diff(result) < 0)

def test_exponential_with_offset_coefficient_b_equal_to_zero():
    """Tests that _exponential function produces expected output
    
    GIVEN: A CurveFitter instance and simple x values with known parameters
    WHEN: The _exponential function is called
    THEN: The output shows decreasing trend and proper shape
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    a, b, c = -1.0, 0, 2.0
    
    with pytest.raises(ValueError):
        result = fitter._exponential(x, a, b, c)
        
def test_exponential_with_offset_coefficient_a_equal_to_zero():
    """Tests that _exponential function produces expected output
    
    GIVEN: A CurveFitter instance and simple x values with known parameters
    WHEN: The _exponential function is called
    THEN: The output shows decreasing trend and proper shape
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    a, b, c = 0.0, 1, 2.0
    
    with pytest.raises(ValueError):
        result = fitter._exponential(x, a, b, c)
    
def test_exponential_with_offset_increasing_trend():
    """Tests that _exponential_with_offset function produces expected output
    
    GIVEN: A CurveFitter instance and simple x values with known parameters
    WHEN: The _exponential_with_offset function is called
    THEN: The output shows decreasing trend and proper shape
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    a, b, c = 1.0, 1.0, 2.0
    
    result = fitter._exponential(x, a, b, c)
    
    # Check shape
    assert result.shape == x.shape
    # Check values are finite
    assert np.all(np.isfinite(result))
    # Check decreasing trend
    assert np.all(np.diff(result) > 0)

def test_exponential_x_scaling():
    """Tests that x values are correctly scaled to prevent overflow
    
    GIVEN: A CurveFitter instance and x values of different magnitudes
    WHEN: The exponential function is called
    THEN: The function handles different scales of x values appropriately
    """
    fitter = CurveFitter()
    x1 = np.array([0, 1, 2])
    x2 = x1 * 1000  # Much larger values
    
    # Same parameters for both calls
    a, b, c = 1.0, 1.0, 0.0
    
    result1 = fitter._exponential(x1, a, b, c)
    result2 = fitter._exponential(x2, a, b, c)
    
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
def test_exponential_bounds(x, a, b, c):
    """Tests that the exponential function produces finite values.
    
    GIVEN: Random arrays of x values and parameters
    WHEN: The exponential function is called
    THEN: All output values should be finite
    """
    fitter = CurveFitter()
    x_arr = np.array(x)
    result = fitter._exponential(x_arr, a, b, c)
    
    assert np.all(np.isfinite(result)), "All outputs should be finite"
    
def test_combination_of_exponential_basic():
    """Tests that _double_exponential function produces expected output
    
    GIVEN: A CurveFitter instance and simple x values with known parameters
    WHEN: The _double_exponential function is called
    THEN: The output has correct shape and finite values
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    a, b, c, d, e, f= 1.0, 1.0, 1.0, 2.0, 1.0, 2.0
    
    result = fitter._combination_of_exponential(x, a, b, c, d, e, f)
    
    assert result.shape == x.shape
    assert np.all(np.isfinite(result))
    
@given(
    x=st.lists(st.floats(min_value=0, max_value=65535), min_size=1),
    a=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude a=0
    b=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude b=0
    c=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude c=0
    d=st.floats(min_value=-10, max_value=10).filter(lambda x: x != 0),  # Exclude d=0
    e=st.floats(min_value=-10, max_value=10),
    f=st.floats(min_value=-10, max_value=10)
    
)
def test_combination_of_exponential_bounds(x, a, b, c, d, e, f):
    """Tests that the exponential function produces finite values.
    GIVEN: Random arrays of x values and parameters
    WHEN: The exponential function is called
    THEN: All output values should be finite
    """
    fitter = CurveFitter()
    x_arr = np.array(x)
    result = fitter._combination_of_exponential(x_arr, a, b, c, d, e, f)
    
    assert np.all(np.isfinite(result)), "All outputs should be finite"

@pytest.mark.parametrize(
    "a, b, c, d, e, f",
    [
        (0, 1.0, 1.0, 1.0, 1.0, 1.0),  # a is zero
        (1.0, 0, 1.0, 1.0, 1.0, 1.0),  # b is zero
        (1.0, 1.0, 0, 1.0, 1.0, 1.0),  # c is zero
        (1.0, 1.0, 1.0, 0, 1.0, 1.0),  # d is zero
    ]
)
def test_combination_of_exponential_with_zero_parameters(a, b, c, d, e, f):
    """Tests that _double_exponential raises ValueError for zero parameters.
    
    GIVEN: A CurveFitter instance and simple x values with at least one zero parameter
    WHEN: The _double_exponential function is called
    THEN: A ValueError is raised, as zero parameters are invalid for the function.
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="Parameters 'a', 'b', 'c', and 'd' must not be zero"):
        result = fitter._combination_of_exponential(x, a, b, c, d, e, f)

def test_combination_of_exponential_edge_case():
    """Tests the exponential difference function with extreme parameter values.
    
    GIVEN: An array of x values and large parameter values that could cause overflow
    WHEN: The exponential difference function is called with these extreme values
    THEN: The function should still return finite values due to its clipping mechanism,
          without any NaN or infinite values
    """
    fitter = CurveFitter()
    x = np.array([0, 1000, -1000])
    result = fitter._combination_of_exponential(x, 1000.0, 1000.0, -1000.0, -1000.0, 1000.0, -1000.0)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    
# Tests for rational function
def test_rational_typical():
    """Tests the rational function with typical input values.
    
    GIVEN: An array of positive x values and typical parameters
    WHEN: The rational function is called
    THEN: The function should return expected rational function values without any singularities
    """
    fitter = CurveFitter()
    x = np.linspace(1, 10, 50)
    result = fitter._generalized_rational(x, 1.0, 2.0, 3.0, 0.0, 0.0)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    

def test_rational_zero_denominator():
    """Tests the rational function with inputs that could make denominator zero.
    
    GIVEN: An x value that would make the denominator zero
    WHEN: The rational function is called
    THEN: The function should handle the singularity appropriately (either by raising
          an exception or returning infinity)
    """
    fitter = CurveFitter()
    x = np.array([3.0])
    d = -3.0  # This will make denominator zero when x = 3
    
    with pytest.raises(ValueError):
        result = fitter._generalized_rational(x, 1.0, 2.0, 1, d, 1.0)
        
        
# Tests for hyperbolic growth
def test_hyperbolic_growth_typical():
    """Tests the hyperbolic growth function with typical input values on the
    generalized rational function
    
    GIVEN: An array of positive x values and positive parameters
    WHEN: The hyperbolic growth function is called
    THEN: The function should return expected saturation curve values
    """
    fitter = CurveFitter()
    x = np.linspace(0, 10, 50)
    result = fitter._generalized_rational(x, 0.0, 2.0, 0.5, 1.0, 1.0)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
    assert np.all(np.diff(result) > 0)
    # Test for saturation by checking if consecutive differences decrease
    differences = np.diff(result)
    assert np.all(np.diff(differences) < 0), "Function should show saturating behavior"

def test_hyperbolic_growth_zero_input():
    """Tests the hyperbolic growth function with zero input.
    
    GIVEN: A zero x value input
    WHEN: The hyperbolic growth function is called
    THEN: The function should return zero (as x=0 is a fixed point)
    """
    fitter = CurveFitter()
    x = np.array([0.0])
    result = fitter._generalized_rational(x, 0.0, 2.0, 5, 1.0, 0.0)
    
    assert result[0] == 0.0

# Property-based test example using Hypothesis
@given(x=st.lists(st.floats(min_value=0.1, max_value=65535), min_size=1),
       a=st.just(0),
       b=st.floats(min_value=0.1, max_value=10),
       c=st.floats(min_value=0.1, max_value=10).filter(lambda x: x != 0), #Exclude c=0
       d=st.just(1),
       e=st.floats(min_value=0.1, max_value=10))
def test_hyperbolic_growth_properties(x, a, b, c, d, e):
    """Tests general properties of the hyperbolic growth function using property-based testing.
    
    GIVEN: Random arrays of positive x values and positive parameters
    WHEN: The hyperbolic growth function is called
    THEN: The function should maintain its mathematical properties:
          - Output should be monotonically increasing
          - Output should be bounded for large x
    """
    fitter = CurveFitter()
    x_arr = np.array(x)
    result = fitter._generalized_rational(x_arr, a, b, c, d, e)
    
    # Test monotonicity for sorted input
    x_sorted = np.sort(x_arr)
    result_sorted = fitter._generalized_rational(x_sorted, a, b, c, d, e)
    assert all(np.diff(result_sorted) >= -1e-10)  # Allow small numerical errors
    
def test_generalized_rational_error_conditions():
    """Tests error handling in generalized rational function
    
    GIVEN: Invalid parameters or inputs that would cause division by zero
    WHEN: The generalized rational function is called
    THEN: Appropriate ValueError exceptions should be raised
    """
    fitter = CurveFitter()
    x = np.array([1.0, 2.0, 3.0])
    
    # Test c = 0 condition
    with pytest.raises(ValueError, match="Parameter 'c' must not be zero"):
        fitter._generalized_rational(x, 1.0, 1.0, 0.0, 1.0, 0.0)
    
    # Test denominator = 0 condition
    with pytest.raises(ValueError, match="Invalid input: denominator would be zero"):
        # If c = 1 and d = -2, denominator will be zero when x = 2
        fitter._generalized_rational(x, 1.0, 1.0, 1.0, -2.0, 0.0)

def test_generalized_rational_offset_case():
    """Tests the special case of rational function with offset
    
    GIVEN: Parameters b=1, e=0 to create (a + x)/(cx + d) form
    WHEN: The generalized rational function is called
    THEN: Function should return expected rational function values
    """
    fitter = CurveFitter()
    x = np.array([1.0, 2.0, 3.0])
    
    # Set b=1, e=0, and some arbitrary values for other parameters
    a = 2.0
    c = 0.5
    d = 1.0
    result = fitter._generalized_rational(x, a, 1.0, c, d, 0.0)
    
    # Calculate expected values manually
    expected = (a + x) / (c * x + d)
    np.testing.assert_array_almost_equal(result, expected)

def test_generalized_rational_edge_cases():
    """Tests edge cases and boundary conditions for generalized rational function
    
    Tests:
    1. Values approaching denominator zero
    2. Very large x values (asymptotic behavior)
    3. Very small x values
    4. Special parameter combinations
    5. Almost-zero c parameter
    """
    fitter = CurveFitter()
    
    def test_near_zero_denominator():
        """Test behavior near points where denominator approaches zero"""
        # For c=1, d=-2, denominator is zero at x=2
        x = np.array([1.99, 2.01])  # Points very close to x=2
        with pytest.raises(ValueError, match="Invalid input: denominator would be zero"):
            fitter._generalized_rational(x, a=1.0, b=1.0, c=1.0, d=-2.0, e=0.0)

    def test_asymptotic_behavior():
        """Test behavior as x approaches infinity"""
        x = np.array([1e6, 1e7])
        result = fitter._generalized_rational(x, a=1.0, b=2.0, c=1.0, d=1.0, e=0.0)
        
        # As x→∞, f(x) → b/c + e
        expected_asymptote = 2.0/1.0 + 0.0
        np.testing.assert_array_almost_equal(result, [expected_asymptote, expected_asymptote], decimal=4)

    def test_very_small_values():
        """Test behavior with very small x values"""
        x = np.array([1e-10, 1e-20])
        result = fitter._generalized_rational(x, a=1.0, b=1.0, c=1.0, d=1.0, e=0.0)
        
        # As x→0, f(x) → a/d + e
        expected_limit = 1.0/1.0 + 0.0
        np.testing.assert_array_almost_equal(result, [expected_limit, expected_limit], decimal=4)

    def test_almost_zero_c():
        """Test behavior when c is very close to zero"""
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Parameter 'c' must not be zero"):
            fitter._generalized_rational(x, a=1.0, b=1.0, c=0.0, d=1.0, e=0.0)
        
        # Test with very small c
        very_small_c = 1e-10
        result = fitter._generalized_rational(x, a=1.0, b=1.0, c=very_small_c, d=1.0, e=0.0)
        assert np.all(np.isfinite(result)), "Function should handle very small c values"
        
    def test_extreme_parameters():
        """Test behavior with extreme parameter values"""
        x = np.array([1.0, 2.0])
        
        # Very large parameters
        result = fitter._generalized_rational(x, a=1e6, b=1e6, c=1e6, d=1e6, e=1e6)
        assert np.all(np.isfinite(result)), "Function should handle large parameters"
        
        # Very small (but valid) parameters
        result = fitter._generalized_rational(x, a=1e-6, b=1e-6, c=1e-6, d=1e-6, e=1e-6)
        assert np.all(np.isfinite(result)), "Function should handle small parameters"

def test_log_function():
    fitter = CurveFitter()
    
    def test_basic_behavior():
        """Test basic function behavior with typical inputs"""
        x = np.array([1.0, 2.0, 3.0])
        result = fitter._log_function(x, a=1.0, b=1.0, c=1.0)
        
        # Check output type and shape
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape
        
        # Check if results are finite
        assert np.all(np.isfinite(result))
        
        # Verify the function formula
        expected = np.log((x + 1.0 + 1e-10)/(1.0 + x + 1e-10)) - 1.0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_edge_cases():
        """Test behavior with edge cases"""
        # Test with zero
        x = np.array([0.0])
        result = fitter._log_function(x, a=1.0, b=1.0, c=1.0)
        assert np.isfinite(result)  # Should handle zero gracefully
        
        # Test with very large values
        x = np.array([1e6])
        result = fitter._log_function(x, a=1.0, b=1.0, c=1.0)
        assert np.isfinite(result)  # Should handle large values
    
    def test_parameter_sensitivity():
        """Test sensitivity to parameter changes"""
        x = np.array([1.0])
        
        # Test different parameter combinations
        result1 = fitter._log_function(x, a=1.0, b=1.0, c=1.0)
        result2 = fitter._log_function(x, a=2.0, b=1.0, c=1.0)
        assert result1 != result2  # Should be sensitive to 'a' parameter
        
        result3 = fitter._log_function(x, a=1.0, b=2.0, c=1.0)
        assert result1 != result3  # Should be sensitive to 'b' parameter
        
        result4 = fitter._log_function(x, a=1.0, b=1.0, c=2.0)
        assert result1 != result4  # Should be sensitive to 'c' parameter
    
    def test_input_types():
        """Test different input types"""
        # Test list input
        result = fitter._log_function([1.0, 2.0], a=1.0, b=1.0, c=1.0)
        assert isinstance(result, np.ndarray)
        
        # Test single float input
        result = fitter._log_function(1.0, a=1.0, b=1.0, c=1.0)
        assert isinstance(result, np.ndarray)
        
    def test_log_function_warnings():
        """Test that appropriate warnings are raised for invalid values"""
        fitter = CurveFitter()
        
        # Test with negative values that should trigger warning
        x = np.array([-1.0, -2.0])
        
        # Use pytest's warning catching mechanism
        with pytest.warns(UserWarning, match="Invalid values found at x positions:"):
            result = fitter._log_function(x, a=1.0, b=1.0, c=2.0)
        
        # Verify the result is NaN for invalid values
        assert np.all(np.isnan(result))
        
def test_generalized_polynomial():
    """Test suite for polynomial scaling function"""
    
    fitter=CurveFitter()

    def test_zero_input():
        """
        GIVEN: Zero input with various parameters
        WHEN: The polynomial scaling function is applied
        THEN: The output should be zero
        """
        assert fitter._generalized_polynomial(0, 1, 1, 2) == 0
        assert fitter._generalized_polynomial(0, -1, 5, 3) == 0
        assert fitter._generalized_polynomial(0, 10, -3, 4) == 0

    def test_unity_scaling():
        """
        GIVEN: Input x=1 with various parameters
        WHEN: The polynomial scaling function is applied
        THEN: The output should be the sum of parameters a and b
        """
        assert fitter._generalized_polynomial(1, 2, 3, 2) == 5  # 2*1 + 3*1^2
        assert fitter._generalized_polynomial(1, -1, 1, 3) == 0  # -1*1 + 1*1^3

    @given(x=st.floats(min_value=-1e3, max_value=1e3),
           a=st.floats(min_value=-1e2, max_value=1e2),
           b=st.floats(min_value=-1e2, max_value=1e2))
    def test_linear_case(x, a, b):
        """
        GIVEN: Random inputs with r=1
        WHEN: The polynomial scaling function is applied
        THEN: The function should behave linearly with slope (a+b)
        """
        assume(not np.isnan(x) and not np.isnan(a) and not np.isnan(b))
        result = fitter._generalized_polynomial(x, a, b, 1)
        expected = x * (a + b)
        assert np.abs(result - expected) < 1e-10
    
    @given(x=st.floats(min_value=0, max_value=1e3), # Only need positive x since we'll test symmetry
           a=st.floats(min_value=-1e2, max_value=1e2),
           b=st.floats(min_value=-1e2, max_value=1e2))

    def test_parabolic_symmetry( x, a, b):
        """
        GIVEN: Random positive inputs with r=2
        WHEN: The polynomial scaling function is evaluated at x and -x
        THEN: The function should display parabolic symmetry properties:
              - If a=0: f(-x) = f(x) (symmetric about y-axis)
              - If b=0: f(-x) = -f(x) (antisymmetric about origin)
              - In general: f(-x) = (-a)x + b*x^2
        """
        assume(not np.isnan(x) and not np.isnan(a) and not np.isnan(b))
        
        # Evaluate function at x and -x
        f_x = fitter._generalized_polynomial(x, a, b, 2)    # a*x + b*x^2
        f_minus_x = fitter._generalized_polynomial(-x, a, b, 2)  # a*(-x) + b*(-x)^2
        
        # Test symmetry properties
        if abs(a) < 1e-10:  # When a ≈ 0
            # Pure quadratic should be symmetric about y-axis
            assert np.abs(f_x - f_minus_x) < 1e-10
        elif abs(b) < 1e-10:  # When b ≈ 0
            # Pure linear should be antisymmetric about origin
            assert np.abs(f_x + f_minus_x) < 1e-10
        else:
            # General case: f(-x) = -ax + bx^2
            expected_symmetry = -a*x + b*x**2
            assert np.abs(f_minus_x - expected_symmetry) < 1e-10
    
    @given(x=st.floats(min_value=1e3, max_value=1e3).filter(lambda x: x != 0),
           a=st.floats(min_value=-1e2, max_value=1e2),
           b=st.floats(min_value=-1e2, max_value=1e2),
           r=st.floats(min_value=0, max_value=1e2))
    def test_polynomial_case(x, a, b, r):
        """
        GIVEN: Random inputs
        WHEN: The polynomial scaling function is applied
        THEN:  The output should be correctly computed for each element
        """
        assume(not np.isnan(x) and not np.isnan(a) and not np.isnan(b) and not np.isnan(r))
        result = fitter._generalized_polynomial(x, a, b, r)
        expected = a*x + b*x**r
        assert np.abs(result - expected) < 1e-10
    
    def test_polynomial_warning_negative_r():
        """Test warning generation for zero values with negative power
        
        GIVEN: Input array containing zero and negative power r
        WHEN: The generalized polynomial function is called  
        THEN: A warning is raised for the zero value positions
        """
        x = np.array([1.0, 0.0, 3.0])
        a, b = 2.0, 3.0
        r = -2.0
        
        with pytest.warns(Warning) as warning_info:
            result = fitter._generalized_polynomial(x, a, b, r)
            assert len(warning_info) == 1
            assert "Invalid values found at x positions: [1]" in str(warning_info[0].message)
        
    def test_polynomial_calculation_negative_r():
        """Test polynomial calculation with negative power on valid inputs
        
        GIVEN: Non-zero input values and negative power r
        WHEN: The generalized polynomial function is called
        THEN: The function correctly computes a*x + b*x^r
        """
        x = np.array([1.0, 2.0, 3.0])
        a, b = 2.0, 3.0
        r = -2.0
        
        # Expected result: 2x + 3x^(-2)
        expected = 2.0 * x + 3.0 * x**(-2.0)
        
        result = fitter.generalized_polynomial(x, a, b, r)
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_polynomial_array_shape():
        """Test output shape matches input shape
        
        GIVEN: Input array of specific shape
        WHEN: The generalized polynomial function is called
        THEN: The output array has the same shape as input
        """
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        a, b = 2.0, 3.0
        r = 2.0
        
        result = fitter.generalized_polynomial(x, a, b, r)
        
        assert result.shape == x.shape


    def test_array_input():
        """
        GIVEN: Array input
        WHEN: The polynomial scaling function is applied
        THEN: The output should be correctly computed for each element
        """
        x = np.array([0, 1, 2])
        result = fitter._generalized_polynomial(x, 1, 2, 2)
        expected = np.array([0, 3, 10])  # [1*0 + 2*0^2, 1*1 + 2*1^2, 1*2 + 2*2^2]
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize("x,a,b,r,expected", [
        (2, 1, 1, 2, 6),    # 1*2 + 1*2^2
        (3, 2, -1, 2, -3),  # 2*3 + (-1)*3^2
        (-1, 1, 1, 3, -2)   # 1*(-1) + 1*(-1)^3
    ])
    def test_specific_values(x, a, b, r, expected):
        """
        GIVEN: Specific input values and parameters
        WHEN: The polynomial scaling function is applied
        THEN: The output should match pre-calculated results
        """
        result = fitter._generalized_polynomial(x, a, b, r)
        assert abs(result - expected) < 1e-10


        
        
        
        



