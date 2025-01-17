# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:32:33 2025

@author: Ele_p
"""

import numpy as np
from Calibration_EBT3 import CurveFitter
import pytest
from hypothesis import given, strategies as st

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
@given(x=st.lists(st.floats(min_value=0.1, max_value=655353), min_size=1),
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
    




