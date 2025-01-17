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
    
def test_double_exponential_basic():
    """Tests that _double_exponential function produces expected output
    
    GIVEN: A CurveFitter instance and simple x values with known parameters
    WHEN: The _double_exponential function is called
    THEN: The output has correct shape and finite values
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    a, b, c, d, e, f= 1.0, 1.0, 1.0, 2.0, 1.0, 2.0
    
    result = fitter._double_exponential(x, a, b, c, d, e, f)
    
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
def test_double_exponential_bounds(x, a, b, c, d, e, f):
    """Tests that the exponential function produces finite values.
    
    GIVEN: Random arrays of x values and parameters
    WHEN: The exponential function is called
    THEN: All output values should be finite
    """
    fitter = CurveFitter()
    x_arr = np.array(x)
    result = fitter._double_exponential(x_arr, a, b, c, d, e, f)
    
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
def test_double_exponential_with_zero_parameters(a, b, c, d, e, f):
    """Tests that _double_exponential raises ValueError for zero parameters.
    
    GIVEN: A CurveFitter instance and simple x values with at least one zero parameter
    WHEN: The _double_exponential function is called
    THEN: A ValueError is raised, as zero parameters are invalid for the function.
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="Parameters 'a', 'b', 'c', and 'd' must not be zero"):
        result = fitter._double_exponential(x, a, b, c, d, e, f)

    




