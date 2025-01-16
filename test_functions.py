# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:32:33 2025

@author: Ele_p
"""

import numpy as np
from Calibration_EBT3 import CurveFitter
import pytest

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
    """Tests that _exponential_with_offset function produces expected output
    
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
    """Tests that _exponential_with_offset function produces expected output
    
    GIVEN: A CurveFitter instance and simple x values with known parameters
    WHEN: The _exponential function is called
    THEN: The output shows decreasing trend and proper shape
    """
    fitter = CurveFitter()
    x = np.array([0, 1, 2])
    a, b, c = -1.0, 0, 2.0
    
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


