# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:19:33 2025

@author: Ele_p
"""

import pytest
import numpy as np
from Calibration_EBT3 import CurveFitter, ProcessingMode



def test_validate_data_typical_case():
    """Tests that validate_data accepts valid input data within normal ranges
    
    GIVEN: A CurveFitter instance and arrays with values within valid ranges
    WHEN: The validate_data method is called with these arrays
    THEN: The method returns True indicating valid data
    """
    fitter = CurveFitter()
    x = np.array([0, 1000, 30000, 65535])
    y = np.array([0, 10, 25, 49])
    
    assert fitter._validate_data(x, y) == True
    
def test_validate_data_missing_y():
    """Tests that validate_data raises TypeError when y parameter is missing
    
    GIVEN: A CurveFitter instance and only x values array
    WHEN: The validate_data method is called without y parameter
    THEN: The method raises TypeError indicating missing required argument
    """
    fitter = CurveFitter()
    x = np.array([0, 1000, 30000, 65535])
    
    with pytest.raises(TypeError, match=r"validate_data\(\) missing 1 required positional argument: 'y'"):
        fitter._validate_data(x)
    
def test_validate_data_none_values():
    """Tests that validate_data raises ValueError when inputs are None
    
    GIVEN: A CurveFitter instance and None values for x or y
    WHEN: The validate_data method is called
    THEN: The method raises ValueError
    """
    fitter = CurveFitter()
    x = np.array([0, 1000, 30000])
    
    with pytest.warns(UserWarning, match="x array cannot be None"):
        fitter._validate_data(None, x)
    
    with pytest.warns(UserWarning, match="y array cannot be None"):
        fitter._validate_data(x, None)

def test_validate_data_empty_arrays():
    """Tests that validate_data raises ValueError when inputs are empty
    
    GIVEN: A CurveFitter instance and empty arrays
    WHEN: The validate_data method is called
    THEN: The method raises ValueError
    """
    fitter = CurveFitter()
    
    with pytest.warns(UserWarning):
        fitter._validate_data(np.array([]), np.array([1, 2, 3]))
    
    with pytest.warns(UserWarning):
        fitter._validate_data(np.array([1, 2, 3]), np.array([]))

def test_validate_data_x_range():
    """Tests that validate_data returns False when x values are out of range
    
    GIVEN: A CurveFitter instance and x values outside [0, 65535]
    WHEN: The validate_data method is called
    THEN: The method returns False
    """
    fitter = CurveFitter()
    y = np.array([1, 2, 3])
    
    # Test x values below 0
    x_below = np.array([-1, 0, 1000])
    with pytest.warns(UserWarning, match="x values must be between 0 and 65535"):
        fitter._validate_data(x_below, y)

    
    # Test x values above 65535
    x_above = np.array([0, 65535, 70000])
    with pytest.warns(UserWarning, match="x values must be between 0 and 65535"):
        fitter._validate_data(x_above, y)

def test_validate_data_y_range():
    """Tests that validate_data returns False when y values are out of range
    
    GIVEN: A CurveFitter instance and y values outside [0, 50]
    WHEN: The validate_data method is called
    THEN: The method returns False
    """
    fitter = CurveFitter()
    x = np.array([0, 1000, 30000])
    
    # Test y values below 0
    y_below = np.array([-1, 0, 10])
    with pytest.warns(UserWarning, match="y values should be between 0 and 50 Gy. Higher doses may lead to inaccurate measurements for EBT3."):
        fitter._validate_data(x, y_below)
    
    # Test y values above 50
    y_above = np.array([0, 50, 51])
    with pytest.warns(UserWarning, match="y values should be between 0 and 50 Gy. Higher doses may lead to inaccurate measurements for EBT3."):
        fitter._validate_data(x, y_above)

def test_validate_data_valid_inputs():
    """Tests that validate_data returns True for valid inputs
    
    GIVEN: A CurveFitter instance and valid x, y arrays
    WHEN: The validate_data method is called
    THEN: The method returns True
    """
    fitter = CurveFitter()
    x = np.array([0, 1000, 65535])
    y = np.array([0, 25, 50])
    
    assert fitter._validate_data(x, y)

@pytest.mark.parametrize("x,y", [
    (np.array([0, 1000, 65535]), np.array([0, 25, 50])),  # Valid case
    (np.array([0]), np.array([0])),  # Minimum valid values
    (np.array([65535]), np.array([50])),  # Maximum valid values
])
def test_validate_data_parametrized(x, y):
    """Tests validate_data with various valid input combinations
    
    GIVEN: A CurveFitter instance and parametrized valid inputs
    WHEN: The validate_data method is called
    THEN: The method returns True for all valid cases
    """
    fitter = CurveFitter()
    assert fitter._validate_data(x, y)
    

def test_process_values_pv_mode():
    """Tests that process_values in PV mode returns unmodified input
    
    GIVEN: A CurveFitter instance and input array
    WHEN: The process_values method is called in PV mode
    THEN: The method returns the input array unchanged
    """
    fitter = CurveFitter()
    x = np.array([1000, 2000, 3000])
    
    result = fitter._process_values(x, mode=ProcessingMode.PV)
    np.testing.assert_array_equal(result, x)

def test_process_values_od_mode():
    """Tests that process_values correctly calculates optical density
    
    GIVEN: A CurveFitter instance and input array
    WHEN: The process_values method is called in OD mode
    THEN: The method returns log10(65535/x) for each value
    """
    fitter = CurveFitter()
    x = np.array([1000, 10000])
    expected = np.log10(65535 / x)
    
    result = fitter._process_values(x, mode=ProcessingMode.OD)
    np.testing.assert_array_almost_equal(result, expected)
        
def test_process_values_od_mode_edge_case_x():
    """Tests that process_values correctly calculates optical density
        for edge case
    
    GIVEN: A CurveFitter instance and input array
    WHEN: The process_values method is called in OD mode
    THEN: The method returns log10(65535/x) for each value
    """
    fitter = CurveFitter()
    x = np.linspace(0, 65535, 20)
    x_PV = fitter._process_values(x, mode=ProcessingMode.PV)
    with pytest.warns(UserWarning, match="Some values in x_values are 0. These values will be ignored "
                      "since the logarithm cannot be computed for zero."):
        x_OD = fitter._process_values(x_PV, mode=ProcessingMode.OD)
    x_PV = x_PV[x_PV > 0]
    expected = np.log10(65535/x_PV)
    np.testing.assert_array_almost_equal(x_OD, expected)
    
def test_process_values_od_mode_edge_case_inverse_function():
    """Tests that process_values correctly calculates optical density 
        for edge case
    
    GIVEN: A CurveFitter instance and input array with edges cases 
    WHEN: The process_values method is called in OD mode
    THEN: x computed by the inverse of the mode OD and the pixel values data are
    the same 
    """
    fitter = CurveFitter()
    x = np.linspace(0, 65535, 20)
    x_PV = fitter._process_values(x, mode=ProcessingMode.PV)
    with pytest.warns(UserWarning, match="Some values in x_values are 0. These values will be ignored "
                      "since the logarithm cannot be computed for zero."):
        x_OD = fitter._process_values(x_PV, mode=ProcessingMode.OD)
    x_PV = x_PV[x_PV > 0]
    expected = (10**x_OD)*x_PV**2/65535
    np.testing.assert_array_almost_equal(x_PV, expected)
    
def test_process_values_net_od_mode():
    """Tests that process_values correctly calculates net optical density
    
    GIVEN: A CurveFitter instance and input arrays including a zero point
    WHEN: The process_values method is called in NET_OD mode
    THEN: The method returns -log10(x/x_zero) where x_zero corresponds to y=0
    """
    fitter = CurveFitter()
    
    x = np.array([1000, 2000, 3000])
    y = np.array([0, 5, 10])  # First point is zero reference
    
    result = fitter._process_values(x, y, mode=ProcessingMode.NET_OD)
    expected = -np.log10(x / x[0])
    np.testing.assert_array_almost_equal(result, expected)

def test_process_values_net_od_missing_y():
    """Tests that process_values raises error when y values are missing in NET_OD mode
    
    GIVEN: A CurveFitter instance and x values without y values
    WHEN: The process_values method is called in NET_OD mode without y values
    THEN: The method raises a ValueError
    """
    fitter = CurveFitter()
    x = np.array([1000, 2000, 3000])
    
    with pytest.raises(ValueError, match="y_values are required for NET_OD mode"):
        fitter._process_values(x, mode=ProcessingMode.NET_OD)

def test_process_values_net_od_no_zero():
    """Tests that process_values raises error when no zero point exists in NET_OD mode
    
    GIVEN: A CurveFitter instance and y values without a zero point
    WHEN: The process_values method is called in NET_OD mode
    THEN: The method raises a ValueError
    """
    fitter = CurveFitter()
    x = np.array([1000, 2000, 3000])
    y = np.array([1, 5, 10])  # No zero point
    
    with pytest.raises(ValueError, match="y_values must contain 0 for NET_OD mode"):
        fitter._process_values(x, y, mode=ProcessingMode.NET_OD)
        

def test_process_values_net_od_no_edge_case():
    """Tests that process_values raises error when no zero point exists in NET_OD mode
    
    GIVEN: A CurveFitter instance and y values without a zero point
    WHEN: The process_values method is called in NET_OD mode
    THEN: The method raises a ValueError
    """
    fitter = CurveFitter()
    x = np.linspace(0, 65535, 20)
    a = 50/65535
    y = - a * x + 50
    with pytest.warns(UserWarning, match="Some values in x_values are 0. These values will be ignored "
                      "since the logarithm cannot be computed for zero."):
        x_net_OD = fitter._process_values(x, y, mode=ProcessingMode.NET_OD)
    
    x_zero = x[np.where(y == 0)[0][0]]
    x = x [x > 0]
    expected =  -np.log10(x / x_zero)
    np.testing.assert_array_almost_equal(expected, x_net_OD)
    
def test_process_values_net_od_x_zero():
    """Tests that process_values raises a ValueError when x is equal to 0 for y equal to 0 in NET_OD mode.
    
    GIVEN: A CurveFitter instance and x with point zero 
    WHEN: The process_values method is called in NET_OD mode
    THEN: The method raises a ValueError
    """
    fitter = CurveFitter()
    x = np.linspace(0, 65535, 20)
    a = 50/65535
    y = a * x  # Zero point
    
    with pytest.warns(UserWarning, match="Some values in x_values are 0. These values will be ignored "
                      "since the logarithm cannot be computed for zero."):
        with pytest.raises(ValueError, match="x at y = 0 must be higher than 0"):
            fitter._process_values(x, y, mode=ProcessingMode.NET_OD)

def test_non_finite_values():
    """Tests that process_values raises error when some points are nan o inf
    GIVEN: Data containing NaN or inf values
    WHEN: validate_data is called
    THEN: Should handle or reject non-finite values appropriately
    """
    fitter = CurveFitter()
    x = np.array([1, 2, np.nan, 4, 5])
    y = np.array([1, 2, 3, np.inf, 5])
    with pytest.warns(UserWarning):
        fitter._validate_data(x,y)
        
def test_invalid_processing_mode():
    """
    Process values with invalid mode
    GIVEN: A CurveFitter instance and array of x values
    WHEN: Processing values with an invalid mode string
    THEN: ValueError should be raised with appropriate message
    """
    fitter = CurveFitter()
    x_values = np.array([1000, 2000, 3000])
    with pytest.raises(ValueError, match="Invalid processing mode: .*"):
        fitter._process_values(x_values, mode= 'invalid') #Intentionally invalid mode

def test_validate_data_invalid_input_types():
    """
    Validate data with invalid types
    GIVEN: A CurveFitter instance and mixed-type input data
    WHEN: Validating a list containing a string value
    THEN: UserWarning should be raised and function should return False
    """
    fitter = CurveFitter()
    # Test with invalid input types
    x = [1, 2, 'a']  #List with a string
    y = [1, 2, 3]
    with pytest.warns(UserWarning, match="Could not convert inputs to numpy arrays"):
        assert not fitter._validate_data(x,y)



        
    
    