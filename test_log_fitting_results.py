# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:49:22 2025

@author: Ele_p
"""
import pytest
from Calibration_EBT3 import CurveFitter
import logging

def test_log_fitting_results_typical_case_polynomial(caplog):
    """
    Test the log of function log_fitting_results for polynomial
    GIVEN: a polynomial fitting result,
    WHEN: log_fitting_results is called,
    THEN: it should log an info message "'Fitting Results Summary:', Polynomial Degree 2: score = 6.34e-01 , mse =  5.26e-01".
    """
    # Arrange: create an instance and a fitting result with a failed polynomial function.
    fitter = CurveFitter()
    fitting_results = [{
        'function': 'polynomial_degree_2',
        'metrics': {
            'mse': 0.526,
            'score': 0.634,
            
        },
        'success': True
    }]

    # Act: call the method while capturing logs from the logger
    with caplog.at_level(logging.INFO,  logger="CurveFitter"):
        fitter.log_fitting_results(fitting_results)

    # Assert: verify that a WARNING log was emitted with the expected message.
    info_logs = [record.message for record in caplog.records if record.levelname == "INFO"]
       # Assert: verify that both parts of the log message are captured
    assert any("Fitting Results Summary:" in msg for msg in info_logs)
    assert any("Polynomial Degree 2: score = 6.34e-01, mse = 5.26e-01" in msg for msg in info_logs)
    
def test_log_fitting_results_typical_case_non_linear_function(caplog):
    """
    Test the log of function log_fitting_results for non-linear function
    GIVEN: a polynomial fitting result,
    WHEN: log_fitting_results is called,
    THEN: it should log an info message "'Fitting Results Summary:', Exponential: score = 6.34e-01 , mse =  5.26e-01".
    """
    # Arrange: create an instance and a fitting result with a failed polynomial function.
    fitter = CurveFitter()
    fitting_results = [{
        'function': '_exponential',
        'metrics': {
            'mse': 0.526,
            'score': 0.634,
            
        },
        'success': True
    }]

    # Act: call the method while capturing logs from the logger
    with caplog.at_level(logging.INFO,  logger="CurveFitter"):
        fitter.log_fitting_results(fitting_results)

    # Assert: verify that a WARNING log was emitted with the expected message.
    info_logs = [record.message for record in caplog.records if record.levelname == "INFO"]
       # Assert: verify that both parts of the log message are captured
    assert any("Fitting Results Summary:" in msg for msg in info_logs)
    assert any("Exponential: score = 6.34e-01, mse = 5.26e-01" in msg for msg in info_logs)


def test_log_fitting_results_failed_polynomial(caplog):
    """
    Test the warning log of function log_fitting_results for failed polynomial
    GIVEN: a failed polynomial fitting result,
    WHEN: log_fitting_results is called,
    THEN: it should log a warning message "Polynomial Degree 2: Fitting Failed".
    """
    # Arrange: create an instance and a fitting result with a failed polynomial function.
    fitter = CurveFitter()
    fitting_results = [{
        'function': 'polynomial_degree_2',
        'metrics': None,
        'success': False
    }]

    # Act: call the method while capturing logs from the logger
    with caplog.at_level(logging.INFO,  logger="CurveFitter"):
        fitter.log_fitting_results(fitting_results)

    # Assert: verify that a WARNING log was emitted with the expected message.
    warning_logs = [record.message for record in caplog.records if record.levelname == "WARNING"]
    assert "Polynomial Degree 2: Fitting Failed" in warning_logs


def test_log_fitting_results_failed_non_polynomial(caplog):
    """
    Test the warning log of function log_fitting_results for failed non-linear function
    GIVEN: a failed non-polynomial fitting result,
    WHEN: log_fitting_results is called,
    THEN: it should log a warning message "Exponential: Fitting Failed".
    """
    # Arrange: create an instance and a fitting result with a failed non-polynomial function.
    fitter = CurveFitter()
    fitting_results = [{
        'function': 'exponential',
        'metrics': None,
        'success': False
    }]

    # Act: call the method while capturing logs from the logger.
    with caplog.at_level(logging.INFO,  logger="CurveFitter"):
        fitter.log_fitting_results(fitting_results)

    # Assert: verify that a WARNING log was emitted with the expected message.
    warning_logs = [record.message for record in caplog.records if record.levelname == "WARNING"]
    assert "Exponential: Fitting Failed" in warning_logs