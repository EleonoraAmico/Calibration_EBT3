# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:12:06 2024

@author: Ele_p
"""

import pytest
from enum import Enum
from Calibration_EBT3 import ProcessingMode

# Esegui tutti i test
#pytest.main(['-v', '-k', 'test_fitting_error_handling', 'C:/Users/Ele_p/EBT3_calibration/test_polynomial_fit.py'])

#'-p', 'no:xdist'
#pytest.main(['-v', 'C:/Users/Ele_p/EBT3_calibration/test_best_fit.py'])
#pytest.main(['-v', '-k', 'test_linear_fit_hypotesis', 'C:/Users/Ele_p/EBT3_calibration/test_best_fit.py'])

pytest.main(['-v', '--hypothesis-show-statistics', '--hypothesis-verbosity=verbose', 
             '-k', 'test_linear_fit_hypotesis', 
             'C:/Users/Ele_p/EBT3_calibration/test_best_fit.py'])

#pytest.main(['--cov=Calibration_EBT3.polynomial_fit', '--cov-report=term-missing', '-v', 'C:/Users/Ele_p/EBT3_calibration/test_polynomial_fit.py'])
#%%

# if __name__ == '__main__':
#     pytest.main([
#         '--cov=Calibration_EBT3.CurveFitter.polynomial_fit', 
#         '--cov-report=html', 
#         'test_polynomial_fit.py'
#     ])
#testpaths = test_polynomial_fit.py
    #%%
# import pytest
# import sys
# import os

# # Add project directory to Python path
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# if __name__ == '__main__':
#     pytest.main([
#         '--cov=Calibration_EBT3',
#         '--cov-report=html',
#         '--cov-report=term-missing',
#         'test_polynomial_fit.py'
#     ])
