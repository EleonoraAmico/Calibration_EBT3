# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:59:35 2025

@author: Ele_p
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from enum import Enum
import enum
import warnings
from sklearn.linear_model import Ridge
import logging
import sys
from scipy.optimize import least_squares
import inspect
from numpy.polynomial import Polynomial
from sklearn.model_selection import KFold
import scipy.stats as st
import platform
from typing import Optional

class ProcessingMode(enum.Enum):
    PV = 1  # Pass-through values
    OD = 2  # Optical Density
    NET_OD = 3  # Net Optical Density

# Windows compatibility for colors
if platform.system() == 'Windows':
    try:
        import colorama
        colorama.init()
    except ImportError:
        pass  # Colors won't work on Windows without colorama

class ColorFormatter(logging.Formatter):
    """Custom formatter with colored output"""
    COLOR_CODES = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[31;1m' # Bold Red
    }
    RESET_CODE = '\033[0m'

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
        self._colored = sys.stderr.isatty()  # Only use colors if output is a terminal

    def format(self, record):
        message = super().format(record)
        if self._colored:
            color = self.COLOR_CODES.get(record.levelname, '')
            return f"{color}{message}{self.RESET_CODE}"
        return message

class LoggerUtility:
    @staticmethod
    def create_logger(
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        verbose: bool = False,
        silent: bool = False
    ) -> logging.Logger:
        """
        Create a configured logger with colored console output and optional file output
        
        Args:
            name (str): Logger name
            level (int): Logging level
            log_file (str, optional): Path to log file
            verbose (bool): Enable verbose logging
            silent (bool): Disable all console output
            
        Returns:
            logging.Logger: Configured logger
        """
        # Handle verbosity levels
        if silent:
            level = logging.CRITICAL + 1  # Disable all logging
        elif verbose:
            level = logging.DEBUG

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers to prevent duplicate logs
        logger.handlers.clear()

        # Console handler with color formatting
        if not silent:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            formatter = ColorFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                '%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler (no colors)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                '%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def configure_cli_logging(parser):
        """Add standard logging options to an argparse parser"""
        log_group = parser.add_mutually_exclusive_group()
        log_group.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Enable verbose output (debug level)'
        )
        log_group.add_argument(
            '-q', '--quiet',
            action='store_true',
            help='Disable non-essential output (warning level)'
        )
        log_group.add_argument(
            '-s', '--silent',
            action='store_true',
            help='Disable all console output'
        )
        parser.add_argument(
            '-l', '--log-file',
            metavar='FILE',
            help='Save log output to file'
        )
# class LoggerUtility:
#     @staticmethod
#     def create_logger(name, level=logging.INFO, log_file=None):
#         """
#         Create a configured logger with optional file output
        
#         Args:
#             name (str): Logger name
#             level (int): Logging level
#             log_file (str, optional): Path to log file
        
#         Returns:
#             logging.Logger: Configured logger
#         """
#         logger = logging.getLogger(f"{name}_{id(name)}")
#         logger.setLevel(level)
        
#         # Clear existing handlers to prevent duplicate logs
#         logger.handlers.clear()
        
#         # Console handler
#         console_handler = logging.StreamHandler(sys.stdout)
#         console_handler.setLevel(level)
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         console_handler.setFormatter(formatter)
#         logger.addHandler(console_handler)
        
#         # Optional file handler
#         if log_file:
#             file_handler = logging.FileHandler(log_file)
#             file_handler.setLevel(level)
#             file_handler.setFormatter(formatter)
#             logger.addHandler(file_handler)
        
#         return logger

class CurveFitter:
    def __init__(self, log_level=logging.INFO, log_file=None):
        self.logger = LoggerUtility.create_logger(
            name=f'CurveFitter_{id(self)}', 
            level=log_level, 
            log_file=log_file
        )
        self.fitting_functions = {
            'exponential': self._exponential,
            'combination_of_exponential': self._combination_of_exponential,
            'generalized_rational': self._generalized_rational,
            #'generalized_polynomial': self._generalized_polynomial,
            'log_function': self._log_function
        }
        
        
    def _process_values(self, x_values, y_values=None, mode=ProcessingMode.PV):
        """
        Process x_values according to different modes.
        
        Args:
            x_values: numpy array of input values
            y_values: optional numpy array of y values, needed for NET_OD mode
            mode: ProcessingMode enum specifying the processing type
            
        Returns:
            processed numpy array
        """
        if not isinstance(mode, ProcessingMode):
            raise ValueError(f"Invalid processing mode: {mode}")

        if mode.name == "PV":
            return x_values
        
        elif mode.name == "OD":
            if np.any(x_values == 0):
                warnings.warn(
                    "Some values in x_values are 0. These values will be ignored "
                    "since the logarithm cannot be computed for zero.",
                    UserWarning
                )
                # Filter positive values
                x_values = x_values[x_values > 0]
            return np.log10(65535 / x_values)
        
        elif mode.name == "NET_OD":
            if y_values is None:
                raise ValueError("y_values are required for NET_OD mode")
            # Find x_value where y = 0
            if 0 not in y_values:
                raise ValueError("y_values must contain 0 for NET_OD mode")
            x_zero = x_values[np.where(y_values == 0)[0][0]]
            if np.any(x_values == 0):
                warnings.warn(
                    "Some values in x_values are 0. These values will be ignored "
                    "since the logarithm cannot be computed for zero.",
                    UserWarning
                )
                # Filter positive values
                x_values = x_values[x_values > 0]
            if x_zero == 0:
                raise ValueError("x at y = 0 must be higher than 0")
            return -np.log10(x_values / x_zero)
            
        else:
            raise ValueError(f"Unknown processing mode: {mode}")
            

    def _validate_data(self, x, y):
        """
        Validates input data ranges.
        
        Parameters:
        -----------
        x : array-like
            Input x values that must be between 0 and 65535
        y : array-like
            Input y values that must be between 0 and 50 Gy
        
        Returns:
        --------
        bool
            True if data is valid, False otherwise
        """
        # Check for NaN or inf values
       
        if x is None:
            warnings.warn("x array cannot be None", UserWarning)
            return False
            
        if y is None:
            warnings.warn("y array cannot be None", UserWarning)
            return False
        
            
        if len(x) == 0:
            warnings.warn("x array cannot be empty", UserWarning)
            return False
            
        if len(y) == 0:
            warnings.warn("y array cannot be empty", UserWarning)
            return False
        if len(x) != len(y):
            warnings.warn("y array and x array must have the same length", UserWarning)
            return False
        
        try:
            x_array = np.asarray(x, dtype = float)
            y_array = np.asarray(y, dtype = float)
        except (ValueError, TypeError) as e:
            warnings.warn(f"Could not convert inputs to numpy arrays: {str(e)}", UserWarning)
            return False
        
        if np.any(np.isnan(x_array)) or np.any(np.isinf(x_array)):
            warnings.warn("x array contains NaN or inf values", UserWarning)
            return False
                
        if not np.all((x_array >= 0) & (x_array <= 65535)):
            x_min = np.min(x)
            x_max = np.max(x)
            warnings.warn(f"x min is {x_min} and x max is {x_max}.x values must be between 0 and 65535", UserWarning)
            return False
            
        if not np.all((y_array >= 0) & (y_array <= 50)):
            logging.info("y values should be between 0 and 50 Gy. Higher doses may lead to inaccurate measurements for EBT3.", UserWarning)
 
        return True

    
    def _normalized_input(self, x):
        """
        Normalize the input array to a [0, 1] range.
    
        This internal function scales the input array `x` to prevent overflow issues 
        when used in other computations. It ensures numerical stability by handling
        edge cases like empty arrays or arrays with identical values.
    
        Parameters:
        -----------
        x : array-like
            Input array to be normalized. Must not be empty.
    
        Returns:
        --------
        array-like
            A normalized array scaled to the [0, 1] range. If all elements in `x` 
            are identical, returns an array of zeros.
    
        Notes:
        ------
        This function is not intended for direct use by end users.
        """
        if len(x) == 0:
            raise ValueError("Input array 'x' must not be empty.")
        
        if np.max(x) - np.min(x) == 0:
            x_scaled = np.zeros_like(x)
        else:
            x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x_scaled


    
    def _exponential(self, x, a, b, c):
        """
        Exponential function with scaling and overflow control.
        
        Parameters:
        -----------
        x : array-like
            values on x axis 
        a, b, c : float
            parameters of the exponential function
            
        Returns:
        --------
        array-like
            values computed by the exponential function 
        
        Raises:
        -------

        ValueError
            If 'a' or 'b' are equal to zero, as this would result in a non-exponential trend.
    
        Description:
        ------------
        This function computes an exponential of the form:
        
            f(x) = a * exp(b * normalized_x) + c
            
        where `normalized_x` scales the input `x` to the range [0, 1] for numerical stability. 
        To prevent overflow in the exponential computation, the exponent is clipped to the range [-700, 700].


        """
        x_scaled = self._normalized_input(x)
        if b == 0:
            raise ValueError("Parameter 'b' must not be zero, as this would result in a constant function.")
        if a == 0: 
            raise ValueError("Parameter 'a' must not be zero, as this would result in a constant function.")
        exp_component = np.exp(np.clip(b * x_scaled, -700, 700))  # Clip the exponent range
    
        return a * exp_component + c


    def _combination_of_exponential(self, x, a, b, c, d):
        """
        Generalized combination of exponential function with scaling and overflow control.
        
        Parameters:
        -----------
        x : array-like
            Values on the x-axis.
        a, b, c, d : float
            Parameters of the exponential terms. Their values determine the behavior of the function:
            - Positive values of a and b contribute positively.
            - Negative values of c and d can introduce subtractive or balancing behavior.
            
        Returns:
        --------
        array-like
            Values computed as the sum or difference of two exponential terms.
            
        Description:
        ------------
        This function computes a generalized combination of exponential function:
            f(x) = a * exp(b * normalized_x + e) + c * exp(d * normalized_x + f)
        where `normalized_x` scales the input `x` to the range [0, 1] for numerical stability.
        """
        if any(param == 0 for param in (a, b, c, d)):
            raise ValueError("Parameters 'a', 'b', 'c', and 'd' must not be zero.")
    
        x_scaled = self._normalized_input(x)
        exp_component_one = np.exp(np.clip(b * x_scaled, -700, 700)) 
        exp_component_two = np.exp(np.clip(d * x_scaled, -700, 700))
       # if np.any(np.abs(exp_component_one)) != np.any(np.abs(exp_component_two)):
        return a * exp_component_one + c * exp_component_two

    

    def _generalized_rational(self, x, a, b, c, d, e):
        """
        Generalized rational function with optional scaling, offset, and saturation behavior.
        
        Parameters:
        -----------
        x : array-like
            Input values on the x-axis.
        a, b : float
            Parameters for the numerator (a + b * x).
        c, d : float
            Parameters for the denominator (c * x + d).
        e : float
            Offset added to the function output.
        
        Returns:
        --------
        array-like
            Output values computed as (a + b * x) / (c * x + d) + e.
        
        Raises:
        -------
        ValueError
            If the denominator is zero for any input value.
            If `c = 0`, as this would result in a linear or constant function rather than rational behavior.
        
        Description:
        ------------
        This function computes a generalized rational relationship of the form:
        
            f(x) = (a + b * x) / (c * x + d) + e
        The behavior of the function varies depending on the parameter values:
    
        1. **Pure offset (constant function) and linear function:**
           -c = 0 is forbitten because it leads to a constant denominator, reducing the function to linear or offset behavior.
    
        2. **Hyperbolic growth:**
           - If  a = 0,  b != 0, and  d = 1, the function behaves like a saturation curve:
             f(x) = (b * x) / (c * x + 1) + e.
    
        3. **Rational function with offset:**
           - If  b = 1  and  e = 0 , the function becomes:
             f(x) = (a + x) / (c * x + d).
    
        4. **Fully generalized rational behavior:**
           - For arbitrary values of  a, b, c, d and  e, the function exhibits
             rational growth or decline based on the interaction between the numerator
             and denominator.
            
        """
        
        if c == 0:
            raise ValueError("Parameter 'c' must not be zero. This would result in constant or linear behavior, not a rational function.")
        if b == 0:
            raise ValueError("Parameter 'b' must not be zero. This would result in constant")
        denominator = c * x + d 
        if np.any(denominator == 0):
            raise ValueError("Invalid input: denominator would be zero")
        k = (a + b*x) / denominator
        if all(k == k[0]):
            raise ValueError("Constant")
        
        return ((a + b * x) / denominator) + e


    def _generalized_polynomial(self, x, a, b, r):
        """
        Function representing polynomial scaling.
    
        Parameters:
        -----------
        x : array-like
            Input values on the x-axis.
        a : float
            Linear scaling parameter.
        b : float
            Coefficient for the polynomial term.
        r : float
            Power of the polynomial term.
    
        Returns:
        --------
        array-like
            Output values computed as a * x + b * x**r.
        """
        # Ensure r is a valid exponent
        if not isinstance(r, (int, float)):
            raise ValueError("Exponent r should be an integer or float.")

        
        # if 0 <= r <= 4:
        #     raise ValueError(f"Error for r={r}. Polynomial fit is more accurate for r between 0 and 4")
        # Check if x is a numpy array
        if not isinstance(x, (int, float, np.ndarray)):
            warnings.warn("Input x is not int, float or a numpy array. Converting to numpy array.", UserWarning)
            x = np.array(x)  # Convert to numpy array if it's not
        # Handle negative exponents with zero check
        if r < 0 or r > 4: 
           # Check for invalid values
           valid_input = (x != 0)
           
           if not np.all(valid_input):
               warnings.warn(f"Invalid values found at x positions: {np.where(~valid_input)[0]}")
               eps = 1e-10
               x = x + eps
           # Compute generalized polynomial
           return  a * x + b * x**r
        


    def _log_function(self, x, a, b, c):
        """
        Enhanced logarithmic function with input validation and numerical stability.
        
        Parameters:
        -----------
        x : array-like
            Input values
        a, b, c : float
            Function parameters
                
        Returns:
        --------
        array-like
            Computed logarithmic values
        """
        eps = 1e-10
        x = np.asarray(x, dtype=float)
        
        numerator = x + c
        denominator = b + x
        
        # Single epsilon addition for numerical stability
        ratio =numerator  / (denominator + eps)
        
        # Check for invalid values
        valid_mask = (numerator > 0) & (denominator > 0)
        
        if not np.all(valid_mask):
            warnings.warn(f"Invalid values found at x positions: {np.where(~valid_mask)[0]}")
                
        result = np.full_like(x, np.nan, dtype=float)
        result[valid_mask] = np.log(ratio[valid_mask]) - a
        
        return result

    def _calculate_metrics(self, y_true, y_pred, degree, coefficients, cv_mse = 0, xerr=None):
        """
        Calculate various metrics for model fitting.
        
        Parameters:
        -----------
        y_true : array-like
            Actual y values
        y_pred : array-like
            Predicted y values
        degree : int
            Polynomial degree
        coefficients : array
            Polynomial coefficients
        xerr : array-like, optional
            Measurement errors
        
        Returns:
        --------
        dict
            Calculated metrics
        """
        # Calculate the residuals
        residuals1 = y_true - y_pred
        # Calculate R-squared
        ss_total1 = np.sum((y_true - np.mean(y_true))**2)
        ss_residual1 = np.sum(residuals1**2)
        r_squared1 = 1 - (ss_residual1 / ss_total1)
       
        
        
        # # Chi-squared calculation
        # chi2 = None
        # if xerr is not None:
        #     residuals = y_true - y_pred
        #     chi2 = np.sum((residuals / xerr) ** 2)
        
        # # Degrees of freedom
        
        # dof = len(y_true) - (len(coefficients))
        
        # if degree != 0:
        #     mse = mean_squared_error(y_true, y_pred)
        #     # Coefficient ratio
        #     coeff_ratio = (np.abs(coefficients[0]) / 
        #                     np.max(np.abs(coefficients[1:])) 
        #                     if len(coefficients) > 1 else np.inf)
        #     # Complexity penalty
        #     complexity_penalty = degree * np.log(len(y_true))
        #     n=len(y_true)
        #     num_params = len(coefficients)
        #     aic = n * np.log(mse) + 2 * num_params
        #     bic = n * np.log(mse) + num_params * np.log(n)
        # else: 
        #     n=len(y_true)
        #     if n > 50:
        #         mse = cv_mse
        #     else: 
        mse = mean_squared_error(y_true, y_pred)
        # Coefficient ratio
        coeff_ratio = (np.min(np.abs(coefficients)) / 
                       np.max(np.abs(coefficients)) 
                       if len(coefficients) > 1 else np.inf)
        
        complexity_penalty = (len(coefficients) - 1) * np.log(len(y_true))
        n=len(y_true)
        num_params = len(coefficients)
        aic = n * np.log(mse) + 2 * num_params + 2 * num_params* (num_params - 1 )/ (n-num_params-1)
        bic = n * np.log(mse) + num_params * np.log(n)
        
        # Modified score with multiple criteria
        score = (mse * 
                 (1 + complexity_penalty / len(y_true)) * 
                 (1 + 1/coeff_ratio))
        
        return {
            'r_squared': r_squared1,
            'mse': mse,
            'aic': aic,
            "bic": bic,
            'score': score,
          # 'chi2': chi2,
            #dof': dof,
            'coeff_ratio': coeff_ratio
        }
    
    
    # def _select_best_fit(self, fitting_results):
    #     """
    #     Select the best fit based on metrics
    #     """
    #     successful_fits = [
    #         result for result in fitting_results 
    #         if result.get('success', False)
    #     ]

    #     # No successful fits
    #     if not successful_fits:
    #         self.logger.warning("No successful fits found")
    #         return None, None, None, fitting_results
        
    #     # Sort by score (assuming lower score is better)
    #     try:
    #         best_fit = min(
    #             successful_fits, 
    #             key=lambda x: x.get('metrics', {}).get('score', float('inf'))
    #         )
    #     except Exception as e:
    #         self.logger.error(f"Error selecting best fit: {str(e)}")
    #         return None, None, None, fitting_results
    #     # Debug print to understand the structure
    #     print("Best Fit Details:", best_fit)

    #     return (
    #         best_fit.get('function'), 
    #         best_fit.get('coefficients'), 
    #         best_fit.get('metrics', {}).get('score'), 
    #         fitting_results
    #     )
    # def _select_best_fit(self, fitting_results):
    #     """
    #     Select the best fit following these steps:
    #     1. Find polynomial with lowest score
    #     2. Compare best polynomial score with other functions' scores
    #     3. If polynomial score is lower -> select polynomial
    #     4. If polynomial score is higher -> compare MSE
    #     5. Select based on lowest MSE between polynomial and other function
        
    #     Returns:
    #     - tuple: (selected function, coefficients, metric value, all fitting results)
    #     """
    #     successful_fits = [
    #         result for result in fitting_results 
    #         if result.get('success', False)
    #     ]
        
    #     if not successful_fits:
    #         self.logger.warning("No successful fits found")
    #         return None, None, None, fitting_results
        
    #     try:
    #         # Step 1: Separate polynomial and other fits
    #         polynomial_fits = []
    #         other_fits = []
            
    #         for fit in successful_fits:
    #             # More explicit check for polynomial fits
    #             is_polynomial = (
    #                 isinstance(fit.get('function'), str) 
    #                 and fit.get('function', '').startswith('polynomial_degree_')
    #             )
    #             if is_polynomial:
    #                 polynomial_fits.append(fit)
    #             else:
    #                 other_fits.append(fit)
            
    #         # Find best polynomial based on score
    #         best_polynomial_score = min(
    #             polynomial_fits,
    #             key=lambda x: x.get('metrics', {}).get('score', float('inf'))
    #         )
    #         # Find best polynomial based on score
    #         best_polynomial_mse = min(
    #             polynomial_fits,
    #             key=lambda x: x.get('metrics', {}).get('mse', float('inf'))
    #         )
    # #         # Find best polynomial based on score
    # #         best_polynomial_r_squared = max(
    # #             polynomial_fits,
    # #             key=lambda x: x.get('metrics', {}).get('r_squared', float('inf'))
    # #         )
            
    #         # Find best non-polynomial based on score
    #         best_other_score = min(
    #             other_fits,
    #             key=lambda x: x.get('metrics', {}).get('score', float('inf'))
    #         )
    #         # Find best non-polynomial based on score
    #         best_other_mse = min(
    #             other_fits,
    #             key=lambda x: x.get('metrics', {}).get('mse', float('inf'))
    #         )
    #         # Find best non-polynomial based on score
    #         best_other_squared = max(
    #             other_fits,
    #             key=lambda x: x.get('metrics', {}).get('r_squared', float('inf'))
    #         )
            
    # #         if best_other_squared > best_polynomial_r_squared:
    #         if polynomial_fits:
    #             # Find best polynomial based on score
    #             best_polynomial = min(
    #                 polynomial_fits,
    #                 key=lambda x: x.get('metrics', {}).get('score', float('inf'))
    #             )
    #             poly_score = best_polynomial.get('metrics', {}).get('score', float('inf'))
    #             poly_mse = best_polynomial.get('metrics', {}).get('mse', float('inf'))
                
    #             if other_fits:
    #                 best_squared = max(
    #                     other_fits,
    #                     key=lambda x: x.get('metrics', {}).get('r_squared', float('inf'))
    #                 )
    #                 # Find best non-polynomial based on score
    #                 best_other_score = min(
    #                     other_fits,
    #                     key=lambda x: x.get('metrics', {}).get('score', float('inf'))
    #                 )
    #                 # print(best_squared)
    #                 # print(best_other_score)
    #                 other_score = best_other_score.get('metrics', {}).get('score', float('inf'))
    #                 other_mse = best_other_score.get('metrics', {}).get('mse', float('inf'))
    #                 # print(f"best other fit is the function {best_other_score} with score {other_score} and mse\
    #                       # {other_mse}")
    #                 # Step 3: If polynomial score is lower, select polynomial
    #                 # print(f"poly_score is {poly_score} while other_mse is {other_score}")
    #                 # print(f"poly_mse is {poly_mse} while other_mse is {other_score}")
    #                 if poly_score < other_score:
    #                 #or np.isclose(poly_mse, other_mse, rtol=1e-2, atol=1e-12):
                        
    #                     best_fit = best_polynomial
    #                     self.logger.debug(f"Selected polynomial due to better score: {poly_score} vs {other_score}")
    #                 # Step 4-6: If polynomial score is higher, compare MSE
    #                 else:
                        
    #                     if poly_mse < other_mse:
    #                     #or np.isclose(poly_mse, other_mse, rtol=1e-2, atol=1e-12):
                            
    #                         best_fit = best_polynomial
    #                         self.logger.debug(f"Selected polynomial due to better MSE: {poly_mse} vs {other_mse}")
    #                     else:
    #                         best_fit = best_other_score
    #                         self.logger.debug(f"Selected other function due to better MSE: {other_mse} vs {poly_mse}")
    #             else:
    #                 best_fit = best_polynomial
    #                 self.logger.debug("Selected polynomial as no other successful fits exist")
    #         else:
    #             # If no polynomial fits, select based on score
    #             best_fit = min(
    #                 successful_fits,
    #                 key=lambda x: x.get('metrics', {}).get('score', float('inf'))
    #             )
    #             self.logger.debug("Selected best fit as no polynomial fits exist")
                
    #     except Exception as e:
    #         self.logger.error(f"Error selecting best fit: {str(e)}")
    #         return None, None, None, fitting_results
    
    #     # Log final selection details
    #     self.logger.debug(f"Final selection:")
    #     self.logger.debug(f"Function: {best_fit.get('function')}")
    #     self.logger.debug(f"Score: {best_fit.get('metrics', {}).get('score')}")
    #     self.logger.debug(f"MSE: {best_fit.get('metrics', {}).get('mse')}")
        
    #     # Return appropriate metric based on function type
    #     selected_metric = (
    #         best_fit.get('metrics', {}).get('score')
    #         if isinstance(best_fit.get('function'), str) and 
    #         'polynomial' in best_fit.get('function', '').lower()
    #         else best_fit.get('metrics', {}).get('mse')
    #     )
    #     print("Best Fit Details:", best_fit)

    #     return (
    #         best_fit.get('function'),
    #         best_fit.get('coefficients'),
    #         selected_metric,
    #         fitting_results
    #         )
    def _select_best_fit(self, fitting_results):
        """
        Select the best fit following a comprehensive comparison strategy:
        1. Filter successful fits
        2. Separate polynomial and non-polynomial fits
        3. Handle cases with empty fit categories
        4. Compare fits based on multiple metrics
        
        Returns:
        - tuple: (selected function, coefficients, metric value, all fitting results)
        """
        # Filter only successful fits
        successful_fits = [
            result for result in fitting_results 
            if result.get('success', False)
        ]
        
        # Handle case of no successful fits
        if not successful_fits:
            self.logger.warning("No successful fits found")
            return None, None, None, fitting_results
        
        try:
            # Categorize fits
            polynomial_fits = [
                fit for fit in successful_fits 
                if isinstance(fit.get('function'), str) 
                and fit.get('function', '').startswith('polynomial_degree_')
            ]
            
            other_fits = [
                fit for fit in successful_fits 
                if fit not in polynomial_fits
            ]
            
            # Handle different fit category scenarios
            if not polynomial_fits and not other_fits:
                self.logger.warning("No categorizable fits found")
                return None, None, None, fitting_results
            
            # Selection logic
            if polynomial_fits and other_fits:
                # Compare polynomial and other fits
                best_polynomial = min(
                    polynomial_fits, 
                    key=lambda x: x.get('metrics', {}).get('score', float('inf'))
                )
                best_other = min(
                    other_fits, 
                    key=lambda x: x.get('metrics', {}).get('score', float('inf'))
                )
                
                # Comparison criteria
                poly_score = best_polynomial.get('metrics', {}).get('score', float('inf'))
                other_score = best_other.get('metrics', {}).get('score', float('inf'))
                
                poly_mse = best_polynomial.get('metrics', {}).get('mse', float('inf'))
                other_mse = best_other.get('metrics', {}).get('mse', float('inf'))
                
                # Selection logic
                if poly_score < other_score or poly_mse < other_mse:
                    best_fit = best_polynomial
                    selection_reason = "Better polynomial score/MSE"
                else:
                    best_fit = best_other
                    selection_reason = "Better non-linear function score/MSE"
            
            elif polynomial_fits:
                # Only polynomial fits available
                best_fit = min(
                    polynomial_fits, 
                    key=lambda x: x.get('metrics', {}).get('score', float('inf'))
                )
                selection_reason = "Only polynomial fits available"
            
            else:
                # Only other fits available
                best_fit = min(
                    other_fits, 
                    key=lambda x: x.get('metrics', {}).get('score', float('inf'))
                )
                selection_reason = "Only other fits available"
            
            # Logging
            self.logger.debug(f"Best Fit Selection Reason: {selection_reason}")
            self.logger.debug(f"Selected Function: {best_fit.get('function')}")
            self.logger.debug(f"Score: {best_fit.get('metrics', {}).get('score')}")
            self.logger.debug(f"MSE: {best_fit.get('metrics', {}).get('mse')}")
            
            # Determine appropriate metric
            selected_metric = (
                best_fit.get('metrics', {}).get('score')
                if isinstance(best_fit.get('function'), str) and 
                'polynomial' in best_fit.get('function', '').lower()
                else best_fit.get('metrics', {}).get('mse')
            )
            
            return (
                best_fit.get('function'),
                best_fit.get('coefficients'),
                selected_metric,
                fitting_results
            )
        
        except Exception as e:
            self.logger.error(f"Error in best fit selection: {str(e)}")
            return None, None, None, fitting_results
    # def _select_best_fit(self, fitting_results):
    #     """
    #     Select the best fit based on r_squared and score:
        
    #     - Compare all successful fits by their r_squared value.
    #     - If one function has the highest r_squared (unique when rounded to 7 decimals), 
    #       select that as the best fit.
    #     - If multiple functions have nearly identical r_squared values (to 7 decimals),
    #       then select the one with the lowest score.
        
    #     Returns:
    #         tuple: (selected function name, coefficients, selected metric, all fitting results)
    #     """
    #     # Filter for successful fits with valid metrics
    #     successful_fits = [
    #         result for result in fitting_results 
    #         if result.get('success', False) and result.get('metrics') is not None
    #     ]
        
    #     if not successful_fits:
    #         self.logger.warning("No successful fits found")
    #         return None, None, None, fitting_results
        
    #     try:
    #         # First, check if any fit has an r_squared of 1 (rounded to 4 decimals)
    #         # fits_with_one = [
    #         #     fit for fit in successful_fits 
    #         #     if round(fit['metrics'].get('r_squared', -float('inf')), 4) == 1.0
    #         # ]
    #         fits_with_one =  [
    #             fit for fit in successful_fits
    #             if fit['metrics'].get('r_squared', -float('inf')) == 1.0
    #         ]
    #         if fits_with_one:
    #             best_r_fits = fits_with_one
    #             self.logger.debug("Found fits with r_squared equal to 1; using these for selection.")
    #         else:
    #             # If no fit has r_squared equal to 1, then find the maximum r_squared.
    #             max_r_squared = max(
    #                 fit['metrics'].get('r_squared', -float('inf')) 
    #                 for fit in successful_fits
    #             )
                
    #             # Identify all fits that have an r_squared value equal to max_r_squared
    #             # when rounded to 4 decimal places.
    #             best_r_fits = [
    #                 fit for fit in successful_fits 
    #                 if round(fit['metrics'].get('r_squared', -float('inf')),3) == round(max_r_squared,3)
    #             ]
    #             self.logger.debug("No fits with r_squared equal to 1; using maximum r_squared for selection.")
            
    #         # Choose the best fit:
    #         if len(best_r_fits) == 1:
    #             # Unique best r_squared: select it.
    #             best_fit = best_r_fits[0]
    #             selected_metric = best_fit['metrics'].get('r_squared')
    #             self.logger.debug("Unique best r_squared found.")
    #         else:
    #             # Multiple fits with almost equal r_squared: choose the one with the lowest score.
    #             best_fit = min(
    #                 best_r_fits,
    #                 key=lambda fit: fit['metrics'].get('mse', float('inf'))
    #             )
    #             selected_metric = best_fit['metrics'].get('score')
    #             self.logger.debug("Multiple fits with nearly identical r_squared; using score to break tie.")
        
    #     except Exception as e:
    #         self.logger.error(f"Error selecting best fit: {str(e)}")
    #         return None, None, None, fitting_results
    
    #     # Log the final selection details
    #     self.logger.debug("Final selection:")
    #     self.logger.debug(f"Function: {best_fit.get('function')}")
    #     self.logger.debug(f"r_squared: {best_fit.get('metrics', {}).get('r_squared')}")
    #     self.logger.debug(f"Score: {best_fit.get('metrics', {}).get('score')}")
    #     self.logger.debug(f"MSE: {best_fit.get('metrics', {}).get('mse')}")
        
    
    #     return (
    #         best_fit.get('function'),
    #         best_fit.get('coefficients'),
    #         selected_metric,
    #         fitting_results
    #     )

    def polynomial_fit(self, x, y, xerr=None, max_degree=4, alpha=1.0, min_points_per_param=3):
        """
        Find the best polynomial fit using multiple criteria to avoid overfitting
        
        Parameters:
        x: array-like, data for the x-axis
        y: array-like, data for the y-axis
        xerr: array-like, measurement errors
        max_degree: int, maximum degree of the polynomial to test
        alpha: float, regularization parameter for coefficient comparison
        
        Returns:
        tuple: (best_mse, best_degree, best_coefficients, fitting_results)
        """
        fitting_results = []
        if not self._validate_data(x, y):
            return [{
                'function': 'No valid data',
                'success': False,
                'error_message': 'Data validation failed',
                'metrics': None,
                'coefficients': None
            }]
        
        
        if alpha < 0:
            raise ValueError ("Alpha must be a positive number")
            
        # Check for constant x or y values
        if np.all(y == y[0]) and len(x) > 1:
            # For constant y, return degree 0 polynomial with that constant
            constant_coeff = np.array([y[0]])
            fitting_results = [{
                'function': 'constant_y',
                'metrics': {
                    'mse': 0,
                    'score': 0,
                    'chi2': 0 if xerr is not None else None
                },
                'coefficients': constant_coeff,
                'degree': 0,
                'polynomial': np.poly1d(constant_coeff),
                'success': True
            }]
            return fitting_results
        
        if np.all(x == x[0]) and len(x) > 1:
            # Cannot fit polynomial if x is constant - undefined
            raise ValueError("Cannot fit polynomial when all x values are constant")
            
        # best_fit = {
        #     'mse': float('inf'),
        #     'score': float('inf'),  # Combined score for selection
        #     'degree': 0,
        #     'coefficients': None
        # }

        if len(x) > 1:
            x_orig = x.copy()  
            x_mean, x_std = np.nanmean(x_orig), np.nanstd(x_orig)
            x_norm = (x_orig - x_mean) / x_std
        
        if max_degree > 4: 
            warnings.warn("Polynomial degrees higher than 4 might lead to overfitting and numerical instability.", UserWarning)
        
        n_points = len(x)
        # Rule 2: Limit maximum degree based on sample size
        suggested_max_degree = int(np.sqrt(n_points))
        actual_max_degree = min(max_degree, suggested_max_degree)
        
        if actual_max_degree < max_degree:
            warnings.warn(f"Reducing maximum polynomial degree from {max_degree} to {actual_max_degree} "
                         f"based on sample size of {n_points} points", UserWarning)
        
        
        for degree in range(1, actual_max_degree + 1):
            # n_params = degree + 1  # number of coefficients for this degree
            # required_points = n_params * min_points_per_param
            
            # if len(x) < required_points:
            #     warnings.warn(f"Skipping degree {degree}: Insufficient data points. "
            #                 f"Need at least {required_points} points for {n_params} parameters")
            #     fitting_results.append({
            #         'function': f'polynomial_degree_{degree}',
            #         'success': False,
            #         'error_message': f"Insufficient data points for reliable fit"
            #     })
            #     continue
            
            if len(x) - (degree) <= 0:
                raise ValueError ("Number of points must be higher than max_degree")
            try:
                # # Fit with normalized x
                # coefficients_norm = Polynomial.fit(x_norm, y, degree)
                # print("coefficients norm", coefficients_norm)
                
                # coefficients =Polynomial.fit(x, y, degree)
                # print("coefficients", coefficients)
                # p = Polynomial(coefficients)
                # print("p", p)
                # test_points = np.array([1])
                # y_pred = p(test_points)
                # print("y_pred", y_pred)
                
                coefficients_norm = np.polyfit(x_norm, y, degree)

                # Convert coefficients back to original scale
                p_norm = np.poly1d(coefficients_norm)
                x_test = np.linspace(min(x), max(x), 100)
                y_test = p_norm((x_test - x_mean) / x_std)
                coefficients = np.polyfit(x_test, y_test, degree)

                p = np.poly1d(coefficients)

                y_pred = p(x)

                # Convert coefficients back to original scale
                # p_norm = Polynomial(coefficients_norm)
                # x_test = np.linspace(min(x), max(x), 100)
                # y_test = p_norm((x_test - x_mean) / x_std)
                # coefficients =Polynomial.fit(x_test, y_test, degree)
                # print("coefficients", coefficients)
                # p = Polynomial(coefficients[::-1])
                # print("p", p)
                # y_pred = p(1)
                # print(y_pred)
                
                # Calculate metrics using the new method
                metrics = self._calculate_metrics(y, y_pred, degree, coefficients, xerr)
            
               
                fitting_results.append({
                    'function': f'polynomial_degree_{degree}',
                    'metrics': metrics,
                    'polynomial': p,
                    'coefficients': coefficients,
                    'degree': degree,
                    'success': True
                })
    
                # # Update best fit if this degree has a better score
                # if metrics['score'] < best_fit['score']:
                #     best_fit.update({
                #         'mse': metrics['mse'],
                #         'score': metrics['score'],
                #         'degree': degree,
                #         'coefficients': coefficients
                #     })
    
            except Exception as e:
                fitting_results.append({
                    'function': f'polynomial_degree_{degree}',
                    'mse': None,
                    'success': False,
                    'error_message': str(e)
                })
    
        # return (best_fit['score'], best_fit['degree'], 
                # best_fit['coefficients'], fitting_results)
        return fitting_results


    
    def _log_fitting_results(self, fitting_results):
        """
        Log a formatted summary of all fitting results.
        
        Parameters:
        -----------
        fitting_results : list
            List of dictionaries containing fitting results for each function
        """
        self.logger.info("\nFitting Results Summary:")
        
        for result in fitting_results:
            # Format the function name to be more readable
            func_name = result['function']
            if func_name.startswith('polynomial_degree_'):
                displayed_name = f"Polynomial Degree {func_name.split('_')[-1]}"
            else:
                # Replace underscores with spaces and capitalize each word
                displayed_name = ' '.join(word.capitalize() for word in func_name.split('_'))
            r_str = f"{result['metrics']['r_squared']:.2e}" if result ["metrics"] is not None else "Failed"
            # Format MSE value
            score_str = f"{result['metrics']['score']:.2e}" if result['metrics'] is not None else "Failed"
            # Format MSE value
            mse_str = f"{result['metrics']['mse']:.2e}" if result['metrics'] is not None else "Failed"
            
            # Use appropriate log level based on success
            if result['success']:
                self.logger.info(f"{displayed_name}: score = {score_str}, mse = {mse_str}, r_str = {r_str}")
            else:
                self.logger.warning(f"{displayed_name}: Fitting Failed")
           
    def calculate_best_fit(self, x, y, mode=ProcessingMode.PV, print_results=False):
        """
        Calculates the best fitting function and its parameters.
        Returns the best function, its coefficients, and score.
        Args:
            x (array-like): Input x values
            y (array-like): Input y values
            mode (ProcessingMode): Processing mode for x values
            print_results (bool): Whether to print detailed fitting results
        
        Returns:
            tuple: Best fitting function, its parameters, and performance metrics
        
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If no valid fit can be found
        """
        # self.logger.info("Starting curve fitting")
        
        if not self._validate_data(x, y):
            return [{
                'function': 'No valid data',
                'success': False,
                'error_message': 'Data validation failed',
                'metrics': None,
                'coefficients': None
            }]
        

        try:
            x_processed = self._process_values(x, y, mode=mode)
        except Exception as e:
            self.logger.error(f"Error processing x values: {str(e)}")
            return None, None, None, None
        
        # best_func = None
        # best_popt = None
        fitting_results = []
        
        # # Polynomial fitting
        # best_score, best_degree, best_coefficients, poly_fitting_results = self.polynomial_fit(x_processed, y, max_degree=4)
        
        # best_func = best_degree
        # best_popt = best_coefficients
        poly_fitting_results = self.polynomial_fit(x_processed, y, max_degree=4)
        
        # Add polynomial results to fitting_results
        fitting_results.extend(poly_fitting_results)
        
        # Implement smarter initial guess based on function characteristics 
        # def generate_initial_guess(func):
        #     signature = inspect.signature(func)
        #     num_params = len(signature.parameters) - 1  # Escludiamo self e x
            
        #     # Controllo specifico per generalized_polynomial
        #     if func.__name__ == "_generalized_polynomial":
        #         # Per generalized_polynomial, impostiamo r=5.0 come terzo parametro
        #         return [1.0] * (num_params - 1) + [5.0]
            
        #     # Per tutte le altre funzioni, usiamo 1.0 per tutti i parametri
        #     return [1.0] * num_params
        
        
        for func_name, func in self.fitting_functions.items():
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Invalid values found at x positions*')
                
                   #initial_guess = self.generate_initial_guess(func, x_processed, y)
                initial_guesses = [
                    None,
                self.generate_initial_guess(func, x_processed, y),
                self.generate_bayesian_initial_guess(func, x_processed, y)
                
            ] 
                for initial_guess in initial_guesses:
                    best_fit = None
                    best_metrics = None
                    fit_succeeded = False
                    
                    try:
                        
                        popt, pcov = curve_fit(func, x_processed, y, p0=initial_guess, maxfev=10000)
                
                        if np.all(np.isfinite(pcov)):
                            y_fit = func(x_processed, *popt)
                            metrics = self._calculate_metrics(y, y_fit, 0, popt)
                            
                            # Keep track of the best fit among all initial guesses
                            if best_metrics is None or metrics['mse'] < best_metrics['mse']:
                                best_fit = {
                                    'function': func_name,
                                    'metrics': metrics,
                                    'success': True,
                                    'error_message': None,
                                    'coefficients': popt
                                }
                                best_metrics = metrics
                            
                            fit_succeeded = True
                            
                    except Exception as e:
                        self.logger.debug(f"Attempt with initial guess {initial_guess} failed for {func_name}: {str(e)}")
                        continue
                
                # After trying all initial guesses, append the best result or failure
                if fit_succeeded:
                    fitting_results.append(best_fit)
                else:
                    # Only add the warning after all initial guesses have failed
                    self.logger.warning(f"All fitting attempts failed for function {func_name}")
                    fitting_results.append({
                        'function': func_name,
                        'metrics': None,
                        'valid_covariance': False,
                        'success': False,
                        'error_message': "All initial guesses failed to converge"
                    })
                    #     popt, pcov = curve_fit(func, x_processed, y,p0=initial_guess, maxfev = 10000)
                        
                    #     # # Define residual function for least_squares
                    #     # def residuals(params):
                    #     #     return func(x_processed, *params) - y
                        
                    #     # # Try fitting without covariance calculation
        
                    #     # result = least_squares(residuals, initial_guess)
        
                    #     if np.all(np.isfinite(pcov)):
                    #         y_fit = func(x_processed, *popt)
                    #         # If fit succeeded, calculate MSE
                    #         #y_fit = func(x_processed, *result.x)
                    #         #metrics = self._calculate_metrics(y, y_fit, 0, result.x)
                    #         # Perform K-Fold Cross-Validation
                    #         # cv_mse = self.cross_validate_curve_fit(func, x_processed, y, initial_guess)
                    #         metrics = self._calculate_metrics(y, y_fit, 0, popt)
                            
                    #         fitting_results.append({
                    #             'function': func_name,
                    #             'metrics': metrics,
                    #             'success': True,
                    #             'error_message': None,
                    #             'coefficients': popt
                    #         })
                            
                    #         # if metrics['score'] < best_score:
                    #         #     best_score = metrics['score']
                    #         #     best_func = func
                    #         #     best_popt = result.x
                    #         break
                    #     else:
                    #         # self.logger.warning(f"Fitting failed for function {func_name}")
                    #         fitting_results.append({
                    #             'function': func_name,
                    #             'metrics': None,
                    #             'valid_covariance': False,
                    #             'success': False,
                    #             'error_message': "Fitting failed"
                    #         })
                            
                    # except Exception as e:
                    #     self.logger.error(f"Exception during fitting for {func_name}: {str(e)}")
                    #     fitting_results.append({
                    #         'function': func_name,
                    #         'metrics': None,
                    #         'valid_covariance': False,
                    #         'success': False,
                    #         'error_message': str(e)
                    #     })
        
        if print_results:
            self._log_fitting_results(fitting_results)
            
        # if best_func is None or best_popt is None:
        #         self.logger.error("No valid fit found")
        #         return
        # elif isinstance(best_func, int):
        #     self.logger.info(f'The best fitting function is Polynomial Degree {best_degree}')
        # else:
        #     # Get the function name without the leading underscore and format it
        #     func_name = best_func.__name__.lstrip('_')
        #     formatted_name = ' '.join(word.capitalize() for word in func_name.split('_'))
        #     self.logger.info(f'The best fitting function is {formatted_name}')
            
        return fitting_results
    
    def cross_validate_curve_fit(self, func, x, y, initial_guess, k=5):
        """ Perform K-Fold Cross-Validation for curve fitting """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        mse_scores = []
        
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            try:
                popt, _ = curve_fit(func, x_train, y_train, p0=initial_guess, maxfev=10000)
                y_pred = func(x_test, *popt)
                mse = mean_squared_error(y_test, y_pred)
                mse_scores.append(mse)
            except Exception as e:
                mse_scores.append(np.inf)  # Assign large error if fitting fails
    
        return np.mean(mse_scores)
    
    def log_best_fitting_function(self, fitting_results, best_func=None, best_degree=None):
        """
        Logs detailed information about the best-fitting function.
    
        Parameters:
        -----------
        best_func : callable or int
            The best-fitting function or polynomial degree
        best_degree : int, optional
            Polynomial degree (used when best_func is an integer)
    
        Returns:
        --------
        str
            A formatted string describing the best-fitting function
        """
        
        # Check if fitting results exist
        if not fitting_results:
            self.logger.warning("No fitting results available. Compute best fit first.")
            return "No fitting results"
    
        try:
            # If no best_func provided, select from existing results
            if best_func is None:
                best_func, best_coeff, best_metrics, _ = self._select_best_fit(fitting_results)
                print(best_func)
            # Check for invalid input
            if best_func is None:
                error_msg = "No valid fit found"
                self.logger.error(error_msg)
                return error_msg
        
            # Handle polynomial fits
            if isinstance(best_func, int) or (best_degree is not None):
                degree = best_func if isinstance(best_func, int) else best_degree
                log_message = f'Best fitting function: Polynomial Degree {degree}'
                self.logger.info(log_message)
                return log_message
        
            # Handle other function types
            try:
                # Get the function name without the leading underscore
                # func_name = best_func.__name__.lstrip('_')
                
                # Format the function name (convert to title case)
                formatted_name = ' '.join(word.capitalize() for word in best_func.split('_'))
                
                log_message = f'Best fitting function: {formatted_name}'
                self.logger.info(log_message)
                return log_message
    
            except AttributeError:
                error_msg = "Unable to determine function name"
                self.logger.error(error_msg)
                return error_msg
        except Exception as e:
            self.logger.error(f"Error in logging best fitting function: {str(e)}")
            return "Error in determining best fit"




    def generate_initial_guess(self, func, x, y):
        """
        Generate an initial guess for curve_fit based on the function name and data.
        
        Args:
            func (callable): The fitting function.
            x (array-like): Input x data.
            y (array-like): Input y data.
        
        Returns:
            list: A list of initial guesses for the parameters.
        """
        # Ensure x and y are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        # Common data statistics
        y_min = np.min(y)
        y_max = np.max(y)
        y_mean = np.mean(y)
        amplitude = y_max - y_min if y_max != y_min else 1.0
    
        if func.__name__ == '_exponential':
            # _exponential(x, a, b, c)
            # Let c be near the lower bound, a be the amplitude, and b be a moderate growth rate.
            c = y_min
            a = amplitude
            b = 1.0 / (np.ptp(x) if np.ptp(x) else 1.0)
            return [a, b, c]
    
        elif func.__name__ == '_combination_of_exponential':
            # _combination_of_exponential(x, a, b, c, d, e, f)
            # Here, we assume the data roughly follows:
            #    y = a*exp(b*x + e) + c*exp(d*x + f)
            #
            # A strategy is to split the amplitude and set b and d based on the x-range.
            #
            # Estimate amplitude split:
            # a_guess = amplitude / 2.0
            # c_guess = amplitude / 2.0
            # # For b and d, a rough estimate is to use the inverse of the x-range,
            # # but note that one exponential might be growing and the other decaying.
            # x_range = np.ptp(x) if np.ptp(x) != 0 else 1.0
            # b_guess = 1.0 / x_range   # for the growing exponential
            # d_guess = -1.0 / x_range  # for the decaying exponential
    
            # For the shifts inside the exponentials, a heuristic is to roughly center them.
            # We can use the logarithm of the amplitudes minus the product with the mean of x.
            x_mean = np.mean(x)
            """Stima i parametri esponenziali usando il logaritmo"""
            # Filtra valori positivi
            # mask = y > 1e-3
            # x_filt = x[mask]
            # y_filt = y[mask]
            
            # Verifica che gli array filtrati non siano vuoti
            # if x_filt.size == 0 or y_filt.size == 0:
            #Estimate amplitude split:
            a_guess = amplitude / 2.0
            c_guess = amplitude / 2.0
            # For b and d, a rough estimate is to use the inverse of the x-range,
            # but note that one exponential might be growing and the other decaying.
            x_range = np.ptp(x) if np.ptp(x) != 0 else 1.0
            b_guess = 1.0 / x_range   # for the growing exponential
            d_guess = -1.0 / x_range  # for the decaying exponential
                
               
            # else: 
            #     # Approssima con log(y) ≈ log(a1) + b1*x + log(1 + (a2/a1)*exp((b2-b1)*x))
            #     logy = np.log(y_filt)
                
            #     # Fit lineare iniziale per il termine dominante
            #     slope, intercept = np.polyfit(x_filt, logy, 1)
            #     b_guess = slope
            #     a_guess = np.exp(intercept)
                
            #     # Stima il secondo termine
            #     residual = y_filt - a_guess * np.exp(b_guess * x_filt)
            #     c_guess = np.max(residual)
            #     d_guess =b_guess * 0.1  # Supponiamo un decadimento più lento
        
       
            e_guess = np.log(np.abs(a_guess)) - b_guess * x_mean
            f_guess = np.log(np.abs(c_guess)) - d_guess * x_mean
        
            return [a_guess, b_guess, c_guess, d_guess, e_guess, f_guess]
    
        elif func.__name__ == '_generalized_rational':
            
            """
            Calculate initial parameters for rational function fitting.
            Function form: f(x) = (a + bx)/(cx + d) + e
            
            Parameters:
            x, y : array-like
                Input x and y data points
            
            Returns:
            tuple of (a, b, c, d, e) as initial parameter guesses
            """
            # Get basic data characteristics
            x_range = np.max(x) - np.min(x)
            
            # Estimate linear trend for numerator
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate improved parameter estimates
            a = intercept  # Intercept term in numerator
            b = slope      # Slope term in numerator
            
            # Initialize denominator terms to create gentle curvature
            c = 1.0 / x_range  # Scale to input range
            d = 1.0           # Prevent division by zero at x=0
            
            # Estimate vertical offset using minimum value
            e = np.min(y) if np.min(y) < 0 else 0.0
            
            return a, b, c, d, e

        elif func.__name__ == '_generalized_polynomial':
            # _generalized_polynomial(x, a, b, r)
            return [1.0, 1.0, 2.0]
    
        elif func.__name__ == '_log_function':
            # _log_function(x, a, b, c)
            return [0.0, 1.0, 1.0]
    
        else:
            num_params = len(inspect.signature(func).parameters) - 1
            return [1.0] * num_params
    
    def generate_bayesian_initial_guess(self, func, x, y, num_samples=1000):
       # Get initial guess
        initial_guess = self.generate_initial_guess(func, x, y)
        
           # Create ranges around initial guess
        param_ranges = []
        for guess in initial_guess:
            # Ensure non-zero, positive ranges
            lower = max(0.001, guess * 0.5)
            upper = max(lower * 2, guess * 1.5)
            param_ranges.append((lower, upper))
        
        # Define prior distributions for each parameter
        priors = [st.uniform(loc=range[0], scale=range[1]-range[0]) for range in param_ranges]

        
        # Sample parameters
        samples = [prior.rvs(num_samples) for prior in priors]
        
        # Generate simulated data for each parameter set
        distances = []
        for params in zip(*samples):
            y_simulated = func(x, *params)
            distance = np.sqrt(np.mean((y - y_simulated)**2))
            distances.append(distance)
        
        # Find parameter set with smallest distance to observed data
        best_index = np.argmin(distances)
        initial_guess = [sample[best_index] for sample in samples]
        
        return initial_guess
    
    def generate_priors(self, func, x, y):
        # Use existing initial guess method as a basis
        initial_guess = self.generate_initial_guess(func, x, y)
        
        priors = []
        for guess in initial_guess:
            # Create uniform prior around the initial guess
            # Adjust the scale based on your data characteristics
            lower = guess * 0.1
            upper = guess * 10
            priors.append(st.uniform(loc=lower, scale=upper-lower))
    
        return priors
    def perfect_sampling_initial_guess(self, func, x, y, num_samples=100_000):
        # Define prior distributions for parameters
        # Adjust these based on your specific function and domain knowledge
        priors = self.generate_priors(func, x, y)
        
        # Sampling process
        accepted_params = []
        for _ in range(num_samples):
            # Sample parameters from priors
            params = [prior.rvs() for prior in priors]
            
            # Generate simulated data
            y_simulated = func(x, *params)
            
            # Define distance metric (similar to lesson's approach)
            distance = np.sqrt(np.mean((y - y_simulated)**2))
            
            # Set a threshold for acceptance (adjust based on your needs)
            if distance < np.std(y):
                accepted_params.append(params)
            
            # Stop if we have enough samples
            if len(accepted_params) > 10:
                break
        
        # If no parameters accepted, fall back to default initial guess
        if not accepted_params:
            return [np.mean(y), 1.0, 0.0][:func.__code__.co_argcount - 2]
        
        # Return the parameter set with the smallest distance
        return accepted_params[0]


                
                
                

