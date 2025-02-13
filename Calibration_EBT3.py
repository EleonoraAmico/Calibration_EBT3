# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:59:35 2025

@author: Ele_p
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import enum
import warnings
import logging
import sys
import inspect
import scipy.stats as st

class ProcessingMode(enum.Enum):
    PV = 1  # Pixel Values
    OD = 2  # Optical Density
    NET_OD = 3  # Net Optical Density
    
class LoggerUtility:
    @staticmethod
    def create_logger(name, level=logging.INFO, log_file=None):
        """
        Create a configured logger with optional file output
        
        Args:
            name (str): Logger name
            level (int): Logging level
            log_file (str, optional): Path to log file
        
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers to prevent duplicate logs
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

class CurveFitter:
    def __init__(self, log_level=logging.INFO, log_file=None):
        self.logger = LoggerUtility.create_logger(
            name='CurveFitter', 
            level=log_level, 
            log_file=log_file
        )
        self.fitting_functions = {
            'exponential': self._exponential,
            'combination_of_exponential': self._combination_of_exponential,
            'generalized_rational': self._generalized_rational,
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
            warnings.warn("x values must be between 0 and 65535", UserWarning)
            return False
            
        if not np.all((y_array >= 0) & (y_array <= 50)):
            warnings.warn("y values should be between 0 and 50 Gy. Higher doses may lead to inaccurate measurements for EBT3.", UserWarning)
            return True
        
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
        
        denominator = c * x + d 
        if np.any(denominator == 0):
            raise ValueError("Invalid input: denominator would be zero")
        
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
        if abs(r) < 0.5:
            raise ValueError("r must be ≥ 0.5 in absolute value")
        # Check if x is a numpy array
        if not isinstance(x, (int, float, np.ndarray)):
            warnings.warn("Input x is not int, float or a numpy array. Converting to numpy array.", UserWarning)
            x = np.array(x)  # Convert to numpy array if it's not
        # Handle negative exponents with zero check
        if r < 0: 
           # Check for invalid values
           valid_input = (x != 0)
           
           if not np.all(valid_input):
               warnings.warn(f"Invalid values found at x positions: {np.where(~valid_input)[0]}")
               x = x[x!=0]
        # Compute generalized polynomial
        result = a * x + b * x**r
        return result


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
        ratio = (numerator + eps) / (denominator + eps)
        
        # Check for invalid values
        valid_mask = (numerator > 0) & (denominator > 0)
        
        if not np.all(valid_mask):
            warnings.warn(f"Invalid values found at x positions: {np.where(~valid_mask)[0]}")
                
        result = np.full_like(x, np.nan, dtype=float)
        result[valid_mask] = np.log(ratio[valid_mask]) - a
        
        return result

    def _calculate_metrics(self, y_true, y_pred, degree, coefficients, xerr=None):
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
        mse = mean_squared_error(y_true, y_pred)
        
        # Chi-squared calculation
        chi2 = None
        if xerr is not None:
            residuals = y_true - y_pred
            chi2 = np.sum((residuals / xerr) ** 2)
        
        # Degrees of freedom
        
        dof = len(y_true) - (len(coefficients))
        
        # Coefficient ratio
        coeff_ratio = (np.abs(coefficients[0]) / 
                       np.mean(np.abs(coefficients[1:])) 
                       if len(coefficients) > 1 else np.inf)
        
        # Complexity penalty
        complexity_penalty = degree * np.log(len(y_true))
        
        # Modified score with multiple criteria
        score = (mse * 
                 (1 + complexity_penalty / len(y_true)) * 
                 (1 + 1/coeff_ratio))
        
        return {
            'mse': mse,
            'score': score,
            'chi2': chi2,
            'dof': dof,
            'coeff_ratio': coeff_ratio
        }
    
    
    def _select_best_fit(self, fitting_results):
        """
        Select the best fit based on metrics
        """
        successful_fits = [
            result for result in fitting_results 
            if result.get('success', False)
        ]

        # No successful fits
        if not successful_fits:
            self.logger.warning("No successful fits found")
            return None, None, None, fitting_results
        
        # Sort by score (assuming lower score is better)
        try:
            best_fit = min(
                successful_fits, 
                key=lambda x: x.get('metrics', {}).get('score', float('inf'))
            )
        except Exception as e:
            self.logger.error(f"Error selecting best fit: {str(e)}")
            return None, None, None, fitting_results
        # Debug print to understand the structure
        print("Best Fit Details:", best_fit)

        return (
            best_fit.get('function'), 
            best_fit.get('coefficients'), 
            best_fit.get('metrics', {}).get('score'), 
            fitting_results
        )
        
    def polynomial_fit(self, x, y, xerr=None, max_degree=4, alpha=1.0):
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
            return None, None, None, None
        
        if len(x) - (max_degree) <= 0:
            raise ValueError ("Number of points must be higher than max_degree")
        if alpha < 0:
            raise ValueError ("Alpha must be a positive number")
            
        # Check for constant x or y values
        if np.all(y == y[0]):
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
        
        if np.all(x == x[0]):
            # Cannot fit polynomial if x is constant - undefined
            raise ValueError("Cannot fit polynomial when all x values are constant")
            
        # best_fit = {
        #     'mse': float('inf'),
        #     'score': float('inf'),  # Combined score for selection
        #     'degree': 0,
        #     'coefficients': None
        # }
    
        # Normalize data for numerical stability
        x_mean, x_std = np.nanmean(x), np.nanstd(x)
        x_norm = (x - x_mean) / x_std
        
        if max_degree > 4: 
            warnings.warn("Polynomial degrees higher than 4 might lead to overfitting and numerical instability.")
            
        
        for degree in range(1, max_degree + 1):
            try:
                # Fit with normalized x
                coefficients_norm = np.polyfit(x_norm, y, degree)
                
                # Convert coefficients back to original scale
                p_norm = np.poly1d(coefficients_norm)
                x_test = np.linspace(min(x), max(x), 100)
                y_test = p_norm((x_test - x_mean) / x_std)
                coefficients = np.polyfit(x_test, y_test, degree)
                
                p = np.poly1d(coefficients)
                y_pred = p(x)
                
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
            
            # Format MSE value
            score_str = f"{result['metrics']['score']:.2e}" if result['metrics'] is not None else "Failed"
            
            # Use appropriate log level based on success
            if result['success']:
                self.logger.info(f"{displayed_name}: score = {score_str}")
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
        self.logger.info("Starting curve fitting")
        
        if not self._validate_data(x, y):
            return None, None, None, None
        
        try:
            x_processed = self._process_values(x, y, mode)
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
        def generate_initial_guess(func):
            signature = inspect.signature(func)
            return [1.0] * (len(signature.parameters) - 1)  # Subtract 2 for self and x
        
        for func_name, func in self.fitting_functions.items():
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Invalid values found at x positions*')
                
                initial_guess = generate_initial_guess(func)
                
                try:
                    # Define residual function for least_squares
                    def residuals(params):
                        return func(x_processed, *params) - y
                    
                    # Try fitting without covariance calculation
    
                    result = least_squares(residuals, initial_guess)
    
                    if result.success:
                        # If fit succeeded, calculate MSE
                        y_fit = func(x_processed, *result.x)
                        metrics = self._calculate_metrics(y, y_fit, 0, result.x)
                       
                        
                        fitting_results.append({
                            'function': func_name,
                            'metrics': metrics,
                            'success': True,
                            'error_message': None,
                            'coefficients': result.x
                        })
                        
                        # if metrics['score'] < best_score:
                        #     best_score = metrics['score']
                        #     best_func = func
                        #     best_popt = result.x
                    else:
                        self.logger.warning(f"Fitting failed for function {func_name}")
                        fitting_results.append({
                            'function': func_name,
                            'metrics': None,
                            'valid_covariance': False,
                            'success': False,
                            'error_message': "Fitting failed"
                        })
                        
                except Exception as e:
                    self.logger.error(f"Exception during fitting for {func_name}: {str(e)}")
                    fitting_results.append({
                        'function': func_name,
                        'metrics': None,
                        'valid_covariance': False,
                        'success': False,
                        'error_message': str(e)
                    })
        
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
                
                
                
                

