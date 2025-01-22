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
from sklearn.linear_model import Ridge

class ProcessingMode(enum.Enum):
    PV = "PV"  # Pass-through values
    OD = "OD"  # Optical Density
    NET_OD = "netOD"  # Net Optical Density


class CurveFitter:
    def __init__(self):
        self.fitting_functions = {
            'exponential': self._exponential,
            'combination_of_exponential': self._combination_of_exponential,
            'generalized_rational': self._generalized_rational,
            'generalized_polynomial': self._generalized_polynomial,
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
        if mode == ProcessingMode.PV:
            return x_values
        
        elif mode == ProcessingMode.OD:
            if np.any(x_values == 0):
                warnings.warn(
                    "Some values in x_values are 0. These values will be ignored "
                    "since the logarithm cannot be computed for zero.",
                    UserWarning
                )
                # Filter positive values
                x_values = x_values[x_values > 0]
            return np.log10(65535 / x_values)
        
        elif mode == ProcessingMode.NET_OD:
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
            x_array = np.asarray(x)
            y_array = np.asarray(y)
        except Exception as e:
            warnings.warn(f"Could not convert inputs to numpy arrays: {str(e)}", UserWarning)
            return False
            
        if not (0 <= x_array.all() <= 65535):
            warnings.warn("x values must be between 0 and 65535", UserWarning)
            return False
            
        if not (0 <= y_array.all() <= 50):
            warnings.warn("y values must be between 0 and 50 Gy", UserWarning)
            return False
        
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


    def _combination_of_exponential(self, x, a, b, c, d, e, f):
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
        exp_component_one = np.exp(np.clip(b * x_scaled + e, -700, 700)) 
        exp_component_two = np.exp(np.clip(d * x_scaled + f, -700, 700))
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
        
        return (a + b * x) / denominator + e


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
        if r < 0: 
           # Check for invalid values
           valid_input = (x != 0)
           
           if not np.all(valid_input):
               warnings.warn(f"Invalid values found at x positions: {np.where(~valid_input)[0]}")
           result = a * valid_input + b * valid_input ** r
        else: 
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
        if not self._validate_data(x, y):
            return None, None, None, None
        
        if len(x) - (max_degree) <= 0:
            raise ValueError ("Number of points must be higher than max_degree")
            
    # Check for constant x or y values
        if np.all(y == y[0]):
            # For constant y, return degree 0 polynomial with that constant
            constant_coeff = np.array([y[0]])
            fitting_results = [{
                'function': 'constant_y',
                'mse': 0,
                'score': 0,
                'coefficients': constant_coeff,
                'degree': 0,
                'polynomial': np.poly1d(constant_coeff),
                'chi2': 0 if xerr is not None else None,
                'dof': len(x) - 1,
                'coeff_ratio': float('inf')
            }]
            return 0, 0, constant_coeff, fitting_results
        
        if np.all(x == x[0]):
            # Cannot fit polynomial if x is constant - undefined
            raise ValueError("Cannot fit polynomial when all x values are constant")
            
        
        
            
        fitting_results = []
        best_fit = {
            'mse': float('inf'),
            'score': float('inf'),  # Combined score for selection
            'degree': 0,
            'coefficients': None
        }
    
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
                
                # Calculate various metrics
                mse = mean_squared_error(y, y_pred)
                chi2 = None
                if xerr is not None:
                    residuals = y - y_pred
                    chi2 = np.sum((residuals / xerr) ** 2)
                
                dof = len(x) - (degree + 1)
                
                # Calculate coefficient ratios
                coeff_ratio = np.abs(coefficients[0]) / np.max(np.abs(coefficients[1:])) if len(coefficients) > 1 else np.inf
                
                # Modified score that combines multiple criteria:
                # 1. MSE for fit quality
                # 2. Coefficient ratio for polynomial behavior
                # 3. Penalty for higher degrees
                # 4. AIC-like term for model complexity
                complexity_penalty = degree * np.log(len(x))
                score = (mse * 
                        (1 + complexity_penalty / len(x)) * 
                        (1 + 1/coeff_ratio) * 
                        (1 + alpha * degree/max_degree))
                print('score', score)
                fitting_results.append({
                    'function': f'polynomial_degree_{degree}',
                    'mse': mse,
                    'score': score,
                    'coefficients': coefficients,
                    'degree': degree,
                    'polynomial': p,
                    'chi2': chi2,
                    'dof': dof,
                    'coeff_ratio': coeff_ratio
                })
    
                # Update best fit if this degree has a better score
                if score < best_fit['score']:
                    best_fit.update({
                        'mse': mse,
                        'score': score,
                        'degree': degree,
                        'coefficients': coefficients
                    })
    
            except Exception as e:
                fitting_results.append({
                    'function': f'polynomial_degree_{degree}',
                    'mse': None,
                    'success': False,
                    'error_message': str(e)
                })
    
        return (best_fit['mse'], best_fit['degree'], 
                best_fit['coefficients'], fitting_results)


    
    def _print_fitting_results(self, fitting_results):
        """
        Print a formatted summary of all fitting results.
        
        Parameters:
        -----------
        fitting_results : list
            List of dictionaries containing fitting results for each function
        """
        print("\nFitting Results Summary:")
        print("-" * 80)
        print(f"{'Function Name':<30} {'MSE':<15} ") #{'Valid Covariance':<20} {'Success'}
        print("-" * 80)
        
        for result in fitting_results:
            # Format the function name to be more readable
            func_name = result['function']
            if func_name.startswith('polynomial_degree_'):
                displayed_name = f"Polynomial Degree {func_name.split('_')[-1]}"
            else:
                # Replace underscores with spaces and capitalize each word
                displayed_name = ' '.join(word.capitalize() for word in func_name.split('_'))
            
            # Format MSE value
            mse_str = f"{result['mse']:.2e}" if result['mse'] is not None else "Failed"
            
            print(f"{displayed_name:<30} {mse_str:<15} ") #{str(result['valid_covariance']):<20} {result['success']}
           
    def calculate_best_fit(self, x, y, mode=ProcessingMode.PV, print_results=False):
        """
        Calculates the best fitting function and its parameters.
        Returns the best function, its coefficients, and MSE.
        """
        if not self._validate_data(x, y):
            return None, None, None, None
        
        try:
            x_processed = self._process_values(x, y, mode)
        except Exception as e:
            print(f"Error processing x values: {str(e)}")
            return None, None, None, None
        
        best_func = None
        best_popt = None
        fitting_results = []
        
        # Polynomial fitting
        poly_mse, best_degree, best_coefficients, poly_fitting_results = self.polynomial_fit(x_processed, y, max_degree=4)
        best_mse = poly_mse
        best_func = best_degree
        best_popt = best_coefficients
        
        # Add polynomial results to fitting_results
        fitting_results.extend(poly_fitting_results)
        
        initial_guess = [1.0] * 6  # Adjust size based on your function parameters
        
        from scipy.optimize import least_squares
        
        for func_name, func in self.fitting_functions.items():
            try:
                # Define residual function for least_squares
                def residuals(params):
                    return func(x_processed, *params) - y
                
                # Try fitting without covariance calculation
                result = least_squares(residuals, initial_guess)
                
                if result.success:
                    # If fit succeeded, calculate MSE
                    y_fit = func(x_processed, *result.x)
                    mse = mean_squared_error(y, y_fit)
                    
                    fitting_results.append({
                        'function': func_name,
                        'mse': mse,
                        'valid_covariance': True,  # We're not actually checking covariance
                        'success': True,
                        'error_message': None,
                        'popt': result.x
                    })
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_func = func
                        best_popt = result.x
                else:
                    fitting_results.append({
                        'function': func_name,
                        'mse': None,
                        'valid_covariance': False,
                        'success': False,
                        'error_message': "Fitting failed"
                    })
                    
            except Exception as e:
                fitting_results.append({
                    'function': func_name,
                    'mse': None,
                    'valid_covariance': False,
                    'success': False,
                    'error_message': str(e)
                })
        
        if print_results:
            self._print_fitting_results(fitting_results)
            
        if best_func is None or best_popt is None:
                print("No valid fit found")
                return
        elif isinstance(best_func, int):
            print(f'The best fitting function is Polynomial Degree {best_degree}')
        else:
            # Get the function name without the leading underscore and format it
            func_name = best_func.__name__.lstrip('_')
            formatted_name = ' '.join(word.capitalize() for word in func_name.split('_'))
            print(f'The best fitting function is {formatted_name}')
            
        return best_func, best_popt, best_mse, fitting_results
                
                
                
                

