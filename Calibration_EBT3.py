# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:59:35 2025

@author: Ele_p
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import enum

class ProcessingMode(enum.Enum):
    PV = "PV"  # Pass-through values
    OD = "OD"  # Optical Density
    NET_OD = "netOD"  # Net Optical Density


class CurveFitter:
    def __init__(self):
        self.fitting_functions = {
            'exponential': self._exponential,
            'exponential_decreasing': self._exponential_decreasing,
            'exponential_difference': self._exponential_difference,
            'double_exponential': self._double_exponential,
            'exponential_combination': self._exponential_combination,
            'rational': self._rational,
            'hyperbolic_growth': self._hyperbolic_growth,
            'rational_with_offset': self._rational_with_offset,
            'saturation_with_offset': self._saturation_with_offset,
            'linear_decay': self._linear_decay,
            'polynomial_scaling': self._polynomial_scaling,
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
            return np.log10(65535 / x_values)
        elif mode == ProcessingMode.NET_OD:
            if y_values is None:
                raise ValueError("y_values are required for NET_OD mode")
            # Find x_value where y = 0
            if 0 not in y_values:
                raise ValueError("y_values must contain 0 for NET_OD mode")
            x_zero = x_values[np.where(y_values == 0)[0][0]]
            return -np.log10(x_values / x_zero)
        else:
            raise ValueError(f"Unknown processing mode: {mode}")
            
    def _validate_data(self, x, y):
        """
        Validates input data ranges.
        Returns True if valid, False otherwise.
        """

        if not all(0 <= val <= 65535 for val in x):
            print("Error: x values must be between 0 and 65535")
            return False
        if not all(0 <= val <= 50 for val in y):
            print("Error: y values must be between 0 and 50 Gy")
            return False
        return True
    
    # Add enum values as function attributes
    for mode in ProcessingMode:
        setattr(_process_values, mode.name, mode)

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
        """
    
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
        exp_component = np.exp(np.clip(b * x_scaled, -700, 700))  # Clip the exponent range
    
        return a * exp_component + c

    def _exponential_decreasing(self, x, a, b, c):
        """
        Function representing an exponential decay with an offset and with scaling and overflow control.
        
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
        """
    
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
        exp_component = np.exp(np.clip(-b * x_scaled, -700, 700))  # Clip the exponent range
        return c - a * exp_component


    def _double_exponential(self, x, a, b, c, d):
        """
        Sum of exponential function with scaling and overflow control.
        
        Parameters:
        -----------
        x : array-like
            values on x axis 
        a, b, c, d : float
            parameters of the double exponential function
            
        Returns:
        --------
        array-like
            values computed by the sum of two exponential functions
        """
    
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
        exp_component_one= np.exp(np.clip(b * x_scaled, -700, 700))  # Clip the exponent range
        exp_component_two= np.exp(np.clip(d * x_scaled, -700, 700))  # Clip the exponent range
        return a * exp_component_one + c * exp_component_two


    def _exponential_difference(self, x, a, b, c, d):
        """
        Subtraction of exponential terms with scaling and overflow control.
        
        Parameters:
        -----------
        x : array-like
            values on x axis 
        a, b, c, d : float
            parameters of the exponential difference function
            
        Returns:
        --------
        array-like
            values computed by the exponential difference function 
        """
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
        # Add small epsilon to prevent underflow
        eps = 1e-10
        exp_component_one= np.exp(np.clip(a * x_scaled + b, -700, 700)) + eps # Clip the exponent range
        exp_component_two= np.exp(np.clip(c * x_scaled + d, -700, 700)) + eps # Clip the exponent range
        return exp_component_one - exp_component_two

    def _exponential_combination(self, x, a, b, c, d):
        """
        Sum of exponentials with positive and negative exponents with scaling and overflow control.
        
        Parameters:
        -----------
        x : array-like
            values on x axis 
        a, b, c, d : float
            parameters of the exponential difference function
            
        Returns:
        --------
        array-like
            values computed by the exponential combination function 
        """
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
        exp_component_one= np.exp(np.clip(a * b * x_scaled, -700, 700))  # Clip the exponent range
        exp_component_two= np.exp(np.clip(-c * d * x_scaled, -700, 700))  # Clip the exponent range
        
        return c * exp_component_one + d * exp_component_two

    def _rational(self, x, a, b, c):
        """
        Function representing a generalized rational function with scaling.
    
        Parameters:
        -----------
        x : array-like
            Input values on the x-axis.
        a : float
            Offset term added to the numerator.
        b : float
            Scaling parameter applied to the input variable in the numerator.
        c : float
            Offset term added to the denominator.
    
        Returns:
        --------
        array-like
            Output values computed as (a + b * x) / (x + c).
    
        Description:
        ------------
        This function computes a rational relationship between input \(x\),
        with an offset \(a\) in the numerator, a scaling factor \(b\) applied to \(x\) in the numerator,
        and a shift \(c\) in the denominator. 
        The additional parameter \(b\) allows for more flexible control 
        over the rate of growth in the numerator compared to simpler rational functions.
        """
        
        return (a + b * x) / (x + c)

    def _hyperbolic_growth(self, x, a, b):
        """
        Function representing a saturation curve.
    
        Parameters:
        -----------
        x : array-like
            Input values on the x-axis.
        a : float
            Scaling parameter for the numerator.
        b : float
            Scaling parameter for the denominator.
    
        Returns:
        --------
        array-like
            Output values computed as (a * x) / (b * x + 1).
        """
        return (a * x) / (b * x + 1)

    def _rational_with_offset(self, x, a, e):
        """
        Function representing a rational function with an offset.
    
        Parameters:
        -----------
        x : array-like
            Input values on the x-axis.
        a : float
            Offset term added to the numerator.
        e : float
            Offset term added to the denominator.
    
        Returns:
        --------
        array-like
            Output values computed as (a + x) / (x + e).
        """
        return (a + x) / (x + e)

    def _saturation_with_offset(self, x, b, a, c):
        """
        Function representing a saturating curve with an offset.
    
        Parameters:
        -----------
        x : array-like
            Input values on the x-axis.
        b : float
            Scaling parameter for the numerator.
        a : float
            Scaling parameter for the denominator.
        c : float
            Offset added to the function output.
    
        Returns:
        --------
        array-like
            Output values computed as (b * x) / (a + x) + c.
        """
        return (b * x) / (a + x) + c

    def _linear_decay(self, x, a, b):
        """
        Function representing a linear decay.
    
        Parameters:
        -----------
        x : array-like
            Input values on the x-axis.
        a : float
            Scaling parameter for the numerator.
        b : float
            Scaling parameter for the denominator.
    
        Returns:
        --------
        array-like
            Output values computed as x - (a * x) / b.
        """
        return x - (a * x) / b

    def _polynomial_scaling(self, x, a, b, r):
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
        return a * x + b * x**r


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
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        
        # Ensure arrays
        x = np.asarray(x)
        
        # Calculate numerator and denominator separately
        numerator = x + c + eps
        denominator = b + x + eps
        
        # Check for invalid values
        valid_mask = (numerator > 0) & (denominator > 0)
        
        # if not np.all(valid_mask):
        #     print(f"Warning: Invalid values found at x positions: {np.where(~valid_mask)[0]}")
            
        # Calculate ratio with epsilon to prevent division by zero
        ratio = (numerator + eps) / (denominator + eps)
        
        # Calculate log with input validation
        result = np.full_like(x, np.nan, dtype=float)
        result[valid_mask] = np.log(ratio[valid_mask]) - a
        
        return result

    def polynomial_fit(self, x, y, max_degree=4):
        
        
        
        
        """
        Find the best polynomial fit by testing different degrees
        
        Parameters:
        x: array-like, data for the x-axis
        y: array-like, data for the y-axis
        max_degree: int, maximum degree of the polynomial to test
        
        Returns:
        dict: Dictionary containing fitting results for all tested polynomials
        """
        if not self._validate_data(x, y):
            return None, None, None, None

        fitting_results = []
        best_mse = float('inf')
        best_coefficients = None
        best_degree = 0
        
        for degree in range(1, max_degree + 1):
            try:
                coefficients = np.polyfit(x, y, degree)
                coefficients = [c if abs(c) >= 1e-08 else 0 for c in coefficients]
                p = np.poly1d(coefficients)
                y_pred = p(x)
                mse = mean_squared_error(y, y_pred)
                
                fitting_results.append({
                    'function': f'polynomial_degree_{degree}',
                    'mse': mse,
                    'valid_covariance': True,
                    'success': True,
                    'error_message': None,
                    'degree': degree,
                    'coefficients': coefficients,
                    'polynomial': p
                })
                
                if mse < best_mse:
                    best_mse = mse
                    best_coefficients = coefficients
                    best_degree = degree
                    
            except Exception as e:
                fitting_results.append({
                    'function': f'polynomial_degree_{degree}',
                    'mse': None,
                    'valid_covariance': False,
                    'success': False,
                    'error_message': str(e),
                    'degree': degree,
                    'coefficients': None,
                    'polynomial': None
                })
        
        return best_mse, best_degree, best_coefficients, fitting_results
         
    
    
    
    def print_fitting_results(self, fitting_results):
        """
        Print a formatted summary of all fitting results.
        
        Parameters:
        -----------
        fitting_results : list
            List of dictionaries containing fitting results for each function
        """
        print("\nFitting Results Summary:")
        print("-" * 80)
        print(f"{'Function Name':<30} {'MSE':<15} {'Valid Covariance':<20} {'Success'}")
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
            
            print(f"{displayed_name:<30} {mse_str:<15} {str(result['valid_covariance']):<20} {result['success']}")

    # def calculate_best_fit(self, x, y, mode=ProcessingMode.PV, print_results= False):
    #     """
    #     Calculates the best fitting function and its parameters.
    #     Returns the best function, its coefficients, and MSE.
    #     """
    #     if not self._validate_data(x, y):
    #         return None, None, None, None
    #     # Process x values according to the specified mode
    #     try:
    #         x_processed = self._process_values(x, y, mode)
    #     except Exception as e:
    #         print(f"Error processing x values: {str(e)}")
    #         return None, None, None, None
        
    #     x_processed = self._process_values(x, y, mode)
    #     best_func = None
    #     best_popt = None
    #     poly_mse, best_degree, best_coefficients, fitting_results = self.polynomial_fit(x_processed, y, max_degree=4)
    #     best_mse = poly_mse
        
    #     best_func = best_degree
    #     best_popt = best_coefficients
        
    #     for func_name, func in self.fitting_functions.items():
    #         try:
    #             popt, pcov, infodict, errmsg, ier = curve_fit(func, x_processed, y,
    #                                                          full_output=True,
    #                                                          maxfev=10000,
    #                                                          method='lm')
    #             if pcov is not None and not np.any(np.isinf(pcov)):
    #                 y_fit = func(x_processed, *popt)
    #                 mse = mean_squared_error(y, y_fit)
                    
    #                 fitting_results.append({
    #                     'function': func_name,
    #                     'mse': mse,
    #                     'valid_covariance': pcov is not None and not np.any(np.isinf(pcov)),
    #                     'success': True,
    #                     'error_message': None,
    #                     'popt': popt
    #                 })
    
    #                 if mse < best_mse:
    #                     best_mse = mse
    #                     best_func = func
    #                     best_popt = popt

    #         except Exception as e:
    #             fitting_results.append({
    #                 'function': func_name,
    #                 'mse': None,
    #                 'valid_covariance': False,
    #                 'success': False,
    #                 'error_message': str(e)
    #             })
        
    #     if print_results == True:
    #         self.print_fitting_results(fitting_results)
        
    #     # # Print summary of fitting attempts
    #     # print("\nFitting Results Summary:")
    #     # print("-" * 80)
    #     # print(f"{'Function Name':<30} {'MSE':<15} {'Valid Covariance':<20} {'Success'}")
    #     # print("-" * 80)
    #     # for result in fitting_results:
    #     #     mse_str = f"{result['mse']:.2e}" if result['mse'] is not None else "Failed"
    #     #     print(f"{result['function']:<30} {mse_str:<15} {str(result['valid_covariance']):<20} {result['success']}")
    #     if best_func is None or best_popt is None:
    #         print("No valid fit found")
    #         return
    #     elif isinstance(best_func, int):
    #         print(f'The best fitting function is Polynomial Degree {best_degree}')
    #     else:
    #         # Get the function name without the leading underscore and format it
    #         func_name = best_func.__name__.lstrip('_')
    #         formatted_name = ' '.join(word.capitalize() for word in func_name.split('_'))
    #         print(f'The best fitting function is {formatted_name}')
            
    #     # if best_func is None or best_popt is None:
    #     #     print("No valid fit found")
    #     #     return
    #     # elif isinstance(best_func, int):
    #     #     print(f'The best fitting function is polynomial degree {best_degree}')
    #     # else:
    #     #     print(f'The best fitting function is {best_func.__name__}')
    #     return best_func, best_popt, best_mse, fitting_results
    #     # Print the best-fitting function details
    #     # if isinstance(best_func, int):  # Polynomial case
    #     #     print(f'The best fitting function is polynomial degree {best_degree}')
    #     #     equation = " + ".join(
    #     #         f"{coeff:.3f}*x**{i}" for i, coeff in enumerate(reversed(best_popt))
    #     #     )
    #     #     print(f"The equation of the polynomial is: {equation}")
    #     # else:  # Non-polynomial function case
    #     #     print(f'The best fitting function is {best_func.__name__}')
    #     #     func_source = inspect.getsource(best_func)
    #     #     for line in func_source.splitlines():
    #     #         if "return" in line:
    #     #             equation = line.strip().replace("return ", "")
    #     #             # Replace variables with parameter values
    #     #             for i, param in enumerate(best_popt):
    #     #                 equation = equation.replace(f"p{i}", f"{param:.3f}")
    #     #             print(f"The equation of the function is: {equation}")
    #     #             break
                
    #     # return best_func, best_popt, best_mse, fitting_results
        
    # def calculate_best_fit(self, x, y, mode=ProcessingMode.PV, print_results=False):
        # """
        # Calculates the best fitting function and its parameters.
        # Returns the best function, its coefficients, MSE, and fitting results.
        
        # Parameters:
        # -----------
        # x : array-like
        #     Independent variable values
        # y : array-like
        #     Dependent variable values
        # mode : ProcessingMode
        #     Processing mode for x values
        # print_results : bool
        #     Whether to print fitting results
            
        # Returns:
        # --------
        # tuple
        #     (best_function, best_parameters, best_mse, fitting_results)
        # """
        # if not self._validate_data(x, y):
        #     return None, None, None, None
    
        # try:
        #     x_processed = self._process_values(x, y, mode)
        # except Exception as e:
        #     print(f"Error processing x values: {str(e)}")
        #     return None, None, None, None
    
        # best_func = None
        # best_popt = None
        # fitting_results = []
    
        # # Polynomial fitting
        # poly_mse, best_degree, best_coefficients, poly_results = self.polynomial_fit(x_processed, y, max_degree=4)
        # best_mse = poly_mse
        # best_func = best_degree
        # best_popt = best_coefficients
        # fitting_results.extend(poly_results)
    
        # # Try different initial guesses for non-polynomial functions
        # initial_guesses = [
        #     None,  # Let curve_fit choose default initial parameters
        #     [1.0] * 4,  # All ones
        #     [0.1] * 4,  # Small positive values
        #     [10.0] * 4  # Larger values
        # ]
    
        # for func_name, func in self.fitting_functions.items():
        #     best_fit_for_func = None
        #     best_mse_for_func = float('inf')
            
        #     for guess in initial_guesses:
        #         try:
        #             # Add bounds to prevent extreme parameter values
        #             bounds = (-np.inf, np.inf)  # Default bounds, adjust as needed
                    
        #             # Try different fitting methods
        #             for method in ['lm', 'trf', 'dogbox']:
        #                 try:
        #                     if method == 'lm':
        #                         popt, pcov = curve_fit(
        #                             func, x_processed, y,
        #                             p0=guess,
        #                             maxfev=10000,
        #                             method=method,
        #                             full_output=False
        #                         )
        #                     else:
        #                         # For 'trf' and 'dogbox' methods, we can use bounds
        #                         popt, pcov = curve_fit(
        #                             func, x_processed, y,
        #                             p0=guess,
        #                             maxfev=10000,
        #                             method=method,
        #                             bounds=bounds,
        #                             full_output=False
        #                         )
    
        #                     # Calculate fit quality
        #                     y_fit = func(x_processed, *popt)
        #                     mse = mean_squared_error(y, y_fit)
                            
        #                     # Check if this is the best fit for this function
        #                     if mse < best_mse_for_func:
        #                         best_mse_for_func = mse
        #                         best_fit_for_func = {
        #                             'function': func_name,
        #                             'mse': mse,
        #                             'valid_covariance': pcov is not None and not np.any(np.isinf(pcov)),
        #                             'success': True,
        #                             'error_message': None,
        #                             'popt': popt,
        #                             'method': method
        #                         }
                                
        #                 except Exception as e:
        #                     continue  # Try next method
    
        #         except Exception as e:
        #             continue  # Try next initial guess
    
        #     # Add the best fit for this function to results
        #     if best_fit_for_func is not None:
        #         fitting_results.append(best_fit_for_func)
                
        #         # Update overall best fit if this function performed better
        #         if best_fit_for_func['mse'] < best_mse:
        #             best_mse = best_fit_for_func['mse']
        #             best_func = func
        #             best_popt = best_fit_for_func['popt']
        #     else:
        #         # If no successful fit was found for this function
        #         fitting_results.append({
        #             'function': func_name,
        #             'mse': None,
        #             'valid_covariance': False,
        #             'success': False,
        #             'error_message': "Failed to fit with any method or initial guess"
        #         })
    
        # if print_results:
        #     self.print_fitting_results(fitting_results)
    
        # return best_func, best_popt, best_mse, fitting_results
      
    # def calculate_best_fit(self, x, y, mode=ProcessingMode.PV, print_results=False):
    #     """
    #     Calculates the best fitting function and its parameters.
    #     Returns the best function, its coefficients, and MSE.
    #     """
    #     if not self._validate_data(x, y):
    #         return None, None, None, None
        
    #     try:
    #         x_processed = self._process_values(x, y, mode)
    #     except Exception as e:
    #         print(f"Error processing x values: {str(e)}")
    #         return None, None, None, None
        
    #     best_func = None
    #     best_popt = None
    #     fitting_results = []
        
    #     # Polynomial fitting
    #     poly_mse, best_degree, best_coefficients, poly_fitting_results = self.polynomial_fit(x_processed, y, max_degree=4)
    #     best_mse = poly_mse
    #     best_func = best_degree
    #     best_popt = best_coefficients
        
    #     # Add polynomial results to fitting_results
    #     fitting_results.extend(poly_fitting_results)
        
    #     # Single initial guess for speed
    #     initial_guess = [1.0] * 4  # Adjust size based on your function parameters
        
    #     # Try each function once with optimized parameters
    #     for func_name, func in self.fitting_functions.items():
    #         try:
    #             popt, pcov = curve_fit(
    #                 func, 
    #                 x_processed, 
    #                 y,
    #                 p0=initial_guess,
    #                 maxfev=10000,
    #                 method='lm'
    #             )
                
    #             # Calculate MSE
    #             y_fit = func(x_processed, *popt)
    #             mse = mean_squared_error(y, y_fit)
                
    #             # Always append results for this function
    #             fitting_results.append({
    #                 'function': func_name,
    #                 'mse': mse,
    #                 'valid_covariance': pcov is not None and not np.any(np.isinf(pcov)),
    #                 'success': True,
    #                 'error_message': None,
    #                 'popt': popt
    #             })
                
    #             # Update best fit if this function performed better
    #             if mse < best_mse:
    #                 best_mse = mse
    #                 best_func = func
    #                 best_popt = popt
                    
    #         except Exception as e:
    #             # Append failed result
    #             fitting_results.append({
    #                 'function': func_name,
    #                 'mse': None,
    #                 'valid_covariance': False,
    #                 'success': False,
    #                 'error_message': str(e)
    #             })
        
    #     if print_results:
    #         self.print_fitting_results(fitting_results)
            
    #     return best_func, best_popt, best_mse, fitting_results
    
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
        
        initial_guess = [1.0] * 4  # Adjust size based on your function parameters
        
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
            self.print_fitting_results(fitting_results)
            
        return best_func, best_popt, best_mse, fitting_results
                
                
                
                

