# -*- coding: utf-8 -*-
"""Created on Mon Jan  6 12:59:35 2025.

@author: Eleonora Cristina Amico
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import enum
import warnings
import logging
import sys
import inspect
import scipy.stats as st

class ProcessingMode(enum.Enum):
    """
    
    Enumeration of processing modes for image analysis.

    This enum defines different modes for processing image data, particularly
    in the context of optical and pixel-based measurements.

    Parameters
    ----------
        PV : int
            Pixel Values mode. Processes raw pixel intensity values.
        OD : int
            Optical Density mode. Converts pixel values to optical density measurements.
        NET_OD : int
            Net Optical Density mode. Calculates optical density with background correction.

    Examples
    --------
        >>> mode = ProcessingMode.PV
        >>> print(mode)
        ProcessingMode.PV
        >>> mode = ProcessingMode.OD
        >>> print(mode.value)
        2
        >>> mode = ProcessingMode.NET_OD
        >>> print(mode.name)
        NET_OD
        
    Notes
    -----
        - PV mode works with raw pixel values and is suitable for basic image analysis
        - OD mode applies logarithmic transformation to calculate optical density
        - NET_OD mode includes background correction in optical density calculations
        
    """
    PV = 1  # Pixel Values
    OD = 2  # Optical Density
    NET_OD = 3  # Net Optical Density
    
class LoggerUtility:
    @staticmethod
    def create_logger(name, level=logging.INFO, log_file=None):
        """
        
        Create a configured logger with optional file output.

        Parameters
        ----------
            name (str): Logger name
            level (int): Logging level
            log_file (str, optional): Path to log file

        Returns
        -------
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
        self.n = 700
        self.l = 100
        self.maxfev = 10000
        
    def _process_values(self, x_values, y_values=None, mode=ProcessingMode.PV):
        """
        
        Process x_values according to different modes.

        This function supports three processing modes:
        - PV (Pixel Value): Returns raw measurement values unchanged
        - OD (Optical Density): Calculates log10(65535/x) for each value
        - NET_OD (Net Optical Density): Calculates -log10(x/x_zero) where x_zero is the x-value 
          corresponding to y=0
        
        Parameters
        ----------
            x_values(numpy.ndarray): Array of measurement values to process. Must contain 
                                     positive values
            y_values: optional numpy array of y values, needed for NET_OD mode.
                      Must contain one zero value to determine the reference point.
                      Default is None.
            mode: ProcessingMode enum specifying the processing type. Must be a valid 
                  ProcessingMode value. Default is ProcessingMode.PV.
            
        Returns
        -------
            numpy.ndarray: Processed values according to the specified mode.
            
        Raises
        ------
            ValueError: If any of these conditions are met:
                - Invalid or unknown processing mode is provided
                - y_values is None when using NET_OD mode
                - y_values doesn't contain 0 when using NET_OD mode
                - The x-value corresponding to y=0 is zero in NET_OD mode
        
        Warnings
        --------
            UserWarning: When x_values contains zeros in OD or NET_OD modes. These values 
                will be filtered out since logarithm cannot be computed for zero.
        
        Examples:
        ---------
            >>> # Pixel Value mode (raw values)
            >>> process_values(np.array([100, 200, 300]), mode=ProcessingMode.PV)
            array([100, 200, 300])
            
            >>> # Optical Density mode
            >>> process_values(np.array([100, 200, 300]), mode=ProcessingMode.OD)
            array([2.81, 2.51, 2.34])  # log10(65535/x)
            
            >>> # Net Optical Density mode
            >>> x_vals = np.array([100, 200, 300])
            >>> y_vals = np.array([1, 0, 2])
            >>> process_values(x_vals, y_vals, mode=ProcessingMode.NET_OD)
            array([0.30, 0.00, 0.48])  # -log10(x/x_zero) where x_zero=200
        
        Notes
        -----
            - For OD and NET_OD modes, any zero values in x_values are filtered out before 
              processing to avoid logarithm computation errors
            - The maximum possible input value is assumed to be 65535 (16-bit)
            - All logarithmic calculations use base 10
            
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
            

    def _validate_data(self, x, y):
        """
        
        Validates input data arrays for dose measurement processing.
        Performs comprehensive validation of array properties and value ranges.

        This function checks:
        - Non-null arrays
        - Non-empty arrays
        - Conversion to numpy float arrays
        - Absence of NaN/inf values
        - Value range constraints
        
        Parameters
        ----------
            x : array-like
                Input measurement values (e.g., pixel intensities). Must be:
                - Non-null and non-empty
                - Convertible to float array
                - Contains no NaN or inf values
                - All values must be between 0 and 65535 (16-bit range)
            y : array-like
                Input dose values in Gray (Gy). Must be:
               - Non-null and non-empty
               - Convertible to float array
               - Contains no NaN or inf values
               - All values should be between 0 and 50 Gy
        
        Returns
        -------
            bool
                True if all validation checks pass, or if only the dose range warning is triggered.
                False if any other validation check fails.
            
        Warnings
        --------
            UserWarning
                Warnings are issued for these conditions:
                - Null arrays (x or y is None)
                - Empty arrays (len(x) or len(y) = 0)
                - Array conversion failures
                - NaN or inf values present
                - x values outside [0, 65535] range
                - y values outside [0, 50] Gy range (warning only, does not cause validation failure)
        
        Notes
        -----
            - The y value range check (0-50 Gy) is a soft validation - it generates a warning
              but returns True, as values above 50 Gy may be valid but could have reduced accuracy
            - All other validation checks are strict and will return False if they fail
            - Input arrays must be convertible to numpy arrays of float type
            
        Examples
        --------
            >>> # Valid data within normal ranges
            >>> validate_data([1000, 2000], [10, 20])
            True
            
            >>> # Invalid x values (outside 16-bit range)
            >>> validate_data([70000, 2000], [10, 20])
            False  # Raises warning about x range
            
            >>> # High but allowed dose values (warning only)
            >>> validate_data([1000, 2000], [10, 60])
            True  # Raises warning about y range but still returns True
            
            >>> # Invalid data (contains NaN)
            >>> validate_data([1000, np.nan], [10, 20])
            False  # Raises warning about NaN values
            
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

        Parameters
        ----------
            x : array-like
                Input array to be normalized. Must not be empty.

        Returns
        -------
            array-like
                A normalized array scaled to the [0, 1] range. If all elements in `x`
                are identical, returns an array of zeros.

        Raises
        ------
            ValueError: if x is an empty array

        Notes
        -----
            This function is not intended for direct use by end users.
            
        """
        if len(x) == 0:
            raise ValueError("Input array 'x' must not be empty.")
        
        if np.max(x) - np.min(x) == 0:
            warnings.warn("Input array has zero range. Returning array of zeros instead of performing scaling.", UserWarning)
            x_scaled = np.zeros_like(x)
            
        else:
            x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x_scaled


    
    def _exponential(self, x, a, b, c):
        """
        
        Exponential function with scaling and overflow control.

        Parameters
        ----------
            x : array-like
                values on x axis
            a, b, c : float
                parameters of the exponential function

        Returns
        -------
            array-like
                values computed by the exponential function

        Raises
        ------

            ValueError
                If 'a' or 'b' are equal to zero, as this would result in a non-exponential trend.

        Description
        -----------
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
        exp_component = np.exp(np.clip(b * x_scaled, -self.n, self.n))  # Clip the exponent range
    
        return a * exp_component + c


    def _combination_of_exponential(self, x, a, b, c, d):
        """
        
        Generalized combination of exponential function with scaling and
        overflow control.

        Parameters
        ----------
            x : array-like
                Values on the x-axis.
            a, b, c, d : float
                Parameters of the exponential terms. Their values determine the behavior of the function:
                - Positive values of a and b contribute positively.
                - Negative values of c and d can introduce subtractive or balancing behavior.

        Returns
        -------
            array-like
                Values computed as the sum or difference of two exponential terms.

        Description
        -----------
            This function computes a generalized combination of exponential function:
                f(x) = a * exp(b * normalized_x) + c * exp(d * normalized_x)
            where `normalized_x` scales the input `x` to the range [0, 1] for numerical stability.
        
        """
        if any(param == 0 for param in (a, b, c, d)):
            raise ValueError("Parameters 'a', 'b', 'c', and 'd' must not be zero.")
    
        x_scaled = self._normalized_input(x)
        exp_component_one = np.exp(np.clip(b * x_scaled, -self.n, self.n)) 
        exp_component_two = np.exp(np.clip(d * x_scaled, -self.n, self.n))
        return a * exp_component_one + c * exp_component_two

    

    def _generalized_rational(self, x, a, b, c, d, e):
        """
        
        Generalized rational function with optional scaling, offset, and
        saturation behavior.

        Parameters
        ----------
            x : array-like
                Input values on the x-axis.
            a, b : float
                Parameters for the numerator (a + b * x).
            c, d : float
                Parameters for the denominator (c * x + d).
            e : float
                Offset added to the function output.

        Returns
        -------
            array-like
                Output values computed as (a + b * x) / (c * x + d) + e.

        Raises
        ------
            ValueError
                If the denominator is zero for any input value.
                If `c = 0`, as this would result in a linear or constant function rather than rational behavior.

        Description
        -----------
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


    def _log_function(self, x, a, b):
        """
        
        Enhanced logarithmic function of the form: f(x) = (ln(b + x)/b)/a with input validation and numerical stability.
        It includes safeguards against numerical instability and invalid inputs.
    
        Parameters
        ----------
            x : array-like
                Input values to be transformed.
                The function will evaluate  (b + x)/(b) for each value.
            a : float
                Scaling parameter for the logarithm. Must be non-zero.
                Controls the slope of the logarithmic curve.
                
            b : float
                Offset parameter that appears in both numerator and denominator.
                Affects the horizontal shift and scaling of the curve.
                
        Returns
        -------
            numpy.ndarray
                Array of computed logarithmic values with the same shape as input x.
                Elements will be NaN where the input results in invalid operations
                (negative or zero arguments to logarithm).
        
        Raises
        -------
            ValueError
                If parameter 'a' is zero, which would result in division by zero.
        
        Warnings
        --------
            UserWarning
                When invalid values are encountered (e.g., when b + x ≤ 0 or b ≤ 0),
                warning includes the indices of invalid values.
        
        Notes
        -----
            - Numerical stability is enhanced by adding a small epsilon (1e-10) to both
              numerator and denominator
            - Invalid results (from negative or zero inputs to logarithm) are replaced
              with NaN values
            - The function preserves input array shape in the output
            - The computation is vectorized for efficient processing of arrays
        
        Mathematical Properties
        -----------------------
            - Domain: x > -b (for real outputs)
            - Range: All real numbers
            - Horizontal asymptote: None
            - Vertical asymptote: x = -b
            - Inflection point: None (function is strictly concave)
            
        Examples
        --------
            >>> # Basic usage with valid inputs
            >>> log_function(np.array([1, 2, 3]), a=1, b=1)
            array([0.693, 1.099, 1.386])  # ln((1 + x)/1)
            
            >>> # Handling invalid inputs
            >>> result = log_function(np.array([-2, 1, 2]), a=1, b=1)
            # Warning is raised for x = -2
            # Returns: array([nan, 0.693, 1.099])
            
            >>> # Zero scaling parameter
            >>> log_function(np.array([1, 2]), a=0, b=1)
            ValueError: Invalid input: denominator would be zero
        
        """
        eps = 1e-10
        
        numerator = b + x
        denominator = b
        
        if a == 0:
            raise ValueError("Invalid input: denominator would be zero")
        
        # Single epsilon addition for numerical stability
        ratio = (numerator + eps) / (denominator + eps)
        
        # Check for invalid values
        valid_mask = (numerator > 0) & (denominator > 0)
        
        if not np.all(valid_mask):
            warnings.warn(f"Invalid values found at x positions: {np.where(~valid_mask)[0]}")
                
        result = np.full_like(x, np.nan, dtype=float)
        result[valid_mask] = np.log(ratio[valid_mask])/ a
        
        return result

    def _calculate_metrics(self, y_true, y_pred, degree, coefficients):
        """
        
        Calculate various metrics for model fitting.

        This function evaluates model performance using a composite score
        that considers prediction accuracy, model complexity, and coefficient relationships.
        
        Parameters
        ----------
            y_true : array-like
                    Actual y values. Must have the same shape as y_pred.
                    Used to calculate prediction error metrics.
            y_pred : array-like
                    Model predictions corresponding to y_true values.
                    Must have the same shape as y_true.
            degree : int
                    Polynomial degree f the model. Used to calculate complexity penalty.
                    Higher degrees indicate more complex models.
            coefficients : array
                    Model coefficients in descending order of degree.
                    For example, for a quadratic function ax² + bx + c, coefficients = [a, b, c].
                    Used to assess model stability and coefficient relationships.
        
        Returns
        -------
        
        dict
            Dictionary containing the following metrics:
            
            'mse' : float
                Mean squared error between y_true and y_pred.
                Lower values indicate better fit accuracy.
                
            'coeff_ratio' : float
                Ratio between the magnitude of the highest-degree coefficient and
                the mean of other coefficients. A measure of coefficient stability.
                - High values suggest dominant high-degree terms
                - Low values suggest more balanced coefficient contributions
                - Returns inf if only one coefficient exists
                
            'score' : float
                Composite score combining MSE, complexity penalty, and coefficient ratio.
                Score = MSE * (1 + complexity_penalty/n) * (1 + 1/coeff_ratio)
                Lower values indicate better overall model performance.
        
        Notes
        -----
            The composite score balances three aspects of model quality:
            1. Prediction accuracy (MSE)
            2. Model complexity penalty (degree * log(n))
            3. Coefficient stability (coefficient ratio)
            
            The complexity penalty increases with:
            - Higher polynomial degrees
            - Larger dataset sizes (logarithmically)
            
            Formula Details:
            - Complexity penalty = degree * log(n)
              where n is the number of data points
            - Coefficient ratio = |a₀| / mean(|a₁, a₂, ...|)
              where a₀ is the highest-degree coefficient
            
        Examples
        --------
            >>> # Linear model example
            >>> metrics = calculate_metrics(
            ...     y_true=[1, 2, 3],
            ...     y_pred=[1.1, 2.1, 2.9],
            ...     degree=1,
            ...     coefficients=[1.0, 0.5]
            ... )
            >>> print(metrics)
            {
                'mse': 0.0167,
                'coeff_ratio': 2.0,
                'score': 0.0209
            }
            
            >>> # Quadratic model example
            >>> metrics = calculate_metrics(
            ...     y_true=[1, 4, 9],
            ...     y_pred=[1.1, 3.9, 8.8],
            ...     degree=2,
            ...     coefficients=[1.0, 0.0, 0.1]
            ... )
            >>> print(metrics)
            {
                'mse': 0.0267,
                'coeff_ratio': 20.0,
                'score': 0.0401
            }
            
        See Also
        --------
            sklearn.metrics.mean_squared_error : Used for MSE calculation
            numpy.log : Used in complexity penalty calculation
        
        """
        
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
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
            'coeff_ratio': coeff_ratio,
            'r2':r2
        }
    
    
    def select_best_fit(self, fitting_results, selection_metric='score'):
        """
        
        Selects the best fitting function from a list of candidate fits
        based on specified metrics.

        The function filters successful fits and chooses the one with the lowest
        metric value.        
        The 'score' metric is particularly recommended for polynomial fits as it effectively
        balances accuracy with model complexity, helping prevent overfitting. It penalizes
        higher-degree polynomials when they don't provide significant improvements in fit quality.
        
        Parameters
        ----------
            fitting_results : list of dict
                List of dictionaries containing fitting results. Each dictionary must contain:
                - 'success' : bool
                    Indicates if the fit was successful
                - 'function' : callable
                    The fitted function
                - 'coefficients' : array-like
                    The fitted parameters
                - 'metrics' : dict
                    Dictionary of metric values including:
                        - 'score' : float
                        - 'mse' : float
                        - Other available metrics
                        
            selection_metric : str, default='score'
                The metric to use for selecting the best fit. Must be a key in the metrics
                dictionary. Lower values are considered better.
        
        Returns
        -------
            function : callable or None
                The best fitting function. None if no successful fits found.
                
            coefficients : array-like or None
                Parameters of the best fit. None if no successful fits found.
                
            best_score : float or None
                Value of the selection metric for the best fit. None if no successful fits found.
                
            fitting_results : list of dict
                The original fitting results list, unchanged.
        
        Notes
        -----
            Selection Process:
            1. Filters out unsuccessful fits
            2. Compares remaining fits using the specified metric
            3. Selects the fit with the lowest metric value
            4. Logs debug information about the selected fit
            
            The function handles several edge cases:
            - Empty fitting_results list
            - No successful fits
            - Missing metric values
            - Errors during comparison
        
        Logging
        -------
            - WARNING level: When no successful fits are found
            - ERROR level: When errors occur during selection
            - DEBUG level: Selected function details and metric values
        
        Examples
        --------
            >>> # Example with polynomial fits of different degrees
            >>> # True function: f(x) = 2x² + 1
            >>> import numpy as np
            >>> x = np.array([0, 1, 2, 3, 4, 5])
            >>> y = 2*x**2 + 1 + np.random.normal(0, 0.1, size=len(x))  # Add some noise
            >>> 
            >>> # Quadratic fit (correct degree)
            >>> quad_fit = {
            ...     'success': True,
            ...     'function': lambda x, a, b, c: a*x**2 + b*x + c,
            ...     'coefficients': [1.98, 0.05, 1.02],  # Close to true values
            ...     'metrics': {
            ...         'score': 0.015,  # Better score
            ...         'mse': 0.01
            ...     }
            ... }
            >>> 
            >>> # Cubic fit (overfitting)
            >>> cubic_fit = {
            ...     'success': True,
            ...     'function': lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d,
            ...     'coefficients': [0.1, 1.85, 0.15, 0.98],
            ...     'metrics': {
            ...         'score': 0.028,  # Worse score due to complexity penalty
            ...         'mse': 0.009    # Slightly better MSE but overfitting
            ...     }
            ... }
            >>> 
            >>> fits = [quad_fit, cubic_fit]
            >>> func, coef, score, results = select_best_fit(fits)
            >>> print(f"Best score: {score}")  # Will select quadratic fit
            Best score: 0.015
            
            In this example, although the cubic fit has a slightly better MSE, the quadratic
            fit is correctly selected because:
            1. It has the right complexity for the true underlying function
            2. The score metric penalizes the unnecessary complexity of the cubic fit
            3. The coefficient ratio is better (closer to true function parameters)
            
        See Also
        --------
            calculate_metrics : Function used to generate the metrics used in selection
        
        Raises
        ------
            Logs errors but does not raise exceptions. Returns None values on failure.
        
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
                key=lambda x: x.get('metrics', {}).get(selection_metric, float('inf'))
            )
        except Exception as e:
            self.logger.error(f"Error selecting best fit: {str(e)}")
            return None, None, None, fitting_results
        # Debug print to understand the structure
        self.logger.debug(f"Selected Function: {best_fit.get('function')}")
        self.logger.debug(f"Score: {best_fit.get('metrics', {}).get('score')}")
        self.logger.debug(f"MSE: {best_fit.get('metrics', {}).get('mse')}")

        return (
            best_fit.get('function'), 
            best_fit.get('coefficients'), 
            best_fit.get('metrics', {}).get(selection_metric), 
            fitting_results
        )
        
    def polynomial_fit(self, x, y, mode=ProcessingMode.PV, max_degree=4):

        """
        
        Calculate the best polynomial fit for a given dataset.
        
        Fits polynomials of varying degrees to the input data with automatic degree 
        optimization and numerical stability measures. The function performs data validation, 
        normalization, and handles various edge cases.
        
        Parameters
        ----------
            x : array-like
                Independent variable values (e.g., measurement values).
                Must contain at least 2 distinct points for fitting.
            
            y : array-like
                Dependent variable values (e.g., dose values).
                Must have the same length as x.
            
            mode : ProcessingMode, default=ProcessingMode.PV
                Processing mode for x values before fitting. Options:
                - PV: Uses raw values
                - OD: Applies optical density transformation
                - NET_OD: Applies net optical density transformation
                
            max_degree : int, default=4
                Maximum polynomial degree to attempt. The actual maximum degree used
                may be lower based on the number of data points (n_points//2).
                Values above 4 trigger a warning about potential overfitting.
        
        Returns
        -------
            list of dict
                List of fitting results for each polynomial degree attempted.
                Each dictionary contains:
                - 'function' : str
                Name of the fit (e.g., 'polynomial_degree_2')
                - 'metrics' : dict
                Fit quality metrics including:
                - 'mse': Mean squared error
                - 'score': Composite score considering complexity
                - 'polynomial' : numpy.poly1d
                Fitted polynomial function
                - 'coefficients' : array
                Polynomial coefficients in descending order
                - 'degree' : int
                Polynomial degree
                - 'success' : bool
                Whether the fit succeeded
                - 'error_message' : str, optional
                Present only if fit failed
            
        Notes
        -----
            Fitting Process:
            1. Data validation using _validate_data
            2. X-value processing according to specified mode
            3. Edge case handling (constant x or y values)
            4. Data normalization for numerical stability
            5. Degree limitation based on sample size
            6. Polynomial fitting with coefficient conversion
            7. Metrics calculation for each successful fit
        
            Numerical Stability:
                - Input data is normalized using mean and standard deviation
                - Coefficients are converted back to original scale
                - Polynomial evaluations use normalized x values internally
                
            Warnings are generated when:
                - max_degree > 4 (overfitting risk)
                - max_degree is reduced due to sample size
            
        Examples
        --------
            >>> # Example with quadratic data and best fit selection
            >>> import numpy as np
            >>> 
            >>> # Generate sample data (quadratic function with noise)
            >>> x = np.array([0, 1, 2, 3, 4, 5])
            >>> y = 2*x**2 + 1 + np.random.normal(0, 0.1, size=len(x))
            >>> 
            >>> # Fit polynomials of different degrees
            >>> fitting_results = polynomial_fit(x, y, max_degree=4)
            >>> 
            >>> # Recommended: Use select_best_fit to choose optimal model
            >>> best_func, best_coeff, best_score, all_results = select_best_fit(
            ...     fitting_results,
            ...     selection_metric='score'  # 'score' metric recommended for polynomials
            ... )
            >>> 
            >>> # The selected model is typically the one that best balances
            >>> # accuracy and complexity. In this case, it should select
            >>> # degree 2 as it matches the true underlying function.
            >>> print(f"Selected polynomial degree: {best_func.split('_')[-1]}")
            Selected polynomial degree: 2
            
            >>> # Not Recommended: Manual selection using only MSE
            >>> # This might lead to overfitting
            >>> worst_fit = min(fitting_results, key=lambda r: r['metrics']['mse'])
            >>> print(f"Degree selected by MSE only: {worst_fit['degree']}")
            Degree selected by MSE only: 4  # Often overfits
            
            >>> # Handling constant y values
            >>> y_const = np.full_like(x, 5.0)
            >>> results = polynomial_fit(x, y_const)
            >>> print(results[0]['degree'])  # Returns degree 0
            0
        
        Raises
        ------
            ValueError
            - When number of points is insufficient for fitting
            - When all x values are constant
            - When data validation fails
            
        See Also
        --------
            select_best_fit : Recommended function for selecting optimal polynomial degree
            numpy.polyfit : Used internally for polynomial fitting
            _validate_data : Data validation function
            _process_values : Value processing function
            _calculate_metrics : Metrics calculation function
            
        Theory
        ------
            Polynomial fitting approximates the relationship between independent and dependent variables 
            by modeling the data as a polynomial function. This method uses the least squares approach 
            to minimize the sum of squared differences between observed data and the polynomial model. 
            The choice of polynomial degree is critical: lower degrees may underfit the data, while 
            higher degrees risk overfitting. To address numerical instability, the function normalizes 
            the data, ensuring that the computations remain stable, especially when working with 
            high-degree polynomials or large magnitude values.
             
        References:
        -----------
            1. Wikipedia contributors, "Polynomial Regression," *Wikipedia, The Free Encyclopedia*, 
               https://en.wikipedia.org/wiki/Polynomial_regression.
            2. NumPy Documentation, "numpy.polyfit", https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html.
            
        """
        fitting_results = []
        if not self._validate_data(x, y):
            return None, None, None, None

        x_processed = self._process_values(x, y, mode)
        
        if len(x_processed) <= 1:
            raise ValueError ("Number of points is not sufficients to fit data")
            
                
        # Check for constant x or y values
        if np.all(y == y[0]):
            # For constant y, return degree 0 polynomial with that constant
            constant_coeff = np.array([y[0]])
            fitting_results = [{
                'function': 'constant_y',
                'metrics': {
                    'mse': 0,
                    'score': 0,
                    
                },
                'coefficients': constant_coeff,
                'degree': 0,
                'polynomial': np.poly1d(constant_coeff),
                'success': True
            }]
            return fitting_results
        
        if np.all(x_processed == x_processed[0]):
            # Cannot fit polynomial if x is constant - undefined
            raise ValueError("Cannot fit polynomial when all x values are constant")
            
        # Normalize data for numerical stability
        x_mean, x_std = np.nanmean(x_processed), np.nanstd(x_processed)
        x_norm = (x_processed - x_mean) / x_std
        
        if max_degree > 4: 
            warnings.warn("Polynomial degrees higher than 4 might lead to overfitting and numerical instability.")
            
        n_points = len(x_processed)
        # Rule 2: Limit maximum degree based on sample size
        suggested_max_degree = int(n_points//2)
        actual_max_degree = min(max_degree, suggested_max_degree)
        
        if actual_max_degree < max_degree:
            warnings.warn(f"Reducing maximum polynomial degree from {max_degree} to {actual_max_degree} "
                         f"based on sample size of {n_points} points", UserWarning)
        
        
            
        for degree in range(1, actual_max_degree + 1):
            try:
                # Fit with normalized x
                coefficients_norm = np.polyfit(x_norm, y, degree)
                
                # Convert coefficients back to original scale
                p_norm = np.poly1d(coefficients_norm)
                x_test = np.linspace(min(x_processed), max(x_processed), self.l)
                y_test = p_norm((x_test - x_mean) / x_std)
                coefficients = np.polyfit(x_test, y_test, degree)
                
                p = np.poly1d(coefficients)
                y_pred = p(x_processed)
                
                # Calculate metrics using the new method
                metrics = self._calculate_metrics(y, y_pred, degree, coefficients)
            
               
                fitting_results.append({
                    'function': f'polynomial_degree_{degree}',
                    'metrics': metrics,
                    'polynomial': p,
                    'coefficients': coefficients,
                    'degree': degree,
                    'success': True
                })
    
            except Exception as e:
                fitting_results.append({
                    'function': f'polynomial_degree_{degree}',
                    'mse': None,
                    'success': False,
                    'error_message': str(e)
                })
    
        return fitting_results


    
    def log_fitting_results(self, fitting_results):
        """
        
        Logs a formatted summary of polynomial and non-linear fitting
        results, highlighting success states and key metrics while flagging
        failed attempts.
        
        Parameters
        ----------
            fitting_results : list of dict
                List of fitting result dictionaries with structure:
                - 'function' : str
                    Function identifier (e.g., 'polynomial_degree_2')
                - 'metrics' : dict or None
                    Quality metrics (present if successful):
                    - 'score': float - Composite score balancing accuracy/complexity
                    - 'mse': float - Mean squared error
                - 'success' : bool
                    Fit success indicator
                - Additional keys may exist but are not logged here
            
        Notes
        -----
            Output Formatting:
            - Polynomial functions: Displayed as 'Polynomial Degree N'
            - Other functions: Converted to Title Case (e.g., '_exponential' -> 'Exponential')
            - Scientific notation: Metrics shown with 2 decimal places (e.g., 1.23e-04)
            - Failed fits: Display 'Failed' and use warning level logging
            
            Logging Levels:
            - INFO: Successfully fitted functions with metrics
            - WARNING: Failed fitting attempts
            
            Display Order:
            - Results are logged in the order they appear in the input list
        
        Examples
        --------
            Typical log output:
                INFO: Fitting Results Summary:
                INFO: Polynomial Degree 2: score = 1.23e-01, mse = 5.67e-03
                WARNING: Exponential: Fitting Failed
        
        See Also
        --------
            polynomial_fit : Generates the fitting results logged by this method
            calculate_non_linear_fit : Generates the fitting results for non-linear function
        
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
            # Format metrics value
            score_str = f"{result['metrics']['score']:.2e}" if result['metrics'] is not None else "Failed"
            mse_str = f"{result['metrics']['mse']:.2e}" if result['metrics'] is not None else "Failed"
            r2_str = f"{result['metrics']['r2']:.2e}" if result['metrics'] is not None else "Failed"
            # Use appropriate log level based on success
            if result['success']:
                self.logger.info(f"{displayed_name}: score = {score_str}, mse = {mse_str}, r2 = {r2_str}")
            else:
                self.logger.warning(f"{displayed_name}: Fitting Failed")
           
    def calculate_non_linear_fit(self, x, y, mode=ProcessingMode.PV, print_results=False):
        """
        
        Calculate the best non-linear fit for a given dataset.

        This function attempts to find the best-fitting function for the provided x and y values by
        iterating over a set of predefined fitting functions. For each candidate function, it generates
        multiple initial guesses (a default guess, a heuristic-based guess, and a Bayesian-inferred guess)
        and then applies non-linear least squares optimization using SciPy's `curve_fit`. The function
        evaluates each fit using performance metrics (such as the Mean Squared Error) and logs detailed
        results if requested.
    
        Parameters
        ----------
            x (array-like): Input x-values.
                Expected to be a sequence of numeric values.
            y (array-like): Input y-values corresponding to x.
                Expected to be a sequence of numeric values.
            mode (ProcessingMode, optional): Specifies the processing mode for transforming x-values.
                Defaults to ProcessingMode.PV.
            print_results (bool, optional): If True, prints detailed fitting results.
                Defaults to False.
    
        Returns
        -------
            list of dict: A list where each dictionary represents the result of fitting using a specific function.
                Each dictionary contains:
                    - 'function' (str): Name of the fitting function used.
                    - 'coefficients' (array-like or None): Best-fit parameters for the function.
                    - 'metrics' (dict or None): Performance metrics (e.g., Mean Squared Error).
                    - 'success' (bool): Indicates whether the fitting was successful.
                    - 'error_message' (str or None): An error message if the fitting failed.
    
        Raises
        ------
            ValueError: If the input data (x or y) is invalid.
            RuntimeError: If no valid fit can be determined after trying all functions and initial guesses.
    
        Examples
        --------
            >>> import numpy as np
            >>> x = np.array([1, 2, 3, 4, 5])
            >>> y = np.array([2.2, 2.8, 3.6, 4.5, 5.1])
            >>> results = fitter.calculate_non_linear_fit(x, y, mode=ProcessingMode.PV, print_results=False)
            >>> for res in results:
            ...     if res['success']:
            ...         print(f"Best fit for {res['function']}: {res['coefficients']}")
            ...     else:
            ...         print(f"Fitting failed for {res['function']}: {res['error_message']}")
    
        Relationships
        -------------
            This method utilizes:
                - self._validate_data() for input data validation.
                - self._process_values() for preprocessing x-values.
                - self._generate_initial_guess() and self._generate_bayesian_initial_guess() to obtain initial guesses.
                - SciPy's `curve_fit` for performing the non-linear least squares optimization.
    
        Theory
        ------
            The non-linear curve fitting is based on the least squares method, which minimizes the sum of
            squared differences between observed and predicted values. This method is a fundamental tool in
            statistical modeling and is widely used in data fitting tasks.
    
        References
        ----------
            - "Least Squares", Wikipedia: https://en.wikipedia.org/wiki/Least_squares
            - SciPy's curve_fit documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        
        """
        self.logger.info("Starting curve fitting")
        
        if not self._validate_data(x, y):
            return None, None, None, None
        
        x_processed = self._process_values(x, y, mode)

        fitting_results = []
        
        for func_name, func in self.fitting_functions.items():
            
            self.logger.info(f"Starting fit for: {func_name}")
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Invalid values found at x positions*')
                initial_guesses = [
                    None,
                self._generate_initial_guess(func, x_processed, y),
                self._generate_bayesian_initial_guess(func, x_processed, y)
                
            ] 
                for initial_guess in initial_guesses:
                    self.logger.info(f"Fit in progress for: {func_name}")
                    best_fit = None
                    best_metrics = None
                    fit_succeeded = False
                    
                    try:
                        
                        popt, pcov = curve_fit(func, x_processed, y, p0=initial_guess, maxfev=self.maxfev)
                        
                        perr = np.sqrt(np.diag(pcov))
                        
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
                                    'coefficients': popt,
                                    'err_coeff': perr
                                }
                                best_metrics = metrics
                            
                            fit_succeeded = True
                     
       
                    except Exception as e:
                        self.logger.debug(f"Attempt with initial guess {initial_guess} failed for {func_name}: {str(e)}")
                        continue
                
                
                # After trying all initial guesses, append the best result or failure
                if fit_succeeded:
                    self.logger.info(f"Fit successfully completed for: {func_name}")
                    fitting_results.append(best_fit)
                else:
                    # Warning after all initial guesses have failed
                    self.logger.warning(f"All fitting attempts failed for function {func_name}")
                    fitting_results.append({
                        'function': func_name,
                        'metrics': None,
                        'valid_covariance': False,
                        'success': False,
                        'error_message': "All initial guesses failed to converge"
                    })
        if print_results:
            self.log_fitting_results(fitting_results)
        return fitting_results
       
    def _generate_initial_guess(self, func, x, y):
        """
        
        Dispatch to specific initialization methods based on the provided
        function.

        This method converts the input x and y values to numpy arrays and constructs a handler 
        method name based on the target function's name (after stripping any leading underscores). 
        It then retrieves the corresponding specialized initial guess method (if available) or falls 
        back to a default initial guess generator. This mechanism provides tailored parameter 
        initialization for non-linear curve fitting, improving the convergence of optimization routines.
     
        Parameters
        -----------
            func : callable
                The target function for which an initial guess is needed. This is typically a model function 
                used in curve fitting.
            x : array-like
                Independent variable data. Must be a numeric sequence convertible to a numpy array.
            y : array-like
                Dependent variable data corresponding to x. Must be a numeric sequence convertible to a numpy array.
     
        Returns
        -------
            list
                A list of numerical initial guess values corresponding to the parameters of the target function.
     
        Raises
        ------
            AttributeError
                If the specialized handler method derived from the function name does not exist and the default 
                handler is not available (this scenario is unlikely).
     
        Examples
        --------
            >>> def exponential:
            ...     return a * np.exp(b * x) + c
            >>> # Assume 'instance' is an instance of the class containing these methods.
            >>> guess = instance._generate_initial_guess(exponential, [1, 2, 3], [2, 4, 6])
            >>> print(guess)
            [amplitude_value, rate_value, offset_value]
     
        Relationships
        -------------
            This method dispatches to specialized methods such as _guess_for_exponential, 
            _guess_for_combination_of_exponential, _guess_for_generalized_rational, and _guess_for_log_function 
            based on the function name. It falls back to _default_initial_guess if no specific method exists.
         
        Theory
        ------
            Generating a good initial guess is critical for the success of non-linear optimization algorithms 
            like those used in curve fitting. Tailored guesses based on function characteristics help in achieving 
            faster and more reliable convergence.
     
        References
        ----------
            1. SciPy documentation on curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        
        """
        x = np.asarray(x)
        y = np.asarray(y)
        # Implement smarter initial guess based on function characteristics 
        handler_name = f"_guess_for_{func.__name__.lstrip('_')}"
        handler = getattr(self, handler_name, self._default_initial_guess)
        
        return handler(x, y, func)

    # --------------------------
    # Specialized Initial Guesses
    # --------------------------
    
    def _guess_for_exponential(self, x, y, func):
        """
        
        Generate an initial guess for an exponential function of the form
        a*exp(b*x) + c.

        This method computes the minimum value of y to estimate the offset (c), calculates the amplitude 
        as the difference between the maximum and minimum y values, and determines a rough estimate for the 
        rate parameter (b) based on the range of x values. These heuristic estimations provide a reasonable 
        starting point for fitting exponential models.
    
        Parameters
        -----------
            x : array-like
                Independent variable data, expected to be a numeric sequence convertible to a numpy array.
            y : array-like
                Dependent variable data corresponding to x, expected to be a numeric sequence convertible to a numpy array.
            func : callable
                The exponential model function, typically of the form a*exp(b*x) + c.
    
        Returns
        -------
            list
                A list containing three numerical values: [a, b, c] corresponding to amplitude, rate, and offset.
    
        Examples
        --------
            >>> import numpy as np
            >>> x = np.array([0, 1, 2])
            >>> y = np.array([1, 2.7, 7.4])
            >>> guess = instance._guess_for_exponential(x, y, lambda x, a, b, c: a*np.exp(b*x)+c)
            >>> print(guess)
            [amplitude, -1.0/x_range, min_y]
    
        Relationships
        -------------
            This method is invoked via _generate_initial_guess when the target function matches an exponential model.
    
        Theory
        ------
            Exponential functions exhibit rapid growth or decay. The offset (c) is approximated by the minimum y value, 
            the amplitude (a) by the y-range, and the rate (b) is inversely related to the x-range. Such heuristics 
            provide a reasonable initialization for non-linear least squares fitting.
    
        References:
        -----------
            1. Wikipedia, "Exponential function": https://en.wikipedia.org/wiki/Exponential_function
        
        """
        y_min = np.min(y)
        amplitude = np.max(y) - y_min
        x_range = np.ptp(x) or 1.0
        return [amplitude, -1.0/x_range, y_min]
    
    def _guess_for_combination_of_exponential(self, x, y, func):
        """
        
        Generate an initial guess for a combined exponential function of the
        form a*exp(b*x) + c*exp(d*x).

        This method estimates the overall amplitude as the range of y values and uses the range of x values 
        to propose rate parameters. The amplitude is equally divided between the two exponential components, 
        with one component assigned a positive rate and the other a negative rate, to capture opposing trends.
    
        Parameters
        ----------
            x : array-like
                Independent variable data, expected to be a numeric sequence convertible to a numpy array.
            y : array-like
                Dependent variable data corresponding to x, expected to be a numeric sequence convertible to a numpy array.
            func : callable
                The combined exponential function, typically of the form a*exp(b*x) + c*exp(d*x).
        
        Returns
        -------
            list
                A list of four numerical values: [a, b, c, d] representing the amplitudes and rates for the two exponentials.
    
        Examples
        --------
            >>> import numpy as np
            >>> x = np.linspace(0, 5, 10)
            >>> y = np.exp(x) + np.exp(-x)
            >>> guess = instance._guess_for_combination_of_exponential(x, y, 
            ...     lambda x, a, b, c, d: a*np.exp(b*x) + c*np.exp(d*x))
            >>> print(guess)
            [half_amplitude, 1.0/x_range, half_amplitude, -1.0/x_range]
        
        Relationships
        -------------
            Called by _generate_initial_guess when the target function corresponds to a combination of exponential terms.
    
        Theory
        ------
            Data exhibiting both growth and decay can often be modeled by a combination of exponentials. 
            Splitting the amplitude and assigning opposite signs to the rate parameters helps in capturing such dual behaviors.
        
        References
        ----------
            1. MathWorld, "Exponential Growth and Decay": http://mathworld.wolfram.com/ExponentialGrowth.html
        
        """
        amplitude = np.ptp(y)
        x_range = np.ptp(x) or 1.0
        
        
        return [
            amplitude/2,        # a
            1.0/x_range,        # b
            amplitude/2,        # c
            -1.0/x_range,       # d
        ]
    
    def _guess_for_generalized_rational(self, x, y, func):
        """
        
        Generate an initial guess for a generalized rational function of the
        form (a + b*x)/(c*x + d) + e.

        This method first estimates a linear trend in the data using a simple linear regression (via np.polyfit) 
        to obtain the slope and intercept. It then uses the range of x to approximate one of the parameters and 
        sets a default for the remaining ones. This heuristic provides a balanced starting point for fitting a 
        rational function that can model more complex relationships.
    
        Parameters
        ----------
            x : array-like
                Independent variable data, expected to be a numeric sequence convertible to a numpy array.
            y : array-like
                Dependent variable data corresponding to x, expected to be a numeric sequence convertible to a numpy array.
            func : callable
                The rational function model, typically of the form (a + b*x)/(c*x + d) + e.
        
        Returns
        -------
            list
                A list containing five numerical values: [a, b, c, d, e] corresponding to the parameters of the rational function.
        
        Examples
        --------
            >>> import numpy as np
            >>> x = np.linspace(1, 10, 10)
            >>> y = (2 + 3*x) / (4*x + 5) + 1
            >>> guess = instance._guess_for_generalized_rational(x, y, 
            ...     lambda x, a, b, c, d, e: (a+b*x)/(c*x+d)+e)
            >>> print(guess)
            [intercept, slope, 1.0/x_range, 1.0, adjusted_min_y]
        
        Relationships
        -------------
            This function is used by _generate_initial_guess when the target model is a rational function.
    
        Theory
        ------
            Generalized rational functions combine linear behaviors in both the numerator and the denominator. 
            A preliminary linear fit helps estimate the linear coefficients, while other parameters are derived from 
            data range heuristics to handle non-linear characteristics.
        
        References
        ----------
            1. Wikipedia, "Rational function": https://en.wikipedia.org/wiki/Rational_function
            2. NumPy polyfit documentation: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
        
        """
        slope, intercept = np.polyfit(x, y, 1)
        x_range = np.ptp(x) or 1.0
        return [
            intercept,   # a
            slope,       # b
            1.0/x_range, # c
            1.0,         # d
            np.min(y) if np.min(y) < 0 else 0.0  # e
        ]
    
    def _guess_for_log_function(self, x, y, func):
        """
       
        Generate an initial guess for a logarithmic function with stability
        safeguards.

        This method ensures that the logarithmic function's domain remains valid by calculating the minimum x 
        value and adjusting the parameter that shifts the x values. This prevents attempting to compute the 
        logarithm of non-positive numbers, thus enhancing numerical stability during curve fitting.
        
        Parameters
        ----------
            x : array-like
                Independent variable data, expected to be a numeric sequence convertible to a numpy array.
            y : array-like
                Dependent variable data corresponding to x (not directly used in this estimation).
            func : callable
                The logarithmic model function, for example, one of the form a * log(x + b).
        
        Returns
        -------
            list
                A list containing two numerical values representing the initial guesses for the logarithmic model's parameters.
            
        Examples
        --------
            >>> import numpy as np
            >>> x = np.array([1, 2, 3])
            >>> guess = instance._guess_for_log_function(x, None, lambda x, a, b: a * np.log(x + b))
            >>> print(guess)
            [1.0, adjusted_value]  # Ensures that x + b remains positive
            
        Relationships
        -------------
            This method is invoked by _generate_initial_guess for logarithmic models to ensure the domain constraints 
            are met.
            
        Theory
        ------
            Logarithmic functions are only defined for positive arguments. By computing the minimum value of x and 
            adjusting the offset accordingly, this method ensures that the argument of the logarithm remains in the 
            valid range, thereby preventing domain errors during optimization.
        
        References
        ----------
            1. Wikipedia, "Logarithm": https://en.wikipedia.org/wiki/Logarithm
        
        """
        min_x = np.min(x)
        return [
            1.0,                   # a
            max(1.0, -min_x + 1e-5),  # b (ensure x + b > 0)
            
        ]
    
    def _default_initial_guess(self, x, y, func):
        """
        
        Generate a default initial guess for an unknown function.

        This fallback method determines the number of parameters required by analyzing the signature 
        of the given function (excluding the independent variable) and returns a list with a default 
        value of 1.0 for each parameter. It serves as a generic initializer when no specialized heuristic 
        is available.
    
        Parameters
        ----------
            x : array-like
                Independent variable data, expected to be a numeric sequence convertible to a numpy array.
            y : array-like
                Dependent variable data corresponding to x, expected to be a numeric sequence convertible to a numpy array.
            func : callable
                The model function for which the initial guess is being computed. The function's signature is inspected 
                to determine the number of parameters (excluding the independent variable).
        
        Returns
        -------
            list
                A list of numerical values (all ones) with a length equal to the number of parameters required by func.
        
        Examples
        --------
            >>> def some_model(x, a, b, c):
            ...     return a * x**2 + b * x + c
            >>> guess = instance._default_initial_guess([0, 1, 2], [1, 3, 7], some_model)
            >>> print(guess)
            [1.0, 1.0, 1.0]
    
        Relationships
        -------------
            This method is used as a fallback in _generate_initial_guess when no specialized initial guess method 
            (e.g., _guess_for_exponential) is found for the target function.
        
        Theory
        ------
            A simple and generic approach for initializing parameters is to assume a default value (1.0) for all. 
            While this may not be optimal, it provides a baseline from which non-linear optimizers can proceed, especially 
            when no domain-specific heuristic is available.
    
        References
        ----------
            1. SciPy curve_fit documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        
        """
        num_params = len(inspect.signature(func).parameters) - 1
        return [1.0] * num_params
    
    def _create_parameter_ranges_and_samples(self, initial_guess, num_samples):
        """Create parameter ranges and sample from prior distributions.
        
        Args:
            initial_guess (list): Initial parameter guesses
            num_samples (int): Number of samples to generate
            
        Returns:
            tuple: (samples, param_ranges) where samples is a list of arrays and 
                   param_ranges is a list of tuples (lower, upper)
        """
        
        # Create ranges around initial guess
        param_ranges = []
        for guess in initial_guess:
            # Add checks for invalid values
            if np.isnan(guess):
                raise ValueError("Guess is NaN")
                lower, upper = -1.0, 1.0
            elif guess == 0:
                lower, upper = -1.0, 1.0
            else:
                scale = max(0.1 * abs(guess), 1e-3)
                lower = guess - 5 * scale
                upper = guess + 5 * scale
                if upper <= lower:
                    upper = lower + 1e-3
            param_ranges.append((lower, upper))
    
        # Define and sample from prior distributions
        priors = []
        for lower, upper in param_ranges:
            scale = upper - lower
            if scale <= 0:
                scale = 1e-3
            priors.append(st.uniform(loc=lower, scale=scale))
    
        try:
            samples = [prior.rvs(num_samples) for prior in priors]
        except ValueError as e:
            print(f"Error in sampling: {e}")
            print(f"Parameter ranges: {param_ranges}")
            raise
    
        return samples, param_ranges
    
    def _generate_bayesian_initial_guess(self, func, x, y, num_samples=5000):
        """
        
        Generate an initial guess for function parameters using Bayesian
        sampling.

        This function refines the initial parameter guess by generating a wide range 
        of candidate values, simulating outputs using the function to be fitted, and 
        selecting the parameter set that minimizes the error between simulated and observed data. 
    
        The process consists of:
        1. Generating an initial deterministic parameter guess.
        2. Defining a search range around the initial guess.
        3. Sampling parameter values from a uniform prior distribution.
        4. Simulating function outputs for each sampled parameter set.
        5. Evaluating the error (distance metric) between simulated and observed outputs.
        6. Selecting the best parameter set with the lowest error.
    
        Parameters
        ----------
            func : callable
                The mathematical function to fit.
            x : array-like
                Independent variable values.
            y : array-like
                Observed dependent variable values corresponding to `x`.
            num_samples : int, optional
                Number of samples drawn for Bayesian estimation, default is 5000.
        
        Returns
        -------
            list of float
                The optimized initial parameter guess that best approximates the given data.
        
        Raises
        ------
            ValueError
                If `x` and `y` are not of the same length.
                If `num_samples` is not a positive integer.
    
        Notes
        -----
            - This method is particularly useful when an analytical approach for determining 
              the initial guess is not available or unreliable.
            - The Bayesian sampling approach allows exploration of multiple plausible parameter sets 
              instead of relying on a single deterministic estimate.
    
        Examples
        --------
            >>> import numpy as np
            >>> from scipy.stats import uniform
            >>> 
            >>> def exponential_function(x, a, b, c):
            ...     return a * np.exp(b * x) + c
            >>> 
            >>> x_data = np.array([1, 2, 3, 4, 5])
            >>> y_data = exponential_function(x_data, 2, -0.5, 1) + np.random.normal(0, 0.1, len(x_data))
            >>> 
            >>> model = ModelFittingClass()
            >>> initial_params = model._generate_bayesian_initial_guess(exponential_function, x_data, y_data)
            >>> print(initial_params)
    
        See Also
        --------
            _generate_initial_guess : Generates a simpler deterministic initial guess.
            scipy.optimize.curve_fit : Optimization method that uses initial guesses.
        
        Theory
        ------
            Bayesian methods are a class of probabilistic techniques based on Bayes' theorem, 
            which describes how prior knowledge is updated with observed data to obtain a 
            posterior distribution. In this case, we do not compute the full posterior 
            distribution but instead use Bayesian-inspired sampling to generate plausible 
            parameter values for function fitting.
         
            The Bayesian approach provides several advantages:
            - It allows the exploration of multiple parameter sets instead of relying on a single 
              deterministic initial guess.
            - It accounts for uncertainty in parameter estimation by considering a range of values 
              rather than fixed assumptions.
            - It is particularly useful when the function has non-linear behavior or when 
              gradient-based methods struggle with poor initial conditions.
         
            The sampling strategy used here is a form of Approximate Bayesian Computation (ABC), 
            where the best parameter set is chosen by minimizing the distance between simulated 
            and observed data.
        
        """
        # Fix the seed for reproducibility
        np.random.seed(42) 
        # Get initial guess
        initial_guess = self._generate_initial_guess(func, x, y)
        
        # Create ranges around initial guess
        param_ranges = []
        # for guess in initial_guess:
        #     if guess == 0:
        #         lower, upper = -1.0, 1.0
        #     else:
        #         # Use relative scaling but allow broader exploration
        #         scale = max(0.1 * abs(guess), 0.1)
        #         lower = guess - 5 * scale
        #         upper = guess + 5 * scale
            
        #     lower, upper = sorted([lower, upper])
        #     param_ranges.append((lower, upper))
        
        # # Define prior distributions for each parameter
        # priors = [st.uniform(loc=range[0], scale=range[1]-range[0]) for range in param_ranges]

        
        # # Sample parameters
        # samples = [prior.rvs(num_samples) for prior in priors]
        samples, param_ranges = self._create_parameter_ranges_and_samples(
            initial_guess, 
            num_samples
        )
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
    

                
                
                
                

