# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:11:39 2025

@author: Ele_p
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from enum import Enum

class ProcessingMode(Enum):
    PV = "PV"
    OD = "OD" 
    NET_OD = "netOD"

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
        
    def process_values(self, x_values, y_values=None, mode=ProcessingMode.PV):
        if mode == ProcessingMode.PV:
            return x_values
        elif mode == ProcessingMode.OD:
            return np.log10(65535 / x_values)
        elif mode == ProcessingMode.NET_OD:
            if y_values is None or 0 not in y_values:
                raise ValueError("y_values must contain 0 for NET_OD mode")
            x_zero = x_values[np.where(y_values == 0)[0][0]]
            return -np.log10(x_values / x_zero)

    def _validate_data(self, x, y):
        return all(0 <= val <= 65535 for val in x) and all(0 <= val <= 50 for val in y)

    # Fitting functions
    def _exponential(self, x, a, b, c):
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        exp_component = np.exp(np.clip(b * x_scaled, -700, 700))
        return a * exp_component + c

    def _exponential_decreasing(self, x, a, b, c):
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        exp_component = np.exp(np.clip(-b * x_scaled, -700, 700))
        return c - a * exp_component

    def _exponential_difference(self, x, a, b, c, d):
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        eps = 1e-10
        exp1 = np.exp(np.clip(a * x_scaled + b, -700, 700)) + eps
        exp2 = np.exp(np.clip(c * x_scaled + d, -700, 700)) + eps
        return exp1 - exp2

    def _double_exponential(self, x, a, b, c, d):
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        exp1 = np.exp(np.clip(b * x_scaled, -700, 700))
        exp2 = np.exp(np.clip(d * x_scaled, -700, 700))
        return a * exp1 + c * exp2

    def _exponential_combination(self, x, a, b, c, d):
        x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
        exp1 = np.exp(np.clip(a * b * x_scaled, -700, 700))
        exp2 = np.exp(np.clip(-c * d * x_scaled, -700, 700))
        return c * exp1 + d * exp2

    def _rational(self, x, a, b, c):
        return (a + b * x) / (x + c)

    def _hyperbolic_growth(self, x, a, b):
        return (a * x) / (b * x + 1)

    def _rational_with_offset(self, x, a, e):
        return (a + x) / (x + e)

    def _saturation_with_offset(self, x, b, a, c):
        return (b * x) / (a + x) + c

    def _linear_decay(self, x, a, b):
        return x - (a * x) / b

    def _polynomial_scaling(self, x, a, b, r):
        return a * x + b * x**r

    def _log_function(self, x, a, b, c):
        eps = 1e-10
        numerator = x + c + eps
        denominator = b + x + eps
        valid_mask = (numerator > 0) & (denominator > 0)
        result = np.full_like(x, np.nan, dtype=float)
        result[valid_mask] = np.log((numerator[valid_mask] + eps) / (denominator[valid_mask] + eps)) - a
        return result

    def polynomial_fit(self, x, y, max_degree=4):
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

    def plot_fits(self, x_data, y_data, fitting_results, title="Fits", mode=ProcessingMode.PV):
        x_processed = self.process_values(x_data, y_data, mode)
        
        for result in fitting_results:
            plt.figure(figsize=(10, 6))
            plt.errorbar(x_processed, y_data, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
            
            x_fit = np.linspace(min(x_processed), max(x_processed), 100)
            if result['coefficients'] is not None:
                p = np.poly1d(result['coefficients'])
                y_fit = p(x_fit)
                plt.plot(x_fit, y_fit, '-', 
                        label=f"Fit: Degree {result['degree']} (MSE: {result['mse']:.2e})")
                
            plt.xlabel('Dose (Gy)')
            plt.ylabel('Response')
            plt.title(f"{title} - Degree {result['degree']}")
            plt.legend()
            plt.grid(True)
            plt.show()

    def calculate_best_fit(self, x, y, title, mode=ProcessingMode.PV):
        if not self._validate_data(x, y):
            return None, None, None, None

        x_processed = self.process_values(x, y, mode)
        
        poly_mse, best_degree, best_coefficients, fitting_results = self.polynomial_fit(x_processed, y, max_degree=4)
        best_mse = poly_mse
        
        best_func = None
        best_popt = None
        
        for func_name, func in self.fitting_functions.items():
            try:
                popt, pcov, infodict, errmsg, ier = curve_fit(func, x_processed, y,
                                                             full_output=True,
                                                             maxfev=10000,
                                                             method='lm')
                
                y_fit = func(x_processed, *popt)
                mse = mean_squared_error(y, y_fit)
                
                fitting_results.append({
                    'function': func_name,
                    'mse': mse,
                    'valid_covariance': pcov is not None and not np.any(np.isinf(pcov)),
                    'success': True,
                    'error_message': None,
                    'popt': popt
                })

                if mse < best_mse:
                    best_mse = mse
                    best_func = func
                    best_popt = popt

            except Exception as e:
                fitting_results.append({
                    'function': func_name,
                    'mse': None,
                    'valid_covariance': False,
                    'success': False,
                    'error_message': str(e)
                })

        return best_func, best_popt, best_mse, fitting_results



def plot_fits(self, x_data, y_data, title="Polynomial Fits", mode=ProcessingMode.PV, fit_function = None):
        """
        Plot all fits along with the data
        
        Parameters:
        x_data: array-like, data for the x-axis
        y_data: array-like, data for the y-axis
        fitting_results: list of dictionaries containing fitting results
        title: str, plot title
        mode: ProcessingMode enum, processing mode for the data
        """
        
        best_func, best_popt, best_mse, fitting_results = self.calculate_best_fit(x_data, y_data, mode=mode)
        try:
            x_processed = self._process_values(x_data, y_data, mode)
        except Exception as e:
            print(f"Error processing x values: {str(e)}")
            return None, None, None, None
            
        if fitting_results is None or len(fitting_results) == 0:
            print("No valid fitting results to plot.")
            return
        
        for i, result in enumerate(fitting_results):
            try:
                
                plt.figure(figsize=(10, 6))
                
                # Plot data points with error bars
                plt.errorbar(x_processed, y_data, fmt='o', ecolor='red', 
                            capsize=5, capthick=2, label='Data')
                
                # Plot polynomial fit
                x_fit = np.linspace(min(x_processed), max(x_processed), 100)
                
                # Try different possible keys for coefficients
                coefficients = None
                if 'coefficients' in result:
                    coefficients = result['coefficients']
                    
                if coefficients is not None:
                    p = np.poly1d(coefficients)
                    y_fit = p(x_fit)
                    
                    # Get degree from coefficients length or from result
                    degree = result.get('degree', len(coefficients)-1)
                    mse = result.get('mse', float('nan'))
                    
                    plt.plot(x_fit, y_fit, '-', 
                            label=f"Fit: Degree {degree} (MSE: {mse:.2e})")
                    
                    plt.xlabel('Dose (Gy)')
                    plt.ylabel('Pixel Value' if mode == ProcessingMode.PV else 'OD')
                    plt.title(f"{title} - Degree {degree}")
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                elif 'popt' in result and result['popt'] is not None:
                    popt = result['popt']
                    func_name = result.get('function', None)
                    if func_name in self.fitting_functions:
                        func = self.fitting_functions[func_name]
                        y_fit = func(x_fit, *popt)
                        mse = result.get('mse', float('nan'))
                        
                        plt.plot(x_fit, y_fit, '-', 
                                label=f"Fit: {func_name} (MSE: {mse:.2e})")
                        
                        plt.xlabel('Pixel Value' if mode == ProcessingMode.PV else 'OD' if mode == ProcessingMode.OD else 'netOD')
                        plt.ylabel('Dose (Gy)')
                        plt.title(f"{title} - {func_name}")
                    else: 
                        print(f"Warning: Function '{func_name}' not found in global namespace for result {i+1}")
                        continue

                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    
            except Exception as e:
                print(f"Error plotting result {i+1}: {str(e)}")
                continue


class PlotType(enum.Enum):
    ALL = "all"
    POLYNOMIAL = "polynomial"
    FUNCTION = "function"

class FitPlotter:
    
    def plot_fits(self, x_data, y_data, title="Fits", plot_type=PlotType.ALL, function_name=None):
        """
        Plot fits based on the specified type
        
        Parameters:
        x_data: array-like, data for the x-axis
        y_data: array-like, data for the y-axis
        title: str, plot title
        plot_type: PlotType, type of plot (ALL, POLYNOMIAL, or FUNCTION)
        function_name: str, optional, specific function to plot when plot_type is FUNCTION
        """
        # Get fitting results
        best_func, best_popt, best_mse, fitting_results = self.calculate_best_fit(x_data, y_data)
        
        if fitting_results is None or len(fitting_results) == 0:
            print("No valid fitting results to plot.")
            return
            
        if plot_type == PlotType.ALL:
            self._plot_all_fits(x_data, y_data, fitting_results, title)
        elif plot_type == PlotType.POLYNOMIAL:
            self._plot_polynomial_fits(x_data, y_data, fitting_results, title)
        elif plot_type == PlotType.FUNCTION:
            if function_name is None:
                print("Function name must be provided for function plotting.")
                return
            self._plot_specific_function(x_data, y_data, fitting_results, function_name, title)

    # Make the PlotType enum available as an attribute of the class
    PlotType = PlotType

    # Rest of the methods remain the same...
    def _plot_all_fits(self, x_data, y_data, fitting_results, title):
        """Plot all available fits including both polynomials and functions"""
        for i, result in enumerate(fitting_results):
            self._create_plot(x_data, y_data, result, title)
            
    def _plot_polynomial_fits(self, x_data, y_data, fitting_results, title):
        """Plot only polynomial fits"""
        poly_results = [result for result in fitting_results if 'coefficients' in result]
        for result in poly_results:
            self._create_plot(x_data, y_data, result, title)
            
    def _plot_specific_function(self, x_data, y_data, fitting_results, function_name, title):
        """Plot a specific named function fit"""
        func_results = [result for result in fitting_results 
                       if 'function' in result and result['function'] == function_name]
        if not func_results:
            print(f"No fitting results found for function: {function_name}")
            return
        for result in func_results:
            self._create_plot(x_data, y_data, result, title)
            
    def _create_plot(self, x_data, y_data, result, title):
        """Create individual plot for a single fit"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot data points
            plt.errorbar(x_data, y_data, fmt='o', ecolor='red', 
                        capsize=5, capthick=2, label='Data')
            
            x_fit = np.linspace(min(x_data), max(x_data), 100)
            
            if 'coefficients' in result:
                self._add_polynomial_fit(x_fit, result, title)
            elif 'popt' in result and 'function' in result:
                self._add_function_fit(x_fit, result, title)
                
            plt.xlabel('X')  # Generic labels - can be customized as needed
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
            
    def _add_polynomial_fit(self, x_fit, result, title):
        """Add polynomial fit to existing plot"""
        coefficients = result['coefficients']
        p = np.poly1d(coefficients)
        y_fit = p(x_fit)
        degree = result.get('degree', len(coefficients)-1)
        mse = result.get('mse', float('nan'))
        
        plt.plot(x_fit, y_fit, '-', 
                label=f"Polynomial Fit: Degree {degree} (MSE: {mse:.2e})")
        plt.title(f"{title} - Degree {degree}")
        
    def _add_function_fit(self, x_fit, result, title):
        """Add function fit to existing plot"""
        popt = result['popt']
        func_name = result['function']
        
        if func_name in self.fitting_functions:
            func = self.fitting_functions[func_name]
            y_fit = func(x_fit, *popt)
            mse = result.get('mse', float('nan'))
            
            plt.plot(x_fit, y_fit, '-', 
                    label=f"Fit: {func_name} (MSE: {mse:.2e})")
            plt.title(f"{title} - {func_name}")
