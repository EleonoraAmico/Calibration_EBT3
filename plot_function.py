# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:34:55 2025

@author: Ele_p
"""

# File: fit_plotter.py
import enum
import numpy as np
import matplotlib.pyplot as plt
from Calibration_EBT3 import CurveFitter, ProcessingMode  # Import the parent class

class PlotType(enum.Enum):
    ALL = "all"
    POLYNOMIAL = "polynomial"
    FUNCTION = "function"
    BEST_FIT = "best fit"

class FitPlotter(CurveFitter):  # Inherit from CurveFitter
    def __init__(self):
        super().__init__()  # Call parent class's __init__
        
    def plot_fits(self, x_data, y_data, title="Fits", plot_type=PlotType.ALL, function_name=None, mode=ProcessingMode.PV):
        """
        Plot fits based on the specified type
        
        Parameters:
        x_data: array-like, data for the x-axis
        y_data: array-like, data for the y-axis
        title: str, plot title
        plot_type: PlotType, type of plot (ALL, POLYNOMIAL, or FUNCTION)
        function_name: str, optional, specific function to plot when plot_type is FUNCTION
        """
        # Now calculate_best_fit is available through inheritance
        best_func, best_popt, best_mse, fitting_results = self.calculate_best_fit(x_data, y_data, mode)
        
        try:
            x_processed = self._process_values(x_data, y_data, mode)
        except Exception as e:
            print(f"Error processing x values: {str(e)}")
            
        if fitting_results is None or len(fitting_results) == 0:
            print("No valid fitting results to plot.")
            return
            
        if plot_type == PlotType.ALL:
            self._plot_all_fits(x_data, y_data, fitting_results, title, mode)
        elif plot_type == PlotType.POLYNOMIAL:
            self._plot_polynomial_fits(x_data, y_data, fitting_results, title, mode)
        elif plot_type == PlotType.FUNCTION:
            if function_name is None:
                print("Function name must be provided for function plotting.")
                return
            self._plot_specific_function(x_data, y_data, fitting_results, function_name, title, mode)
        elif plot_type == PlotType.BEST_FIT:
            self._plot_best_fit(x_processed, y_data, best_func, best_popt, best_mse, title, mode)

    def _plot_best_fit(self, x_processed, y_data, best_func, best_popt, best_mse, title, mode):
        """Plot only the best fit"""
        if best_func is None or best_popt is None:
            print("No valid best fit found")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot data points with error bars
        plt.errorbar(x_processed, y_data, fmt='o', ecolor='red', 
                    capsize=5, capthick=2, label='Data')
        
        x_fit = np.linspace(min(x_processed), max(x_processed), 100)
        
        if isinstance(best_func, int):
            # Polynomial case
            coefficients = best_popt
            p = np.poly1d(coefficients)
            y_fit = p(x_fit)
            plt.plot(x_fit, y_fit, '-', 
                    label=f"Best Fit: Polynomial Degree {best_func} (MSE: {best_mse:.2e})")
        else:
            # Function case
            y_fit = best_func(x_fit, *best_popt)
            plt.plot(x_fit, y_fit, '-', 
                    label=f'Best Fit: {best_func.__name__} (MSE: {best_mse:.2e})')
        
        plt.xlabel('Pixel Value' if mode == ProcessingMode.PV 
                  else 'OD' if mode == ProcessingMode.OD 
                  else 'netOD')
        plt.ylabel('Dose (Gy)')
        plt.title(f'{title} - Best Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

    PlotType = PlotType

    def _plot_all_fits(self, x_data, y_data, fitting_results, title, mode):
        """Plot all available fits including both polynomials and functions"""
        for i, result in enumerate(fitting_results):
            self._create_plot(x_data, y_data, result, title, mode)
            
    def _plot_polynomial_fits(self, x_data, y_data, fitting_results, title, mode):
        """Plot only polynomial fits"""
        poly_results = [result for result in fitting_results if 'coefficients' in result]
        for result in poly_results:
            self._create_plot(x_data, y_data, result, title, mode)
            
    def _plot_specific_function(self, x_data, y_data, fitting_results, function_name, title, mode):
        """Plot a specific named function fit"""
        func_results = [result for result in fitting_results 
                       if 'function' in result and result['function'] == function_name]
        if not func_results:
            print(f"No fitting results found for function: {function_name}")
            return
        for result in func_results:
            self._create_plot(x_data, y_data, result, title, mode)
            
    def _create_plot(self, x_data, y_data, result, title, mode):
        """Create individual plot for a single fit"""
        try:
            x_processed = self._process_values(x_data, y_data, mode)
        except Exception as e:
            print(f"Error processing x values: {str(e)}")
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot data points
            plt.errorbar(x_processed, y_data, fmt='o', ecolor='red', 
                        capsize=5, capthick=2, label='Data')
            
            x_fit = np.linspace(min(x_processed), max(x_processed), 100)
            
            if 'coefficients' in result:
                self._add_polynomial_fit(x_fit, result, title)
            elif 'popt' in result and 'function' in result:
                self._add_function_fit(x_fit, result, title)
                
            plt.xlabel('Pixel Value' if mode == ProcessingMode.PV else 'OD' if mode == ProcessingMode.OD  else 'netOD')  
            plt.ylabel('Dose (Gy)')
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
