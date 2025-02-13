# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:34:55 2025

@author: Ele_p
"""

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
        Generate plots for various curve fitting results based on the specified plot type.

        This function coordinates different types of curve fitting and plotting operations. It can generate
        plots for all fits, polynomial fits only, a specific function fit, or the best fit. The function
        performs both non-linear and polynomial fitting before creating the requested plots.
    
        Args:
            x_data (numpy.ndarray): Raw x-values for fitting (typically pixel values)
            y_data (numpy.ndarray): Y-values representing dose measurements in Gy
            title (str, optional): Title for the plot(s). Defaults to "Fits"
            plot_type (PlotType, optional): Type of plot to generate. Defaults to PlotType.ALL
                Available options:
                - PlotType.ALL: Plot all available fits
                - PlotType.POLYNOMIAL: Plot only polynomial fits
                - PlotType.FUNCTION: Plot a specific function fit
                - PlotType.BEST_FIT: Plot only the best fit
            function_name (str, optional): Name of specific function to plot when plot_type is FUNCTION.
                Required if plot_type is PlotType.FUNCTION. Defaults to None
            mode (ProcessingMode, optional): Processing mode for x-values. Defaults to ProcessingMode.PV
                Available options:
                - ProcessingMode.PV: Process as pixel values
                - ProcessingMode.OD: Process as optical density
                - ProcessingMode.netOD: Process as net optical density
    
        Returns:
            None: Displays the requested plots
    
        Raises:
            Exception: If there's an error processing x values, the error will be caught and printed
    
        Notes:
            - Requires calculate_non_linear_fit and polynomial_fit methods to be available through inheritance
            - Will print error messages and return early if:
                - No valid fitting results are available
                - Function name is not provided when plot_type is FUNCTION
                - Error occurs during x value processing
        """
        # Now calculate_best_fit is available through inheritance
        fitting_results = self.calculate_non_linear_fit(x_data, y_data, mode = mode)
        print(fitting_results)
        fitting_results_poly = self.polynomial_fit(x_data, y_data, mode)
        best_func, best_popt, best_mse, fitting_results_poly = self.select_best_fit(fitting_results_poly)
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
            self._plot_polynomial_fits(x_data, y_data, fitting_results_poly, title, mode)
        elif plot_type == PlotType.FUNCTION:
            if function_name is None:
                print("Function name must be provided for function plotting.")
                return
            self._plot_specific_function(x_data, y_data, fitting_results, function_name, title, mode)
        elif plot_type == PlotType.BEST_FIT:
            self._plot_best_fit(x_processed, y_data, best_func, best_popt, best_mse, title, mode)

    def _plot_best_fit(self, x_processed, y_data, best_func, best_popt, best_mse, title, mode):
        """
        Plot the best fitting curve among all attempted fits with the original data points.

        Creates a matplotlib figure showing the original data points with error bars and the best-fitting 
        curve overlay. Handles both polynomial and function fits with appropriate labeling and metrics.
    
        Args:
            x_processed (numpy.ndarray): Processed x-values (pixel values, OD, or netOD)
            y_data (numpy.ndarray): Y-values representing dose measurements in Gy
            best_func (callable or int): Function object for function fits or integer for polynomial fits
            best_popt (numpy.ndarray): Optimized parameters for the best fit
            best_mse (float): Mean squared error of the best fit
            title (str): Title for the plot
            mode (ProcessingMode): Enum indicating the processing mode (PV, OD, or netOD)
    
        Returns:
            None: Displays the plot using plt.show()
    
        Notes:
            Will print "No valid best fit found" if best_func or best_popt is None
        """
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
            displayed_name = f"Polynomial Degree {best_func.split('_')[-1]}"
            plt.plot(x_fit, y_fit, '-', 
                    label=f"Best Fit: Polynomial Degree {displayed_name} (MSE: {best_mse:.2e})")
        else:
            # Function case
            y_fit = best_func(x_fit, *best_popt)
            func_name = best_func.__name__.lstrip('_')
            formatted_name = ' '.join(word.capitalize() for word in func_name.split('_'))
            plt.plot(x_fit, y_fit, '-', 
                    label=f'Best Fit: {formatted_name} (MSE: {best_mse:.2e})')
        
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
        """
        Plot all available fits including both polynomials and functions.

        Generates individual plots for each fit in the fitting results, showing original data
        points and fitted curves.
    
        Args:
            x_data (numpy.ndarray): Raw x-values
            y_data (numpy.ndarray): Y-values representing dose measurements in Gy
            fitting_results (list): List of dictionaries containing fitting results
            title (str): Base title for all plots
            mode (ProcessingMode): Enum indicating the processing mode
    
        Returns:
            None: Displays multiple plots
        """
        for i, result in enumerate(fitting_results):
            self._create_plot(x_data, y_data, result, title, mode)
            
    def _plot_polynomial_fits(self, x_data, y_data, fitting_results_poly, title, mode):
        """
        Plot only polynomial fits.

        Filters fitting results to include only polynomial fits and generates individual plots
        for each polynomial degree.
    
        Args:
            x_data (numpy.ndarray): Raw x-values
            y_data (numpy.ndarray): Y-values representing dose measurements in Gy
            fitting_results_poly (list): List of dictionaries containing polynomial fitting results
            title (str): Base title for all plots
            mode (ProcessingMode): Enum indicating the processing mode
    
        Returns:
            None: Displays multiple plots
        """
        poly_results = [result for result in fitting_results_poly if 'coefficients' in result]
        for result in poly_results:
            self._create_plot(x_data, y_data, result, title, mode)
            
    def _plot_specific_function(self, x_data, y_data, fitting_results, function_name, title, mode):
        """
        Plot a specific named function fit.

        Searches for and plots only the fits matching the specified function name.
    
        Args:
            x_data (numpy.ndarray): Raw x-values
            y_data (numpy.ndarray): Y-values representing dose measurements in Gy
            fitting_results (list): List of dictionaries containing all fitting results
            function_name (str): Name of the function to plot
            title (str): Base title for all plots
            mode (ProcessingMode): Enum indicating the processing mode
    
        Returns:
            None: Displays plots for matching functions
    
        Notes:
            Prints a message if no matching function is found
        """
        func_results = [result for result in fitting_results 
                       if 'function' in result and result['function'] == function_name]
        if not func_results:
            print(f"No fitting results found for function: {function_name}")
            return
        for result in func_results:
            self._create_plot(x_data, y_data, result, title, mode)
            
    def _create_plot(self, x_data, y_data, result, title, mode):
        """
        Create an individual plot for a single fit result.

       Handles the creation of a single plot with data points, error bars, and fitted curve.
       Supports both polynomial and function fits.
    
       Args:
           x_data (numpy.ndarray): Raw x-values
           y_data (numpy.ndarray): Y-values representing dose measurements in Gy
           result (dict): Dictionary containing fitting results and parameters
           title (str): Plot title
           mode (ProcessingMode): Enum indicating the processing mode
    
       Returns:
           None: Displays the plot
    
       Raises:
           Exception: Catches and prints any exceptions during value processing or plot creation
       """

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
            
            if result.get('function', '').startswith('polynomial_degree'):
                self._add_polynomial_fit(x_fit, result, title)
            else:
                self._add_function_fit(x_fit, result, title)
                
            plt.xlabel('Pixel Value' if mode == ProcessingMode.PV else 'OD' if mode == ProcessingMode.OD  else 'netOD')  
            plt.ylabel('Dose (Gy)')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
            
    def _add_polynomial_fit(self, x_fit, result, title):
        """
        Add a polynomial fit curve to an existing plot.

        Calculates and plots the polynomial fit curve using the provided coefficients,
        including degree and MSE in the label.
    
        Args:
            x_fit (numpy.ndarray): X-values for the fit curve
            result (dict): Dictionary containing polynomial fitting results
            title (str): Plot title
    
        Returns:
            None: Modifies existing plot
        """
        coefficients = result['coefficients']
        p = np.poly1d(coefficients)
        y_fit = p(x_fit)
        degree = result.get('degree', len(coefficients)-1)
        mse = f"{result['metrics']['mse']:.2e}" if result['metrics'] is not None else "Failed"
        
        plt.plot(x_fit, y_fit, '-', 
                label=f"Polynomial Fit: Degree {degree} (MSE: {mse})")
        plt.title(f"{title} - Degree {degree}")
        
    def _add_function_fit(self, x_fit, result, title):
        """
        Add a function fit curve to an existing plot.

        Calculates and plots the function fit curve using the provided function and 
        optimized parameters, including function name and MSE in the label.
     
        Args:
            x_fit (numpy.ndarray): X-values for the fit curve
            result (dict): Dictionary containing function fitting results
            title (str): Plot title
     
        Returns:
            None: Modifies existing plot
     
        Notes:
            Requires function to exist in self.fitting_functions
        """
        popt = result['coefficients']
        func_name = result['function']
        
        if func_name in self.fitting_functions:
            func = self.fitting_functions[func_name]
            y_fit = func(x_fit, *popt)
            mse = f"{result['metrics']['mse']:.2e}" if result['metrics'] is not None else "Failed"
            
            plt.plot(x_fit, y_fit, '-', 
                    label=f"Fit: {func_name} (MSE: {mse})")
            plt.title(f"{title} - {func_name}")
