# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:14:20 2025

@author: Eleonora Cristina Amico
"""


"""
Example Script: Red Channel Calibration and Curve Fitting

This script demonstrates how to:
1. Read a CSV file containing calibration data for the red channel.
2. Extract and preprocess the relevant data.
3. Perform non-linear and polynomial curve fitting.
4. Log the fitting results.
5. Generate plots to visualize the fitting outcomes.

Note:
- The CSV file should be placed in the "Example dates" folder.
- The CSV file is expected to have at least two columns:
  • The first column contains the dose values.
  • The second column contains the pixel values (or netOD values).
- Adjust the scaling of the dose values as needed (here we divide by 100).

Before running this script, ensure that you have implemented and can import the following:
    CurveFitter, ProcessingMode, PlotType, FitPlotter

"""
import pandas as pd
# Import the necessary classes from Calibration_EBT3 and plot_function.
#import calibration_ebt3
from Calibration_EBT3 import CurveFitter, ProcessingMode
from Calibration_EBT3.plot_function import FitPlotter, PlotType

# -----------------------------------------------------------------------------
# Step 1: Read the CSV file containing the red channel calibration data.
# -----------------------------------------------------------------------------
# We use a relative path so that the code remains portable and does not expose
# any local file system paths.
import os

tutorial_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(tutorial_dir, "example", "channel_red.csv")
df = pd.read_csv(data_file)


# -----------------------------------------------------------------------------
# Step 2: Extract the red channel data from the CSV.
# -----------------------------------------------------------------------------
# - Extract the pixel values from the second column and convert them to floats.
# - Extract the dose values from the first column, convert them to floats, and
#   apply scaling (here, dividing by 100) if required because dose should be 
#   expressed in Gray.
x_data_red = df.iloc[:, 1].astype(float)      # Pixel values (netOD)
y_data_red = df.iloc[:, 0].astype(float) / 100  # Dose values (scaled)
#%%
# -----------------------------------------------------------------------------
# Step 3: Perform non-linear curve fitting using the CurveFitter.
# -----------------------------------------------------------------------------
# Create an instance of the CurveFitter class and perform a non-linear fit on
# the red channel data. The 'mode' parameter (ProcessingMode.PV) specifies that
# the fitting should be based on pixel values. It is possible to change 'mode'
# parameter in ProcessingMode.OD to process x values in Optical Density or in 
# ProcessingMode.NET_OD to process x values in net Optical Density.

fitter = CurveFitter()
fitting_results = fitter.calculate_non_linear_fit(
    x_data_red, y_data_red, mode=ProcessingMode.NET_OD, print_results=True
)
# Note: Setting print_results=True will automatically log the fitting results.
#%%
# -----------------------------------------------------------------------------
# Step 4: Perform polynomial curve fitting and log the results.
# -----------------------------------------------------------------------------
# Compute a polynomial fit for the red channel data and log the fitting results.

fitting_results_polynomial = fitter.polynomial_fit(
    x_data_red, y_data_red, mode=ProcessingMode.PV
)
fitter.log_fitting_results(fitting_results_polynomial)
#%%
# Select the best fit based on a specified metric (e.g., score)
best_funct, coeff, score, fitting_results = fitter.select_best_fit(
    fitting_results, selection_metric='mse'
)
#%%
print(best_funct)
print(coeff)
#%%

# -----------------------------------------------------------------------------
# Step 5: Generate plots to visualize the fitting results.
# -----------------------------------------------------------------------------
# Create an instance of the FitPlotter class to generate custom plots.
plot = FitPlotter()

# Example: Plot polynomial fits with a custom title for the red channel calibration.
plot.plot_fits(
    x_data_red, y_data_red,
    title="Calibration Red Channel (Polynomial Fit)",
    plot_type=PlotType.POLYNOMIAL,
    mode=ProcessingMode.PV
)

# Example: Plot an exponential decreasing function fit for the red channel data.
plot.plot_fits(
    x_data_red, y_data_red,
    title="Exponential Fit (Red Channel)",
    plot_type=PlotType.FUNCTION,
    function_name="exponential",
    mode=ProcessingMode.PV
)

# Plot all fitted curves (both polynomial and non-linear fits).
# PlotType.ALL will visualize all the fitting functions together.
plot.plot_fits(
    x_data_red, y_data_red,
    title="All Fits for Red Channel",
    plot_type=PlotType.ALL,
    mode=ProcessingMode.PV
)
# Note: PlotType.ALL will display all polynomial and non-linear fitted curves.
#%%
# Plot the best fit function (determined by the lowest mean squared error, MSE).
plot.plot_fits(
    x_data_red, y_data_red,
    title="Best Fit for Red Channel",
    plot_type=PlotType.BEST_FIT,
    mode=ProcessingMode.NET_OD
)
# Note: PlotType.BESTFIT will show only the function with the lowest MSE.
#%%
# Example: Plot polynomial fits with a custom title for the red channel calibration.
plot.plot_fits(
    x_data_red, y_data_red,
    title="Calibration Red Channel (Polynomial Fit)",
    plot_type=PlotType.POLYNOMIAL,
    mode=ProcessingMode.NET_OD
)

# Example: Plot an exponential function fit for the red channel data.
plot.plot_fits(
    x_data_red, y_data_red,
    title="Exponential Fit (Red Channel)",
    plot_type=PlotType.FUNCTION,
    function_name="exponential",
    mode=ProcessingMode.OD
)