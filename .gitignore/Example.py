# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:12:53 2025

@author: Ele_p
"""


import os
print(os.listdir("C:/Users/Ele_p/EBT3_calibration"))

#%%
import sys
sys.path.append("C:/Users/Ele_p/EBT3_calibration")

#%%

from Calibration_EBT3 import CurveFitter, ProcessingMode
from plot_function import FitPlotter, PlotType
import pandas as pd
from single_channel_calibration import SingleChannelCalibration
import numpy as np
import matplotlib.pyplot as plt
from multi_calibration import MultiChannelCalibration
from skimage.io import imread
#%%
# Leggi il CSV
df = pd.read_csv('C:/Users/Ele_p/EBT3_calibration/Channel_red_DvsPV.csv')
# Leggi il CSV
df_red = pd.read_csv('C:/Users/Ele_p/EBT3_calibration/Channel_red_DvsPV')
# Leggi il CSV
df_green = pd.read_csv('C:/Users/Ele_p/EBT3_calibration/Channel_green_DvsPV')
# Leggi il CSV
df_blue = pd.read_csv('C:/Users/Ele_p/EBT3_calibration/Channel_blue_DvsPV')


#%%
# Estrai i dati
x_data = df.iloc[:, 1].astype(float) # Pixel Value/netOD
y_data = df.iloc[:, 0].astype(float)/100 #Dose

#%%
# Estrai i dati
x_data_red = df_red.iloc[:, 1].astype(float) # Pixel Value/netOD
y_data_red = df_red.iloc[:, 0].astype(float)/100 #Dose

# Estrai i dati
x_data_blue = df_blue.iloc[:, 1].astype(float) # Pixel Value/netOD
y_data_blue = df_blue.iloc[:, 0].astype(float)/100 #Dose

# Estrai i dati
x_data_green = df_green.iloc[:, 1].astype(float) # Pixel Value/netOD
y_data_green = df_green.iloc[:, 0].astype(float)/100 #Dose

#%%
red_data=(x_data, y_data)
green_data=(x_data_green, y_data_green)
blue_data=(x_data_blue, y_data_blue)
#%%
# Create an instance of CurveFitter
fitter = CurveFitter()
fitting_results = fitter.calculate_non_linear_fit(x_data_red, y_data_red, mode=ProcessingMode.PV, print_results=True)
#%%
fitting_results_polynomial = fitter.polynomial_fit(x_data_red, y_data_red, mode=ProcessingMode.PV)
fitter.log_fitting_results(fitting_results_polynomial)
#%%
fitter.plot_all_fits(x_data, y_data)
#%%
fitter.plot_best_fits(x_data, y_data, mode=ProcessingMode.OD)
#%%
best_func, best_popt, best_mse, fitting_results = fitter.polynomial_fit(x_data, y_data)
#%%
fitter.plot_polynomial_fits(x_data, y_data)
#%%
plot = FitPlotter()

#plot.plot_fits(x_data, y_data, title="Fits", plot_type=PlotType.ALL)

#%%
plot.plot_fits(x_data, y_data, title="Fits", plot_type=PlotType.POLYNOMIAL, mode=ProcessingMode.PV)

#%%
plot.plot_fits(x_data, y_data, title="exponential decreasing", plot_type=PlotType.FUNCTION, function_name = "exponential_decreasing",  mode=ProcessingMode.PV)

#%%
plot.plot_fits(x_data, y_data, title = "Double exponential", plot_type=PlotType.FUNCTION,
               function_name = "double_exponential", mode=ProcessingMode.NET_OD)
#%%
plot.plot_fits(x_data_red, y_data_red, title="Calibration Red Channel", plot_type=PlotType.POLYNOMIAL, mode=ProcessingMode.NET_OD)
#%%
plot.plot_fits(x_data_green, y_data_green, title="Calibration Green Channel", plot_type=PlotType.POLYNOMIAL, mode=ProcessingMode.NET_OD)
plot.plot_fits(x_data_blue, y_data_blue, title="Calibration Blue Channel", plot_type=PlotType.POLYNOMIAL, mode=ProcessingMode.NET_OD)

#%%
plot.plot_fits(x_data, y_data, plot_type=PlotType.BEST_FIT, mode=ProcessingMode.PV)


#%%
from skimage.io import imread
image_down=imread("C:/Users/Ele_p/EBT3_calibration/CR-AVG_DOWN_Orizzontale.tif")
image_r_A=imread("C:/Users/Ele_p/EBT3_calibration/a001r.tif")
image_r_B=imread("C:/Users/Ele_p/EBT3_calibration/b001r.tif")
image_r_C=imread("C:/Users/Ele_p/EBT3_calibration/c001r.tif")


#%%
single_calibration = SingleChannelCalibration(red_data)
results_red=single_calibration.calibrate(red_data, mode=ProcessingMode.NET_OD)


#%%
calibrated_image_C=single_calibration.calculate_dose(image_r_C, mode=ProcessingMode.NET_OD)
# Calcola statistiche
mean_dose_gy = np.mean(calibrated_image_C)
median_dose_gy = np.median(calibrated_image_C)
max_dose_gy = np.max(calibrated_image_C)

# Visualizza l'immagine e stampa la dose media
plt.imshow(calibrated_image_C, cmap=plt.cm.bone)
plt.colorbar(label='Dose (Gy)')
plt.title('Dose Map "B" Single Channel Calibration - Red Channel')
plt.show()

# Stampa statistiche
print('Media di dose_gy:', mean_dose_gy)
print('Mediana di dose_gy:', median_dose_gy)
print('Max di dose_gy:', max_dose_gy)
print(np.shape(calibrated_image_C))

#%%

image_down=imread("C:/Users/Ele_p/EBT3_calibration/CR-AVG_DOWN_Orizzontale.tif")

calibrated_image=single_calibration.calculate_dose(image_down, mode=ProcessingMode.OD)

#%%
# Calcola statistiche
mean_dose_gy = np.mean(calibrated_image)
median_dose_gy = np.median(calibrated_image)
max_dose_gy = np.max(calibrated_image)

# Visualizza l'immagine e stampa la dose media
plt.imshow(calibrated_image, cmap=plt.cm.bone)
plt.colorbar(label='Dose (Gy)')
plt.title('Dose Map BG Combined Channel')
plt.show()

# Stampa statistiche
print('Media di dose_gy:', mean_dose_gy)
print('Mediana di dose_gy:', median_dose_gy)
print('Max di dose_gy:', max_dose_gy)
print(np.shape(calibrated_image))
#multi_calibration.calibrate_channels(blue_data, red_data, green_data, mode = ProcessingMode.PV)
#%%
# Leggi il CSV
df_red = pd.read_csv('C:/Users/Ele_p/EBT3_calibration/Channel_red_DvsPV')
df_green = pd.read_csv('C:/Users/Ele_p/EBT3_calibration/Channel_green_DvsPV')
df_blue = pd.read_csv('C:/Users/Ele_p/EBT3_calibration/Channel_blue_DvsPV')
# Estrai i dati
x_data_red = df_red.iloc[:, 1].astype(float) # Pixel Value/netOD
y_data_red = df_red.iloc[:, 0].astype(float)/100 #Dose
x_data_green = df_green.iloc[:, 1].astype(float) # Pixel Value/netOD
y_data_green = df_green.iloc[:, 0].astype(float)/100 #Dose
x_data_blue = df_blue.iloc[:, 1].astype(float) # Pixel Value/netOD
y_data_blue = df_blue.iloc[:, 0].astype(float)/100 #Dose
#%% 
blue_data=(x_data_blue, y_data_blue)
red_data=(x_data_red, y_data_red)
green_data=(x_data_green, y_data_green)
multi_calibration = MultiChannelCalibration()

calibration_results= multi_calibration.calibrate_channels(
    red_data, green_data, blue_data, mode= ProcessingMode.PV)

#%%
image_red=imread("C:/Users/Ele_p/EBT3_calibration/CR-crop_down_orizzontale.tif")
image_green=imread("C:/Users/Ele_p/EBT3_calibration/CG-crop_down_orizzontale.tif")
image_blue=imread("C:/Users/Ele_p/EBT3_calibration/CB-crop_down_orizzontale.tif")

dose_combined_down, dose_gy_r, dose_gy_g, dose_gy_b=multi_calibration.calculate_combined_dose( 
    image_red, image_green, image_blue, mode=ProcessingMode.NET)
# Calcola statistiche
mean_dose_gy = np.mean(dose_combined_down)
median_dose_gy = np.median(dose_combined_down)
max_dose_gy = np.max(dose_combined_down)

# Visualizza l'immagine e stampa la dose media
plt.imshow(dose_combined_down, cmap=plt.cm.bone)
plt.colorbar(label='Dose (Gy)')
plt.title('Dose Map Down Combined Channel')
plt.show()

# Stampa statistiche
print('Media di dose_gy:', mean_dose_gy)
print('Mediana di dose_gy:', median_dose_gy)
print('Max di dose_gy:', max_dose_gy)
print(np.shape(dose_combined_down))
#multi_calibration.calibrate_channels(blue_data, red_data, green_data, mode = ProcessingMode.PV)

#%%
image_red=imread("C:/Users/Ele_p/EBT3_calibration/CR-crop_up_orizzontale.tif")
image_green=imread("C:/Users/Ele_p/EBT3_calibration/CG-crop_up_orizzontale.tif")
image_blue=imread("C:/Users/Ele_p/EBT3_calibration/CB-crop_up_orizzontale.tif")

dose_combined_up, dose_gy_r, dose_gy_g, dose_gy_b=multi_calibration.calculate_combined_dose( 
    image_red, image_green, image_blue, mode=ProcessingMode.PV)
# Calcola statistiche
mean_dose_gy = np.mean(dose_combined_up)
median_dose_gy = np.median(dose_combined_up)
max_dose_gy = np.max(dose_combined_up)

# Visualizza l'immagine e stampa la dose media
plt.imshow(dose_combined_up, cmap=plt.cm.bone)
plt.colorbar(label='Dose (Gy)')
plt.title('Dose Map Up Combined Channel')
plt.show()

# Stampa statistiche
print('Media di dose_gy:', mean_dose_gy)
print('Mediana di dose_gy:', median_dose_gy)
print('Max di dose_gy:', max_dose_gy)
print(np.shape(dose_combined_up))

#%%
from PIL import Image, TiffImagePlugin

matrix = calibrated_image_C
image = Image.fromarray(matrix)

# Add metadata (e.g., resolution in DPI, description, etc.)
metadata = TiffImagePlugin.ImageFileDirectory_v2()
metadata[256] = matrix.shape[1]  # Image width
metadata[257] = matrix.shape[0]  # Image height
metadata[282] = (72)         # X resolution (e.g., 300 DPI)
metadata[283] = (72)         # Y resolution (e.g., 300 DPI)
metadata[305] = "Generated by Python"  # Software

# Save the TIFF file with metadata
output_file = "C:/Users/Ele_p/EBT3_calibration/Tiff/second_experiment/dose_map_C_single_channel_Red.tiff"
image.save(output_file, tiffinfo=metadata)

print(f"TIFF image saved as {output_file} with metadata.")

#%%

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

For example, if they are in a module named "calibration", you might uncomment:
    from calibration import CurveFitter, ProcessingMode, PlotType, FitPlotter
"""

import pandas as pd

# Import the necessary classes from your calibration module.
# Uncomment and modify the following line based on your project structure:
# from calibration import CurveFitter, ProcessingMode, PlotType, FitPlotter

# -----------------------------------------------------------------------------
# Step 1: Read the CSV file containing the red channel calibration data.
# -----------------------------------------------------------------------------
# We use a relative path so that the code remains portable and does not expose
# any local file system paths.
data_file = 'Example dates/Channel_red_DvsPV.csv'
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
    x_data_red, y_data_red, mode=ProcessingMode.PV, print_results=True
)

# -----------------------------------------------------------------------------
# Step 4: Perform polynomial curve fitting and log the results.
# -----------------------------------------------------------------------------
# Compute a polynomial fit for the red channel data and log the fitting results.
fitting_results_polynomial = fitter.polynomial_fit(
    x_data_red, y_data_red, mode=ProcessingMode.PV
    )
fitter.log_fitting_results(fitting_results_polynomial)

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
    mode=ProcessingMode.NET_OD
)

# Example: Plot an exponential decreasing function fit for the red channel data.
plot.plot_fits(
    x_data_red, y_data_red,
    title="Exponential Decreasing Fit (Red Channel)",
    plot_type=PlotType.FUNCTION,
    function_name="exponential_decreasing",
    mode=ProcessingMode.PV
)

# (Optional) Add more plot types or fitting examples as needed to demonstrate the
# full functionality of your code.

