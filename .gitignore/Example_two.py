# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:28:20 2025

@author: Ele_p
"""

import pandas as pd

from Calibration_EBT3 import CurveFitter, ProcessingMode
from plot_function import FitPlotter, PlotType
import pandas as pd
from single_channel_calibration import SingleChannelCalibration
import numpy as np
import matplotlib.pyplot as plt
from multi_calibration import MultiChannelCalibration
from skimage.io import imread

# Leggi il file txt come se fosse un csv, specificando lo spazio come separatore
df = pd.read_csv('calibration_blu_new_scanner.txt', sep='\t', header=None)

# Estrai la seconda colonna (indice 1) e convertila in float
x_data_blue = df.iloc[:, 1].astype(float)

# Se vuoi vedere i primi valori
print(x_data_blue.head())
# Se vuoi alcune statistiche base
print(x_data_blue.describe())

# Leggi il CSV
df = pd.read_csv('C:/Users/Ele_p/EBT3_calibration/Channel_red_DvsPV.csv')

y_data = df.iloc[:, 0].astype(float)/100 #Dose

blue_data = [x_data_blue, y_data]
#%%
# Create an instance of CurveFitter
print(len(x_data_blue))
fitter = CurveFitter()
#%%
fitting_results = fitter.calculate_non_linear_fit(x_data_blue, y_data, mode=ProcessingMode.NET_OD, print_results=False)
fitter.log_fitting_results(fitting_results)
#%%
fitter = CurveFitter()
fitting_results = fitter.polynomial_fit(x_data_blue, y_data, mode=ProcessingMode.NET_OD)
fitter.log_fitting_results(fitting_results)


#%%
best_func, best_coeffs, best_score, fitting_results = fitter._select_best_fit(fitting_results)
#%%
print(best_func)
print(best_coeffs)
#%%
fitter1 = CurveFitter()
fitter2 = CurveFitter()
assert fitter1.logger is not fitter2.logger  # Should be True if it's stateless
#%%




