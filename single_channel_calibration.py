# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:36:15 2025

@author: Ele_p
"""

import numpy as np
from typing import Tuple, Dict, Any
from Calibration_EBT3 import CurveFitter, ProcessingMode
import matplotlib.pyplot as plt


class SingleChannelCalibration(CurveFitter):
    """Single channel calibration that inherits from CurveFitter"""
    def __init__(self, channel_name: str):
        super().__init__()  # Initialize the parent CurveFitter class
        self.channel_name = channel_name
        self.calibration_results = {}
        self.background_mean = None
        
    def _find_background_mean(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """Find the x value (background mean) that corresponds to y=0."""
        idx_below = np.where(y_data <= 0)[0]
        idx_above = np.where(y_data > 0)[0]
        
        if len(idx_below) == 0 or len(idx_above) == 0:
            raise ValueError(f"Cannot find background mean for {self.channel_name}: calibration data doesn't cross y=0")
            
        idx_below = idx_below[-1]
        idx_above = idx_above[0]
        
        x1, y1 = x_data[idx_below], y_data[idx_below]
        x2, y2 = x_data[idx_above], y_data[idx_above]
        
        return x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
        
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], mode: ProcessingMode = ProcessingMode.NET_OD) -> Dict[str, Any]:
        """Calibrate single channel using inherited CurveFitter functionality."""
        # Using calculate_best_fit directly from CurveFitter (parent class)
        best_func, best_popt, best_mse, _ = self.calculate_best_fit(
            data[0], data[1], mode
        )
        
        self.calibration_results = {
            'function': best_func.__name__,  # Store the actual function since we inherit from CurveFitter
            'parameters': best_popt,
            'mse': best_mse
        }
        
        self.background_mean = self._find_background_mean(data[0], data[1])
        print(self.background_mean)
        return self.calibration_results
        
    def calculate_dose(self, image: np.ndarray) -> np.ndarray:
        """Calculate dose for single channel."""
        if not self.calibration_results or self.background_mean is None:
            raise ValueError(f"{self.channel_name} channel must be calibrated before calculating dose")
            
        log_PV = np.where(image / self.background_mean > 0,
                         -np.log10(image / self.background_mean), 0)
        print(log_PV)
        self.func = self.calibration_results['function']
        print(self.func)
        
        red_coeffs = self.calibration_results['parameters']
        # Apply calibration curves to obtain dose maps
        dose_gy_r = self.func(log_PV, *red_coeffs)
        
        return dose_gy_r


        
        