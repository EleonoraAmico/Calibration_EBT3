# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:36:15 2025

@author: Ele_p
"""

import numpy as np
from typing import Tuple, Dict, Any
from Calibration_EBT3 import CurveFitter, ProcessingMode



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
        
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], mode: ProcessingMode = ProcessingMode.PV) -> Dict[str, Any]:
        """Calibrate single channel using inherited CurveFitter functionality."""
        # Using calculate_best_fit directly from CurveFitter (parent class)
        best_func, best_popt, best_mse, _ = self.calculate_best_fit(
            data[0], data[1], mode
        )
        # Handle case when best_func is a polynomial degree number
        if isinstance(best_func, int):
            function_name = f'polynomial_degree_{best_func}'
            print(f"Function name: {function_name}, Type: {type(function_name)}")
        else:
            function_name = best_func.__name__
        
        self.calibration_results = {
            'function': function_name,  # Store the actual function since we inherit from CurveFitter
            'parameters': best_popt,
            'mse': best_mse,
            'mode': mode
        }

        self.background_mean = self._find_background_mean(data[0], data[1])
        print(self.background_mean)
        return self.calibration_results
        
    def calculate_dose(self, image: np.ndarray, mode: ProcessingMode = ProcessingMode.PV) -> np.ndarray:
        """Calculate dose for single channel."""
        if not self.calibration_results or self.background_mean is None:
            raise ValueError(f"{self.channel_name} channel must be calibrated before calculating dose")
        
        # Check if modes match
        if mode != self.calibration_results['mode']:
            print(f"Warning: Processing mode mismatch. Calibration used {self.calibration_results['mode']}, but calculating dose with {mode}. They should be the same for accurate results.")
        # self.func = getattr(CurveFitter, self.calibration_results['function'], None)

        # # Check if the function is valid
        # if not callable(self.func):
        #     raise ValueError(f"Invalid function '{self.calibration_results['function']}' for {self.channel_name} channel")
        # # self.func = self.calibration_results['function']
        # print(self.func)
        function_name = self.calibration_results['function']
        print(function_name)
        coeffs = self.calibration_results['parameters']
        # Prepare the input data based on mode
        if mode == ProcessingMode.PV:
            input_data = image
        elif mode == ProcessingMode.OD:
            input_data = np.log10(65535 / image)
        elif mode == ProcessingMode.NET_OD:
            input_data = np.where(image / self.background_mean > 0,
                               -np.log10(image / self.background_mean), 0)
        
        # Handle polynomial case
        function_name = function_name.strip().lower()
        if function_name.startswith('polynomial'):
            print("Polynomial function detected.")
        else:
            print(f"Function '{function_name}' does not start with 'polynomial'.")
        if function_name.startswith('polynomial'):
            print(coeffs)
            p = np.poly1d(coeffs)
            dose_gy_r = p(input_data)
        else:
            # Handle regular function case
            self.func = getattr(CurveFitter, function_name, None)
            if not callable(self.func):
                raise ValueError(f"Invalid function '{function_name}' for {self.channel_name} channel")
            dose_gy_r = self.func(self, input_data, *coeffs)
        # red_coeffs = self.calibration_results['parameters']
        
        # if mode == ProcessingMode.PV:
        #     dose_gy_r=self.func(self, image, *red_coeffs)
            
        # elif mode == ProcessingMode.OD:
        #     OD_image = np.log10(65535 / image)
        #     dose_gy_r=self.func(self, OD_image, *red_coeffs)
            
        # elif mode == ProcessingMode.NET_OD:
        #     log_PV = np.where(image / self.background_mean > 0,
        #                      -np.log10(image / self.background_mean), 0)
        #     dose_gy_r = self.func(self, log_PV, *red_coeffs)
        
        return dose_gy_r


        
        