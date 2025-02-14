# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:26:36 2025

@author: Ele_p
"""
import numpy as np
from typing import Tuple, Dict, Any
from Calibration_EBT3 import CurveFitter, ProcessingMode
from single_channel_calibration import SingleChannelCalibration


class MultiChannelCalibration(CurveFitter):
    """Multi-channel calibration that inherits from CurveFitter"""
    def __init__(self):
        super().__init__()  # Initialize the parent CurveFitter class
        self.channels = {
            'blue': SingleChannelCalibration('Blue'),
            'red': SingleChannelCalibration('Red'),
            'green': SingleChannelCalibration('Green')
        }
        self.weights = None
        
    def calibrate_channels(self, 
                          red_data: Tuple[np.ndarray, np.ndarray],
                          green_data: Tuple[np.ndarray, np.ndarray],
                          blue_data: Tuple[np.ndarray, np.ndarray],
                          mode: ProcessingMode = ProcessingMode.PV) -> Dict[str, Dict[str, Any]]:
        """Calibrate all channels and calculate weights."""
        calibration_results = {}
        
        # Calibrate each channel using SingleChannelCalibration instances
        for channel_name, data in [('blue', blue_data), ('red', red_data), ('green', green_data)]:
            calibration_results[channel_name] = self.channels[channel_name].calibrate(data, mode)
        
        # Calculate weights based on MSE
        mse_values = [results['mse'] for results in calibration_results.values()]
        print(mse_values)
        weights = 1 / np.array(mse_values)
        self.weights = weights / weights.sum()
        
        return calibration_results
    
    def calculate_combined_dose(self,
                              image_red: np.ndarray,
                              image_green: np.ndarray,
                              image_blue: np.ndarray, mode = ProcessingMode.PV) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate combined dose from all channels."""
        if self.weights is None:
            raise ValueError("Channels must be calibrated before calculating combined dose")
        
        # Calculate individual doses
        dose_gy_b = self.channels['blue'].calculate_dose(image_blue, mode)
        dose_gy_r = self.channels['red'].calculate_dose(image_red, mode)
        dose_gy_g = self.channels['green'].calculate_dose(image_green, mode)
        
        # Calculate weighted average
        weights_dict = {
            'Blue': self.weights[0],
            'Red': self.weights[1],
            'Green': self.weights[2]
        }
        print(weights_dict)
        
        dose_combined = (weights_dict['Blue'] * dose_gy_b +
                        weights_dict['Red'] * dose_gy_r +
                        weights_dict['Green'] * dose_gy_g)
        
        return dose_combined, dose_gy_r, dose_gy_g, dose_gy_b


    
    # def predict_dose(self, red_value: float, green_value: float, blue_value: float) -> float:
    #     """
    #     Predict dose using weighted combination of all channels.
    #     """
    #     if not self.calibration_results or self.weights is None:
    #         raise ValueError("Must run calibrate_channels first")
            
    #     predictions = []
    #     for i, (channel, (func, params, _, _)) in enumerate(self.calibration_results.items()):
    #         value = {'blue': blue_value, 'red': red_value, 'green': green_value}[channel]
    #         predictions.append(func(value, *params) * self.weights[i])
            
    #     return sum(predictions)

    # def plot_dose_predictions(true_dose, results):
    #     """
    #     Create a scatter plot comparing predicted doses with true doses.
        
    #     Parameters:
    #     -----------
    #     true_dose : array-like
    #         True dose values
    #     results : dict
    #         Dictionary containing prediction results from calculate_weights()
    #     """
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10, 6))
        
    #     plt.scatter(true_dose, results['pred_dose_blue'], 
    #                 color='blue', label='Predizione Dose Blue', alpha=0.6)
    #     plt.scatter(true_dose, results['pred_dose_red'], 
    #                 color='red', label='Predizione Dose Red', alpha=0.6)
    #     plt.scatter(true_dose, results['pred_dose_green'], 
    #                 color='green', label='Predizione Dose Green', alpha=0.6)
    #     plt.scatter(true_dose, results['pred_dose_combined'], 
    #                 color='purple', label='Predizione Dose Combinata', alpha=0.8)
        
    #     # Add ideal fit line
    #     min_dose = true_dose.min()
    #     max_dose = true_dose.max()
    #     plt.plot([min_dose, max_dose], [min_dose, max_dose], 
    #             color='black', linestyle='--', label='Ideal Fit')
        
    #     plt.xlabel('Dose Reale')
    #     plt.ylabel('Dose Predetta')
    #     plt.title('Confronto tra le predizioni della Dose e la Dose Reale')
    #     plt.legend()
    #     plt.show()