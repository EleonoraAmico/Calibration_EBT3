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
    """
    Multi-channel calibration that inherits from CurveFitter
    Implements multichannel calibration for radiochromic film dosimetry.
    Combines information from all three color channels (RGB) to improve accuracy 
    with a weighted average.

    This class implements a comprehensive approach to radiochromic film dosimetry
    by leveraging data from all three color channels (RGB) simultaneously. It creates
    individual calibrations for each channel and then combines their outputs using an
    optimal weighting scheme based on calibration performance.

    The process consists of:
    1. Calibrating each color channel (red, green, blue) independently.
    2. Calculating optimal weights for each channel based on fitting performance.
    3. Combining dose calculations from all channels for improved accuracy.
    4. Providing both individual channel results and weighted combined results.
    
    This multi-channel approach typically yields more accurate and robust dose measurements
    than single-channel methods, especially in regions where one channel may have poor sensitivity
    or non-linear response characteristics.
    
    Attributes
    ----------
        channels : Dict[str, SingleChannelCalibration]
            Dictionary of SingleChannelCalibration instances for each color channel.
        weights : np.ndarray
            Array of normalized weights for each channel, calculated during calibration.
    
    Raises
    ------
        ValueError
            If attempting to calculate dose before calibration.
        Various exceptions from SingleChannelCalibration may propagate.
    
    Notes
    -----
        - This class inherits from CurveFitter which provides the underlying fitting algorithms.
        - The weighting scheme automatically gives higher influence to channels with better
          calibration performance (lower mean squared error).
        - The multiple channel approach provides redundancy and can reduce uncertainty
          in dose measurements compared to single-channel methods.
        - The class automatically handles all the complexity of multi-channel calibration
          and weighted combination.
    
    Examples
    --------
        >>> import numpy as np
        >>> # Create calibration data for each channel
        >>> dose_values = np.array([0, 0.5, 1, 2, 5, 10])
        >>> red_pixels = np.array([65000, 48000, 35000, 20000, 8000, 3000])
        >>> green_pixels = np.array([65000, 50000, 40000, 25000, 12000, 5000])
        >>> blue_pixels = np.array([65000, 52000, 45000, 30000, 15000, 8000])
        >>> 
        >>> # Initialize multi-channel calibration
        >>> multi_calib = MultiChannelCalibration()
        >>> 
        >>> # Perform calibration for all channels
        >>> calib_results, weights = multi_calib.calibrate_channels(
        ...     (dose_values, red_pixels),
        ...     (dose_values, green_pixels),
        ...     (dose_values, blue_pixels),
        ...     ProcessingMode.PV
        ... )
        >>> 
        >>> # Calculate dose from a new image with all channels
        >>> combined_dose, red_dose, green_dose, blue_dose = multi_calib.calculate_combined_dose(
        ...     red_image, green_image, blue_image, ProcessingMode.PV
        ... )
    
    See Also
    --------
        SingleChannelCalibration : Class for calibrating individual color channels.
        CurveFitter : Parent class providing curve fitting functionality.
        ProcessingMode : Enum defining different processing modes for film data.
    
    Theory
    ------
        Radiochromic film responds to radiation by darkening across multiple wavelengths,
        resulting in changes to all three RGB channels when scanned. Each channel exhibits
        different sensitivity and saturation characteristics depending on dose levels:
        
        - Red channel: Typically most sensitive at low to medium doses
        - Green channel: Often optimal for medium dose ranges
        - Blue channel: Sometimes more effective at higher dose ranges
        
        The multi-channel approach leverages the complementary nature of these responses.
        By weighting each channel according to its calibration performance (inverse of MSE),
        the combined result automatically emphasizes channels with better performance while
        reducing the influence of channels with poorer response.
        
        This weighted combination approach has been shown in the literature to provide superior
        accuracy and precision compared to single-channel methods, particularly across wide
        dose ranges or when dealing with film batches with non-standard response characteristics.
    """
    def __init__(self):
        """
        Initialize the MultiChannelCalibration instance.
         
        Creates three SingleChannelCalibration objects for Red, Green, and Blue
        channels and initializes weights to None.
         
        Parameters
        ----------
             None
             
        Returns
        -------
             None
             
        Notes
        -----
             Initializes the parent CurveFitter class using super().__init__()
             Creates a dictionary with SingleChannelCalibration instances for each RGB channel
             Sets weights to None initially, to be calculated during calibration
        """
        super().__init__()  # Initialize the parent CurveFitter class
        self.channels = {
            'blue': SingleChannelCalibration('Blue'),
            'red': SingleChannelCalibration('Red'),
            'green': SingleChannelCalibration('Green')
        }
        self.weights = None
        
    def calibrate_channels(
        self, 
        red_data: Tuple[np.ndarray, np.ndarray],
        green_data: Tuple[np.ndarray, np.ndarray],
        blue_data: Tuple[np.ndarray, np.ndarray],
        mode: ProcessingMode = ProcessingMode.PV
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calibrate all channels and calculate weights.
        
        Performs calibration for each RGB channel independently and calculates
        optimal weights for combining them based on the mean squared error (MSE)
        of each channel's calibration.
        
        Parameters
        ----------
            red_data: Tuple[np.ndarray, np.ndarray]
                Tuple containing (signal_values, dose_values) for the red channel
            green_data: Tuple[np.ndarray, np.ndarray]
                Tuple containing (signal_values, dose_values) for the green channel
            blue_data: Tuple[np.ndarray, np.ndarray]
                Tuple containing (signal_values, dose_values) for the blue channel
            mode: ProcessingMode
                Processing mode to use during calibration. Default is ProcessingMode.PV
                
        Returns
        -------
            Dict[str, Dict[str, Any]]: 
                Dictionary containing calibration results for each channel
                
            numpy.ndarray:
                Array of normalized weights for each channel
                
        Raises
        ------
            Errors from underlying SingleChannelCalibration.calibrate() may propagate
            
        Notes
        -----
            - Weights are calculated as the inverse of the MSE for each channel
            - Weights are normalized to sum to 1.0
            - The weights are stored in the instance for later use in dose calculation
            - Channel order in the weights array: [Blue, Red, Green]
        """
        calibration_results = {}
        
        # Calibrate each channel using SingleChannelCalibration instances
        for channel_name, data in [('blue', blue_data), ('red', red_data), ('green', green_data)]:
            calibration_results[channel_name] = self.channels[channel_name].calibrate(data, mode)
        
        # Calculate weights based on MSE
        mse_values = [results['mse'] for results in calibration_results.values()]
        weights = 1 / np.array(mse_values)
        self.weights = weights / weights.sum()
        
        return calibration_results, self.weights
    
    def calculate_combined_dose(
        self,
        image_red: np.ndarray,
        image_green: np.ndarray,
        image_blue: np.ndarray, 
        mode: ProcessingMode = ProcessingMode.PV
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate combined dose from all channels.
        
        Computes dose from each channel independently and then combines them
        using the previously calculated optimal weights to produce a final
        dose estimate with improved accuracy.
        
        Parameters
        ----------
            image_red: np.ndarray
                Array of red channel values from the radiochromic film scan
            image_green: np.ndarray
                Array of green channel values from the radiochromic film scan  
            image_blue: np.ndarray
                Array of blue channel values from the radiochromic film scan
            mode: ProcessingMode
                Processing mode to use for dose calculation. Default is ProcessingMode.PV
                
        Returns
        -------
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - dose_combined: Weighted average dose from all channels
                - dose_gy_r: Dose calculated from red channel only
                - dose_gy_g: Dose calculated from green channel only
                - dose_gy_b: Dose calculated from blue channel only
                
        Raises
        ------
            ValueError: 
                If weights are None, meaning channels have not been calibrated
            Errors from underlying SingleChannelCalibration.calculate_dose() may propagate
            
        Examples
        --------
            >>> # After calibration
            >>> calib = MultiChannelCalibration()
            >>> calib.calibrate_channels(red_data, green_data, blue_data)
            >>> 
            >>> # Calculate dose from a film scan
            >>> combined, red, green, blue = calib.calculate_combined_dose(
            ...     scan_red, scan_green, scan_blue
            ... )
            
        Notes
        -----
            - Channels must be calibrated before this method can be used
            - Channel weights are based on the calibration MSE, giving higher
              weight to channels with better performance
        """
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
        
        dose_combined = (weights_dict['Blue'] * dose_gy_b +
                        weights_dict['Red'] * dose_gy_r +
                        weights_dict['Green'] * dose_gy_g)
        
        return dose_combined, dose_gy_r, dose_gy_g, dose_gy_b


    
   