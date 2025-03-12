# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 18:36:15 2025

@author: Ele_p
"""

import numpy as np
from typing import Tuple, Dict, Any
from Calibration_EBT3 import CurveFitter, ProcessingMode



class SingleChannelCalibration(CurveFitter):
    """
    Single channel calibration that inherits from CurveFitter.
    
    This class manages the calibration process for a single radiochromic film channel,
    providing functionality to determine the optimal mathematical relationship between
    pixel values (optical density or netOD) and radiation dose. It leverages the curve-fitting
    capabilities of its parent class to identify the best-performing function for 
    calibration and applies this function to convert image data to dose values.
    
    The process consists of:
    1. Taking calibration data (known dose values and corresponding pixel/OD/netOD measurements).
    2. Fitting multiple mathematical functions to the data using the parent CurveFitter class.
    3. Selecting the best-fitting function based on mean squared error.
    4. Storing the calibration results for subsequent dose calculations.
    5. Calculating dose values from new image data using the established calibration.
    
    Parameters
    ----------
        channel_name : str
            The name of the film channel being calibrated (e.g., 'red', 'green', 'blue').
    
    Attributes
    ----------
        channel_name : str
            The name of the film channel.
        calibration_results : dict
            Stores calibration results including the best function, its parameters, and error metrics.
        background_mean : float
            The mean pixel value for background (zero dose) measurements.
    
    Raises
    ------
        ValueError
            If attempting to calculate dose before calibration.
            If there's a mismatch between calibration mode and dose calculation mode.
            If an invalid function name is encountered during dose calculation.
    
    Notes
    -----
        - This class inherits from CurveFitter which provides the underlying fitting algorithms.
        - Different processing modes (PV, OD, NET_OD) offer flexibility in how film data is processed.
        - The class automatically determines the best mathematical function for the calibration curve
          from a set of candidate functions defined in the parent class.
        - Polynomial fitting is supported alongside other predefined functions.
    
    Examples
    --------
        >>> import numpy as np
        >>> # Create calibration data
        >>> dose_values = np.array([0, 0.5, 1, 2, 5, 10])
        >>> pixel_values = np.array([65000, 48000, 35000, 20000, 8000, 3000])
        >>> 
        >>> # Initialize calibration for red channel
        >>> red_channel = SingleChannelCalibration("red")
        >>> 
        >>> # Perform calibration using pixel value mode
        >>> calibration_results = red_channel.calibrate((dose_values, pixel_values), 
        ...                                            ProcessingMode.PV)
        >>> 
        >>> # Calculate dose from a new image
        >>> new_image = np.array([[64000, 45000], [32000, 7000]])
        >>> dose_map = red_channel.calculate_dose(new_image, ProcessingMode.PV)
        >>> print(f"Calculated dose map: {dose_map}")
    
    See Also
    --------
        CurveFitter : Parent class providing curve fitting functionality.
        ProcessingMode : Enum defining different processing modes for film data.
    
    Theory
    ------
        Radiochromic film dosimetry relies on the relationship between absorbed radiation dose
        and the film's color change, which can be quantified as pixel values, optical density or netOD.
        This relationship is typically non-linear and varies between film batches, requiring
        calibration for each measurement session.
        
        The calibration process involves exposing film pieces to known radiation doses,
        scanning them, and establishing a mathematical relationship between the measured
        quantities (pixel value, optical density) and dose. This relationship can then be
        used to convert measured values from unknown exposures to dose values.
        
        Multiple processing modes offer different approaches:
        - Pixel Value (PV): Direct use of scanner pixel values
        - Optical Density (OD): Logarithmic transformation of pixel values
        - Net Optical Density (NET_OD): OD relative to unexposed film
        
        The choice of processing mode depends on film type, dose range, and measurement protocol.
    """
    def __init__(self, channel_name: str):
        """
       Initialize the SingleChannelCalibration object.
       
       Parameters
       ----------
           channel_name : str
               The name of the film channel being calibrated (e.g., 'red', 'green', 'blue').
       """
        super().__init__()  # Initialize the parent CurveFitter class
        self.channel_name = channel_name
        self.calibration_results = {}
        self.background_mean = None
        
    def calibrate(self, data: Tuple[np.ndarray, np.ndarray], mode: ProcessingMode = ProcessingMode.PV) -> Dict[str, Any]:
        """
        Calibrate single channel using inherited CurveFitter functionality.
        
        This method determines the optimal mathematical relationship between the known
        dose values and corresponding measurements (pixel values or optical density).
        It evaluates multiple fitting functions and selects the best one based on
        mean squared error.
        
        Parameters
        ----------
            data : Tuple[np.ndarray, np.ndarray]
                A tuple containing (dose_values, pixel_values) as numpy arrays.
            mode : ProcessingMode, optional
                The processing mode for calibration (PV, OD, or NET_OD), default is PV.
                
        Returns
        -------
            Dict[str, Any]
                A dictionary containing calibration results: function name, parameters,
                mean squared error, and the processing mode used.
                
        Notes
        -----
            - Uses both non-linear fitting and polynomial fitting from the parent class.
            - Automatically selects the best-performing function.
            - Stores the background mean (zero dose pixel value) for NET_OD calculations.
            - Prints the name of the best-fitting function for user information.
        """
        # Using calculate_best_fit directly from CurveFitter (parent class)
        fitting_results = self.calculate_non_linear_fit(data[0], data[1], mode)
        fitting_results_poly = self.polynomial_fit(data[0], data[1], mode)
        fitting_results_all = fitting_results + fitting_results_poly
        best_func, best_popt, best_mse, _ = self.select_best_fit(fitting_results_all)
        # Handle case when best_func is a polynomial degree number
        
        if best_func.startswith('polynomial_degree_'):
            displayed_name = f"Polynomial Degree {best_func.split('_')[-1]}"
        else:
            # Replace underscores with spaces and capitalize each word
            displayed_name = ' '.join(word.capitalize() for word in best_func.split('_'))
        print("The best fitting function is ", displayed_name)
        self.calibration_results = {
            'function': best_func,  # Store the actual function since we inherit from CurveFitter
            'parameters': best_popt,
            'mse': best_mse,
            'mode': mode
        }
        x_values = data[0]
        y_values = data[1]
        self.background_mean =  x_values[np.where(y_values == 0)[0][0]]

        return self.calibration_results
        
    def calculate_dose(self, image: np.ndarray, mode: ProcessingMode = ProcessingMode.PV) -> np.ndarray:
        """
        Calculate dose for single channel based on calibration results.
        
        This method applies the calibration function to convert image pixel values
        (or derived quantities like OD or NET_OD) to radiation dose values.
        
        Parameters
        ----------
            image : np.ndarray
                The input image array containing pixel values to be converted to dose.
            mode : ProcessingMode, optional
                The processing mode for dose calculation, default is PV.
                Must match the mode used during calibration.
                
        Returns
        -------
            np.ndarray
                Array of calculated dose values corresponding to the input image.
                
        Raises
        ------
            ValueError
                If the channel has not been calibrated.
                If the mode for dose calculation doesn't match the calibration mode.
                If the function name in calibration_results is invalid.
                
        Notes
        -----
            - Different processing modes require different transformations of the input data.
            - For NET_OD mode, the background_mean value is used in the calculation.
            - Supports both polynomial functions and predefined fitting functions.
        """
        if not self.calibration_results or self.background_mean is None:
            raise ValueError(f"{self.channel_name} channel must be calibrated before calculating dose")
        
        # Check if modes match
        if mode != self.calibration_results['mode']:
            raise ValueError(
                f"Processing mode mismatch: "
                f"Calibration used {self.calibration_results['mode']}, "
                f"but calculating dose with {mode}. "
                f"They must be the same for accurate results."
            )
        function_name = self.calibration_results['function']
        coeffs = self.calibration_results['parameters']
        # Prepare the input data based on mode
        if mode == ProcessingMode.PV:
            input_data = image
        elif mode == ProcessingMode.OD:
            input_data = np.log10(65535 / image)
        elif mode == ProcessingMode.NET_OD:
            input_data = np.where(image / self.background_mean > 0,
                               -np.log10(image / self.background_mean), 0)
        else:
            raise ValueError(f"Unsupported processing mode: {mode}")
        
        # Handle polynomial case
        function_name = function_name.strip().lower()

        if function_name.startswith('polynomial'):
            print("Polynomial function detected.")
            p = np.poly1d(coeffs)
            dose_gy_r = p(input_data)
        else:
            if function_name in self.fitting_functions:
                print(f"Best Function is '{function_name}' .")
                dose_gy_r = self.fitting_functions[function_name](input_data, *coeffs)
            else:
                raise ValueError(f"Invalid function '{function_name}' for {self.channel_name} channel")
        
        return dose_gy_r


        
        