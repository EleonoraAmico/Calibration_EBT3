# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:59:35 2025

@author: Ele_p
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import enum

class ProcessingMode(enum.Enum):
    PV = "PV"  # Pass-through values
    OD = "OD"  # Optical Density
    NET_OD = "netOD"  # Net Optical Density

def process_values(x_values, y_values=None, mode=ProcessingMode.PV):
    """
    Process x_values according to different modes.
    
    Args:
        x_values: numpy array of input values
        y_values: optional numpy array of y values, needed for NET_OD mode
        mode: ProcessingMode enum specifying the processing type
        
    Returns:
        processed numpy array
    """
    if mode == ProcessingMode.PV:
        return x_values
    elif mode == ProcessingMode.OD:
        return np.log10(65535 / x_values)
    elif mode == ProcessingMode.NET_OD:
        if y_values is None:
            raise ValueError("y_values are required for NET_OD mode")
        # Find x_value where y = 0
        if 0 not in y_values:
            raise ValueError("y_values must contain 0 for NET_OD mode")
        x_zero = x_values[np.where(y_values == 0)[0][0]]
        return -np.log10(x_values / x_zero)
    else:
        raise ValueError(f"Unknown processing mode: {mode}")

# Add enum values as function attributes
for mode in ProcessingMode:
    setattr(process_values, mode.name, mode)

def exponential(x, a, b, c):
    """
    Exponential function with scaling and overflow control.
    
    Parameters:
    -----------
    x : array-like
        values on x axis 
    a, b, c : float
        parameters of the exponential function
        
    Returns:
    --------
    array-like
        values computed by the exponential function 
    """

    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
    exp_component = np.exp(np.clip(b * x_scaled, -700, 700))  # Clip the exponent range

    return a * exp_component + c

def exponential_decreasing_function(x, a, b, c):
    """
    Function representing an exponential decay with an offset and with scaling and overflow control.
    
    Parameters:
    -----------
    x : array-like
        values on x axis 
    a, b, c : float
        parameters of the exponential function
        
    Returns:
    --------
    array-like
        values computed by the exponential function 
    """

    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
    exp_component = np.exp(np.clip(-b * x_scaled, -700, 700))  # Clip the exponent range
    return c - a * exp_component


def double_exponential(x, a, b, c, d):
    """
    Sum of exponential function with scaling and overflow control.
    
    Parameters:
    -----------
    x : array-like
        values on x axis 
    a, b, c, d : float
        parameters of the double exponential function
        
    Returns:
    --------
    array-like
        values computed by the sum of two exponential functions
    """

    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
    exp_component_one= np.exp(np.clip(b * x_scaled, -700, 700))  # Clip the exponent range
    exp_component_two= np.exp(np.clip(d * x_scaled, -700, 700))  # Clip the exponent range
    return a * exp_component_one + c * exp_component_two


def exponential_difference(x, a, b, c, d):
    """
    Subtraction of exponential terms with scaling and overflow control.
    
    Parameters:
    -----------
    x : array-like
        values on x axis 
    a, b, c, d : float
        parameters of the exponential difference function
        
    Returns:
    --------
    array-like
        values computed by the exponential difference function 
    """
    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
    # Add small epsilon to prevent underflow
    eps = 1e-10
    exp_component_one= np.exp(np.clip(a * x_scaled + b, -700, 700)) + eps # Clip the exponent range
    exp_component_two= np.exp(np.clip(c * x_scaled + d, -700, 700)) + eps # Clip the exponent range
    return exp_component_one - exp_component_two

def exponential_combination(x, a, b, c, d):
    """
    Sum of exponentials with positive and negative exponents with scaling and overflow control.
    
    Parameters:
    -----------
    x : array-like
        values on x axis 
    a, b, c, d : float
        parameters of the exponential difference function
        
    Returns:
    --------
    array-like
        values computed by the exponential combination function 
    """
    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))  # Normalize x
    exp_component_one= np.exp(np.clip(a * b * x_scaled, -700, 700))  # Clip the exponent range
    exp_component_two= np.exp(np.clip(-c * d * x_scaled, -700, 700))  # Clip the exponent range
    
    return c * exp_component_one + d * exp_component_two

def rational(x, a, b, c):
    """
    Function representing a generalized rational function with scaling.

    Parameters:
    -----------
    x : array-like
        Input values on the x-axis.
    a : float
        Offset term added to the numerator.
    b : float
        Scaling parameter applied to the input variable in the numerator.
    c : float
        Offset term added to the denominator.

    Returns:
    --------
    array-like
        Output values computed as (a + b * x) / (x + c).

    Description:
    ------------
    This function computes a rational relationship between input \(x\),
    with an offset \(a\) in the numerator, a scaling factor \(b\) applied to \(x\) in the numerator,
    and a shift \(c\) in the denominator. 
    The additional parameter \(b\) allows for more flexible control 
    over the rate of growth in the numerator compared to simpler rational functions.
    """
    
    return (a + b * x) / (x + c)

# Additional functions
def hyperbolic_growth(x, a, b):
    """
    Function representing a saturation curve.

    Parameters:
    -----------
    x : array-like
        Input values on the x-axis.
    a : float
        Scaling parameter for the numerator.
    b : float
        Scaling parameter for the denominator.

    Returns:
    --------
    array-like
        Output values computed as (a * x) / (b * x + 1).
    """
    return (a * x) / (b * x + 1)

def rational_function_with_offset(x, a, e):
    """
    Function representing a rational function with an offset.

    Parameters:
    -----------
    x : array-like
        Input values on the x-axis.
    a : float
        Offset term added to the numerator.
    e : float
        Offset term added to the denominator.

    Returns:
    --------
    array-like
        Output values computed as (a + x) / (x + e).
    """
    return (a + x) / (x + e)

def saturation_function_with_offset(x, b, a, c):
    """
    Function representing a saturating curve with an offset.

    Parameters:
    -----------
    x : array-like
        Input values on the x-axis.
    b : float
        Scaling parameter for the numerator.
    a : float
        Scaling parameter for the denominator.
    c : float
        Offset added to the function output.

    Returns:
    --------
    array-like
        Output values computed as (b * x) / (a + x) + c.
    """
    return (b * x) / (a + x) + c

def linear_decay(x, a, b):
    """
    Function representing a linear decay.

    Parameters:
    -----------
    x : array-like
        Input values on the x-axis.
    a : float
        Scaling parameter for the numerator.
    b : float
        Scaling parameter for the denominator.

    Returns:
    --------
    array-like
        Output values computed as x - (a * x) / b.
    """
    return x - (a * x) / b

def polynomial_scaling(x, a, b, r):
    """
    Function representing polynomial scaling.

    Parameters:
    -----------
    x : array-like
        Input values on the x-axis.
    a : float
        Linear scaling parameter.
    b : float
        Coefficient for the polynomial term.
    r : float
        Power of the polynomial term.

    Returns:
    --------
    array-like
        Output values computed as a * x + b * x**r.
    """
    return a * x + b * x**r


def log_function(x, a, b, c):
    """
    Enhanced logarithmic function with input validation and numerical stability.
    
    Parameters:
    -----------
    x : array-like
        Input values
    a, b, c : float
        Function parameters
        
    Returns:
    --------
    array-like
        Computed logarithmic values
    """
    # Add small epsilon to prevent log(0)
    eps = 1e-10
    
    # Ensure arrays
    x = np.asarray(x)
    
    # Calculate numerator and denominator separately
    numerator = x + c + eps
    denominator = b + x + eps
    
    # Check for invalid values
    valid_mask = (numerator > 0) & (denominator > 0)
    
    # if not np.all(valid_mask):
    #     print(f"Warning: Invalid values found at x positions: {np.where(~valid_mask)[0]}")
        
    # Calculate ratio with epsilon to prevent division by zero
    ratio = (numerator + eps) / (denominator + eps)
    
    # Calculate log with input validation
    result = np.full_like(x, np.nan, dtype=float)
    result[valid_mask] = np.log(ratio[valid_mask]) - a
    
    return result
def validate_data(x, y):
    """
    Validates input data ranges.
    Returns True if valid, False otherwise.
    """
    if not all(0 <= val <= 65535 for val in x):
        print("Error: x values must be between 0 and 65535")
        return False
    if not all(0 <= val <= 50 for val in y):
        print("Error: y values must be between 0 and 50 Gy")
        return False
    return True
def polynomial_fit(x, y, fitting_results, max_degree=4):
    """
    Find the best polynomial fit by testing different degrees
    
    Parameters:
    x: array-like, data for the x-axis
    y: array-like, data for the y-axis
    max_degree: int, maximum degree of the polynomial to test
    
    Returns:
    dict: Dictionary containing fitting results for all tested polynomials
    """
    if not validate_data(x, y):
        return None, None, None

    fitting_results = []
    best_mse = float('inf')
    best_coefficients = None
    best_degree = 0
    
    # Test different degrees
    for degree in range(1, max_degree + 1):
        try:
            # Polynomial fit
            coefficients = np.polyfit(x, y, degree)
            
            # Filter out very small coefficients
            coefficients = [c if abs(c) >= 1e-08 else 0 for c in coefficients]
            
            # Create polynomial function
            p = np.poly1d(coefficients)
            y_pred = p(x)
            
            # Calculate MSE and other metrics
            mse = mean_squared_error(y, y_pred)
            #metrics = evaluate_metrics(y, y_pred)
            
            # Store results
            fitting_results.append({
                'function': f'polynomial_degree_{degree}',
                'mse': mse,
                'valid_covariance': True,  # Polynomials always have valid covariance
                'success': True,
                'error_message': None,
                'degree': degree,
                'coefficients': coefficients,  # Convert numpy array to list
                #'metrics': metrics,
                'polynomial': p
            })
            
            # Update best fit if necessary
            if mse < best_mse:
                best_mse = mse
                best_coefficients = coefficients
                best_degree = degree
                
        except Exception as e:
            fitting_results.append({
                'function': f'polynomial_degree_{degree}',
                'mse': None,
                'valid_covariance': False,
                'success': False,
                'error_message': str(e),
                'degree': degree,
                'coefficients': None,
                'metrics': None,
                'polynomial': None
            })
    
    # # Print summary
    # print("\nPolynomial Fitting Results Summary:")
    # print("-" * 80)
    # print(f"{'Degree':<10} {'MSE':<15} {'Success':<10}")
    # print("-" * 80)
    
    # for result in fitting_results:
    #     mse_str = f"{result['mse']:.2e}" if result['mse'] is not None else "Failed"
    #     print(f"{result['degree']:<10} {mse_str:<15} {result['success']:<10}")
    
    print(f'\nBest fit: Polynomial degree {best_degree} with MSE {best_mse:.2e}')
    
    return best_mse, best_degree, best_coefficients, fitting_results

def plot_polynomial_fits(x_data, y_data, fitting_results, title="Polynomial Fits", mode=ProcessingMode.PV ):
    """
    Plot all polynomial fits along with the data
    
    Parameters:
    x: array-like, data for the x-axis
    y: array-like, data for the y-axis
    fitting_results: list of dictionaries containing fitting results
    title: str, plot title
    """
    try:
        x_processed = process_values(x_data, y_data, mode)
    except Exception as e:
        print(f"Error processing x values: {str(e)}")
        return None, None, None, None
    if fitting_results is None:
        print("No valid fitting results to plot.")
        return

    
    for result in fitting_results:
        #if not result['success']:
            #continue
            
        plt.figure(figsize=(10, 6))
        
        # Plot data points with error bars
        plt.errorbar(x_processed, y_data, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
        
        # Plot polynomial fit
        x_fit = np.linspace(min(x_processed), max(x_processed), 100)
        if result['coefficients'] is not None: 
            coefficients = result['coefficients']
            p = np.poly1d(coefficients)
            y_fit = p(x_fit)
            
            #y_fit = result['polynomial'](x_fit)
            plt.plot(x_fit, y_fit, '-', 
                    label=f"Fit: Degree {result['degree']} (MSE: {result['mse']:.2e})")
            
            plt.xlabel('Dose (Gy)')
            plt.ylabel('Response')
            plt.title(f"{title} - Degree {result['degree']}")
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # # Print detailed metrics for this fit
        # print(f"\nMetrics for polynomial degree {result['degree']}:")
        # for key, value in result['metrics'].items():
        #     print(f"{key}: {value:.4f}")
def calculate_best_fit(x, y, title, mode=ProcessingMode.PV ):
    """
    Calculates the best fitting function and its parameters.
    Returns the best function, its coefficients, and MSE.
    """
    if not validate_data(x, y):
        return None, None, None, None
    # Process x values according to the specified mode
    try:
        x_processed = process_values(x, y, mode)
    except Exception as e:
        print(f"Error processing x values: {str(e)}")
        return None, None, None, None
    
    functions = [
        exponential, 
        exponential_decreasing_function,
        exponential_difference,
        double_exponential, 
        exponential_combination,
        rational, 
        hyperbolic_growth, 
        rational_function_with_offset, 
        saturation_function_with_offset, 
        linear_decay, 
        polynomial_scaling, 
        log_function
    ]

    best_func = None
    best_popt = None
    best_mse = float('inf')
    
    fitting_results = []
    
    # Get polynomial fit results
    poly_mse, best_degree, best_coefficients, fitting_results = polynomial_fit(x_processed, y, fitting_results)
    best_mse = poly_mse

    # Try each function and choose the one with the best MSE
    for func in functions:
        try:
            popt, pcov, infodict, errmsg, ier = curve_fit(func, x_processed, y,
                                                         full_output=True,
                                                         maxfev=10000,
                                                         method='lm')
            
            
            y_fit = func(x_processed, *popt)
            mse = mean_squared_error(y, y_fit)
            
            has_valid_covariance = pcov is not None and not np.any(np.isinf(pcov))
            
            fitting_results.append({
                'function': func.__name__,
                'mse': mse,
                'valid_covariance': has_valid_covariance,
                'success': True,
                'error_message': None,
                'popt': popt,
                'coefficient': None
            })

            if mse < best_mse:
                best_mse = mse
                best_func = func
                best_popt = popt

        except Exception as e:
            fitting_results.append({
                'function': func.__name__,
                'mse': None,
                'valid_covariance': False,
                'success': False,
                'error_message': str(e)
            })
            continue
    # Print summary of fitting attempts
    print("\nFitting Results Summary:")
    print("-" * 80)
    print(f"{'Function Name':<30} {'MSE':<15} {'Valid Covariance':<20} {'Success'}")
    print("-" * 80)

    for result in fitting_results:
        mse_str = f"{result['mse']:.2e}" if result['mse'] is not None else "Failed"
        print(f"{result['function']:<30} {mse_str:<15} {str(result['valid_covariance']):<20} {result['success']}")
    if best_func is not None:
        print(f'The best fitting function is {best_func.__name__}')
        return best_func, best_popt, best_mse, fitting_results
    elif best_degree is not None: 
        print(f'The best fitting function is polynomial degree {best_degree}')
        return best_degree, best_coefficients, best_mse, fitting_results
    else:
        print("No function found")
        return None, None, None, fitting_results
    

def plot_best_fits(x_data, y_data, best_func, best_popt, best_mse, title="Fitting Results", x_name="Dose", y_name="Response", mode=ProcessingMode.PV):
    """
    Plots the best fit curve along with the data points
    
    Parameters:
    x_data: array-like, data for the x-axis
    y_data: array-like, data for the y-axis
    best_func: best fitting function
    best_popt: parameters for best fitting function
    best_mse: mean squared error of the best fit
    title: str, plot title
    x_name: str, name for x-axis
    y_name: str, name for y-axis
    """
    # Process x values according to the specified mode
    try:
        x_processed = process_values(x_data, y_data, mode)
    except Exception as e:
        print(f"Error processing x values: {str(e)}")
        return None, None, None, None
    
    if best_func is None or best_popt is None:
        print(f"No valid fit found for {title}.")
        return
    elif isinstance(best_func, int):
        

        # Plot polynomial fit
        x_fit = np.linspace(min(x_processed), max(x_processed), 100)
        coefficients = best_popt
        p = np.poly1d(coefficients)
        y_fit = p(x_fit)
        
        #y_fit = result['polynomial'](x_fit)
        plt.plot(x_fit, y_fit, '-', 
                label=f"Fit: Degree {best_func} (MSE: {best_mse:.2e})")

    else:
        # Convert x values to Gray if they aren't already

        print(f"Best fit function: {best_func.__name__}")
        print(f'MSE: {best_mse:.2e}')
        
        plt.figure(figsize=(10, 6))
        
        
        
        # Plot best fit curve
        x_fit = np.linspace(min(x_processed), max(x_processed), 100)
        y_fit = best_func(x_fit, *best_popt)
        plt.plot(x_fit, y_fit, '-', label=f'Best Fit: {best_func.__name__} (MSE: {best_mse:.2e})')
        
    # Plot data points with error bars
    plt.errorbar(x_processed, y_data, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')   
    plt.xlabel('Pixel Value' if mode == ProcessingMode.PV else 'OD' if mode == ProcessingMode.OD  else 'netOD')
    plt.ylabel('Dose (Gy)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_all_fits(x_data, y_data, fitting_results, title="Polynomial Fits", mode=ProcessingMode.PV):
    """
    Plot all polynomial fits along with the data
    
    Parameters:
    x_data: array-like, data for the x-axis
    y_data: array-like, data for the y-axis
    fitting_results: list of dictionaries containing fitting results
    title: str, plot title
    mode: ProcessingMode enum, processing mode for the data
    """
    try:
        x_processed = process_values(x_data, y_data, mode)
    except Exception as e:
        print(f"Error processing x values: {str(e)}")
        return None, None, None, None
        
    if fitting_results is None or len(fitting_results) == 0:
        print("No valid fitting results to plot.")
        return
    
    for i, result in enumerate(fitting_results):
        try:
            
            plt.figure(figsize=(10, 6))
            
            # Plot data points with error bars
            plt.errorbar(x_processed, y_data, fmt='o', ecolor='red', 
                        capsize=5, capthick=2, label='Data')
            
            # Plot polynomial fit
            x_fit = np.linspace(min(x_processed), max(x_processed), 100)
            
            # Try different possible keys for coefficients
            coefficients = None
            if 'coefficients' in result:
                coefficients = result['coefficients']
            elif 'coef' in result:
                coefficients = result['coef']
            elif 'params' in result:
                coefficients = result['params']
                
            if coefficients is not None:
                p = np.poly1d(coefficients)
                y_fit = p(x_fit)
                
                # Get degree from coefficients length or from result
                degree = result.get('degree', len(coefficients)-1)
                mse = result.get('mse', float('nan'))
                
                plt.plot(x_fit, y_fit, '-', 
                        label=f"Fit: Degree {degree} (MSE: {mse:.2e})")
                
                plt.xlabel('Dose (Gy)')
                plt.ylabel('Pixel Value' if mode == ProcessingMode.PV else 'OD')
                plt.title(f"{title} - Degree {degree}")
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                popt = None
                if 'popt' in result:
                    popt = result['popt']

                    
                if popt is not None:
                    func_name = result.get('function', None)
                    # Get the function from globals
                    if isinstance(func_name, str):
                        # Try to get the function from the global namespace
                        func = globals().get(func_name)
                        if func is None:
                            print(f"Warning: Function '{func_name}' not found in global namespace for result {i+1}")
                            continue
                    else:
                        func = func_name  # If it's already a function object
                    y_fit = func(x_fit, *popt)
                    mse = result.get('mse', float('nan'))  # Get MSE if available
                    
                    
                    plt.plot(x_fit, y_fit, '-', 
                            label=f"Fit: {func.__name__} (MSE: {mse:.2e})")
                    
                    plt.xlabel('Pixel Value' if mode == ProcessingMode.PV else 'OD' if mode == ProcessingMode.OD  else 'netOD')
                    plt.ylabel('Dose (Gy)')
                    plt.title(f"{title} - Degree {degree}")
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                
        except Exception as e:
            print(f"Error plotting result {i+1}: {str(e)}")
            continue

        
#%%



#%%
df = pd.read_csv('synthetic_data_degree_1.csv')
x = df.iloc[:,0]
y = df.iloc[:,1]



#%%
df = pd.read_csv('synthetic_data_degree_1_1.2.csv')
x = df.iloc[:, 0].astype(float) # Pixel Value/netOD
y = df.iloc[:, 1].astype(float) #Dose

best_degree, best_coefficients, best_mse, fitting_results = calculate_best_fit(x, y, 'Best fit', mode=ProcessingMode.PV )
plot_best_fits(x, y, best_degree, best_coefficients, best_mse)
#%%
# Leggi il CSV
df = pd.read_csv('Channel_red_DvsPV')

# Estrai i dati
x_data = df.iloc[:, 1].astype(float) # Pixel Value/netOD
y_data = df.iloc[:, 0].astype(float)/100 #Dose

best_degree, best_coefficients, best_mse, fitting_results = calculate_best_fit(x_data, y_data, 'Best fit', mode=ProcessingMode.PV )
plot_best_fits(x_data, y_data, best_degree, best_coefficients, best_mse, mode=ProcessingMode.PV)
plot_all_fits(x_data, y_data, fitting_results, mode=ProcessingMode.PV)
#%%
best_degree, best_coefficients, best_mse, fitting_results = calculate_best_fit(x_data, y_data, 'Best fit', mode=ProcessingMode.OD )
plot_best_fits(x_data, y_data, best_degree, best_coefficients, best_mse, mode=ProcessingMode.OD)
plot_all_fits(x_data, y_data, fitting_results, mode=ProcessingMode.OD)
#%%
best_degree, best_coefficients, best_mse, fitting_results = calculate_best_fit(x_data, y_data, 'Best fit', mode=ProcessingMode.NET_OD )
plot_best_fits(x_data, y_data, best_degree, best_coefficients, best_mse, mode=ProcessingMode.NET_OD)
plot_all_fits(x_data, y_data, fitting_results, mode=ProcessingMode.NET_OD)
#%%
# Leggi il CSV
df = pd.read_csv('Channel_green_DvsPV')

# Estrai i dati
x_data = df.iloc[:, 1].astype(float) # Pixel Value/netOD
y_data = df.iloc[:, 0].astype(float)/100 #Dose
best_degree, best_coefficients, best_mse, fitting_results = calculate_best_fit(x_data, y_data, 'Best fit', mode=ProcessingMode.NET_OD )
plot_best_fits(x_data, y_data, best_degree, best_coefficients, best_mse, mode=ProcessingMode.NET_OD)
plot_all_fits(x_data, y_data, fitting_results, mode=ProcessingMode.NET_OD)

#%%
# for i, result in enumerate(fitting_results):
#     func = result.get('exponential_difference', None)
#     print (func)
data = []
for result in fitting_results:
        if result.get('function') == 'polynomial_degree_4':
            coefficients = result.get('coefficients', [])
            print("Coefficients for 'exponential_difference':", coefficients)
            data.append({'function': 'polynomial_degree_4', 'coefficients': coefficients})
    
# Convert to DataFrame
df = pd.DataFrame(data)
print(df)
output_file="CG_parameters_net_OD.csv"

# Save to CSV
df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")

import pandas as pd

def save_exponential_difference_coefficients(fitting_results, output_file):
    """
    Saves the coefficients of the 'exponential_difference' curve from fitting results to a CSV file.

    Parameters:
    -----------
    fitting_results : list of dict
        A list of dictionaries containing fitting results. Each dictionary should have:
        - 'function_name' : str : The name of the function (e.g., 'exponential_difference').
        - 'popt' : list or array : The coefficients of the fitted function.
    output_file : str
        The file path where the CSV should be saved.

    Returns:
    --------
    None
        Saves the data to the specified CSV file.
    """
    data = []
    for result in fitting_results:
        if result.get('function_name') == 'exponential_difference':
            coefficients = result.get('popt', [])
            # Create a row for each coefficient set
            data.append({'function_name': result.get('function_name'), 'coefficients': coefficients})
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
