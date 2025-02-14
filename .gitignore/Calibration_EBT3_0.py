# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:59:55 2025

@author: Ele_p
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

def evaluate_metrics(observed, expected):
    """
    Calculate Chi-Square, R-Squared, and RMSE metrics.

    Parameters:
        observed (array-like): Observed data points.
        expected (array-like): Expected or predicted data points.

    Returns:
        dict: A dictionary containing chi-square, r-squared, and rmse values.
    """
    # Ensure input is converted to numpy arrays
    observed = np.array(observed)
    expected = np.array(expected)

    # Validate that observed and expected have the same length
    if len(observed) != len(expected):
        raise ValueError("Observed and expected data must have the same length.")

    # Chi-Square calculation
    chi_square = np.sum((observed - expected) ** 2 / expected)

    # R-Squared calculation
    r_squared  = r2_score(observed, expected)

    # RMSE calculation
    rmse = root_mean_squared_error(observed, expected)

    return {
        "chi_square": chi_square,
        "r_squared": r_squared,
        "rmse": rmse
    }


def polynomial_fit(x, y, max_degree=5):
    """
    Find the best polynomial fit by testing different degrees
    
    Parameters:
    x: array-like, data for the x-axis
    y: array-like, data for the y-axis
    max_degree: int, maximum degree of the polynomial to test
    plot: bool, if True, generates a plot of the best fit
    
    Returns:
    best_coefficients: coefficients of the best polynomial
    best_mse: MSE of the best fit
    best_degree: degree of the best polynomial
    """

    best_mse = float('inf')
    best_coefficients = None
    best_degree = 0
    mse_history = []
    
    # Test diversi gradi
    for degree in range(1, max_degree + 1):
        print('degree', degree)
        # Fit polinomiale
        coefficients = np.polyfit(x, y, degree)
        
        coefficients = [c if abs(c) >= 1e-08 else 0 for c in coefficients]
        
        # Output dei risultati
        print("Coefficients:", coefficients)
        
        p = np.poly1d(coefficients)
        y_pred = p(x)
        #print('y', y)
        #print('y_pred', y_pred)
        
        #print('coefficients', coefficients)
        # Calculate metrics
        results = evaluate_metrics(y, y_pred)
        
        # Print results
        print("Evaluation Metrics:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")
        # Calcola MSE
        mse = mean_squared_error(y,y_pred)
        print('mse', mse)
        #print('r_squared', r_squared)
        #rmse = np.sqrt(mse)
        #print(rmse)
        mse_history.append(mse)
        
        #Plot dei dati con barre di errore
        plt.errorbar(x, y, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
        
        # Plot della curva di fit
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = p(x_fit)
        plt.plot(x_fit, y_fit, '-', label=f'Fit: {degree}')
        
        plt.xlabel('PV')
        plt.ylabel('Dose')
        plt.title("Polynomial fit")
        plt.legend()
        plt.show()
        
        # Aggiorna il migliore se necessario
        if mse < best_mse:
            best_mse = mse
            best_coefficients = coefficients
            best_degree = degree
    print(f'best mse {best_mse}, best degree {best_degree}')
        
    return best_mse, best_degree, best_coefficients
#%%
def polynomial_fit_0(x, y, fitting_results, max_degree=5):
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
            metrics = evaluate_metrics(y, y_pred)
            
            # Store results
            fitting_results.append({
                'function': f'polynomial_degree_{degree}',
                'mse': mse,
                'valid_covariance': True,  # Polynomials always have valid covariance
                'success': True,
                'error_message': None,
                'degree': degree,
                'coefficients': coefficients,  # Convert numpy array to list
                'metrics': metrics,
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

def plot_polynomial_fits(x, y, fitting_results, title="Polynomial Fits"):
    """
    Plot all polynomial fits along with the data
    
    Parameters:
    x: array-like, data for the x-axis
    y: array-like, data for the y-axis
    fitting_results: list of dictionaries containing fitting results
    title: str, plot title
    """
    if fitting_results is None:
        print("No valid fitting results to plot.")
        return

    
    for result in fitting_results:
        #if not result['success']:
            #continue
            
        plt.figure(figsize=(10, 6))
        
        # Plot data points with error bars
        plt.errorbar(x, y, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
        
        # Plot polynomial fit
        x_fit = np.linspace(min(x), max(x), 100)
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
        
        # Print detailed metrics for this fit
        print(f"\nMetrics for polynomial degree {result['degree']}:")
        for key, value in result['metrics'].items():
            print(f"{key}: {value:.4f}")

# Example usage:

#%%
best_mse, best_degree, best_coefficients, fitting_results = polynomial_fit_0(x_data, y_data, fitting_results)
plot_polynomial_fits(x_data, y_data, fitting_results)
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
    return (a + b * x) / (x + c)

# Additional functions
def func1(x, a, b):
    return (a * x) / (b * x + 1)

def func2(x, a, e):
    return (a + x) / (x + e)

def func3(x, b, a, c):
    return (b * x) / (a + x) + c

def func4(x, a, b):
    return x - (a * x) / b

def func5(x, a, b, r):
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

# Funzione di fitting e plotting considerando la dose come variabile dipendente 
def best_fit(x, y, title, x_name, y_name):
    
    
        
    # Funzione da provare (in questo caso solo rational_new)
    functions = [
    exponential, 
    exponential_decreasing_function,
    exponential_difference,
    double_exponential, 
    exponential_combination,
    rational, 
    func1, 
    func2, 
    func3, 
    func4, 
    func5, 
    log_function
]

    best_func = None
    best_popt = None
    best_mse = float('inf')
    best_degree = None
    best_coefficients = None
    fitting_results = []
    
    best_mse, best_degree, best_coefficients = polynomial_fit(x,y)
    # Prova ogni funzione e scegli quella con il miglior errore quadratico medio
    for func in functions:
        try:
            # Ottieni i parametri ottimali con curve_fit
            popt, pcov, infodict, errmsg, ier = curve_fit(func, x, y,
                                full_output=True,
                                maxfev=10000,  # Increase maximum number of function evaluations
                                method='lm' )
            
            # Calcola la curva di fitting
            y_fit = func(x, *popt)
            
            # Calcola l'errore quadratico medio (MSE)
            mse = mean_squared_error(y, y_fit)
            
            # Check if covariance matrix is valid
            has_valid_covariance = pcov is not None and not np.any(np.isinf(pcov))
            
            # Store results
            fitting_results.append({
                'function': func.__name__,
                'mse': mse,
                'valid_covariance': has_valid_covariance,
                'success': True,
                'error_message': None
            })
            #Plot dei dati con barre di errore
            plt.errorbar(x, y, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
            
            # Plot della curva di fit
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = func(x_fit, *popt)
            plt.plot(x_fit, y_fit, '-', label=f'Fit: {func.__name__}')
            
            plt.xlabel(f'{x_name}')
            plt.ylabel(f'{y_name}')
            plt.title(title)
            plt.legend()
            plt.show()
            # Se l'MSE è migliore di quello precedente, salva i parametri
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
    # Se è stata trovata una funzione adatta, procedi al plotting
    if best_func and best_popt is not None:
        print(f"Best fit function for {title}: {best_func.__name__}")
        print(f'MSE={best_mse}')
        # Plot dei dati con barre di errore
        plt.errorbar(x, y, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
        
        # Plot della curva di fit
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = best_func(x_fit, *best_popt)
        plt.plot(x_fit, y_fit, '-', label=f'Best Fit: {best_func.__name__}')
        
        plt.xlabel(f'{x_name}')
        plt.ylabel(f'{y_name}')
        plt.title(title)
        plt.legend()
        plt.show()
    else:
        print('best degree', best_degree)
        print('best coefficients', best_coefficients)
        print(f"No valid fit found for {title}.")
    # Print summary of fitting attempts
    print("\nFitting Results Summary:")
    print("-" * 80)
    print(f"{'Function Name':<30} {'MSE':<15} {'Valid Covariance':<20} {'Success'}")
    print("-" * 80)

    for result in fitting_results:
        mse_str = f"{result['mse']:.2e}" if result['mse'] is not None else "Failed"
        print(f"{result['function']:<30} {mse_str:<15} {str(result['valid_covariance']):<20} {result['success']}")
    
    
    return best_popt
#%%
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

def calculate_best_fit(x, y, title):
    """
    Calculates the best fitting function and its parameters.
    Returns the best function, its coefficients, and MSE.
    """
    if not validate_data(x, y):
        return None, None, None
    
    functions = [
        exponential, 
        exponential_decreasing_function,
        exponential_difference,
        double_exponential, 
        exponential_combination,
        rational, 
        func1, 
        func2, 
        func3, 
        func4, 
        func5, 
        log_function
    ]

    best_func = None
    best_popt = None
    best_mse = float('inf')
    
    fitting_results = []
    
    # Get polynomial fit results
    poly_mse, best_degree, best_coefficients, fitting_results = polynomial_fit_0(x, y, fitting_results)
    best_mse = poly_mse

    # Try each function and choose the one with the best MSE
    for func in functions:
        try:
            popt, pcov, infodict, errmsg, ier = curve_fit(func, x, y,
                                                         full_output=True,
                                                         maxfev=10000,
                                                         method='lm')
            
            y_fit = func(x, *popt)
            mse = mean_squared_error(y, y_fit)
            
            has_valid_covariance = pcov is not None and not np.any(np.isinf(pcov))
            
            fitting_results.append({
                'function': func.__name__,
                'mse': mse,
                'valid_covariance': has_valid_covariance,
                'success': True,
                'error_message': None,
                'popt': popt
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
    print(best_func.__name__)
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
        return best_degree, best_coefficients, best_mse
    else:
        print("No function found")
        return None, None, None
def plot_all_fit(x, y, fitting_results, best_func=None, best_popt=None, title="Fitting Results", x_name="Dose", y_name="Response"):
    """
    Plots all successful fits from fitting_results, including both polynomial and function fits
    
    Parameters:
    x: array-like, data for the x-axis
    y: array-like, data for the y-axis
    fitting_results: list of dictionaries containing fitting results
    best_func: best fitting function (optional)
    best_popt: parameters for best fitting function (optional)
    title: str, base title for plots
    x_name: str, name for x-axis
    y_name: str, name for y-axis
    """
    if fitting_results is None:
        print("No valid fitting results to plot.")
        return
    
    for result in fitting_results:
        if not result['success']:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Plot data points with error bars
        plt.errorbar(x, y, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
        
        x_fit = np.linspace(min(x), max(x), 100)
        
        # Handle polynomial fits
        if 'polynomial' in result:
            y_fit = result['polynomial'](x_fit)
            fit_label = f"Polynomial Degree {result['degree']} (MSE: {result['mse']:.2e})"
            plot_title = f"{title} - Polynomial Degree {result['degree']}"
            
            # Print metrics if available
            if 'metrics' in result:
                print(f"\nMetrics for polynomial degree {result['degree']}:")
                for key, value in result['metrics'].items():
                    print(f"{key}: {value:.4f}")
                    
        # Handle function fits
        else:
            # Skip if this is just the function name without parameters
            if 'parameters' not in result or result['parameters'] is None:
                continue
                
            func_name = result['function']
            if func_name == best_func.__name__ and best_popt is not None:
                # Use the best parameters if this is the best function
                y_fit = best_func(x_fit, *best_popt)
            else:
                # Try to get parameters from the result
                try:
                    y_fit = eval(func_name)(x_fit, *result['parameters'])
                except:
                    print(f"Could not plot {func_name} - missing or invalid parameters")
                    continue
                    
            fit_label = f"{func_name} (MSE: {result['mse']:.2e})"
            plot_title = f"{title} - {func_name}"
            
        # Plot the fit
        plt.plot(x_fit, y_fit, '-', label=fit_label)
        
        plt.xlabel(f'{x_name} (Gy)')
        plt.ylabel(y_name)
        plt.title(plot_title)
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_best_fits(x_data, y_data, best_func, best_popt, best_mse, title="Fitting Results", x_name="Dose", y_name="Response"):
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
    if best_func is None or best_popt is None:
        print(f"No valid fit found for {title}.")
        return
    elif best_func is int: 
        # Plot polynomial fit
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        coefficients = best_popt
        p = np.poly1d(coefficients)
        y_fit = p(x_fit)
        
        #y_fit = result['polynomial'](x_fit)
        plt.plot(x_fit, y_fit, '-', 
                label=f"Fit: Degree {best_degree} (MSE: {best_mse:.2e})")

    else:
        # Convert x values to Gray if they aren't already
        x = x_data 
        y = y_data
        print(f"Best fit function: {best_func.__name__}")
        print(f'MSE: {best_mse:.2e}')
        
        plt.figure(figsize=(10, 6))
        
        # Plot data points with error bars
        plt.errorbar(x, y, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
        
        # Plot best fit curve
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = best_func(x_fit, *best_popt)
        plt.plot(x_fit, y_fit, '-', label=f'Best Fit: {best_func.__name__} (MSE: {best_mse:.2e})')
        
        plt.xlabel(f'{x_name} (Gy)')
        plt.ylabel(y_name)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage:
# best_func, best_popt, best_mse = calculate_best_fit(x_data, y_data, "My Fit")
# plot_all_fits(x_data, y_data, best_func, best_popt, best_mse, "My Fitting Analysis", "Pixel Value", "Dose")

# Example usage:
# best_func, best_popt, best_mse = calculate_best_fit(x, y, "My Fit")
# plot_all_fits(x, y, fitting_results, best_func, best_popt, "My Fitting Analysis")

#%%
# Leggi il CSV
df = pd.read_csv('Channel_red_DvsPV')

# Estrai i dati
x_data = df.iloc[:, 1].astype(float) # Pixel Value/netOD
#xerr = df.iloc[:, 2].astype(float) #StDev
y_data = df.iloc[:, 0].astype(float)/100 #Dose
print(x_data)
print(y_data)
#%%
# Fitting channel red 
#params_r=best_fit(x,y, 'Channel Red', 'PV', 'Dose (Gy)')
# Example usage:
best_func, best_popt, best_mse, fitting_results = calculate_best_fit(x, y, "Function of best fit")
#%%
#plot_polynomial_fits(x_data, y_data, fitting_results)
plot_all_fit(x_data, y_data, best_func, best_popt, best_mse, "Best Fit", "Pixel Value", "Dose")
#%%
print(best_func)
plot_best_fits(x, y, best_func, best_popt, best_mse)
#%%
df = pd.read_csv('Channel_red_DvsPV')
x = df.iloc[:, 1].astype(float) # Pixel Value/netOD
y = df.iloc[:, 0].astype(float) #Dose
polynomial_fit(x,y)
#%%
params_r=best_fit('synthetic_data_decrescent_x.csv', 'Synthetic_data', 'PV', 'Dose (Gy)')
#%%
df = pd.read_csv('synthetic_data_degree_1.csv')
x = df.iloc[:,0]
y = df.iloc[:,1]
print(x)
print(y)
#%%
polynomial_fit(x,y)
#%%
params_degree_2 = best_fit('synthetic_data_degree_2.csv', "Synthetic data degree 2", "PV", "Dose(Gy)")

#%%
params_degree_3 = best_fit('synthetic_data_degree_3.csv', "Synthetic data degree 3", "PV", "Dose(Gy)")

#%%
params_degree_4 = best_fit('synthetic_data_degree_4.csv', "Synthetic data degree 4", "PV", "Dose(Gy)")

#%%
params_degree_1 = best_fit('synthetic_data_degree_1_1.2.csv', "Synthetic data degree 1", "PV", "Dose(Gy)" )
#%%
df = pd.read_csv('synthetic_data_degree_1_1.2.csv')
x = df.iloc[:, 0].astype(float) # Pixel Value/netOD
y = df.iloc[:, 1].astype(float) #Dose
polynomial_fit(x,y)
#%%
print(df)
x = df.iloc[:,0].astype(float)
print(x)