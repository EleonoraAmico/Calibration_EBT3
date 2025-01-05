# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:59:55 2025

@author: Ele_p
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error



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
        print(degree)
        # Fit polinomiale
        coefficients = np.polyfit(x, y, degree)
        p = np.poly1d(coefficients)
        y_pred = p(x)
        
        # Calcola MSE
        mse = mean_squared_error(y,y_pred)
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
        print(mse_history)
        # Aggiorna il migliore se necessario
        if mse < best_mse:
            best_mse = mse
            best_coefficients = coefficients
            best_degree = degree
    return best_mse, best_degree, best_coefficients

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
def best_fit(csv_file, title, x_name, y_name):
    # Leggi il CSV
    df = pd.read_csv(csv_file)
    
    # Estrai i dati
    x = df.iloc[:, 0].astype(float) # Pixel Value/netOD
    #xerr = df.iloc[:, 2].astype(float) #StDev
    y = df.iloc[:, 1].astype(float)/100 #Dose

    
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
# Fitting channel red 
params_r=best_fit('Channel_red_DvsPV', 'Channel Red', 'PV', 'Dose (Gy)')

#%%
df = pd.read_csv('Channel_red_DvsPV')
x = df.iloc[:, 1].astype(float) # Pixel Value/netOD
y = df.iloc[:, 0].astype(float) #Dose
polynomial_fit(x,y)
#%%
params_r=best_fit('synthetic_data_decrescent_x.csv', 'Synthetic_data', 'PV', 'Dose (Gy)')

