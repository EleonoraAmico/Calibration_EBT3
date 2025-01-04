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


# Funzione di fitting (per esempio, razionale)
def rational_new(x, a, b, c):
    return (a + b * x) / (x + c)
# Define various functions to fit
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def rational(x, a, b, c):
    return (a + b * x) / (x + c)

def double_exponential(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)

# Additional functions from the image
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

def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def func7(x, a, b, c, d):
    return np.exp(a * x + b) - np.exp(c * x + d)

def func8(x, a, b, c):
    return np.log((x + c) / (b + x)) - a

def func9(x, a, b, c, d):
    return c * np.exp(a * b * x) + d * np.exp(-c * d * x)

def func10(x, a, b, c):
    return c - a * np.exp(-b * x)
def funct11(x, a, b, c, d, e):
    return ( a*x + b*x**2+c*x**3+d*x**4+e)

# Funzione di fitting e plotting considerando la dose come variabile dipendente 
def best_fit(csv_file, title, x_name, y_name):
    # Leggi il CSV
    df = pd.read_csv(csv_file)
    
    # Estrai i dati
    x = df.iloc[:, 1].astype(float) # Pixel Value/netOD
    xerr = df.iloc[:, 2].astype(float) #StDev
    y = df.iloc[:, 0].astype(float)/100 #Dose

    
    # Funzione da provare (in questo caso solo rational_new)
    functions = [
    linear, 
    quadratic, 
    cubic, 
    exponential, 
    rational, 
    double_exponential, 
    func1, 
    func2, 
    func3, 
    func4, 
    func5, 
    polynomial, 
    func7, 
    func8, 
    func9, 
    func10,
    funct11,
]

    best_func = None
    best_popt = None
    best_mse = float('inf')
    
    # Prova ogni funzione e scegli quella con il miglior errore quadratico medio
    for func in functions:
        try:
            # Ottieni i parametri ottimali con curve_fit
            popt, _ = curve_fit(func, x, y)
            
            # Calcola la curva di fitting
            y_fit = func(x, *popt)
            
            # Calcola l'errore quadratico medio (MSE)
            mse = mean_squared_error(y, y_fit)
            
            # Se l'MSE è migliore di quello precedente, salva i parametri
            if mse < best_mse:
                best_mse = mse
                best_func = func
                best_popt = popt
        except Exception as e:
            # Stampa l'errore ma continua con il prossimo tentativo
            print(f"Fitting failed for function {func.__name__} with error: {e}")
            continue
    
    # Se è stata trovata una funzione adatta, procedi al plotting
    if best_func and best_popt is not None:
        print(f"Best fit function for {title}: {best_func.__name__}")
        print(f'MSE={best_mse}')
        # Plot dei dati con barre di errore
        plt.errorbar(x, y, xerr=xerr, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
        
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
        print(f"No valid fit found for {title}.")
    return best_popt

# Fitting channel red 
params_r=best_fit('Dati_calibrazione/Channel_red_DvsPV', 'Channel Red', 'PV', 'Dose (Gy)')

