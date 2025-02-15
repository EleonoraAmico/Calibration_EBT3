# CurveFitter: Calibration Library for Gafchromic EBT3 Dosimetry
## Table of Contents  
1. [Description](#description)  
2. [Key Features](#key-features)  
3. [Installation](#installation)  
   - [Requirements](#requirements)  
   - [Installing Dependencies](#installing-dependencies)  
4. [Tutorial](#tutorial)  
5. [Processing Modes](#processing-modes)  
6. [Supported Non-Linear Functions](#supported-non-linear-functions)  
7. [Advanced Methods](#advanced-methods)  
8. [Multiple Metric Calculation](#multiple-metric-calculation)  
9. [Testing](#testing) 
11. [Considerations and Limitations](#considerations-and-limitations)
12. [Documentation](#documentation)
13. [Project overview](#project-overview)

## Description
CurveFitter is a Python library designed to calibrate Gafchromic EBT3 radiochromic films used in dosimetry applications. The main challenge in calibrating these films arises from the lack of a well-known calibration function. In addition, each batch of films must be individually calibrated.

To address this, CurveFitter provides an automated framework for testing various calibration functions found in the scientific literature. The goal is to identify the calibration function that minimizes the Mean Squared Error (MSE) between the measured and expected dose-response data. By comparing different calibration models, this library helps users find the most accurate calibration for their specific set of data.

The calibration process involves two main variables:

- The X-axis represents the pixel values measured from the film.
- The Y-axis represents the corresponding relative dose associated with those pixel values.

However, because there are different methods to derive the calibration curve, the library also includes tools to process the data in multiple ways. Specifically, it supports methods to calculate the optical density (OD) and net optical density (netOD), as these are widely used in the literature for calibrating Gafchromic films. This flexibility allows the user to choose the appropriate method depending on the dataset and the specific application.

Additionally, a separate class called FitPlotter has been implemented to facilitate the visualization of the calibration results. This class inherits from CurveFitter and generates plots of the fitting functions, providing an intuitive way to visually assess the performance of different calibration models. The FitPlotter class includes the plot_fits method, which allows users to plot the fitting functions based on different plot types, including:

- *ALL*: Displays all fitting functions.
- *POLYNOMIAL*: Displays only the polynomial fit.
- *FUNCTION*: Displays a specific function, if provided.
- *BEST FIT*: Displays the best fitting function selected based on the minimization of MSE.  
The plotting functionality supports the display of the fitted curves and the data points, making it easier for users to compare and validate their calibration choices.

The library is designed to be flexible and can easily be adapted to different datasets. Its primary purpose is to streamline the calibration process, reducing both manual effort and the risk of errors in the calibration procedure.

## Key Features

- Support for calibration methods using:
  * Pixel Values (PV)
  * Optical Density (OD)
  * Net Optical Density (netOD)
- Data processing and normalization for both polynomial and non-linear fits
- Automatic model selection through minimization of multiple metrics (score and MSE)
- Advanced parameter estimation techniques

## Installation

### Requirements
- Python 3.8+
- Required libraries:
   * numpy==1.24.4
   * scipy==1.10.1
   * scikit-learn==1.5.1
   * pytest==7.4.0
   * pytest-cov==6.0.0
   * hypothesis==6.124.0
   * matplotlib==3.8.4

### Installing Dependencies
`bash pip install -r requirements.txt`


# Tutorial
To get started with the CurveFitter package, check out the tutorial.py file located in the Tutorial folder for a complete usage example.

Additionally, you can download the sample data file channel_red.csv from the example/ folder. This will allow you to test the library with real data and see the results.

### Quick Instructions:

1. **Download the sample data**:
   - Go to the `example/` folder and download the `channel_red.csv` file containing the sample data.

2. **Check out the `tutorial.py` file**:
   - The `tutorial.py` file in the `Tutorial` folder contains a full example of how to use the `CurveFitter` library, including code for non-linear and polynomial fitting, as well as how to generate plots.

# Processing Modes

The library supports different processing modes for handling calibration data. The available modes are:

| Mode   | Description |
|--------|------------|
| **PV** (Pixel Values) | Uses raw pixel values directly for calibration without conversion. Suitable for simple calibrations where optical density is not required. |
| **OD** (Optical Density) | Converts pixel values into optical density (OD), which is often used in dosimetry for better correlation with radiation dose. |
| **NET_OD** (Net Optical Density) | Uses net optical density, calculated as the difference between exposed and unexposed film optical density. This method helps reduce variations due to scanner fluctuations and background noise. |


### *Supported Non-Linear Functions* 

*CurveFitter* includes several non-linear fitting functions to perform calibration:

- **Exponential Fit**:
  - $f(x) = a \cdot e^{b \cdot x} + c$
  - A simple exponential function to model the relationship between pixel values and relative dose.

- **Combination of Exponentials**:
  - $f(x) = a \cdot \exp(b \cdot x) + c \cdot \exp(d \cdot x)$
  - A combination of two exponential functions to model more complex relationships between pixel values and dose.

- **Generalized Rational Function**:
  - $f(x) = \frac{a + x}{c \cdot x + d} + e$
  - A generalized rational function to provide a flexible model for the dose-response relationship.

- **Logarithmic Function**:
  - $f(x) = \frac{\ln\left(\frac{x + b}{b}\right)}{a}$
  - A logarithmic function that models non-linear relationships by calculating the log of the ratio between two linear terms, with numerical stability adjustments.

### **Advanced Methods** 

Generating a good initial guess is critical for the success of non-linear optimization algorithms like those used in curve fitting.In addition to basic fitting, *CurveFitter* offers advanced features to improve calibration accuracy:

1. *Initial Guess Generation*:
   -  Tailored guesses based on function characteristics help in achieving faster and more reliable convergence.
3. *Bayesian Approach for Parameter Selection*:
      - Bayesian methods are a class of probabilistic techniques based on Bayes' theorem, which describes how prior knowledge is updated with observed data to obtain a posterior distribution. In this case, we do         not compute the full posterior distribution but instead use Bayesian-inspired sampling to generate plausible parameter values for function fitting.  
        The Bayesian approach provides several advantages:  
        - It allows the exploration of multiple parameter sets instead of relying on a single deterministic initial guess.  
        - It accounts for uncertainty in parameter estimation by considering a range of values rather than fixed assumptions.  
        - It is particularly useful when the function has non-linear behavior or when gradient-based methods struggle with poor initial conditions.  
         
        The sampling strategy used here is a form of Approximate Bayesian Computation (ABC), where the best parameter set is chosen by minimizing the distance between simulated and observed data.

### **Multiple Metric Calculation**

**CurveFitter** evaluates the fitting results using multiple metrics, with a primary focus on the **Mean Squared Error (MSE)** and a **Score**. The **Score** metric is particularly useful for polynomial fits and is calculated based on several criteria, which provide a more comprehensive evaluation of the fitting quality. The formula for the score is as follows:

1. **Coefficient Ratio**: Measures the ratio of the absolute value of the first coefficient to the mean of the absolute values of the other coefficients. This helps assess the dominance of the first coefficient in the polynomial fitting.  
    coeff_ratio = $\frac{\left|\text{coefficients}[0]\right|}{\text{mean}\left(\left|\text{coefficients}[1:]\right|\right)} \quad \text{if} \quad \text{len(coefficients)} > 1 \quad \text{else} \quad \infty$


   Where:
   - The numerator is the absolute value of the **first coefficient** (`coefficients[0]`).
   - The denominator is the **mean of the absolute values** of all remaining coefficients (`coefficients[1:]`).
   
   This ratio helps identify if the first coefficient (typically the highest degree term in polynomials) dominates the model.

3. **Complexity Penalty**: A penalty term that increases with the degree of the polynomial and the size of the dataset. This prevents overfitting by discouraging excessively complex models.

   complexity_penality = $\text{degree} \cdot \log(\text{len}(y_{\text{true}}))$

4. **Modified Score**: Combines the MSE, complexity penalty, and coefficient ratio to form the final score, which balances goodness of fit with model complexity.

   $\text{score} = \text{mse} \cdot \left( 1 + \frac{\text{complexity penalty}}{\text{len}(y_{\text{true}})} \right) \cdot \left( 1 + \frac{1}{\text{coeff ratio}} \right)$

This metric helps evaluate the fitting process by considering not only the error between the fitted and actual data points (via MSE) but also the complexity of the model and the relative significance of the coefficients. It ensures that the chosen model is both accurate and appropriately complex, preventing overfitting.


## Testing
Tests have been implemented to ensure the reliability and robustness of the library. These tests validate various aspects of the fitting and data processing methods, ensuring consistent and accurate results for different datasets.
The tests are located in the `tests` folder and can be executed using [pytest](https://docs.pytest.org/).

## Considerations and Limitations

While *CurveFitter* is a powerful tool, there are some known limitations:

- *Polynomial Fits*:
  - The maximum polynomial degree is limited to 4 to prevent overfitting.
  - The recommended metric for evaluating polynomial fits is the *score*, which takes into account both Mean Squared Error (MSE) and the complexity of the function.

- *Non-linear Fits*: 
  - While the Mean Squared Error (MSE) is a widely used metric for evaluating model performance, it does not always guarantee that the selected best-fit function      aligns with the expected non-linear behavior. During testing, it was observed that in some cases, the MSE-based selection chose a different function rather        than the expected one.  

This suggests that further investigation is needed to determine the most suitable metric for selecting the best model.    
Users are encouraged to analyze the fitting results carefully and consider additional criteria, such as function interpretability and physical meaning, when choosing the most appropriate calibration function. The provided plotting methods can be useful for visually inspecting the fits and validating model selection.

## Documentation

The full documentation for this project is available in the `docs` folder. You can view it by opening the `index.html` file in your browser.

If you want to build the documentation locally, you can use Sphinx. Simply navigate to the `docs` folder and run:

`bash
sphinx-build -b html . _build`      


## Project Overview

The goal of this project was to create an open-source code to assist users in calibrating Gafchromic EBT3 radiochromic films for dosimetry applications. The library provides methods to process and fit data, ensuring accurate and reliable results for each batch of films.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

