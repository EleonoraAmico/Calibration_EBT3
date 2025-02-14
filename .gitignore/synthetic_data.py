# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 19:49:27 2025

@author: Ele_p
"""
#%%
import numpy as np
# Genera dati di test sintetici
x = np.linspace(0, 10, 100)
a = np.random.normal(-100,100)
b = np.random.normal(-100,100)
y_true = a * x + b
y_noisy = y_true + np.random.normal(0, 0.1, x.shape)
print(y_noisy)
print(x)
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
def generate_synthetic_data(n_points=10):
    """
    Generate synthetic data with specific conditions:
    1. Points between 0-5 and 5-30 on y-axis are equally distributed within their ranges
    2. Linear relationship between x and y
    3. X values decrease from 45000 to 5000
    4. Y values range from 0 to 30
    """
    # Calculate how many points should be in each y range
    n_points_lower = 6 # Points for 0-5 range
    n_points_upper = n_points - n_points_lower  # Points for 5-30 range
    
    # Generate y values
    y_lower = np.linspace(0, 5, n_points_lower)
    y_upper = np.linspace(5, 30, n_points_upper)
    y = np.concatenate([y_lower, y_upper])
    
    # Generate corresponding x values
    # Using linear relationship: x = mx + b
    x_max, x_min = 45000, 5000
    x = np.linspace(x_max, x_min, n_points)
    
    # Create DataFrame and sort by x in descending order
    df = pd.DataFrame({
        'x': x,
        'y': y
    }).sort_values('x', ascending=False)
    
    return df

# Generate example data
data = generate_synthetic_data(10)

# Display first few rows and basic statistics
print("First few rows:")
print(data.head())
print("\nBasic statistics:")
print(data.describe())
#%%
x_fit = data.iloc[:,0]
y_fit = data.iloc[:,1]
plt.plot(x_fit, y_fit)
plt.errorbar(x_fit, y_fit, fmt='o', ecolor='red', capsize=5, capthick=2, label='Data')
plt.title('Synthetic data')
plt.legend()
plt.show()
#%%
# Salva i dati in un file CSV
data.to_csv("synthetic_data_decrescent_x.csv", index=False)

#%%

def generate_synthetic_data(n_points=10):
    """
    Generate synthetic data with specific conditions:
    1. Single linear relationship between x and y (inverse relationship)
    2. Points concentrated differently in two regions of y (0-5 and 5-30)
    3. X values increase from 5000 to 45000
    4. Y values decrease from 30 to 0
    """
    # First, create the straight line equation
    x_min, x_max = 5000, 45000
    y_max, y_min = 30, 0
    
    # Calculate slope (negative now) and intercept
    m = (y_min - y_max) / (x_max - x_min)  # This will give us a negative slope
    b = y_max - m * x_min  # Using x_min since we want y_max at x_min
    
    # Generate more points in the 0-5 range and fewer in the 5-30 range
    n_points_lower = n_points // 2
    n_points_upper = n_points - n_points_lower
    
    # Calculate x values that correspond to y = 5 using the line equation
    # y = mx + b -> x = (y-b)/m
    x_at_y5 = (5 - b) / m
    
    # Generate x values with different densities
    x_lower = np.linspace(x_at_y5, x_max, n_points_lower)  # More points here
    x_upper = np.linspace(x_min, x_at_y5, n_points_upper)  # Fewer points here
    x = np.concatenate([x_upper, x_lower])
    
    # Calculate y values using the same linear equation for all points
    y = m * x + b
    
    # Create DataFrame and sort by x in ascending order
    df = pd.DataFrame({
        'x': x,
        'y': y
    }).sort_values('x', ascending=True)
    
    return df

# Generate example data
data = generate_synthetic_data(10)

# Display first few rows and basic statistics
print("First few rows:")
print(data.head())
print("\nBasic statistics:")
print(data.describe())

# Optional: Plot to verify the linear relationship
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(data['x'], data['y'])
plt.title('Synthetic Data Distribution')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True)
plt.show()

#%%

def generate_polynomial_data(degree=2, n_points=10):
    """
    Generate synthetic data with polynomial relationship of specified degree:
    1. Polynomial relationship between x and y
    2. Points concentrated differently in two regions of y (0-5 and 5-30)
    3. X values increase from 5000 to 45000
    4. Y values range approximately from 0 to 30
    
    Parameters:
    degree (int): Degree of the polynomial (1=linear, 2=quadratic, 3=cubic, etc.)
    n_points (int): Number of total points to generate
    """
    # Generate x values with different densities
    x_min, x_max = 5000, 45000
    n_points_lower = n_points // 2
    n_points_upper = n_points - n_points_lower
    
    # Normalize x to [0,1] for better numerical stability
    x_norm_min, x_norm_max = 0, 1
    
    # Generate coefficients for a polynomial that will give us roughly the desired y range
    if degree == 1:
        coeffs = [30, 0]  # Linear case (ax + b)
    else:
        # For higher degrees, we'll create coefficients that give a reasonable curve
        coeffs = np.zeros(degree + 1)
        coeffs[degree] = 30  # Highest degree coefficient
        coeffs[0] = 0       # Constant term
    
    # Function to transform normalized x back to original scale
    def transform_x(x_norm):
        return x_min + (x_max - x_min) * x_norm
    
    # Function to calculate y values using the polynomial
    def polynomial(x_norm):
        return np.polyval(coeffs, 1 - x_norm)  # 1 - x_norm to get decreasing relationship
    
    # Generate normalized x values
    x_norm_upper = np.linspace(0, 0.4, n_points_upper)  # More points in upper range
    x_norm_lower = np.linspace(0.4, 1, n_points_lower)  # Fewer points in lower range
    x_norm = np.concatenate([x_norm_upper, x_norm_lower])
    
    # Calculate actual x and y values
    x = transform_x(x_norm)
    y = polynomial(x_norm)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    }).sort_values('x', ascending=True)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(df['x'], df['y'])
    plt.title(f'Polynomial Relationship (degree={degree})')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.grid(True)
    
    # Plot with normalized x for better visualization of point distribution
    plt.subplot(122)
    plt.scatter(x_norm, y)
    plt.title('Normalized X Scale')
    plt.xlabel('Normalized X values')
    plt.ylabel('Y values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return df

# Generate examples with different polynomial degrees
degrees = [1, 2, 3, 4]
for deg in degrees:
    print(f"\nGenerating polynomial of degree {deg}:")
    data = generate_polynomial_data(degree=deg)
    print("\nFirst few rows:")
    print(data.head())
    print("\nBasic statistics:")
    print(data.describe())
    
#%%


def generate_polynomial_data(curve_type='default', n_points=10):
    """
    Generate synthetic data with different polynomial curve shapes:
    1. Different types of polynomial relationships
    2. Points concentrated differently in two regions of y (0-5 and 5-30)
    3. X values increase from 5000 to 45000
    4. Y values range approximately from 0 to 30
    
    Parameters:
    curve_type (str): Type of curve to generate:
        'default': Standard decreasing polynomial
        'sigmoid': S-shaped curve
        'convex': Convex decreasing curve
        'concave': Concave decreasing curve
        'wave': Wave-like curve
    n_points (int): Number of total points to generate
    """
    # Generate x values with different densities
    x_min, x_max = 5000, 45000
    n_points_lower = n_points // 2
    n_points_upper = n_points - n_points_lower
    
    # Normalize x to [0,1] for better numerical stability
    x_norm_min, x_norm_max = 0, 1
    
    # Define different coefficient sets for different curve shapes
    curve_coefficients = {
        'default': [30, 0, 0],  # Standard quadratic
        'sigmoid': [30, -60, 30],  # S-shaped curve
        'convex': [10, 0, 20],  # Convex decreasing
        'concave': [20, 0, 10],  # Concave decreasing
        'wave': [15, 30, -30, 15]  # Wave-like pattern
    }
    
    coeffs = curve_coefficients.get(curve_type, curve_coefficients['default'])
    
    # Function to transform normalized x back to original scale
    def transform_x(x_norm):
        return x_min + (x_max - x_min) * x_norm
    
    # Function to calculate y values using the polynomial
    def polynomial(x_norm):
        return np.polyval(coeffs, 1 - x_norm)  # 1 - x_norm to get decreasing relationship
    
    # Generate normalized x values
    x_norm_upper = np.linspace(0, 0.4, n_points_upper)  # More points in upper range
    x_norm_lower = np.linspace(0.4, 1, n_points_lower)  # Fewer points in lower range
    x_norm = np.concatenate([x_norm_upper, x_norm_lower])
    
    # Calculate actual x and y values
    x = transform_x(x_norm)
    y = polynomial(x_norm)
    
    # Ensure y values stay within reasonable range
    y = np.clip(y, 0, 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    }).sort_values('x', ascending=True)
    
    # Plot the results
    plt.figure(figsize=(15, 5))
    
    # Plot actual values
    plt.subplot(131)
    plt.scatter(df['x'], df['y'])
    plt.title(f'Polynomial Shape: {curve_type}')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.grid(True)
    
    # Plot with normalized x
    plt.subplot(132)
    plt.scatter(x_norm, y)
    plt.title('Normalized X Scale')
    plt.xlabel('Normalized X values')
    plt.ylabel('Y values')
    plt.grid(True)
    
    # Plot the point density
    plt.subplot(133)
    plt.hist(y, bins=20, orientation='horizontal')
    plt.title('Point Density')
    plt.xlabel('Count')
    plt.ylabel('Y values')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df

# Demonstrate different curve shapes
curve_types = ['default', 'sigmoid', 'convex', 'concave', 'wave']
for curve_type in curve_types:
    print(f"\nGenerating {curve_type} curve:")
    data = generate_polynomial_data(curve_type=curve_type)
    print("\nBasic statistics:")
    print(data.describe())



#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_polynomial_data(degree=2, n_points=10):
    """
    Generate synthetic data with polynomial relationships:
    1. True polynomial relationships of different degrees
    2. Points concentrated differently in two y regions (0-5 and 5-30)
    3. X values increase from 5000 to 45000
    4. Y values range approximately from 0 to 30
    
    Parameters:
    degree (int): Degree of polynomial (1=linear, 2=quadratic, 3=cubic, etc.)
    n_points (int): Number of total points to generate
    """
    x_min, x_max = 5000, 45000
    
    # First, generate a dense grid of points to find where y=5 occurs
    x_grid = np.linspace(x_min, x_max, 1000)
    x_grid_norm = 2 * (x_grid - x_min) / (x_max - x_min) - 1
    
    # Generate y values based on polynomial degree
    if degree == 1:
        y_grid = -30 * x_grid_norm + 15
    elif degree == 2:
        y_grid = 15 * (x_grid_norm ** 2) - 15
    elif degree == 3:
        y_grid = -10 * (x_grid_norm ** 3) + 15 * x_grid_norm
    elif degree == 4:
        y_grid = 5 * (x_grid_norm ** 4) - 15 * (x_grid_norm ** 2) + 10
    else:
        raise ValueError("Degree must be between 1 and 4")
    
    # Scale y values to [0, 30] range
    y_grid = ((y_grid - y_grid.min()) / (y_grid.max() - y_grid.min())) * 30
    
    # Find x values where y ≈ 5
    y5_indices = np.where(np.abs(y_grid - 5) < 0.1)[0]
    if len(y5_indices) > 0:
        x_at_y5 = x_grid[y5_indices[0]]
    else:
        x_at_y5 = (x_min + x_max) / 2
    
    # Generate points with different densities based on y regions
    n_points_lower = n_points // 2  # More points for y ≤ 5
    n_points_upper = n_points - n_points_lower  # Fewer points for y > 5
    
    # Generate x values with different densities
    x_lower = np.linspace(x_at_y5, x_max, n_points_lower)
    x_upper = np.linspace(x_min, x_at_y5, n_points_upper)
    x = np.concatenate([x_upper, x_lower])
    
    # Calculate normalized x and corresponding y values
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    
    # Generate y values based on polynomial degree
    if degree == 1:
        y = -30 * x_norm + 15
    elif degree == 2:
        y = 15 * (x_norm ** 2) - 15
    elif degree == 3:
        y = -10 * (x_norm ** 3) + 15 * x_norm
    elif degree == 4:
        y = 5 * (x_norm ** 4) - 15 * (x_norm ** 2) + 10
    
    # Scale y values to [0, 30] range
    y = ((y - y.min()) / (y.max() - y.min())) * 30
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    }).sort_values('x', ascending=True)
    
    # Plot the results
    plt.figure(figsize=(15, 5))
    
    # Plot actual values
    plt.subplot(131)
    plt.scatter(df['x'], df['y'], alpha=0.6)
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.title(f'Polynomial Relationship (degree={degree})')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.grid(True)
    
    # Plot with normalized x
    plt.subplot(132)
    plt.scatter(x_norm, y, alpha=0.6)
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.title('Normalized X Scale')
    plt.xlabel('Normalized X values')
    plt.ylabel('Y values')
    plt.grid(True)
    
    # Plot point density
    plt.subplot(133)
    plt.hist(y, bins=20, orientation='horizontal')
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.title('Point Density Distribution')
    plt.xlabel('Count')
    plt.ylabel('Y values')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df

# Show examples of different polynomial degrees
for degree in [1, 2, 3, 4]:
    print(f"\nGenerating polynomial of degree {degree}:")
    data = generate_polynomial_data(degree=degree)
    print("\nBasic statistics:")
    print(data.describe())


        
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_polynomial_data(curve_type='default', degree=2, n_points=15):
    """
    Generate synthetic data with customizable polynomial relationships:
    1. Different types of polynomial shapes and degrees
    2. Points concentrated differently in two y regions (0-5 and 5-30)
    3. X values increase from 5000 to 45000
    4. Y values range approximately from 0 to 30
    
    Parameters:
    curve_type (str): Type of curve to generate:
        'default': Standard polynomial
        'sigmoid': S-shaped curve
        'convex': Convex decreasing curve
        'concave': Concave decreasing curve
        'wave': Wave-like pattern
    degree (int): Base degree of the polynomial (1-4)
    n_points (int): Number of total points to generate
    """
    x_min, x_max = 5000, 45000
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def generate_y_values(x_normalized, curve_type, degree):
        if curve_type == 'default':
            return -(x_normalized ** degree)
        # It will change the degree of the function, while the default 
        # parameter doesn't do that 
        # elif curve_type == 'sigmoid':
        #     return sigmoid(3 * x_normalized) * 30
        
        # elif curve_type == 'convex':
        #     return -(x_normalized ** (degree/2)) * 30
        
        # elif curve_type == 'concave':
        #     return -(x_normalized ** (degree*1.5)) * 30
        
        # elif curve_type == 'wave':
        #     return np.sin(3 * x_normalized) * 10 - x_normalized * 15
        
        # else:
        #     return -(x_normalized ** degree)
    
    # First, generate a dense grid of points to find where y=5 occurs
    x_grid = np.linspace(x_min, x_max, 1000)
    x_grid_norm = (x_grid - x_min) / (x_max - x_min)
    
    # Generate y values for the grid
    y_grid = generate_y_values(x_grid_norm, curve_type, degree)
    
    # Scale y values to [0, 30] range
    y_grid = ((y_grid - y_grid.min()) / (y_grid.max() - y_grid.min())) * 30
    
    # Find x values where y ≈ 5
    y5_indices = np.where(np.abs(y_grid - 5) < 0.1)[0]
    if len(y5_indices) > 0:
        x_at_y5 = x_grid[y5_indices[0]]
    else:
        x_at_y5 = (x_min + x_max) / 2
    
    # Generate points with different densities based on y regions
    n_points_lower = n_points // 2  # More points for y ≤ 5
    n_points_upper = n_points - n_points_lower  # Fewer points for y > 5
    
    # Generate x values with different densities
    x_lower = np.linspace(x_at_y5, x_max, n_points_lower)
    x_upper = np.linspace(x_min, x_at_y5, n_points_upper)
    x = np.concatenate([x_upper, x_lower])
    
    # Calculate normalized x and corresponding y values
    x_norm = (x - x_min) / (x_max - x_min)
    
    # Generate y values
    y = generate_y_values(x_norm, curve_type, degree)
    
    # Scale y values to [0, 30] range
    y = ((y - y.min()) / (y.max() - y.min())) * 30
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    }).sort_values('x', ascending=True)
    # Salva i dati in un file CSV
    df.to_csv(f"synthetic_data_degree_{degree}.csv", index=False)
    # Plot the results
    plt.figure(figsize=(15, 5))
    
    # Plot actual values
    plt.subplot(131)
    plt.scatter(df['x'], df['y'], alpha=0.6)
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.title(f'Curve Type: {curve_type}\nDegree: {degree}')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.grid(True)
    
    # Plot with normalized x
    plt.subplot(132)
    plt.scatter(x_norm, y, alpha=0.6)
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.title('Normalized X Scale')
    plt.xlabel('Normalized X values')
    plt.ylabel('Y values')
    plt.grid(True)
    
    # Plot point density
    plt.subplot(133)
    plt.hist(y, bins=20, orientation='horizontal')
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.title('Point Density Distribution')
    plt.xlabel('Count')
    plt.ylabel('Y values')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df

# Show examples of different curve types and degrees
curve_types = ['default']
degrees = [1, 2, 3, 4]  # You can try different degrees

for curve_type in curve_types:
    for degree in degrees:
        print(f"\nGenerating {curve_type} curve with degree {degree}:")
        data = generate_polynomial_data(curve_type=curve_type, degree=degree)
        print("\nBasic statistics:")
        print(data.describe())

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_polynomial_data(degree=2, n_points=100, noise_level=0.5):
    """
    Generate synthetic data with polynomial relationships and noise:
    1. Basic polynomial of specified degree
    2. Points concentrated differently in two y regions (0-5 and 5-30)
    3. X values increase from 5000 to 45000
    4. Y values range approximately from 0 to 30
    5. Random noise added to points
    
    Parameters:
    degree (int): Degree of the polynomial (1=linear, 2=quadratic, 3=cubic, etc.)
    n_points (int): Number of total points to generate
    noise_level (float): Amount of random noise to add (0 = no noise, higher = more noise)
    """
    x_min, x_max = 5000, 45000
    
    def generate_y_values(x_normalized, degree):
        return -(x_normalized ** degree)
    
    # First, generate a dense grid of points to find where y=5 occurs
    x_grid = np.linspace(x_min, x_max, 1000)
    x_grid_norm = (x_grid - x_min) / (x_max - x_min)
    
    # Generate y values for the grid
    y_grid = generate_y_values(x_grid_norm, degree)
    
    # Scale y values to [0, 30] range
    y_grid = ((y_grid - y_grid.min()) / (y_grid.max() - y_grid.min())) * 30
    
    # Find x values where y ≈ 5
    y5_indices = np.where(np.abs(y_grid - 5) < 0.1)[0]
    if len(y5_indices) > 0:
        x_at_y5 = x_grid[y5_indices[0]]
    else:
        x_at_y5 = (x_min + x_max) / 2
    
    # Generate points with different densities based on y regions
    n_points_lower = n_points // 2  # More points for y ≤ 5
    n_points_upper = n_points - n_points_lower  # Fewer points for y > 5
    
    # Generate x values with different densities
    x_lower = np.linspace(x_at_y5, x_max, n_points_lower)
    x_upper = np.linspace(x_min, x_at_y5, n_points_upper)
    x = np.concatenate([x_upper, x_lower])
    
    # Calculate normalized x and corresponding y values
    x_norm = (x - x_min) / (x_max - x_min)
    
    # Generate base y values
    y = generate_y_values(x_norm, degree)
    
    # Scale y values to [0, 30] range
    y = ((y - y.min()) / (y.max() - y.min())) * 30
    
    # Add random noise
    noise = np.random.normal(0, noise_level, size=len(y))
    y = y + noise
    
    # Clip y values to ensure they stay in [0, 30] range after adding noise
    y = np.clip(y, 0, 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    }).sort_values('x', ascending=True)
    
    print(df)
    df.to_csv(f"synthetic_data_degree_{degree}_{noise_level}.csv", index=False)
    # Plot the results
    plt.figure(figsize=(15, 5))
    
    # Plot actual values
    plt.subplot(131)
    plt.scatter(df['x'], df['y'], alpha=0.6)
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.title(f'Polynomial Degree: {degree}\nNoise Level: {noise_level}')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.grid(True)
    
    # Plot with normalized x
    plt.subplot(132)
    plt.scatter(x_norm, y, alpha=0.6)
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.title('Normalized X Scale')
    plt.xlabel('Normalized X values')
    plt.ylabel('Y values')
    plt.grid(True)
    
    # Plot point density
    plt.subplot(133)
    plt.hist(y, bins=20, orientation='horizontal')
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.title('Point Density Distribution')
    plt.xlabel('Count')
    plt.ylabel('Y values')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return df

# Example usage with different degrees and noise levels
degrees = [1, 2, 3]
noise_levels = [0.2, 0.5, 1.0]

# Generate example with degree=2 and moderate noise
data = generate_polynomial_data(degree=1, n_points=15, noise_level=1.0)
print("\nFirst few rows:")
print(data.head())
print("\nBasic statistics:")
print(data.describe())

# Uncomment to test different combinations
# for degree in degrees:
#     for noise in noise_levels:
#         print(f"\nGenerating polynomial degree {degree} with noise level {noise}:")
#         data = generate_polynomial_data(degree=degree, noise_level=noise)