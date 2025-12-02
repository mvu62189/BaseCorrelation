import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

def parametric_model(x, a, b, c):
    """Simple Quadratic Model: rho(x) = a + b*x + c*x^2"""
    return a + b*x + c*x**2

def compare_interpolation():
    df = pd.read_csv('res/base_correlations.csv')
    
    # Pick one tenor to visualize
    tenor = 5.0
    subset = df[df['Tenor'] == tenor].sort_values('Detachment')
    
    x_known = subset['Detachment'].values
    y_known = subset['Correlation'].values
    
    # Add anchor at 0 (flat extrapolation)
    x_known = np.insert(x_known, 0, 0.0)
    y_known = np.insert(y_known, 0, y_known[0])
    
    # 1. PCHIP Interpolation
    pchip = PchipInterpolator(x_known, y_known)
    
    # 2. Parametric Fit (Quadratic)
    # We constrain the fit to be somewhat reasonable
    popt, _ = curve_fit(parametric_model, x_known, y_known)
    
    # Generate Grid
    x_fine = np.linspace(0.0, 1.0, 100)
    y_pchip = pchip(x_fine)
    y_param = parametric_model(x_fine, *popt)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_known*100, y_known, color='black', label='Calibrated Points', zorder=5)
    plt.plot(x_fine*100, y_pchip, label='PCHIP Interpolation', linestyle='-')
    plt.plot(x_fine*100, y_param, label=f'Quadratic Fit', linestyle='--')
    
    plt.title(f'Interpolation vs Parametric Fit ({int(tenor)}Y)')
    plt.xlabel('Detachment (%)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig('res/parametric_comparison.png')
    print("Saved plot to res/parametric_comparison.png")
    
    print("\n--- Fit Parameters ---")
    print(f"Quadratic: {popt[0]:.2f} + {popt[1]:.2f}*K + {popt[2]:.2f}*K^2")

if __name__ == "__main__":
    compare_interpolation()