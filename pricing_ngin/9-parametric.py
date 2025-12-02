import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
import os

def parametric_model(x, a, b, c):
    """Simple Quadratic Model: rho(x) = a + b*x + c*x^2"""
    return a + b*x + c*x**2

def compare_interpolation_restricted():
    print("Running Parametric Comparison (Focused on 0-15% region)...")
    
    # 1. Load Data
    res_dir = 'res/'
    base_df = pd.read_csv(os.path.join(res_dir, 'base_correlations.csv'))
    
    # Pick one tenor to visualize (e.g. 5Y, usually the most liquid)
    tenor = 5.0
    
    # FILTER: Only keep points <= 15% Detachment
    # This ignores the 15-100% point which can skew the quadratic fit
    subset = base_df[(base_df['Tenor'] == tenor) & (base_df['Detachment'] <= 0.15)].sort_values('Detachment')
    
    if subset.empty:
        print(f"No data found for Tenor {tenor}Y in 0-15% range.")
        return

    x_known = subset['Detachment'].values
    y_known = subset['Correlation'].values
    
    # Add anchor at 0% (Flat extrapolation from first point)
    if len(x_known) > 0 and x_known[0] > 0.0:
        x_known = np.insert(x_known, 0, 0.0)
        y_known = np.insert(y_known, 0, y_known[0])
    
    # 2. PCHIP Interpolation (The "Exact" Benchmark for these points)
    pchip = PchipInterpolator(x_known, y_known)
    
    # 3. Parametric Fit (Quadratic)
    # Fit only to these points (0, 3, 7, 10, 15)
    try:
        popt, _ = curve_fit(parametric_model, x_known, y_known)
    except:
        print("Parametric fit failed.")
        popt = [0, 0, 0]
    
    # Generate Grid for Plotting (Show up to 20% to visualize trajectory)
    x_fine = np.linspace(0.0, 0.20, 100)
    y_pchip = pchip(x_fine)
    y_param = parametric_model(x_fine, *popt)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Plot the Knots
    plt.scatter(x_known*100, y_known, color='black', label='Calibrated Knots (<=15%)', zorder=5, s=60)
    
    # Plot PCHIP
    plt.plot(x_fine*100, y_pchip, 'b-', label='PCHIP (Exact Interpolation)', linewidth=2, alpha=0.6)
    
    # Plot Parametric
    plt.plot(x_fine*100, y_param, 'r--', label='Quadratic Fit (Smoothed)', linewidth=2)
    
    plt.title(f'Interpolation vs Quadratic Fit ({int(tenor)}Y) - 0-15% Focus')
    plt.xlabel('Detachment (%)')
    plt.ylabel('Base Correlation')
    plt.legend()
    plt.grid(True)
    
    # Explicitly limit x-axis to zoom in on the relevant region
    plt.xlim(0, 16) 
    
    save_path = os.path.join(res_dir, 'parametric_comparison_restricted.png')
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    
    print("\n--- Quadratic Fit Parameters (rho = a + b*K + c*K^2) ---")
    print(f"a (Intercept): {popt[0]:.4f}")
    print(f"b (Slope):     {popt[1]:.4f}")
    print(f"c (Convexity): {popt[2]:.4f}")
    
    # Calculate fit error on the knots
    y_pred_param = parametric_model(x_known, *popt)
    sse = np.sum((y_known - y_pred_param)**2)
    print(f"Sum of Squared Errors (on knots): {sse:.6f}")

if __name__ == "__main__":
    compare_interpolation_restricted()