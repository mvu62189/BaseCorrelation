import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator # Preserves monotonicity

def generate_continuous_smile():
    # 1. Load the discrete calibration points
    df = pd.read_csv('res/base_correlations.csv')
    
    # 2. Prepare plot
    plt.figure(figsize=(10, 6))
    
    continuous_results = []

    # 3. Process each tenor
    for tenor, group in df.groupby('Tenor'):
        # Add (0,0) point if needed, or assume first point is boundary. 
        # Base correlation at Detachment=0 is theoretically 0, but often treated asymptotically.
        # It's safer to start interpolation from the first equity tranche (e.g., 3%).
        
        x_known = group['Detachment'].values
        y_known = group['Correlation'].values
        
        # Sort to ensure strictly increasing x for interpolation
        idx = np.argsort(x_known)
        x_known = x_known[idx]
        y_known = y_known[idx]

        if len(x_known) < 2:
            print(f"Skipping Tenor {tenor}Y: Not enough data points to interpolate.")
            continue

        # Create Continuous Function (Smile)
        # PchipInterpolator is often preferred for Base Corr to avoid "swinging" arbitrage
        smile_func = PchipInterpolator(x_known, y_known, extrapolate=True)
        
        # 4. Resample on fine grid (continuous space every 0.5%)
        x_fine = np.round(np.arange(0.005, .155, 0.005), 3)
        y_fine = smile_func(x_fine)
        
        # Store for output
        for d, r in zip(x_fine, y_fine):
            continuous_results.append({'Tenor': tenor, 'Detachment': d, 'Correlation': r})
            
        # Plot
        plt.plot(x_fine * 100, y_fine, label=f'{int(tenor)}Y Smile')
        plt.scatter(x_known * 100, y_known, s=30) # Show original knots

    plt.title('Continuous Base Correlation Smiles (PCHIP Interpolation)')
    plt.xlabel('Detachment (%)')
    plt.ylabel('Base Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig('res/continuous_correlation_smiles.png')
    
    # Save continuous data for checking
    pd.DataFrame(continuous_results).to_csv('res/continuous_base_correlations.csv', index=False)
    print("Saved continuous smiles to res/continuous_base_correlations.csv")

if __name__ == "__main__":
    generate_continuous_smile()