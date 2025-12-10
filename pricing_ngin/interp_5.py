import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator # Preserves monotonicity
import os

def generate_continuous_smile(output_dir):
    print(f"Running Interpolation in: {output_dir}")
    res_dir = os.path.join(output_dir, 'res')
    df = pd.read_csv(os.path.join(res_dir, 'base_correlations.csv'))
    results = []
    
    for tenor, group in df.groupby('Tenor'):
        x = group['Detachment'].values
        y = group['Correlation'].values
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        
        if len(x) < 2: continue
        
        # Add Anchor? Standard practice for PCHIP
        # The script originally did PCHIP on existing points. Assuming that logic.
        smile = PchipInterpolator(x, y, extrapolate=True)
        x_fine = np.round(np.arange(0.005, 0.155, 0.005), 3)
        y_fine = smile(x_fine)
        
        for d, r in zip(x_fine, y_fine):
            results.append({'Tenor': tenor, 'Detachment': d, 'Correlation': r})
            
    pd.DataFrame(results).to_csv(os.path.join(res_dir, 'continuous_base_correlations.csv'), index=False)

if __name__ == "__main__":
    generate_continuous_smile('./')