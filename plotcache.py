import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_cache_surface(date_str, cache_root='./daily_output_cache/'):
    """
    Loads base correlations from the cache (BASE scenario) and plots/saves the surface.
    """
    # 1. Construct Path (Corrected to look in 'base/res/')
    # We plot the 'base' (unbumped) surface as it represents the actual market fit.
    base_dir = os.path.join(cache_root, date_str, 'base')
    file_path = os.path.join(base_dir, 'res', 'continuous_base_correlations.csv')
    
    # 2. Check Existence
    if not os.path.exists(file_path):
        print(f"[ERROR] No cache found for date '{date_str}'. Expected file at: {file_path}")
        return

    print(f"Loading data for {date_str}...")
    df = pd.read_csv(file_path)
    
    # 3. Pivot Data for 3D Plotting
    pivot = df.pivot(index='Tenor', columns='Detachment', values='Correlation')
    x_tenors = pivot.index.values
    y_detach = pivot.columns.values
    X, Y = np.meshgrid(x_tenors, y_detach * 100) # Detachment in %
    Z = pivot.values.T 

    # 4. Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.5, alpha=0.8)
    
    ax.set_title(f'Continuous Base Correlation Surface: {date_str}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Tenor (Years)', fontsize=11)
    ax.set_ylabel('Detachment (%)', fontsize=11)
    ax.set_zlabel('Correlation', fontsize=11)
    ax.view_init(elev=25, azim=-130)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label='Base Correlation')
    plt.tight_layout()
    
    # 5. Save Plot
    # We save it into the same folder as the data for easy reference
    # e.g. ./daily_output_cache/1119/base/res/surface_plot_1119.png
    output_dir = os.path.dirname(file_path) 
    save_path = os.path.join(output_dir, f'continuous_surface_plot_{date_str}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")
    
    # Show plot (optional, comment out if running in batch)
    plt.show()

if __name__ == "__main__":
    # Add the dates you want to visualize here
    dates_to_plot = ['1119', '1120', '1121', '1125', '1126', '1201', '1203'] 
    
    for date in dates_to_plot:
        plot_cache_surface(date)