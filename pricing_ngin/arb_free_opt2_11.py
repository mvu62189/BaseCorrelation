import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from gaussian_3_copula_pricing import GaussianCopulaPricer
import os

def arb_free_optimization(tenor=5.0):
    print(f"Running Arbitrage-Free Optimization for {int(tenor)}Y Tenor (0-15% Focus)...")
    
    # 1. Load Data & Initialize Pricer
    res_dir = 'res/'
    base_df = pd.read_csv(os.path.join(res_dir, 'base_correlations.csv'))
    pricer = GaussianCopulaPricer('outdata/', 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
    
    # FILTER: Strictly keep points <= 15% Detachment
    # We ignore the 100% detachment point to prevent Super Senior distortion.
    subset = base_df[(base_df['Tenor'] == tenor) & (base_df['Detachment'] <= 0.15)].sort_values('Detachment')
    
    if subset.empty:
        print(f"No data found for {tenor}Y <= 15%.")
        return

    # Add anchor at 0%
    x_mkt = subset['Detachment'].values
    y_mkt = subset['Correlation'].values
    if x_mkt[0] > 0.0:
        x_mkt = np.insert(x_mkt, 0, 0.0)
        y_mkt = np.insert(y_mkt, 0, y_mkt[0]) # Flat extrapolation to 0
        
    # 2. Define Optimization Grid
    # We stop the grid shortly after 15% to let the boundary derivative settle,
    # but we don't extend to 100%.
    opt_grid_x = list(x_mkt)
    
    # Add a buffer tail point (e.g., 20%) to stabilize the spline at 15%
    if 0.20 > opt_grid_x[-1]:
        opt_grid_x.append(0.20)
            
    opt_grid_x = np.array(sorted(list(set(opt_grid_x)))) # Deduplicate and sort
    
    # Initial Guess: PCHIP through the restricted market points
    initial_pchip = PchipInterpolator(x_mkt, y_mkt)
    opt_grid_y_init = initial_pchip(opt_grid_x)

    # 3. Define Objective Function
    # Identify which grid points correspond to actual market quotes
    market_indices = [i for i, x in enumerate(opt_grid_x) if x in x_mkt]
    
    def objective(y_new):
        # A. Distance from Prior (Stability)
        penalty_prior = np.sum((y_new - opt_grid_y_init)**2)
        
        # B. Distance from Market Quotes (Accuracy - High Penalty)
        y_at_mkt = y_new[market_indices]
        penalty_mkt = np.sum((y_at_mkt - y_mkt)**2) * 1000.0 
        
        # C. Smoothness (Penalize 2nd derivative)
        # Calculates (y_{i+1} - 2y_i + y_{i-1}) approximate curvature
        d2 = np.diff(y_new, 2)
        penalty_smooth = np.sum(d2**2) * 10.0
        
        return penalty_prior + penalty_mkt + penalty_smooth

    # 4. Define Arbitrage Constraint (EL Monotonicity)
    def calculate_el_vector(y_current):
        els = []
        for k, rho in zip(opt_grid_x, y_current):
            if k == 0:
                els.append(0.0)
                continue
            
            tranche = {
                'Attachment': 0.0, 'Detachment': k, 'Maturity': tenor,
                'Spread_bps': 0, 'Type': 'Upfront'
            }
            # Clip rho to safe bounds
            rho = np.clip(rho, 0.001, 0.999)
            
            # PV Protection
            upfront = pricer.price_tranche(tranche, rho)
            pv_loss = upfront * k
            els.append(pv_loss)
        return np.array(els)

    def constraint_monotonic_el(y_new):
        # Constraint: EL[i] - EL[i-1] >= 0
        els = calculate_el_vector(y_new)
        return np.diff(els)

    # 5. Run Optimization
    cons = ({'type': 'ineq', 'fun': constraint_monotonic_el})
    bounds = [(0.01, 0.999) for _ in opt_grid_x]
    
    print("Optimizing... (Restricted Grid)")
    res = minimize(objective, opt_grid_y_init, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'ftol': 1e-4, 'disp': True, 'maxiter': 50})

    if res.success:
        print("Optimization Successful!")
    else:
        print(f"Optimization Failed: {res.message}")

    # 6. Plotting
    opt_y_final = res.x
    final_interpolator = PchipInterpolator(opt_grid_x, opt_y_final)
    
    x_plot = np.linspace(0.0, 0.20, 100) # Plot only up to 20%
    y_plot_pchip = initial_pchip(x_plot)
    y_plot_opt = final_interpolator(x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_mkt*100, y_mkt, color='black', label='Market Quotes (<=15%)', zorder=5, s=60)
    plt.plot(x_plot*100, y_plot_pchip, 'r--', label='Initial PCHIP')
    plt.plot(x_plot*100, y_plot_opt, 'g-', label='Arb-Free Optimized', linewidth=2)
    
    plt.title(f'Arbitrage-Free Optimization ({int(tenor)}Y) - 0-15% Focus')
    plt.xlabel('Detachment (%)')
    plt.ylabel('Base Correlation')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 20) # Zoomed in
    
    save_path = os.path.join(res_dir, 'arb_free_optimization_restricted.png')
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    arb_free_optimization()