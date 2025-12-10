import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from gaussian_3_copula_pricing import GaussianCopulaPricer
import os

def run_constrained_comparison(tenor=5.0):
    print(f"Running 3-Way Comparison for {int(tenor)}Y Tenor...")
    print("1. Unconstrained PCHIP (0-15% only)")
    print("2. Constrained PCHIP (0-15% + 100% Anchor)")
    print("3. Arb-Free Optimization")

    # --- SETUP & DATA LOADING ---
    res_dir = 'res/'
    base_df = pd.read_csv(os.path.join(res_dir, 'base_correlations.csv'))
    pricer = GaussianCopulaPricer('outdata/', 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
    
    # Filter 0-15% Data (The "Active" Region)
    subset = base_df[(base_df['Tenor'] == tenor) & (base_df['Detachment'] <= 0.15)].sort_values('Detachment')
    
    if subset.empty:
        print("No data found.")
        return

    # Base Knots
    x_mkt = subset['Detachment'].values
    y_mkt = subset['Correlation'].values
    if x_mkt[0] > 0.0:
        x_mkt = np.insert(x_mkt, 0, 0.0)
        y_mkt = np.insert(y_mkt, 0, y_mkt[0]) # Flat extrapolation

    # --- MODEL 1: UNCONSTRAINED PCHIP ---
    # Fits only to the local data. Extrapolates slope naturally.
    pchip_unconstrained = PchipInterpolator(x_mkt, y_mkt)

    # --- MODEL 2: CONSTRAINED PCHIP (With 100% Anchor) ---
    # Forces the curve to hit 1.0 correlation at 100% detachment.
    # This simulates the "Cap" constraint.
    x_constr = np.append(x_mkt, 1.0)
    y_constr = np.append(y_mkt, 1.0) # Assume correlation converges to 1.0
    pchip_constrained = PchipInterpolator(x_constr, y_constr)

    # --- MODEL 3: ARB-FREE OPTIMIZATION ---
    # (Simplified version of File 10 logic)
    
    # Grid: Market points + buffer at 20%
    opt_grid_x = list(x_mkt)
    if 0.20 > opt_grid_x[-1]: opt_grid_x.append(0.20)
    opt_grid_x = np.array(sorted(list(set(opt_grid_x))))
    
    # Init guess (Unconstrained)
    opt_grid_y_init = pchip_unconstrained(opt_grid_x)
    market_indices = [i for i, x in enumerate(opt_grid_x) if x in x_mkt]
    
    # Optimization Functions
    def calculate_el_vector(y_current):
        els = []
        for k, rho in zip(opt_grid_x, y_current):
            if k == 0:
                els.append(0.0)
                continue
            # Clip rho
            rho = np.clip(rho, 0.001, 0.999)
            # Price
            t = {'Attachment': 0.0, 'Detachment': k, 'Maturity': tenor, 'Spread_bps': 0, 'Type': 'Upfront'}
            upfront = pricer.price_tranche(t, rho)
            els.append(upfront * k)
        return np.array(els)

    def objective(y_new):
        # Priors + Mkt Penalty + Smoothness
        pen_prior = np.sum((y_new - opt_grid_y_init)**2)
        pen_mkt = np.sum((y_new[market_indices] - y_mkt)**2) * 1000.0
        d2 = np.diff(y_new, 2)
        pen_smooth = np.sum(d2**2) * 10.0
        return pen_prior + pen_mkt + pen_smooth

    def constraint(y_new):
        return np.diff(calculate_el_vector(y_new))

    # Run Opt
    cons = ({'type': 'ineq', 'fun': constraint})
    bounds = [(0.01, 0.999) for _ in opt_grid_x]
    res = minimize(objective, opt_grid_y_init, method='SLSQP', bounds=bounds, constraints=cons)
    
    if res.success:
        opt_y_final = res.x
        pchip_opt = PchipInterpolator(opt_grid_x, opt_y_final)
    else:
        print("Optimization failed, skipping Model 3.")
        pchip_opt = None

    # --- VISUALIZATION ---
    # Plotting Grid (Zoomed in 0-20% to see the distortion)
    x_plot = np.linspace(0.0, 0.20, 100)
    
    y_unc = pchip_unconstrained(x_plot)
    y_con = pchip_constrained(x_plot)
    y_opt = pchip_opt(x_plot) if pchip_opt else np.zeros_like(x_plot)
    
    plt.figure(figsize=(12, 7))
    
    # Market Knots
    plt.scatter(x_mkt*100, y_mkt, color='black', s=80, label='Market Quotes', zorder=10)
    
    # Lines
    plt.plot(x_plot*100, y_unc, 'b--', label='Unconstrained PCHIP', linewidth=2)
    plt.plot(x_plot*100, y_con, 'r:', label='Constrained PCHIP (with 100% Anchor)', linewidth=2.5)
    plt.plot(x_plot*100, y_opt, 'g-', label='Arb-Free Optimized', linewidth=2.5, alpha=0.8)
    
    plt.title(f'Comparison of Smoothing Methods ({int(tenor)}Y)', fontsize=14)
    plt.xlabel('Detachment (%)', fontsize=12)
    plt.ylabel('Base Correlation', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlim(0, 20) # Focus on the active region
    
    save_path = os.path.join(res_dir, 'method_comparison_constrained.png')
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    run_constrained_comparison()