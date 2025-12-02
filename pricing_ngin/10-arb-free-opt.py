import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from gaussian_3_copula_pricing import GaussianCopulaPricer

def arb_free_optimization(tenor=5.0):
    print(f"Running Arbitrage-Free Optimization for {int(tenor)}Y Tenor...")
    
    # 1. Load Data & Initialize Pricer
    base_df = pd.read_csv('res/base_correlations.csv')
    pricer = GaussianCopulaPricer('outdata/', 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
    
    # Get Market Knots for this Tenor
    subset = base_df[base_df['Tenor'] == tenor].sort_values('Detachment')
    
    # Add anchor at 0%
    x_mkt = subset['Detachment'].values
    y_mkt = subset['Correlation'].values
    if x_mkt[0] > 0.0:
        x_mkt = np.insert(x_mkt, 0, 0.0)
        y_mkt = np.insert(y_mkt, 0, y_mkt[0]) # Flat extrapolation to 0
        
    # 2. Define Optimization Grid (The "Variables")
    # We optimize correlation at these knots to ensure the whole shape is valid.
    # Too many knots = slow. Too few = rigid.
    # We use Market Knots + some intermediate/tail points.
    
    # Start with market knots
    opt_grid_x = list(x_mkt)
    
    # Add tail points to define the rest of the curve (e.g. 20%, 30% ... 100%)
    tail_points = np.arange(0.20, 1.01, 0.10)
    for p in tail_points:
        if p > opt_grid_x[-1] + 0.01: # Avoid duplicates near market points
            opt_grid_x.append(p)
            
    opt_grid_x = np.array(sorted(opt_grid_x))
    
    # Initial Guess: Use PCHIP to fill the tail points
    initial_pchip = PchipInterpolator(x_mkt, y_mkt)
    opt_grid_y_init = initial_pchip(opt_grid_x)

    # 3. Define Objective Function
    # We want to stay close to the Initial PCHIP guess (Prior) 
    # BUT we heavily penalize deviating from actual Market Quotes.
    
    market_indices = [i for i, x in enumerate(opt_grid_x) if x in x_mkt]
    
    def objective(y_new):
        # 1. Distance from Prior (Regularization to keep shape stable)
        penalty_prior = np.sum((y_new - opt_grid_y_init)**2)
        
        # 2. Distance from Market Quotes (Harder penalty)
        # We extract the values at market indices
        y_at_mkt = y_new[market_indices]
        penalty_mkt = np.sum((y_at_mkt - y_mkt)**2) * 1000.0 
        
        # 3. Smoothness (Penalize 2nd derivative/jaggedness)
        d2 = np.diff(y_new, 2)
        penalty_smooth = np.sum(d2**2) * 10.0
        
        return penalty_prior + penalty_mkt + penalty_smooth

    # 4. Define Arbitrage Constraint
    # EL(K_i) - EL(K_{i-1}) >= 0
    
    # Memoize/Cache to speed up constraint checking if needed, 
    # but optimizer changes Y every time, so must re-calc.
    
    def calculate_el_vector(y_current):
        els = []
        for k, rho in zip(opt_grid_x, y_current):
            if k == 0:
                els.append(0.0)
                continue
            
            # EL = PV_Protection (assuming 0 coupon)
            tranche = {
                'Attachment': 0.0, 'Detachment': k, 'Maturity': tenor,
                'Spread_bps': 0, 'Type': 'Upfront'
            }
            # We constrain rho to be within [0, 1] bounds later
            rho = np.clip(rho, 0.001, 0.999)
            
            # Get Price (Upfront points)
            upfront = pricer.price_tranche(tranche, rho)
            pv_loss = upfront * k
            els.append(pv_loss)
        return np.array(els)

    def constraint_monotonic_el(y_new):
        # Returns vector that must be non-negative
        els = calculate_el_vector(y_new)
        # Diff must be >= 0
        return np.diff(els)

    # 5. Run Optimization
    # Constraints dictionary
    cons = ({'type': 'ineq', 'fun': constraint_monotonic_el})
    
    # Bounds for correlation (0 to 1)
    bounds = [(0.01, 0.999) for _ in opt_grid_x]
    
    print("Optimizing... (This may take a minute)")
    res = minimize(objective, opt_grid_y_init, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'ftol': 1e-4, 'disp': True, 'maxiter': 50})

    if res.success:
        print("Optimization Successful!")
    else:
        print(f"Optimization Failed: {res.message}")

    # 6. Generate Final Curves for Plotting
    opt_y_final = res.x
    
    # Interpolate the optimized knots for smooth plotting
    final_interpolator = PchipInterpolator(opt_grid_x, opt_y_final)
    
    x_plot = np.linspace(0.0, 1.0, 100)
    y_plot_pchip = initial_pchip(x_plot)
    y_plot_opt = final_interpolator(x_plot)
    
    # Plot Correlation Smile
    plt.figure(figsize=(10, 6))
    plt.scatter(x_mkt*100, y_mkt, color='black', label='Market Quotes', zorder=5, s=50)
    plt.plot(x_plot*100, y_plot_pchip, 'r--', label='Initial PCHIP (Unconstrained)')
    plt.plot(x_plot*100, y_plot_opt, 'g-', label='Arb-Free Optimized', linewidth=2)
    
    plt.title(f'Arbitrage-Free Optimization ({int(tenor)}Y)')
    plt.xlabel('Detachment (%)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.savefig('res/arb_free_optimization.png')
    print("Saved plot to res/arb_free_optimization.png")
    
    # Plot Expected Loss (Arbitrage Check)
    el_pchip = []
    el_opt = []
    
    # Calculate EL on the fine grid for visualization
    # (We just calculate a few points to show the shape)
    check_points = np.linspace(0.03, 0.30, 10) # Zoom in on the risky part
    
    print("\nVerifying Arbitrage on key points...")
    for k in check_points:
        # PCHIP EL
        rho_p = initial_pchip(k)
        t_p = {'Attachment': 0.0, 'Detachment': k, 'Maturity': tenor, 'Spread_bps': 0, 'Type': 'Upfront'}
        el_pchip.append(pricer.price_tranche(t_p, rho_p) * k)
        
        # Optimized EL
        rho_o = final_interpolator(k)
        el_opt.append(pricer.price_tranche(t_p, rho_o) * k)

    plt.figure(figsize=(10, 6))
    plt.plot(check_points*100, el_pchip, 'r--o', label='PCHIP EL')
    plt.plot(check_points*100, el_opt, 'g-o', label='Optimized EL')
    plt.title(f'Expected Loss Monotonicity Check ({int(tenor)}Y)')
    plt.xlabel('Detachment (%)')
    plt.ylabel('Expected Loss (PV)')
    plt.legend()
    plt.grid(True)
    plt.savefig('res/arb_free_el_check.png')
    print("Saved EL check to res/arb_free_el_check.png")

if __name__ == "__main__":
    arb_free_optimization()