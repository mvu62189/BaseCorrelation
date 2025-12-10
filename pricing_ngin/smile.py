# pricing_ngin/smile_factory.py

import pandas as pd
import numpy as np
import os
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

# Core imports for the Arb-Free Optimization
from pricing_ngin.gaussian_3_copula_pricing import GaussianCopulaPricer 

def generate_interpolator(output_dir, tenor, method='ARB_FREE'):
    """
    Generates and returns the selected correlation interpolator for a given tenor and method.

    Args:
        output_dir (str): Path to the daily output cache (contains /res/base_correlations.csv).
        tenor (float): The target maturity tenor (e.g., 5.0).
        method (str): 'UNCONSTRAINED', 'CONSTRAINED', or 'ARB_FREE'.
        
    Returns:
        PchipInterpolator: The selected correlation function, or None if optimization fails.
    """
    res_dir = os.path.join(output_dir, 'res')
    base_df = pd.read_csv(os.path.join(res_dir, 'base_correlations.csv'))
    
    # Filter and prepare market knots (logic adapted from 12-constrained-pchip.py)
    subset = base_df[(base_df['Tenor'] == tenor) & (base_df['Detachment'] <= 0.15)].sort_values('Detachment')
    
    x_mkt = subset['Detachment'].values
    y_mkt = subset['Correlation'].values
    if x_mkt.size == 0: raise ValueError(f"No market knots found for {tenor}Y.")
        
    if x_mkt[0] > 0.0:
        x_mkt = np.insert(x_mkt, 0, 0.0)
        y_mkt = np.insert(y_mkt, 0, y_mkt[0])

    # --- 1. UNCONSTRAINED PCHIP ---
    if method == 'UNCONSTRAINED':
        return PchipInterpolator(x_mkt, y_mkt, extrapolate=True)

    # --- 2. CONSTRAINED PCHIP ---
    if method == 'CONSTRAINED':
        x_constr = np.append(x_mkt, 1.0)
        y_constr = np.append(y_mkt, 1.0) 
        return PchipInterpolator(x_constr, y_constr, extrapolate=True)

    # --- 3. ARB-FREE OPTIMIZED (Default) ---
    if method == 'ARB_FREE':
        # Must re-initialize the pricer context for the current day's curves
        outdata_dir = os.path.join(output_dir, 'outdata')
        pricer = GaussianCopulaPricer(outdata_dir, 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
        
        # Optimization Grid Setup (adapted from 12-constrained-pchip.py)
        opt_grid_x = list(x_mkt)
        if 0.20 > opt_grid_x[-1]: opt_grid_x.append(0.20)
        opt_grid_x = np.array(sorted(list(set(opt_grid_x))))
        
        initial_pchip = PchipInterpolator(x_mkt, y_mkt)
        opt_grid_y_init = initial_pchip(opt_grid_x)
        market_indices = [i for i, x in enumerate(opt_grid_x) if x in x_mkt]

        def calculate_el_vector(y_current):
            # Calculates PV Expected Loss for 0-K tranches
            els = []
            for k, rho in zip(opt_grid_x, y_current):
                if k == 0:
                    els.append(0.0)
                    continue
                rho = np.clip(rho, 0.001, 0.999)
                t = {'Attachment': 0.0, 'Detachment': k, 'Maturity': tenor, 'Spread_bps': 0, 'Type': 'Upfront'}
                upfront = pricer.price_tranche(t, rho)
                els.append(upfront * k)
            return np.array(els)

        def objective(y_new):
            # Combination of Prior, Market Fit, and Smoothness penalties
            pen_prior = np.sum((y_new - opt_grid_y_init)**2)
            pen_mkt = np.sum((y_new[market_indices] - y_mkt)**2) * 1000.0
            d2 = np.diff(y_new, 2)
            pen_smooth = np.sum(d2**2) * 10.0
            return pen_prior + pen_mkt + pen_smooth

        def constraint_monotonic_el(y_new):
            # Constraint: EL[i] - EL[i-1] >= 0
            return np.diff(calculate_el_vector(y_new))

        cons = ({'type': 'ineq', 'fun': constraint_monotonic_el})
        bounds = [(0.01, 0.999) for _ in opt_grid_x]
        
        res = minimize(objective, opt_grid_y_init, method='SLSQP', bounds=bounds, constraints=cons)
        
        if res.success:
            opt_y_final = res.x
            return PchipInterpolator(opt_grid_x, opt_y_final, extrapolate=True)
        else:
            print(f"ARB_FREE optimization failed: {res.message}. Falling back to CONSTRAINED.")
            return generate_interpolator(output_dir, tenor, method='CONSTRAINED') # Fallback

    raise ValueError(f"Unknown smile method: {method}")