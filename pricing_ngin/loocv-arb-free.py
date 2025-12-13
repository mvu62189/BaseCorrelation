import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
import os
from gaussian_3_copula_pricing import GaussianCopulaPricer

class ArbFreeLOOCV:
    def __init__(self, res_dir='res/', data_dir='indata/', out_dir='outdata/'):
        print("Initializing Arb-Free LOOCV Validator...")
        self.res_dir = res_dir
        self.base_df = pd.read_csv(os.path.join(res_dir, 'base_correlations.csv'))
        self.market_df = pd.read_csv(os.path.join(data_dir, 'CDX.NA.IG.45.csv'))
        self.pricer = GaussianCopulaPricer(out_dir, 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
        
        # Map Detachment -> Tranche Info for Pricing
        # We need this to know which tranche to reprice when we hide a specific point
        self.tranche_map = {
            0.07: {'Att': 0.03, 'Name': '3-7%', 'Type': 'Spread', 'Running': 100, 'Col': '3-7'},
            0.10: {'Att': 0.07, 'Name': '7-10%', 'Type': 'Spread', 'Running': 100, 'Col': '7-10'}
        }

    def run_loocv_optimization(self, tenor=5.0, hide_det=0.07):
        print(f"\n--- Running Arb-Free LOOCV (Tenor: {int(tenor)}Y, Hide: {hide_det*100}%) ---")
        
        # 1. Prepare Training Data (Remove the hidden point)
        subset = self.base_df[self.base_df['Tenor'] == tenor].sort_values('Detachment')
        
        # Full set for reference
        x_full = subset['Detachment'].values
        y_full = subset['Correlation'].values
        
        # Training set (exclude hidden)
        mask = ~np.isclose(x_full, hide_det)
        x_train = x_full[mask]
        y_train = y_full[mask]
        
        # Add Anchor at 0%
        if x_train[0] > 0.0:
            x_train = np.insert(x_train, 0, 0.0)
            y_train = np.insert(y_train, 0, y_train[0])
            
        print(f"Training Knots: {x_train}")
        
        # 2. Create Initial Guess (PCHIP on Training Data)
        # This creates a curve that ignores the existence of the 7% point entirely
        initial_pchip = PchipInterpolator(x_train, y_train)
        
        # 3. Setup Optimization Grid
        # We need a fine enough grid to enforce monotonicity, but aligned with knots
        opt_grid_x = list(x_train)
        # Add the hidden point to the grid so we explicitly optimize its value
        opt_grid_x.append(hide_det) 
        # Add tail points
        tail_points = np.arange(0.20, 1.01, 0.10)
        for p in tail_points:
            if p > max(opt_grid_x) + 0.01:
                opt_grid_x.append(p)
                
        opt_grid_x = np.array(sorted(list(set(opt_grid_x))))
        
        # Initial Y values from the PCHIP guess
        opt_grid_y_init = initial_pchip(opt_grid_x)
        
        # Identify indices of the TRAINING knots (we want to lock these in/penalize deviation)
        train_indices = [i for i, x in enumerate(opt_grid_x) if x in x_train]
        
        # 4. Define Objective Function
        def objective(y_new):
            # A. Stability: Stay close to the PCHIP shape (Regularization)
            penalty_prior = np.sum((y_new - opt_grid_y_init)**2)
            
            # B. Calibration: Match the KNOWN market points strictly
            # We do NOT penalize deviation at 'hide_det' because it's unknown to the model
            y_at_train = y_new[train_indices]
            penalty_mkt = np.sum((y_at_train - y_train)**2) * 10000.0 
            
            # C. Smoothness
            d2 = np.diff(y_new, 2)
            penalty_smooth = np.sum(d2**2) * 10.0
            
            return penalty_prior + penalty_mkt + penalty_smooth

        # 5. Define Constraints (Monotonic Expected Loss)
        def calculate_el_vector(y_current):
            els = []
            for k, rho in zip(opt_grid_x, y_current):
                if k == 0:
                    els.append(0.0)
                    continue
                
                # Price Protection Leg (PV of EL)
                tranche = {'Attachment':0.0, 'Detachment':k, 'Maturity':tenor, 'Spread_bps':0, 'Type':'Upfront'}
                rho = np.clip(rho, 0.001, 0.999)
                upfront = self.pricer.price_tranche(tranche, rho)
                els.append(upfront * k)
            return np.array(els)

        def constraint_monotonic_el(y_new):
            return np.diff(calculate_el_vector(y_new))

        # 6. Run Optimization
        print("Optimizing...")
        cons = ({'type': 'ineq', 'fun': constraint_monotonic_el})
        bounds = [(0.01, 0.999) for _ in opt_grid_x]
        
        res = minimize(objective, opt_grid_y_init, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'ftol': 1e-4, 'maxiter': 50})
        
        if res.success:
            print("Optimization Successful.")
        else:
            print(f"Optimization Failed: {res.message}")
            
        # 7. Create Final Curve
        final_curve = PchipInterpolator(opt_grid_x, res.x)
        
        # 8. Reprice the Hidden Tranche
        self.reprice_tranche(tenor, hide_det, final_curve)

    def reprice_tranche(self, tenor, det, curve_func):
        """
        Reprices the specific tranche associated with 'det' using the provided curve.
        """
        if det not in self.tranche_map:
            print(f"No standard tranche definition found for detachment {det}.")
            return

        info = self.tranche_map[det]
        att = info['Att']
        col_name = info['Col']
        
        # Get Correlations from the Optimized Curve
        # Note: We use the curve for BOTH points.
        # Even though we "know" the attachment point (if it was in training), 
        # using the optimized curve ensures internal consistency of the price.
        rho_att = float(curve_func(att))
        rho_det = float(curve_func(det))
        
        print(f"\n--- Pricing Check: {info['Name']} Tranche ---")
        print(f"Optimized Rho (Att {att:.0%}): {rho_att:.4f}")
        print(f"Optimized Rho (Det {det:.0%}): {rho_det:.4f}")
        
        # Pricing Logic
        # 1. Attachment PVs
        if att == 0.0:
            pv_prot_att = 0.0
            pv_prem_att = 0.0
        else:
            t_prot_a = {'Attachment':0.0, 'Detachment':att, 'Maturity':tenor, 'Spread_bps':0, 'Type':'Upfront'}
            pv_prot_att = self.pricer.price_tranche(t_prot_a, rho_att) * att
            
            t_prem_a = {'Attachment':0.0, 'Detachment':att, 'Maturity':tenor, 'Spread_bps':100, 'Type':'Upfront'}
            upfront_100_a = self.pricer.price_tranche(t_prem_a, rho_att)
            pv_prem_att = (pv_prot_att - upfront_100_a * att) / 0.01

        # 2. Detachment PVs
        t_prot_d = {'Attachment':0.0, 'Detachment':det, 'Maturity':tenor, 'Spread_bps':0, 'Type':'Upfront'}
        pv_prot_det = self.pricer.price_tranche(t_prot_d, rho_det) * det
        
        t_prem_d = {'Attachment':0.0, 'Detachment':det, 'Maturity':tenor, 'Spread_bps':100, 'Type':'Upfront'}
        upfront_100_d = self.pricer.price_tranche(t_prem_d, rho_det)
        pv_prem_det = (pv_prot_det - upfront_100_d * det) / 0.01
        
        # 3. Tranche PV
        leg_prot = pv_prot_det - pv_prot_att
        leg_prem = pv_prem_det - pv_prem_att
        width = det - att
        
        # Calculate Model Price (Spread)
        # Spread = Prot / Prem * 10000
        if leg_prem < 1e-9:
            model_price = 0.0
        else:
            model_price = (leg_prot / leg_prem) * 10000
            
        # Get Market Price
        mkt_row = self.market_df[self.market_df['Tenor'] == f'{int(tenor)}Y']
        if mkt_row.empty:
            print("Market data not found.")
            return
            
        mkt_price = mkt_row.iloc[0][col_name]
        
        print("-" * 40)
        print(f"{'Metric':<15} | {'Value':<10}")
        print("-" * 40)
        print(f"{'Market Quote':<15} | {mkt_price:<10.2f} bps")
        print(f"{'Arb-Free Model':<15} | {model_price:<10.2f} bps")
        print(f"{'Difference':<15} | {model_price - mkt_price:<10.2f} bps")
        print("-" * 40)

if __name__ == "__main__":
    validator = ArbFreeLOOCV()
    # Run the test: Hide 7% (The 3-7% Tranche)
    validator.run_loocv_optimization(tenor=5.0, hide_det=0.07)