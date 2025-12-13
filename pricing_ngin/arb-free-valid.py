import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
import os
from gaussian_3_copula_pricing import GaussianCopulaPricer

class ArbFreeValidator:
    def __init__(self, res_dir='res/', data_dir='indata/', out_dir='outdata/'):
        self.res_dir = res_dir
        self.base_df = pd.read_csv(os.path.join(res_dir, 'base_correlations.csv'))
        self.market_df = pd.read_csv(os.path.join(data_dir, 'CDX.NA.IG.45.csv'))
        self.pricer = GaussianCopulaPricer(out_dir, 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
        
        # Standard Tranches for 5Y
        self.tranche_map = [
            (0.00, 0.03, 'Upfront', '0-3 upfront', 500),
            (0.03, 0.07, 'Spread', '3-7', 100),
            (0.07, 0.10, 'Spread', '7-10', 100),
            (0.10, 0.15, 'Spread', '10-15', 100)
        ]

    def get_optimized_curve(self, tenor=5.0):
        """Runs the Arb-Free Optimization routine and returns the fitted interpolator."""
        print(f"Optimizing 5Y Curve (Arb-Free)...")
        
        # 1. Prepare Data
        subset = self.base_df[self.base_df['Tenor'] == tenor].sort_values('Detachment')
        x_mkt = subset['Detachment'].values
        y_mkt = subset['Correlation'].values
        
        # Add anchor at 0%
        if x_mkt[0] > 0.0:
            x_mkt = np.insert(x_mkt, 0, 0.0)
            y_mkt = np.insert(y_mkt, 0, y_mkt[0])
            
        # Define Grid (Mkt + Tail)
        opt_grid_x = list(x_mkt)
        tail_points = np.arange(0.20, 1.01, 0.10)
        for p in tail_points:
            if p > opt_grid_x[-1] + 0.01:
                opt_grid_x.append(p)
        opt_grid_x = np.array(sorted(opt_grid_x))
        
        # Initial Guess
        initial_pchip = PchipInterpolator(x_mkt, y_mkt)
        opt_grid_y_init = initial_pchip(opt_grid_x)
        
        # Indices of market points in the grid
        market_indices = [i for i, x in enumerate(opt_grid_x) if x in x_mkt]

        # 2. Objective Function
        def objective(y_new):
            penalty_prior = np.sum((y_new - opt_grid_y_init)**2)
            y_at_mkt = y_new[market_indices]
            penalty_mkt = np.sum((y_at_mkt - y_mkt)**2) * 1000.0 
            d2 = np.diff(y_new, 2)
            penalty_smooth = np.sum(d2**2) * 10.0
            return penalty_prior + penalty_mkt + penalty_smooth

        # 3. Constraints
        def calculate_el_vector(y_current):
            els = []
            for k, rho in zip(opt_grid_x, y_current):
                if k == 0:
                    els.append(0.0)
                    continue
                tranche = {'Attachment':0.0, 'Detachment':k, 'Maturity':tenor, 'Spread_bps':0, 'Type':'Upfront'}
                rho = np.clip(rho, 0.001, 0.999)
                upfront = self.pricer.price_tranche(tranche, rho)
                els.append(upfront * k)
            return np.array(els)

        def constraint_monotonic_el(y_new):
            return np.diff(calculate_el_vector(y_new))

        # 4. Run Optimization
        cons = ({'type': 'ineq', 'fun': constraint_monotonic_el})
        bounds = [(0.01, 0.999) for _ in opt_grid_x]
        
        res = minimize(objective, opt_grid_y_init, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'ftol': 1e-4, 'maxiter': 50})
        
        if not res.success:
            print(f"Optimization Warning: {res.message}")
            
        return PchipInterpolator(opt_grid_x, res.x)

    def calculate_price(self, tenor, att, det, rho_att, rho_det, running_bps, quote_type):
        """Standard Tranche Pricing Helper"""
        # Attachment PVs
        if att == 0.0:
            pv_prot_att = 0.0
            pv_prem_att = 0.0
        else:
            t_prot_a = {'Attachment':0.0, 'Detachment':att, 'Maturity':tenor, 'Spread_bps':0, 'Type':'Upfront'}
            pv_prot_att = self.pricer.price_tranche(t_prot_a, rho_att) * att
            
            t_prem_a = {'Attachment':0.0, 'Detachment':att, 'Maturity':tenor, 'Spread_bps':100, 'Type':'Upfront'}
            upfront_100_a = self.pricer.price_tranche(t_prem_a, rho_att)
            pv_prem_att = (pv_prot_att - upfront_100_a * att) / 0.01

        # Detachment PVs
        t_prot_d = {'Attachment':0.0, 'Detachment':det, 'Maturity':tenor, 'Spread_bps':0, 'Type':'Upfront'}
        pv_prot_det = self.pricer.price_tranche(t_prot_d, rho_det) * det
        
        t_prem_d = {'Attachment':0.0, 'Detachment':det, 'Maturity':tenor, 'Spread_bps':100, 'Type':'Upfront'}
        upfront_100_d = self.pricer.price_tranche(t_prem_d, rho_det)
        pv_prem_det = (pv_prot_det - upfront_100_d * det) / 0.01
        
        # Combine
        leg_prot = pv_prot_det - pv_prot_att
        leg_prem = pv_prem_det - pv_prem_att
        width = det - att

        if quote_type == 'Upfront':
            return (leg_prot - (running_bps/10000)*leg_prem) / width * 100
        else:
            if leg_prem < 1e-9: return 0.0
            return (leg_prot / leg_prem) * 10000

    def run_check(self):
        tenor = 5.0
        # 1. Get the Optimized Curve
        smile_func = self.get_optimized_curve(tenor)
        
        # 2. Get Market Data
        mkt_row = self.market_df[self.market_df['Tenor'] == '5Y'].iloc[0]
        
        print(f"\n--- Repricing 5Y Tranches using Arb-Free Optimized Correlations ---")
        print(f"{'Tranche':<10} | {'Mkt Quote':<10} | {'Model Px':<10} | {'Diff':<10} | {'Rho Att':<7} {'Rho Det':<7}")
        
        for att, det, q_type, col, running in self.tranche_map:
            mkt_val = mkt_row[col]
            
            # Get Optimized Correlations
            rho_att = float(smile_func(att)) if att > 0 else 0.0
            rho_det = float(smile_func(det))
            
            # Calculate Price
            model_val = self.calculate_price(tenor, att, det, rho_att, rho_det, running, q_type)
            
            diff = model_val - mkt_val
            label = f"{att*100:g}-{det*100:g}%"
            
            print(f"{label:<10} | {mkt_val:<10.2f} | {model_val:<10.2f} | {diff:<10.2f} | {rho_att:.4f}  {rho_det:.4f}")

if __name__ == "__main__":
    validator = ArbFreeValidator()
    validator.run_check()