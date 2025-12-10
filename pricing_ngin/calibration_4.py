import pandas as pd
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Import the Pricer class from the previous file
# Ensure gaussian_copula_pricing.py is in the same folder

from pricing_ngin.gaussian_3_copula_pricing import GaussianCopulaPricer

def calibrate_base_correlations(input_dir, output_dir):
    print(f"Running Calibration: {input_dir} -> {output_dir}")
    
    cdx_df = pd.read_csv(os.path.join(input_dir, 'CDX.NA.IG.45.csv'))
    outdata_dir = os.path.join(output_dir, 'outdata')
    res_dir = os.path.join(output_dir, 'res')
    os.makedirs(res_dir, exist_ok=True)
    
    # Init Pricer with the new curves
    pricer = GaussianCopulaPricer(outdata_dir, 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
    
    standard_tranches = [
        (0.00, 0.03, 'Upfront', '0-3 upfront', 500),
        (0.03, 0.07, 'Spread', '3-7', 100),
        (0.07, 0.10, 'Spread', '7-10', 100),
        (0.10, 0.15, 'Spread', '10-15', 100),
        (0.15, 1.00, 'Spread', '15-100', 100)
    ]
    
    results = []
    for t_str in ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y']:
        row = cdx_df[cdx_df['Tenor'] == t_str]
        if row.empty: continue
        row = row.iloc[0]
        T = float(t_str.replace('Y', ''))
        
        # Cache {Detachment: {rho, pv_prot, pv01}}
        base_cache = {0.0: {'pv_prot': 0.0, 'pv01': 0.0}}
        prev_det = 0.0
        
        for _, det, qtype, col, running in standard_tranches:
            val = row[col]
            if pd.isna(val): break
            
            # Objective: Match (PV_Prot_D - PV_Prot_A) - MktQuote * (PV01_D - PV01_A) = 0
            # Where PV_Prot_A and PV01_A are fixed from previous step
            
            def solve_rho(rho):
                # 1. Calculate Base Tranche [0, det] legs
                t0 = {'Attachment':0.0, 'Detachment':det, 'Maturity':T, 'Spread_bps':0, 'Type':'Upfront'}
                pv_prot_D = pricer.price_tranche(t0, rho) * det
                
                t1 = {'Attachment':0.0, 'Detachment':det, 'Maturity':T, 'Spread_bps':100, 'Type':'Upfront'}
                # PV01 = (Prot - Upfront_at_100)/0.01 (Simplified backout)
                # Actually simpler: price_tranche internal logic computes PV01. 
                # Ideally expose get_legs, but sticking to existing public API:
                upfront_with_prem = pricer.price_tranche(t1, rho)
                pv01_D = (pv_prot_D - upfront_with_prem * det) / 0.01
                
                # 2. Get Fixed [0, prev_det] legs
                pv_prot_A = base_cache[prev_det]['pv_prot']
                pv01_A = base_cache[prev_det]['pv01']
                
                # 3. Mismatch
                if qtype == 'Upfront':
                    return pv_prot_D - (running/10000)*pv01_D - (val/100)*det
                else:
                    return (pv_prot_D - pv_prot_A) - (val/10000)*(pv01_D - pv01_A)

            try: sol = brentq(solve_rho, 0.001, 0.999)
            except: sol = np.nan
            
            if not np.isnan(sol):
                # Recalculate legs to cache
                t0 = {'Attachment':0.0, 'Detachment':det, 'Maturity':T, 'Spread_bps':0, 'Type':'Upfront'}
                prot = pricer.price_tranche(t0, sol) * det
                t1 = {'Attachment':0.0, 'Detachment':det, 'Maturity':T, 'Spread_bps':100, 'Type':'Upfront'}
                prem = (prot - pricer.price_tranche(t1, sol)*det)/0.01
                
                base_cache[det] = {'pv_prot': prot, 'pv01': prem}
                results.append({'Tenor': T, 'Detachment': det, 'Correlation': sol})
                prev_det = det
            else:
                break
                
    pd.DataFrame(results).to_csv(os.path.join(res_dir, 'base_correlations.csv'), index=False)
    return pricer

if __name__ == "__main__":
    calibrate_base_correlations('indata/', 'outdata/')