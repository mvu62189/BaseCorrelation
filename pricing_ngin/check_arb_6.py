import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gaussian_3_copula_pricing import GaussianCopulaPricer

def check_arbitrage():
    # Load calibrated correlations
    base_corr_df = pd.read_csv('res/base_correlations.csv')
    
    # Initialize Pricer
    pricer = GaussianCopulaPricer('outdata/', 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
    
    # Store Expected Losses
    # Structure: {Tenor: {Detachment: EL}}
    el_results = {}
    
    print("Calculating Expected Losses (EL) for Base Tranches...")
    
    for tenor in base_corr_df['Tenor'].unique():
        el_results[tenor] = {}
        subset = base_corr_df[base_corr_df['Tenor'] == tenor].sort_values('Detachment')
        
        # Add 0% detachment (EL=0) for completeness
        el_results[tenor][0.0] = 0.0
        
        for _, row in subset.iterrows():
            det = row['Detachment']
            rho = row['Correlation']
            
            # Setup Tranche (0 - Detachment)
            tranche = {
                'Attachment': 0.0,
                'Detachment': det,
                'Maturity': tenor,
                'Spread_bps': 0, # irrelevant for EL calc
                'Type': 'Upfront'
            }
            
            # Hack: Get Expected Loss directly.
            # In your pricer, 'price_tranche' returns a price. 
            # We can calculate EL by asking for the 'Protection PV' (Upfront * Notional)
            # assuming 0 coupon.
            # PV_Prot = EL_discounted. 
            # But strictly, Base Correlation monotonicity applies to the raw Expected Loss or PV(Protection).
            # Let's use PV(Protection) as the proxy for arbitrage check.
            
            upfront = pricer.price_tranche(tranche, rho)
            # Upfront = PV_Prot / Notional -> PV_Prot = Upfront * Notional
            pv_protection = upfront * det
            
            el_results[tenor][det] = pv_protection

    # --- Check 1: Tranche Monotonicity (Strike Arbitrage) ---
    print("\n--- Tranche Arbitrage Check (EL vs Detachment) ---")
    for tenor, data in el_results.items():
        detachments = sorted(data.keys())
        els = [data[k] for k in detachments]
        
        # Check if strictly increasing
        diffs = np.diff(els)
        if np.all(diffs >= 0):
            print(f"Tenor {tenor}Y: PASS (Monotonic)")
        else:
            print(f"Tenor {tenor}Y: FAIL (Non-monotonic at indices {np.where(diffs < 0)[0]})")

    # --- Check 2: Calendar Monotonicity (Time Arbitrage) ---
    print("\n--- Calendar Arbitrage Check (EL vs Tenor) ---")
    # Get common detachments
    common_dets = [0.03, 0.07, 0.1, 0.15]
    sorted_tenors = sorted(el_results.keys())
    
    for det in common_dets:
        els = []
        valid_tenors = []
        for t in sorted_tenors:
            if det in el_results[t]:
                els.append(el_results[t][det])
                valid_tenors.append(t)
        
        diffs = np.diff(els)
        if np.all(diffs >= 0):
            print(f"Detachment {det*100}%: PASS (Monotonic over time)")
        else:
            print(f"Detachment {det*100}%: FAIL")
            
    # Plotting
    plt.figure(figsize=(10,6))
    for t in sorted_tenors:
        d = sorted(el_results[t].keys())
        e = [el_results[t][k] for k in d]
        plt.plot(d, e, marker='o', label=f'{t}Y')
    plt.xlabel('Detachment')
    plt.ylabel('PV Expected Loss')
    plt.title('Base Expected Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('res/arbitrage_check.png')
    print("Saved plot to res/arbitrage_check.png")

if __name__ == "__main__":
    check_arbitrage()