import pandas as pd
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the Pricer class from the previous file
# Ensure gaussian_copula_pricing.py is in the same folder

from gaussian_copula_pricing import GaussianCopulaPricer


def calibrate_base_correlations():
    print("Loading Market Data...")
    cdx_df = pd.read_csv('CDX.NA.IG.45.csv')
    
    # Initialize Pricer
    # We use the adjusted curves from Step 2
    pricer = GaussianCopulaPricer('discount_curve.csv', 'adjusted_constituent_survival_curves.csv')

    # Define Tranche Structure (Attachment, Detachment)
    # We map the CSV columns to these detachments
    # CSV Cols: 0-3 upfront, 3-7, 7-10, 10-15, 15-100
    
    base_tranches = [0.03, 0.07, 0.10, 0.15, 1.00]
    
    # Map for standard tranches to calculate Base Targets
    # (Att, Det, 'Type', Spread_Col_Name, Running_Bps)
    standard_structure = [
        (0.00, 0.03, 'Upfront', '0-3 upfront', 500),
        (0.03, 0.07, 'Spread', '3-7', 100),
        (0.07, 0.10, 'Spread', '7-10', 100),
        (0.10, 0.15, 'Spread', '10-15', 100),
        (0.15, 1.00, 'Spread', '15-100', 100)
    ]

    results = []

    # Loop over Tenors
    target_tenors = ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y'] 
    
    for t_str in target_tenors:
        print(f"\nProcessing Tenor: {t_str}...")
        
        row = cdx_df[cdx_df['Tenor'] == t_str]
        if row.empty:
            print(f"No data for {t_str}")
            continue
        row = row.iloc[0]
        
        # Parse Tenor to float
        T = float(t_str.replace('Y', ''))
        
        # Store Base Tranche PVs and Correlations
        # Key: Detachment Point -> {'rho': float, 'pv_prot': float, 'pv_prem_risky': float}
        base_cache = {0.0: {'rho': 0, 'pv_prot': 0.0, 'pv01': 0.0}}
        
        previous_detachment = 0.0
        
        for i, (att, det, q_type, col, running_bps) in enumerate(standard_structure):
            
            # Get Market Quote
            val = row[col]
            if pd.isna(val):
                print(f"  Missing quote for {col}. Skipping chain.")
                break
                
            # Base Correlation Logic:
            # We want to find rho_base (for 0 - det) such that:
            # Value_Model(0, det, rho_base) = Value_Model(0, att, rho_prev) + Value_Market(att, det)
            
            # 1. Calculate Value_Market(att, det)
            # We need to Value the standard tranche using the market quote.
            # Issue: To value the standard tranche (PV), we need a PV01 (Risky Duration).
            # Consistency Rule: Use the PV01 calculated using the CANDIDATE rho_base.
            # 
            # Equation:
            # PV_Prot(0, det, rho) = PV_Prot(0, att, rho_prev) + Market_Val_Standard
            # where Market_Val_Standard = Market_Quote_Terms
            
            # Let's decompose:
            # PV_Prot(0, det) - Running_Bps * PV01(0, det)  <-- This is NOT what we equate.
            # We equate the "Loss Leg" minus "Premium Leg" to the market value?
            
            # Simpler formulation:
            # PV_Prot(0, det, rho) = PV_Prot(0, att, rho_prev) + PV_Prot(att, det, rho??)
            # 
            # Standard Base Correlation Equation:
            # PV_Prot(0, D, rho) - S_mkt * PV01(A, D, rho) - (PV_Prot(0, A, rho_prev) - S_mkt * PV01(0, A, rho_prev)) = 0
            # (Assuming S_mkt is the fair spread of A-D)
            
            # If Quote is Upfront (Equity 0-3):
            # PV_Prot(0, 3, rho) - (Upfront + 500bps * PV01(0, 3, rho)) = 0
            
            prev_data = base_cache[previous_detachment]
            
            def objective(rho):
                # Current Base Tranche (0 to det)
                tranche_base = {
                    'Attachment': 0.0, 'Detachment': det, 'Maturity': T,
                    'Spread_bps': running_bps, 'Type': 'Spread' # Type doesn't matter for raw legs
                }
                
                # We manually call internal pricing parts to get raw PVs to build equation
                # But our pricer returns "Price". We need helper or just reverse calculate.
                # Let's use the pricer's price_tranche to get Fair Spread or Upfront, then compare?
                # No, simpler to access components. But price_tranche returns final number.
                # Let's Modify logic:
                #   Price_Model(0, D, rho) is implied by legs.
                
                # Let's use a Hack: Price the tranche (0, det) with rho.
                # Get PV_Prot(0, D) and PV01(0, D).
                # Since pricer returns "Spread" or "Upfront", we can back out PVs if we knew which one.
                
                # Better: Just implement the calculation inside the objective using the pricer's internal methods.
                # But we can't access "self" easily outside.
                # Alternative: Use the public `price_tranche` but we need legs.
                #
                # Let's assume we can trust `price_tranche(..., type='Spread')` returns Par Spread (bps).
                # Par Spread S_model = PV_Prot / PV01.
                # So PV_Prot = S_model * PV01.
                # We still need PV01.
                
                # RE-IMPLEMENTING small PV helper here using the pricer instance?
                # It's cleaner to just add a method to Pricer, but I cannot edit the previous file in this turn.
                # I will rely on the `price_tranche` logic being robust and standard.
                
                # WORKAROUND:
                # We use the equation:
                # Loss(0, Det) = Loss(0, Att) + Loss(Att, Det)
                # The Market implies a specific relationship for Loss(Att, Det).
                # Loss_Leg_Market = Premium_Leg_Market (at fair spread)
                # PV_Prot(A, D) = S_mkt * PV01(A, D, rho)
                #
                # So we find rho such that:
                # PV_Prot(0, D, rho) - PV_Prot(0, A, rho_prev) = S_mkt * [PV01(0, D, rho) - PV01(0, A, rho_prev)]
                #
                # Rearranging for Root Finding:
                # F(rho) = PV_Prot(0, D) - PV_Prot(0, A)_FIXED - S_mkt * (PV01(0, D) - PV01(0, A)_FIXED) = 0
                #
                # Note: PV01(0, A)_FIXED means we calculate the annuity of the *subordinate* tranche using the *current* rho?
                # NO. Base Correlation framework:
                # The terms for (0, A) are calculated using rho_prev (Fixed).
                # The terms for (0, D) are calculated using rho (Variable).
                # The PV01(A, D) is (PV01(0, D, rho) - PV01(0, A, rho_prev)).
                # This ensures the "Model Price" of the standard tranche matches market.
                
                # 1. Calculate (0, D) quantities with rho
                # To get raw PVs from `price_tranche`:
                #   Set spread = 0. Price 'Upfront'. 
                #   Upfront = PV_Prot / Notional. -> PV_Prot = Upfront * Notional
                #   Set spread = 10000. Price 'Upfront'.
                #   Upfront_2 = (PV_Prot - 1 * PV_Prem) / Notional.
                #   PV_Prem = PV_Prot - Upfront_2 * Notional.
                
                tranche_0_D = {'Attachment': 0.0, 'Detachment': det, 'Maturity': T, 'Spread_bps': 0, 'Type': 'Upfront'}
                
                # PV Protection (0, D)
                upfront_prot_only = pricer.price_tranche(tranche_0_D, rho)
                pv_prot_0_D = upfront_prot_only * det # Notional is (Det - 0) = Det
                
                # PV01 (0, D)
                # PV_Prem is for 1 bps? No, let's calculate PV01 directly.
                # PV_Riskless = ... hard to get.
                # Let's use a 100bps spread quote to back it out.
                tranche_0_D_100 = {'Attachment': 0.0, 'Detachment': det, 'Maturity': T, 'Spread_bps': 100, 'Type': 'Upfront'}
                upfront_with_prem = pricer.price_tranche(tranche_0_D_100, rho)
                # Upfront = (Prot - 0.01 * PV01) / Notional
                # 0.01 * PV01 = Prot - Upfront * Notional
                pv01_0_D = (pv_prot_0_D - upfront_with_prem * det) / 0.01
                
                # 2. Get Fixed (0, A) quantities (calculated with rho_prev)
                pv_prot_0_A = prev_data['pv_prot']
                pv01_0_A = prev_data['pv01']
                
                # 3. Calculate Mismatch
                if q_type == 'Upfront':
                    # Equity Tranche Case (Att=0)
                    # Market Upfront Quote U is given. Running is 500.
                    # Mkt Value = 0 (by definition of fair quote) -> PV_Prot - PV_Prem - Upfront_Cash = 0
                    # PV_Prot(0, D) - 0.05 * PV01(0, D) - Quote_U * D = 0
                    # (Note: Quote in CSV is percentage points, e.g., 30.0 for 30%. Need /100)
                    
                    quote_dec = val / 100.0
                    mismatch = pv_prot_0_D - (running_bps/10000.0) * pv01_0_D - quote_dec * det
                    
                else:
                    # Spread Tranche Case (Att > 0)
                    # Quote is S_mkt (bps).
                    # PV_Prot(A, D) = S_mkt * PV01(A, D)
                    # (Prot_D - Prot_A) = S_mkt * (PV01_D - PV01_A)
                    
                    s_mkt_dec = val / 10000.0
                    
                    pv_prot_A_D = pv_prot_0_D - pv_prot_0_A
                    pv01_A_D = pv01_0_D - pv01_0_A
                    
                    mismatch = pv_prot_A_D - s_mkt_dec * pv01_A_D

                return mismatch

            # Solve
            try:
                # Bounds for correlation
                sol = brentq(objective, 0.001, 0.999, xtol=1e-4)
            except Exception as e:
                print(f"  Calibration failed for {att}-{det}%: {e}")
                sol = np.nan
            
            print(f"  {att*100:g}-{det*100:g}% | Quote: {val} | Implied Rho: {sol:.4f}")
            
            # Store results for next step
            # We must re-calculate the (0, D) leg values using the FOUND solution to freeze them
            if not np.isnan(sol):
                # Recalculate exactly as in objective
                tranche_0_D = {'Attachment': 0.0, 'Detachment': det, 'Maturity': T, 'Spread_bps': 0, 'Type': 'Upfront'}
                upfront_prot = pricer.price_tranche(tranche_0_D, sol)
                prot_val = upfront_prot * det
                
                tranche_0_D_100 = {'Attachment': 0.0, 'Detachment': det, 'Maturity': T, 'Spread_bps': 100, 'Type': 'Upfront'}
                upfront_prem = pricer.price_tranche(tranche_0_D_100, sol)
                pv01_val = (prot_val - upfront_prem * det) / 0.01
                
                base_cache[det] = {'rho': sol, 'pv_prot': prot_val, 'pv01': pv01_val}
                results.append({'Tenor': T, 'Detachment': det, 'Correlation': sol})
                
                previous_detachment = det
            else:
                # If failed, cannot proceed down the structure reliably
                break

    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv('base_correlations.csv', index=False)
    print("\nSaved 'base_correlations.csv'")
    
    # Plot Surface
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare Grid
    pivot = res_df.pivot(index='Tenor', columns='Detachment', values='Correlation')
    X_tenors = pivot.index.values
    Y_detach = pivot.columns.values
    X, Y = np.meshgrid(X_tenors, Y_detach * 100) # Detachment in %
    Z = pivot.values.T # Transpose to match meshgrid
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Gaussian Base Correlation Surface')
    ax.set_xlabel('Tenor (Years)')
    ax.set_ylabel('Detachment (%)')
    ax.set_zlabel('Correlation')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig('base_correlation_surface.png')
    print("Saved 'base_correlation_surface.png'")
    plt.show()

if __name__ == "__main__":
    calibrate_base_correlations()