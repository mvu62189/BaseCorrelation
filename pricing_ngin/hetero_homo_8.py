import pandas as pd
import numpy as np
import os
from gaussian_3_copula_pricing import GaussianCopulaPricer

def create_homogeneous_data(outdir):
    """Creates a dummy survival file where every company is the average of the portfolio."""
    df = pd.read_csv(os.path.join(outdir, 'adjusted_constituent_survival_curves.csv'))
    
    # Calculate average survival probabilities for each tenor column
    numeric_cols = [c for c in df.columns if c.startswith('S_')]
    avg_curve = df[numeric_cols].mean(axis=0)
    
    # Create homogeneous dataframe
    hom_df = df.copy()
    for col in numeric_cols:
        hom_df[col] = avg_curve[col]
        
    hom_df.to_csv(os.path.join(outdir, 'homogeneous_survival_curves.csv'), index=False)
    print("Created homogeneous_survival_curves.csv")

def compare_models():
    outdir = 'outdata/'
    # 1. Create Homogeneous Input
    create_homogeneous_data(outdir)
    
    # 2. Initialize Two Pricers
    pricer_het = GaussianCopulaPricer(outdir, 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
    pricer_hom = GaussianCopulaPricer(outdir, 'discount_curve.csv', 'homogeneous_survival_curves.csv')
    
    # 3. Compare Expected Loss on Equity Tranche (0-3%)
    # Using a fixed correlation (e.g., 0.4) to isolate the impact of curve dispersion
    test_rho = 0.4
    tranche = {'Attachment': 0.0, 'Detachment': 0.03, 'Maturity': 5.0, 'Spread_bps': 500, 'Type': 'Upfront'}
    
    print("\n--- Model Comparison (5Y Equity Tranche, Rho=0.4) ---")
    
    # Heterogeneous Price
    price_het = pricer_het.price_tranche(tranche, test_rho)
    print(f"Heterogeneous Upfront: {price_het*100:.4f}%")
    
    # Homogeneous Price
    price_hom = pricer_hom.price_tranche(tranche, test_rho)
    print(f"Homogeneous Upfront:   {price_hom*100:.4f}%")
    
    diff = (price_hom - price_het) * 100
    print(f"Difference: {diff:.4f}%")

if __name__ == "__main__":
    compare_models()