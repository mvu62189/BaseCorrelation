# pricing_ngin/utility_wrappers.py

import os
import shutil
from pricing_ngin.gaussian_3_copula_pricing import GaussianCopulaPricer # Core Pricer Class
from scipy.interpolate import PchipInterpolator
import pandas as pd
import numpy as np

import importlib.util

def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

curves_mod = import_module_from_path('curves', 'pricing_ngin/curves_1.py')
basis_mod = import_module_from_path('basis', 'pricing_ngin/basis_adjustments_2.py')
calib_mod = import_module_from_path('calib', 'pricing_ngin/calibration_4.py')
interp_mod = import_module_from_path('interp', 'pricing_ngin/interp_5.py')
# -------------------------------

def run_full_calibration_chain(input_dir, output_dir):
    """
    Orchestrates the pipeline by calling the refactored functions.
    """
    # 1. Bootstrapping
    curves_mod.run_curve_bootstrapping(input_dir, output_dir)
    
    # 2. Basis Adjustment
    basis_mod.calculate_basis_adjustment(output_dir)
    
    # 3. Calibration (Returns pricer object)
    pricer = calib_mod.calibrate_base_correlations(input_dir, output_dir)
    
    # 4. Interpolation
    interp_mod.generate_continuous_smile(output_dir)
    
    return pricer

def apply_spread_bump(src_path, dst_path, bump_bps=1.0):
    df = pd.read_csv(src_path)
    # Identify spread columns (exclude Tenor, etc)
    # CDX.NA.IG columns usually: Bid, Ask, Last, 0-3, 0-3 upfront, etc.
    cols_to_bump = [c for c in df.columns if c not in ['Tenor', 'Last', '0-100', 'Time']]
    
    bump_dec = bump_bps # Assuming quotes are bps or handled consistently
    # NOTE: Check if Upfront is % or points. Usually points (0-100) or decimal.
    # If 500 bps = 5%, bumping by 1.0 might be huge if 1.0 = 100%. 
    # Standard convention: bump_bps=1 means add 1 to bps columns.
    
    for c in cols_to_bump:
        if df[c].dtype in [float, int]:
            df[c] += bump_dec
            
    df.to_csv(dst_path, index=False)

def load_interpolator(output_dir, tenor):
    res_path = os.path.join(output_dir, 'res', 'base_correlations.csv')
    df = pd.read_csv(res_path)
    subset = df[df['Tenor'] == tenor].sort_values('Detachment')
    
    x = subset['Detachment'].values
    y = subset['Correlation'].values
    
    if x[0] > 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0]) # Flat anchor
        
    return PchipInterpolator(x, y, extrapolate=True)