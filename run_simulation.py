# run_simulation.py
import pandas as pd
import os
import shutil

from pricing_ngin.delta_hedge import DeltaDV01Calculator

# --- CONFIGURATION ---
SIMULATION_DATES = [
    '1119', '1120', '1121',
    '1125', '1126', '1201', '1203'
]

# BESPOKE TRANCHE DEFINITION (5Y, 4-6%)
TR_TENOR = 5.0 
TR_ATTACH = 0.03  
TR_DETACH = 0.07  
TR_NAME = f"{TR_ATTACH*100:.1f}-{TR_DETACH*100:.1f}%"
TR_NOTIONAL = 10_000_000 

SMILE_METHOD = 'ARB_FREE' # Options: 'ARB_FREE', 'CONSTRAINED', 'UNCONSTRAINED'

# Directory setup
RAW_INPUT_ROOT = './historical_data/'
DAILY_CACHE_ROOT = './daily_output_cache/'
OUTPUT_FILE = f'simulation_out/delta_hedge_results_{TR_NAME}.csv'

def run_dynamic_hedge_simulation():
    # Setup folders
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(DAILY_CACHE_ROOT, exist_ok=True)
    
    dv01_calculator = DeltaDV01Calculator(temp_base_dir=DAILY_CACHE_ROOT)
    results = []
    
    prev_tranche_pv = None
    prev_index_pv = None
    prev_hedge_notional = 0.0
    cumulative_pnl = 0.0

    print(f"Starting Rigorous Delta Hedge Simulation for {TR_NAME} using {SMILE_METHOD} smile...")

    for i, date in enumerate(SIMULATION_DATES):
        print(f"Processing Day {i+1}: {date}...")
        raw_input_path = os.path.join(RAW_INPUT_ROOT, date)
        
        # --- 1. CALCULATE DV01 & BASE PV ---
        metrics = dv01_calculator.calculate_full_dv01(
            date, TR_TENOR, TR_ATTACH, TR_DETACH, raw_input_path,
            smile_method=SMILE_METHOD
        )
        
        # --- 2. HEDGE RATIO ---
        current_hedge_ratio = metrics['tranche_dv01'] / metrics['index_dv01'] if abs(metrics['index_dv01']) > 1e-9 else 0.0
        
        # Target Index Notional = -HR * Tranche Notional (Shorting the hedge index protection)
        current_hedge_notional = -current_hedge_ratio * TR_NOTIONAL 
        
        # --- 3. DYNAMIC PNL ---
        daily_pnl = 0.0
        
        if i > 0:
            # Tranche PnL: (PV_Today - PV_Yesterday) * -Tranche_Notional (Sold Protection)
            tranche_pnl = (metrics['base_tranche_pv'] - prev_tranche_pv) * -TR_NOTIONAL
            
            # Hedge PnL: (PV_Today - PV_Yesterday) * Hedge_Notional_Yesterday (Held Overnight)
            hedge_pnl = (metrics['base_index_pv'] - prev_index_pv) * prev_hedge_notional
            
            daily_pnl = tranche_pnl + hedge_pnl
            cumulative_pnl += daily_pnl
        
        # --- 4. STORE RESULTS & REBALANCE ---
        results.append({
            'date': date,
            'tranche_pv': metrics['base_tranche_pv'],
            'index_pv': metrics['base_index_pv'],
            'tranche_dv01': metrics['tranche_dv01'],
            'index_dv01': metrics['index_dv01'],
            'hedge_ratio': current_hedge_ratio,
            'hedge_notional': current_hedge_notional,
            'daily_pnl': daily_pnl,
            'cumulative_pnl': cumulative_pnl
        })
        
        prev_tranche_pv = metrics['base_tranche_pv']
        prev_index_pv = metrics['base_index_pv']
        prev_hedge_notional = current_hedge_notional
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSimulation Complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_dynamic_hedge_simulation()