import pandas as pd
import numpy as np
import scipy.interpolate
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Load Data and Helper Functions
# ==========================================

def parse_tenor(tenor):
    """Converts tenor strings (e.g., '6M', '1Y') to years (float)."""
    if isinstance(tenor, str):
        if 'W' in tenor:
            return float(tenor.replace('W', '')) / 52.0
        elif 'M' in tenor:
            return float(tenor.replace('M', '')) / 12.0
        elif 'Y' in tenor:
            return float(tenor.replace('Y', ''))
    return float(tenor)

data_in = 'indata/'
data_out = 'outdata/'
plot = 'plot/'

# Load datasets
print("Loading data...")
ois_df = pd.read_csv(f'{data_in}OIS_CURVE.csv')
constituents_df = pd.read_csv(f'{data_in}125constituents.csv')
cdx_df = pd.read_csv(f'{data_in}CDX.NA.IG.45.csv')

# ==========================================
# 2. Build Discount Curve from OIS
# ==========================================

print("Building Discount Curve...")
ois_df['Time'] = ois_df['Tenor'].apply(parse_tenor)
ois_df['Rate'] = ois_df['OIS Curve Mid'] / 100.0  # Convert to decimal

# Linear interpolation of Zero Rates
rate_interp = scipy.interpolate.interp1d(ois_df['Time'], ois_df['Rate'], kind='linear', fill_value="extrapolate")

def get_discount_factor(t):
    """Returns DF(t) = exp(-r(t) * t)"""
    t = np.array(t)
    # Handle small t to avoid division by zero or similar issues if any, though linear interp is safe
    r = rate_interp(t)
    return np.exp(-r * t)

# Save Discount Curve for target tenors
target_tenors = [1, 2, 3, 5, 7, 10]
df_output = pd.DataFrame({'Tenor': target_tenors, 'DF': get_discount_factor(target_tenors)})
df_output.to_csv(os.path.join(data_out, 'discount_curve.csv'), index=False)
print("Saved discount_curve.csv")

# ==========================================
# 3. Bootstrapping Logic
# ==========================================

def bootstrap_curve(tenors, spreads_bps, recovery_rate, discount_func):
    """
    Bootstraps survival probabilities from spreads using a piecewise constant hazard rate model.
    
    Parameters:
    - tenors: list of maturities in years (sorted)
    - spreads_bps: list of spreads in basis points
    - recovery_rate: recovery rate (e.g., 0.4)
    - discount_func: function returning DF(t)
    
    Returns:
    - lambdas: calculated hazard rates for each segment
    - curve_points: list of (time, survival_probability) tuples
    """
    spreads = np.array(spreads_bps) / 10000.0 # Convert bps to decimal
    dt = 0.25 # Quarterly integration steps
    
    lambdas = []
    results_surv = [(0.0, 1.0)] # (Time, Probability)
    
    # Current state variables
    curr_t = 0.0
    curr_s = 1.0
    
    for i, T in enumerate(tenors):
        S_market = spreads[i]
        seg_start = tenors[i-1] if i > 0 else 0.0
        seg_end = T
        
        # Objective function to find lambda for this segment
        def objective(lam):
            # Calculate PV of Protection and Premium
            # 1. From previous segments (already fixed)
            pv_prot = 0.0
            pv_prem = 0.0
            
            temp_t = 0.0
            temp_s = 1.0
            
            # Re-calculate PVs for known segments to get accumulated PV up to seg_start
            for j in range(i):
                prev_seg_end = tenors[j]
                prev_lam = lambdas[j]
                
                n_steps = int(np.ceil((prev_seg_end - temp_t) / dt))
                if n_steps == 0: n_steps = 1
                steps = np.linspace(temp_t, prev_seg_end, n_steps + 1)
                
                for k in range(len(steps)-1):
                    t0, t1 = steps[k], steps[k+1]
                    dt_step = t1 - t0
                    
                    s_next = temp_s * np.exp(-prev_lam * dt_step)
                    df_mid = discount_func((t0 + t1)/2)
                    
                    # PV Protection: (1-R) * (S_prev - S_next) * DF
                    pv_prot += (1 - recovery_rate) * (temp_s - s_next) * df_mid
                    # PV Premium: Spread * Avg_S * dt * DF
                    pv_prem += S_market * ((temp_s + s_next)/2) * dt_step * df_mid
                    
                    temp_s = s_next
                    temp_t = t1
            
            # 2. Current segment (solving for lam)
            n_steps = int(np.ceil((seg_end - seg_start) / dt))
            if n_steps == 0: n_steps = 1
            steps = np.linspace(seg_start, seg_end, n_steps + 1)
            
            for k in range(len(steps)-1):
                t0, t1 = steps[k], steps[k+1]
                dt_step = t1 - t0
                
                s_next = temp_s * np.exp(-lam * dt_step)
                df_mid = discount_func((t0 + t1)/2)
                
                pv_prot += (1 - recovery_rate) * (temp_s - s_next) * df_mid
                pv_prem += S_market * ((temp_s + s_next)/2) * dt_step * df_mid
                
                temp_s = s_next
            
            return pv_prot - pv_prem

        try:
            # Solve for lambda, bounded reasonably
            lam_sol = brentq(objective, -0.1, 5.0)
        except Exception as e:
            lam_sol = 0.0 # Default fallback
            
        lambdas.append(lam_sol)
        
        # Update survival probability for the end of this segment
        surv_end = results_surv[-1][1] * np.exp(-lam_sol * (T - seg_start))
        results_surv.append((T, surv_end))
        
    return lambdas, results_surv

def get_survival_prob(t_req, curve_points):
    """Interpolates survival probability for specific times using log-linear interpolation."""
    times = [x[0] for x in curve_points]
    probs = [x[1] for x in curve_points]
    
    # Avoid log(0)
    probs = np.maximum(probs, 1e-9)
    log_probs = np.log(probs)
    
    interp = scipy.interpolate.interp1d(times, log_probs, kind='linear', fill_value="extrapolate")
    
    t_req = np.array(t_req)
    return np.exp(interp(t_req))

# ==========================================
# 4. Process Index Curve
# ==========================================

print("Processing Index Curve...")
cdx_tenors_map = {'1Y':1, '2Y':2, '3Y':3, '5Y':5, '7Y':7, '10Y':10}
cdx_data = []

for i, row in cdx_df.iterrows():
    t_label = row['Tenor']
    if t_label in cdx_tenors_map:
        t_val = cdx_tenors_map[t_label]
        mid_spread = (row['Bid'] + row['Ask']) / 2
        cdx_data.append((t_val, mid_spread))

cdx_data.sort(key=lambda x: x[0])
index_tenors = [x[0] for x in cdx_data]
index_spreads = [x[1] for x in cdx_data]

index_lambdas, index_curve_points = bootstrap_curve(index_tenors, index_spreads, 0.4, get_discount_factor)
index_surv_probs = get_survival_prob(target_tenors, index_curve_points)

pd.DataFrame({'Tenor': target_tenors, 'Survival': index_surv_probs}).to_csv(os.path.join(data_out, 'index_survival_curve.csv'), index=False)
print("Saved index_survival_curve.csv")

# ==========================================
# 5. Process Constituent Curves
# ==========================================

print("Processing Constituent Curves...")
constituent_tenors_map = {
    '6 Mo': 0.5, '1 Yr': 1.0, '2 Yr': 2.0, '3 Yr': 3.0, 
    '4 Yr': 4.0, '5 Yr': 5.0, '7 Yr': 7.0, '10 Yr': 10.0
}
# Ensure input tenors are sorted
sorted_c_tenors = sorted(constituent_tenors_map.items(), key=lambda x: x[1])
c_tenors_labels = [x[0] for x in sorted_c_tenors]
c_tenors_vals = [x[1] for x in sorted_c_tenors]

results = []

for idx, row in constituents_df.iterrows():
    spreads = [row[label] for label in c_tenors_labels]
    rec_rate = row['Recovery rate']
    
    try:
        lambdas, curve_points = bootstrap_curve(c_tenors_vals, spreads, rec_rate, get_discount_factor)
        surv_probs = get_survival_prob(target_tenors, curve_points)
        
        res_dict = {'Company': row['Company'], 'Recovery': rec_rate}
        for i, t in enumerate(target_tenors):
            res_dict[f'S_{t}Y'] = surv_probs[i]
        results.append(res_dict)
    except Exception as e:
        print(f"Error bootstrapping {row['Company']}: {e}")

const_surv_df = pd.DataFrame(results)
const_surv_df.to_csv(os.path.join(data_out, 'constituent_survival_curves.csv'), index=False)
print("Saved constituent_survival_curves.csv")

# ==========================================
# 6. Calculate Losses and Plot
# ==========================================

print("Calculating Losses and Plotting...")
# Average Constituent Loss
loss_data = []
for t_str in ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y']:
    t = int(t_str.replace('Y',''))
    col = f'S_{t}Y'
    # Loss = (1 - Survival) * (1 - Recovery)
    losses = (1 - const_surv_df[col]) * (1 - const_surv_df['Recovery'])
    loss_data.append(losses.mean())

# Index Loss
index_loss = (1 - index_surv_probs) * (1 - 0.4) # Index Recovery fixed at 0.4

# Save Loss Data
pd.DataFrame({
    'Tenor': target_tenors,
    'Index_Loss': index_loss,
    'Avg_Constituent_Loss': loss_data
}).to_csv(os.path.join(data_out,'loss_curves.csv'), index=False)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(target_tenors, index_loss, marker='o', label='Index Expected Loss')
plt.plot(target_tenors, loss_data, marker='x', linestyle='--', label='Avg Constituent Expected Loss')
plt.title('Expected Loss Curves: Index vs. Constituent Average')
plt.xlabel('Tenor (Years)')
plt.ylabel('Expected Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot,'loss_comparison_plot.png'))
plt.show()
print("Process Complete. All files saved.")