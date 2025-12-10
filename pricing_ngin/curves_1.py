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

def bootstrap_curve(tenors, spreads_bps, recovery_rate, discount_func):
    # ... [Keep existing bootstrap_curve logic unchanged] ...
    # (Copy the existing function body here)
    spreads = np.array(spreads_bps) / 10000.0
    dt = 0.25
    lambdas = []
    results_surv = [(0.0, 1.0)]
    
    for i, T in enumerate(tenors):
        S_market = spreads[i]
        seg_start = tenors[i-1] if i > 0 else 0.0
        curr_s = results_surv[-1][1]
        
        def objective(lam):
            pv_prot = 0.0
            pv_prem = 0.0
            temp_t, temp_s = 0.0, 1.0
            # Previous segments
            for j in range(i):
                prev_seg_end = tenors[j]
                prev_lam = lambdas[j]
                n_steps = int(np.ceil((prev_seg_end - temp_t) / dt)) or 1
                steps = np.linspace(temp_t, prev_seg_end, n_steps + 1)
                for k in range(len(steps)-1):
                    t0, t1 = steps[k], steps[k+1]
                    s_next = temp_s * np.exp(-prev_lam * (t1 - t0))
                    df_mid = discount_func((t0 + t1)/2)
                    pv_prot += (1 - recovery_rate) * (temp_s - s_next) * df_mid
                    pv_prem += S_market * ((temp_s + s_next)/2) * (t1 - t0) * df_mid
                    temp_s, temp_t = s_next, t1
            
            # Current segment
            n_steps = int(np.ceil((T - seg_start) / dt)) or 1
            steps = np.linspace(seg_start, T, n_steps + 1)
            temp_t, temp_s = seg_start, curr_s
            for k in range(len(steps)-1):
                t0, t1 = steps[k], steps[k+1]
                s_next = temp_s * np.exp(-lam * (t1 - t0))
                df_mid = discount_func((t0 + t1)/2)
                pv_prot += (1 - recovery_rate) * (temp_s - s_next) * df_mid
                pv_prem += S_market * ((temp_s + s_next)/2) * (t1 - t0) * df_mid
                temp_s = s_next
            return pv_prot - pv_prem

        try: lam_sol = brentq(objective, -0.1, 5.0)
        except: lam_sol = 0.0
        lambdas.append(lam_sol)
        results_surv.append((T, results_surv[-1][1] * np.exp(-lam_sol * (T - seg_start))))
    return lambdas, results_surv

def get_survival_prob(t_req, curve_points):
    # ... [Keep existing get_survival_prob logic unchanged] ...
    times = [x[0] for x in curve_points]
    probs = [x[1] for x in curve_points]
    probs = np.maximum(probs, 1e-9)
    interp = scipy.interpolate.interp1d(times, np.log(probs), kind='linear', fill_value="extrapolate")
    return np.exp(interp(t_req))

# --- MAIN EXECUTION WRAPPER ---
def run_curve_bootstrapping(input_dir, output_dir):
    print(f"Running Curve Bootstrapping: {input_dir} -> {output_dir}")
    
    # 1. Load Data
    ois_df = pd.read_csv(os.path.join(input_dir, 'OIS_CURVE.csv'))
    constituents_df = pd.read_csv(os.path.join(input_dir, '125constituents.csv'))
    cdx_df = pd.read_csv(os.path.join(input_dir, 'CDX.NA.IG.45.csv'))
    
    outdata_path = os.path.join(output_dir, 'outdata')
    os.makedirs(outdata_path, exist_ok=True)

    # 2. Build Discount Curve
    ois_df['Time'] = ois_df['Tenor'].apply(parse_tenor)
    ois_df['Rate'] = ois_df['OIS Curve Mid'] / 100.0
    rate_interp = scipy.interpolate.interp1d(ois_df['Time'], ois_df['Rate'], kind='linear', fill_value="extrapolate")
    
    def get_discount_factor_local(t):
        return np.exp(-rate_interp(t) * t)

    target_tenors = [1, 2, 3, 5, 7, 10]
    pd.DataFrame({'Tenor': target_tenors, 'DF': get_discount_factor_local(target_tenors)}).to_csv(
        os.path.join(outdata_path, 'discount_curve.csv'), index=False
    )

    # 3. Index Curve
    cdx_tenors_map = {'1Y':1, '2Y':2, '3Y':3, '5Y':5, '7Y':7, '10Y':10}
    cdx_data = []
    for _, row in cdx_df.iterrows():
        if row['Tenor'] in cdx_tenors_map:
            cdx_data.append((cdx_tenors_map[row['Tenor']], (row['Bid'] + row['Ask']) / 2))
    cdx_data.sort(key=lambda x: x[0])
    
    _, idx_pts = bootstrap_curve([x[0] for x in cdx_data], [x[1] for x in cdx_data], 0.4, get_discount_factor_local)
    idx_surv = get_survival_prob(target_tenors, idx_pts)
    pd.DataFrame({'Tenor': target_tenors, 'Survival': idx_surv}).to_csv(
        os.path.join(outdata_path, 'index_survival_curve.csv'), index=False
    )

    # 4. Constituent Curves
    c_map = {'6 Mo': 0.5, '1 Yr': 1.0, '2 Yr': 2.0, '3 Yr': 3.0, '4 Yr': 4.0, '5 Yr': 5.0, '7 Yr': 7.0, '10 Yr': 10.0}
    c_sorted = sorted(c_map.items(), key=lambda x: x[1])
    c_labels, c_vals = [x[0] for x in c_sorted], [x[1] for x in c_sorted]
    
    results = []
    for _, row in constituents_df.iterrows():
        try:
            _, pts = bootstrap_curve(c_vals, [row[l] for l in c_labels], row['Recovery rate'], get_discount_factor_local)
            probs = get_survival_prob(target_tenors, pts)
            res = {'Company': row['Company'], 'Recovery': row['Recovery rate']}
            for i, t in enumerate(target_tenors): res[f'S_{t}Y'] = probs[i]
            results.append(res)
        except: pass
        
    const_df = pd.DataFrame(results)
    const_df.to_csv(os.path.join(outdata_path, 'constituent_survival_curves.csv'), index=False)

    # 5. Loss Curves
    loss_data = []
    for t in target_tenors:
        loss = (1 - const_df[f'S_{t}Y']) * (1 - const_df['Recovery'])
        loss_data.append(loss.mean())
    
    pd.DataFrame({'Tenor': target_tenors, 'Index_Loss': (1-idx_surv)*(1-0.4), 'Avg_Constituent_Loss': loss_data}).to_csv(
        os.path.join(outdata_path, 'loss_curves.csv'), index=False
    )

if __name__ == "__main__":
    # Backward compatibility
    run_curve_bootstrapping('indata/', 'outdata/')