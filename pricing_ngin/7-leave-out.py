import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt
import os
from gaussian_3_copula_pricing import GaussianCopulaPricer

class LOOCV_Validator:
    def __init__(self, res_dir='res/', data_dir='indata/', out_dir='outdata/'):
        print("Initializing LOOCV Validator...")
        self.res_dir = res_dir
        self.base_df = pd.read_csv(os.path.join(res_dir, 'base_correlations.csv'))
        self.market_df = pd.read_csv(os.path.join(data_dir, 'CDX.NA.IG.45.csv'))
        self.pricer = GaussianCopulaPricer(out_dir, 'discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
        
        # Standard Tranche Definitions for Pricing Checks
        # Map: Detachment -> (Attachment, Type, ColName, RunningBps)
        self.tranche_map = {
            0.03: (0.00, 'Upfront', '0-3 upfront', 500),
            0.07: (0.03, 'Spread', '3-7', 100),
            0.10: (0.07, 'Spread', '7-10', 100),
            0.15: (0.10, 'Spread', '10-15', 100)
        }
        
        # RESTRICTION: Only test these detachments for Point-LOOCV
        self.testable_detachments = [0.07, 0.10]

    # ----------------------------------------------------------------
    # INTERPOLATION HELPERS
    # ----------------------------------------------------------------
    def predict_rho_time(self, target_t, target_d, train_df):
        """Predicts correlation using Time Interpolation (Linear) from other tenors."""
        subset = train_df[train_df['Detachment'] == target_d].sort_values('Tenor')
        if len(subset) < 2: return np.nan
        f = interp1d(subset['Tenor'], subset['Correlation'], kind='linear', fill_value='extrapolate')
        return float(f(target_t))

    def predict_rho_smile(self, target_t, target_d, train_df):
        """
        Predicts correlation using Smile Interpolation (PCHIP) from other detachments.
        Strictly assumes interpolation (no extrapolation) for stability.
        """
        subset = train_df[train_df['Tenor'] == target_t].sort_values('Detachment')
        if len(subset) < 2: return np.nan
        
        x = subset['Detachment'].values
        y = subset['Correlation'].values
        
        # Add Anchor at 0%
        if x[0] > 0.0:
            x = np.insert(x, 0, 0.0)
            y = np.insert(y, 0, y[0])
            
        try:
            f = PchipInterpolator(x, y, extrapolate=False)
            return float(f(target_d))
        except:
            return np.nan

    # ----------------------------------------------------------------
    # TEST 1: Hide Entire Tenor (Time LOOCV) - HONEST VERSION
    # ----------------------------------------------------------------
    def run_tenor_loocv(self, hidden_tenor=7.0):
        print(f"\n--- Running Tenor LOOCV (Hiding {hidden_tenor}Y) ---")
        
        # 1. Train set: Everything except hidden_tenor
        train_df = self.base_df[self.base_df['Tenor'] != hidden_tenor]
        
        # 2. Predict Correlations for ALL detachments first
        # We need the full curve to price standard tranches correctly
        detachments = [0.03, 0.07, 0.10, 0.15]
        predicted_corrs = {}
        
        for det in detachments:
            pred_rho = self.predict_rho_time(hidden_tenor, det, train_df)
            predicted_corrs[det] = pred_rho
            # print(f"  Det {det:.0%}: Pred Rho = {pred_rho:.4f}")
            
        # 3. Calculate Base PVs (Protection and Premium) for 0-Det using Predicted Rho
        # Initialize with 0 for the start
        base_pvs = {0.0: {'prot': 0.0, 'prem': 0.0}}
        
        for det in detachments:
            rho = predicted_corrs[det]
            if np.isnan(rho):
                base_pvs[det] = {'prot': np.nan, 'prem': np.nan}
                continue

            # A. Protection PV (Using 0-coupon hack)
            t_prot = {'Attachment':0.0, 'Detachment':det, 'Maturity':hidden_tenor, 'Spread_bps':0, 'Type':'Upfront'}
            upfront_prot = self.pricer.price_tranche(t_prot, rho)
            pv_prot = upfront_prot * det
            
            # B. Premium PV (Annuity) - Using 100bps hack
            t_prem = {'Attachment':0.0, 'Detachment':det, 'Maturity':hidden_tenor, 'Spread_bps':100, 'Type':'Upfront'}
            upfront_100 = self.pricer.price_tranche(t_prem, rho)
            # Annuity = (PV_Prot - Upfront*Notional) / 0.01
            pv_ann = (pv_prot - upfront_100 * det) / 0.01
            
            base_pvs[det] = {'prot': pv_prot, 'prem': pv_ann}

        # 4. Price Standard Tranches and Compare
        print(f"{'Tranche':<10} | {'Mkt Price':<10} | {'Pred Price':<10} | {'Diff':<10}")
        
        mkt_row = self.market_df[self.market_df['Tenor'] == f'{int(hidden_tenor)}Y']
        if mkt_row.empty:
            print("No market data found for this tenor.")
            return []

        results = []
        
        # Loop through our known standard tranches
        for det, (att, q_type, col_name, running_bps) in self.tranche_map.items():
            
            # Get Market Price
            mkt_val = mkt_row.iloc[0][col_name]
            
            # Get Model Price components
            pv_prot_det = base_pvs[det]['prot']
            pv_prem_det = base_pvs[det]['prem']
            pv_prot_att = base_pvs[att]['prot']
            pv_prem_att = base_pvs[att]['prem']
            
            if np.isnan(pv_prot_det) or np.isnan(pv_prot_att):
                model_val = np.nan
            else:
                # Tranche PV = (Prot_D - Prot_A) - Coupon * (Prem_D - Prem_A)
                leg_prot = pv_prot_det - pv_prot_att
                leg_prem = pv_prem_det - pv_prem_att
                width = det - att
                
                if q_type == 'Upfront':
                    # Upfront %
                    model_val = (leg_prot - (running_bps/10000)*leg_prem) / width * 100
                else:
                    # Spread bps
                    if leg_prem < 1e-9: model_val = 0.0
                    else: model_val = (leg_prot / leg_prem) * 10000

            label = f"{att*100:g}-{det*100:g}%"
            diff = model_val - mkt_val if not np.isnan(model_val) else np.nan
            print(f"{label:<10} | {mkt_val:<10.2f} | {model_val:<10.2f} | {diff:<10.2f}")
            
            results.append({'Tranche': label, 'Price_Diff': diff})
            
        return results

    # ----------------------------------------------------------------
    # TEST 2: Hide Single Point (Restricted Point LOOCV)
    # ----------------------------------------------------------------
    def run_point_loocv(self):
        print(f"\n--- Running Restricted Point-Based LOOCV (Middle Tranches: {self.testable_detachments}) ---")
        
        results = []
        candidates = self.base_df[self.base_df['Detachment'].isin(self.testable_detachments)]
        
        for idx, row in candidates.iterrows():
            t = row['Tenor']
            d = row['Detachment']
            actual_rho = row['Correlation']
            
            # 1. Smile Prediction
            train_smile = self.base_df[(self.base_df['Tenor'] == t) & (self.base_df['Detachment'] != d)]
            pred_smile = self.predict_rho_smile(t, d, train_smile)
            
            # 2. Time Prediction
            train_time = self.base_df[(self.base_df['Detachment'] == d) & (self.base_df['Tenor'] != t)]
            pred_time = self.predict_rho_time(t, d, train_time)
            
            results.append({
                'Tenor': t, 'Detachment': d, 'Actual_Rho': actual_rho,
                'Smile_Pred': pred_smile, 'Time_Pred': pred_time,
                'Smile_Err': pred_smile - actual_rho if not np.isnan(pred_smile) else np.nan,
                'Time_Err': pred_time - actual_rho if not np.isnan(pred_time) else np.nan
            })
            
        res_df = pd.DataFrame(results)
        if res_df.empty:
            print("No testable points found.")
            return

        print(res_df[['Tenor', 'Detachment', 'Actual_Rho', 'Smile_Pred', 'Time_Pred']].round(4))
        
        mae_smile = res_df['Smile_Err'].abs().mean()
        mae_time = res_df['Time_Err'].abs().mean()
        print(f"\nMAE Smile Interp: {mae_smile:.4f}")
        print(f"MAE Time Interp:  {mae_time:.4f}")
        return res_df

    # ----------------------------------------------------------------
    # VISUALIZATION: Aggregated 6-Panel Plot
    # ----------------------------------------------------------------
    def plot_aggregated_loocv(self):
        print("\nGenerating Aggregated LOOCV Smile Comparison Plot...")
        
        target_tenors = [5.0, 7.0, 10.0]
        # Rows = Detachments (7%, 10%), Cols = Tenors (5Y, 7Y, 10Y)
        fig, axes = plt.subplots(len(self.testable_detachments), len(target_tenors), figsize=(18, 10))
        fig.suptitle('LOOCV: Calibrated vs. Interpolated Correlation Smiles', fontsize=16)
        
        x_fine = np.linspace(0.0, 0.20, 100)
        
        for col_idx, tenor in enumerate(target_tenors):
            for row_idx, hide_det in enumerate(self.testable_detachments):
                ax = axes[row_idx, col_idx]
                
                # 1. Full Data
                subset = self.base_df[self.base_df['Tenor'] == tenor].sort_values('Detachment')
                x_full = subset['Detachment'].values
                y_full = subset['Correlation'].values
                
                if len(x_full) > 0 and x_full[0] > 0.0:
                    x_full = np.insert(x_full, 0, 0.0)
                    y_full = np.insert(y_full, 0, y_full[0])
                    
                try:
                    f_full = PchipInterpolator(x_full, y_full, extrapolate=True)
                    y_plot_full = f_full(x_fine)
                    ax.plot(x_fine*100, y_plot_full, 'b-', label='Calibrated', linewidth=2, alpha=0.7)
                except: pass
                
                # 2. Interpolated Data (Hiding target point)
                x_train, y_train = [], []
                target_y = np.nan
                
                for x, y in zip(x_full, y_full):
                    if np.isclose(x, hide_det):
                        target_y = y
                        continue
                    x_train.append(x)
                    y_train.append(y)
                
                pred_val = np.nan
                if len(x_train) >= 2:
                    try:
                        f_interp = PchipInterpolator(x_train, y_train, extrapolate=False)
                        y_plot_interp = f_interp(x_fine)
                        pred_val = float(f_interp(hide_det))
                        ax.plot(x_fine*100, y_plot_interp, 'r--', label='Interpolated', linewidth=1.5)
                    except: pass
                
                # 3. Points
                ax.scatter(np.array(x_train)*100, y_train, color='black', s=30, label='Known', zorder=5)
                
                if not np.isnan(target_y):
                    ax.scatter([hide_det*100], [target_y], color='blue', s=80, edgecolors='black', label='Actual', zorder=10)
                
                if not np.isnan(pred_val):
                    ax.scatter([hide_det*100], [pred_val], color='red', s=80, marker='x', linewidth=3, label='Predicted', zorder=10)
                    err = pred_val - target_y
                    ax.text(0.05, 0.05, f"Err: {err:.4f}", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

                if row_idx == 0: ax.set_title(f'{int(tenor)}Y Tenor')
                if col_idx == 0: ax.set_ylabel(f'Corr (Hide {int(hide_det*100)}%)')
                ax.grid(True)
                
                if row_idx == len(self.testable_detachments)-1 and col_idx == len(target_tenors)-1:
                    ax.legend(loc='lower right', fontsize='small')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(self.res_dir, 'loocv_smiles_comparison.png'))
        print(f"Saved aggregated plot to loocv_smiles_comparison.png")

if __name__ == "__main__":
    validator = LOOCV_Validator()
    
    # 1. Run Tenor LOOCV
    validator.run_tenor_loocv(7.0)
    
    # 2. Run Restricted Point LOOCV
    validator.run_point_loocv()
    
    # 3. Generate Aggregated Plot
    validator.plot_aggregated_loocv()