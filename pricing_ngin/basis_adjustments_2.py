import pandas as pd
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import os

def calculate_basis_adjustment(data_dir):
    print(f"Running Basis Adjustment in: {data_dir}")
    inoutdir = os.path.join(data_dir, 'outdata')
    
    loss_df = pd.read_csv(os.path.join(inoutdir, 'loss_curves.csv'))
    const_df = pd.read_csv(os.path.join(inoutdir, 'constituent_survival_curves.csv'))
    
    tenors = [1, 2, 3, 5, 7, 10]
    adjusted_data = const_df.copy()
    betas = []

    for t in tenors:
        target_loss = loss_df.loc[loss_df['Tenor'] == t, 'Index_Loss'].values[0]
        S_i = const_df[f'S_{t}Y'].values
        R_i = const_df['Recovery'].values
        
        def objective(beta):
            return np.mean((1 - R_i) * (1 - np.power(S_i, beta))) - target_loss
            
        try: beta = brentq(objective, 0.01, 10.0)
        except: beta = 1.0
        betas.append(beta)
        adjusted_data[f'S_{t}Y'] = np.power(S_i, beta)

    pd.DataFrame({'Tenor': tenors, 'Beta': betas}).to_csv(os.path.join(inoutdir, 'beta_curve.csv'), index=False)
    adjusted_data.to_csv(os.path.join(inoutdir, 'adjusted_constituent_survival_curves.csv'), index=False)

if __name__ == "__main__":
    calculate_basis_adjustment('./')