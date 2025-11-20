import pandas as pd
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def calculate_basis_adjustment():
    print("Loading data...")
    # Load the inputs generated from the previous step
    try:
        loss_df = pd.read_csv('loss_curves.csv')
        const_df = pd.read_csv('constituent_survival_curves.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'loss_curves.csv' and 'constituent_survival_curves.csv' are in the current directory.")
        return

    # Define tenors of interest
    tenors = [1, 2, 3, 5, 7, 10]
    tenor_cols = [f'S_{t}Y' for t in tenors]

    betas = []
    adjusted_data = const_df.copy()

    print("Calculating Beta(t) for each tenor...")
    
    for t, col in zip(tenors, tenor_cols):
        # 1. Get Target Index Loss for this tenor
        target_row = loss_df.loc[loss_df['Tenor'] == t]
        if target_row.empty:
            print(f"Warning: No loss data for {t}Y. Skipping.")
            betas.append(1.0)
            continue
            
        target_loss = target_row['Index_Loss'].values[0]
        
        # 2. Get Constituent Data
        S_i = const_df[col].values
        R_i = const_df['Recovery'].values
        
        # 3. Define Objective Function
        # We want to find beta such that:
        # Average( (1 - R_i) * (1 - S_i^beta) ) == Target_Index_Loss
        
        def objective(beta):
            # Apply power adjustment to survival probabilities
            # S_adj = S ^ beta
            S_adj = np.power(S_i, beta)
            
            # Calculate individual losses
            losses = (1 - R_i) * (1 - S_adj)
            
            # Calculate average loss
            avg_loss = np.mean(losses)
            
            return avg_loss - target_loss
        
        # 4. Solve for Beta
        try:
            # Beta is typically between 0 and 5. 
            # beta < 1 implies we are increasing survival (reducing spread)
            # beta > 1 implies we are reducing survival (increasing spread)
            beta_sol = brentq(objective, 0.01, 10.0)
        except Exception as e:
            print(f"Optimization failed for Tenor {t}Y: {e}. Defaulting to 1.0")
            beta_sol = 1.0
        
        betas.append(beta_sol)
        
        # 5. Apply Adjustment
        adjusted_data[col] = np.power(S_i, beta_sol)
        print(f"Tenor {t}Y: Beta = {beta_sol:.6f}")

    # Create Beta DataFrame
    beta_df = pd.DataFrame({'Tenor': tenors, 'Beta': betas})

    # Save Outputs
    print("\nSaving outputs...")
    beta_df.to_csv('beta_curve.csv', index=False)
    adjusted_data.to_csv('adjusted_constituent_survival_curves.csv', index=False)
    print(" - beta_curve.csv")
    print(" - adjusted_constituent_survival_curves.csv")

    # Plot Beta Curve
    plt.figure(figsize=(10, 6))
    plt.plot(beta_df['Tenor'], beta_df['Beta'], marker='o', linestyle='-', color='b')
    plt.title('Basis Adjustment Function beta(t)')
    plt.xlabel('Tenor (Years)')
    plt.ylabel('Beta')
    plt.grid(True)
    plt.savefig('beta_curve_plot.png')
    print(" - beta_curve_plot.png")
    
    # Verify results
    print("\nVerification (Adjusted Avg Loss vs Target):")
    verified_losses = []
    for t, col in zip(tenors, tenor_cols):
        S_adj = adjusted_data[col]
        R = adjusted_data['Recovery']
        loss = (1 - R) * (1 - S_adj)
        verified_losses.append(loss.mean())
        
    verification_df = pd.DataFrame({
        'Tenor': tenors,
        'Target_Index': loss_df['Index_Loss'],
        'Adjusted_Avg': verified_losses
    })
    print(verification_df)
    print("\nDone.")

if __name__ == "__main__":
    calculate_basis_adjustment()