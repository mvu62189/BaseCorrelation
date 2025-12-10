import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Results
# Replace with your actual output filename
file_path = 'simulation_out/delta_hedge_results_4.0-6.0%.csv'
df = pd.read_csv(file_path)

# 2. Define Notional (Must match what you used in simulation)
TR_NOTIONAL = 10_000_000 

# 3. Calculate Unhedged PnL
# We assume Short Protection (-Notional). 
# If PV goes UP, you lose money.
df['Unhedged_Daily_PnL'] = -df['tranche_pv'].diff() * TR_NOTIONAL

# 4. Compare with Hedged PnL (already in CSV)
# The first day is usually 0 or NaN because there's no history
df_clean = df.dropna().iloc[1:].copy()

# Cumulative PnL for plotting
df_clean['Unhedged_Cumulative_PnL'] = df_clean['Unhedged_Daily_PnL'].cumsum()
df_clean['Hedged_Cumulative_PnL'] = df_clean['daily_pnl'].cumsum()

# 5. Calculate Hedging Effectiveness
var_unhedged = df_clean['Unhedged_Daily_PnL'].var()
var_hedged = df_clean['daily_pnl'].var()

effectiveness = 1 - (var_hedged / var_unhedged)

print(f"--- Hedging Effectiveness ---")
print(f"Variance Reduction: {effectiveness:.2%}")
print(f"Unhedged Volatility (Daily): ${df_clean['Unhedged_Daily_PnL'].std():,.0f}")
print(f"Hedged Volatility (Daily):   ${df_clean['daily_pnl'].std():,.0f}")

# 6. Plot
plt.figure(figsize=(12, 6))
plt.plot(df_clean['date'], df_clean['Unhedged_Cumulative_PnL'], label='Unhedged Portfolio', linestyle='--', color='red')
plt.plot(df_clean['date'], df_clean['Hedged_Cumulative_PnL'], label='Delta Hedged Portfolio', color='blue')
plt.title(f'Hedging Performance (Effectiveness: {effectiveness:.1%})')
plt.ylabel('Cumulative PnL ($)')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()