import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Load the data
def load_data(file_path):
    # Read CSV into a pandas dataframe
    df = pd.read_csv(file_path)
    
    # Ensure date column is treated as string for consistent filtering/parsing
    df['date'] = df['date'].astype(str)
    
    # --- FIX 1: Exclude the date '1025' ---
    #df = df[df['date'] != '1025']
    
    # Convert date column to datetime format
    # format='%m%d' will default to year 1900, which is fine for plotting relative order
    df['date'] = pd.to_datetime(df['date'], format='%m%d')
    return df

# Plotting the results
def plot_hedging_results(df):
    plt.figure(figsize=(14, 8))
    
    # --- FIX 2: Define the mmdd Date Formatter ---
    my_date_fmt = mdates.DateFormatter('%m%d')

    # Plot Daily PnL
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df['date'], df['daily_pnl'], color='b', label='Daily PnL')
    ax1.set_title('Daily PnL')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily PnL')
    ax1.xaxis.set_major_formatter(my_date_fmt) # Apply format
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot Cumulative PnL
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df['date'], df['cumulative_pnl'], color='g', label='Cumulative PnL')
    ax2.set_title('Cumulative PnL')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative PnL')
    ax2.xaxis.set_major_formatter(my_date_fmt) # Apply format
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot Hedge Ratio
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(df['date'], df['hedge_ratio'], color='r', label='Hedge Ratio')
    ax3.set_title('Hedge Ratio')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Hedge Ratio')
    ax3.xaxis.set_major_formatter(my_date_fmt) # Apply format
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot Hedge Notional
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(df['date'], df['hedge_notional'], color='purple', label='Hedge Notional')
    ax4.set_title('Hedge Notional')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Hedge Notional')
    ax4.xaxis.set_major_formatter(my_date_fmt) # Apply format
    plt.xticks(rotation=45)
    plt.grid(True)

    # Layout adjustments
    plt.tight_layout()
    plt.show()

def main():
    # Path to the CSV file
    # Ensure this matches your actual file name
    file_path = 'simulation_out/delta_hedge_results_0.0-3.0%.csv'  

    # Load the data from CSV
    df = load_data(file_path)

    # Visualize the hedging results
    plot_hedging_results(df)

if __name__ == '__main__':
    main()