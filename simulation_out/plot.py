import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
def load_data(file_path):
    # Read CSV into a pandas dataframe
    df = pd.read_csv(file_path)
    # Convert date column to datetime format
    df['date'] = pd.to_datetime(df['date'], format='%m%d')
    return df

# Plotting the results
def plot_hedging_results(df):
    plt.figure(figsize=(14, 8))

    # Plot Daily PnL
    plt.subplot(2, 2, 1)
    plt.plot(df['date'], df['daily_pnl'], color='b', label='Daily PnL')
    plt.title('Daily PnL')
    plt.xlabel('Date')
    plt.ylabel('Daily PnL')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot Cumulative PnL
    plt.subplot(2, 2, 2)
    plt.plot(df['date'], df['cumulative_pnl'], color='g', label='Cumulative PnL')
    plt.title('Cumulative PnL')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot Hedge Ratio
    plt.subplot(2, 2, 3)
    plt.plot(df['date'], df['hedge_ratio'], color='r', label='Hedge Ratio')
    plt.title('Hedge Ratio')
    plt.xlabel('Date')
    plt.ylabel('Hedge Ratio')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot Hedge Notional
    plt.subplot(2, 2, 4)
    plt.plot(df['date'], df['hedge_notional'], color='purple', label='Hedge Notional')
    plt.title('Hedge Notional')
    plt.xlabel('Date')
    plt.ylabel('Hedge Notional')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Layout adjustments
    plt.tight_layout()
    plt.show()

def main():
    # Path to the CSV file
    file_path = 'delta_hedge_results_4.0-6.0%.csv'  # Change this to your actual file path

    # Load the data from CSV
    df = load_data(file_path)

    # Visualize the hedging results
    plot_hedging_results(df)

if __name__ == '__main__':
    main()
