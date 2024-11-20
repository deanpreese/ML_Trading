import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    close_data = df['SeqClose'] + df['output']
    change_data = df[['output']].values
    return change_data, close_data.values

# Function to train the HMM model
def train_hmm_model(train_data, val_data, n_states=4):
    model = hmm.GaussianHMM(n_components=n_states, 
                            #covariance_type="diag", 
                            n_iter=1000, init_params="")
    
    model.startprob_ = np.array([1.0 / n_states] * n_states)
    model.transmat_ = np.array([[0.5 if i == j else 0.5 / (n_states - 1) for j in range(n_states)] for i in range(n_states)])
    model.means_ = np.array([train_data.mean(axis=0) for _ in range(n_states)])
    
    # Corrected covars_ initialization
    model.covars_ = np.array([np.var(train_data, axis=0)] * n_states)

    best_val_loss = float('inf')
    no_improvement_count = 0

    for iteration in range(100):  # Maximum training iterations
        model.fit(train_data)
        val_states = model.predict(val_data)
        val_reconstructed = model.means_[val_states].flatten()
        val_loss = np.mean((val_data.flatten() - val_reconstructed) ** 2)

        print(val_loss)

        if val_loss < best_val_loss + 0.0001:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= 10:  # Early stopping patience
            print(f"Early stopping at iteration {iteration}. Best validation loss: {best_val_loss:.4f}")
            break

    return model

# Function to predict states
def predict_states(model, data):
    return model.predict(data)

# Function to identify buy and sell points based on state transitions
def identify_buy_sell_points(states):
    buy_points = [i for i in range(1, len(states)) if states[i - 1] < states[i]]
    sell_points = [i for i in range(1, len(states)) if states[i - 1] > states[i]]
    return buy_points, sell_points

# Function to evaluate trading performance
def evaluate_trading_performance(buy_points, sell_points, original_data):
    if not buy_points or not sell_points:
        print("No buy or sell points identified.")
        return

    trade_results = []
    num_trades = min(len(buy_points), len(sell_points))
    for i in range(num_trades):
        if buy_points[i] < len(original_data) and sell_points[i] < len(original_data):
            buy_price = original_data[buy_points[i]]
            sell_price = original_data[sell_points[i]]
            trade_result = sell_price - buy_price
            trade_results.append(trade_result)
        else:
            print(f"Skipping trade at indices {buy_points[i]}, {sell_points[i]} due to index out of range")

    if not trade_results:
        print("No valid trades to evaluate.")
        return

    winning_trades = sum(1 for result in trade_results if result > 0)
    win_rate = (winning_trades / len(trade_results)) * 100 if trade_results else 0

    total_profits = sum(result for result in trade_results if result > 0)
    total_losses = sum(abs(result) for result in trade_results if result < 0)
    profit_factor = total_profits / total_losses if total_losses != 0 else 0

    initial_investment = sum(original_data[buy_points[i]] for i in range(num_trades) if buy_points[i] < len(original_data))
    total_return = sum(trade_results)
    roi = (total_return / initial_investment) * 100 if initial_investment != 0 else 0

    print(f"Win Rate: {win_rate}%")
    print(f"Profit Factor: {profit_factor}")
    print(f"ROI: {roi}%")

# Function to plot buy and sell points
def plot_buy_sell_points(data, states, buy_points, sell_points, x_last):
    data = data[-x_last:]
    states = states[-x_last:]

    buy_points = [i for i in buy_points if i >= len(states) - x_last]
    sell_points = [i for i in sell_points if i >= len(states) - x_last]

    relative_buy_points = [i - (len(states) - x_last) for i in buy_points]
    relative_sell_points = [i - (len(states) - x_last) for i in sell_points]

    relative_buy_points = [i for i in relative_buy_points if 0 <= i < len(data)]
    relative_sell_points = [i for i in relative_sell_points if 0 <= i < len(data)]

    plt.figure(figsize=(14, 7))
    plt.plot(data, label="Stock Price", color="blue")

    if relative_buy_points:
        plt.scatter(relative_buy_points, [data[i] for i in relative_buy_points], color="lime", label="Buy", marker="^", s=100)
    if relative_sell_points:
        plt.scatter(relative_sell_points, [data[i] for i in relative_sell_points], color="red", label="Sell", marker="v", s=100)

    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.title(f"Last {x_last} Values: Stock Price with Buy/Sell Points")
    plt.show()

# Main function
def main():
    datafile = [
        'data/NewModel_3070_oos.csv',
        'data/NewModel_3070.csv',
        'data/NewModel_ALL_oos.csv',
        'data/NewModel_ALL.csv',
        'data/NewModel_span3_3070_oos.csv',
        'data/NewModel_span3_3070.csv',
    ]

    filepath = datafile[3]
    normalized_data, original_data = load_and_preprocess_data(filepath)

    train_data, test_data = train_test_split(normalized_data, test_size=0.3, random_state=42, shuffle=False)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42, shuffle=False)

    model = train_hmm_model(train_data, val_data)
    historical_states = predict_states(model, normalized_data)

    buy_points, sell_points = identify_buy_sell_points(historical_states)

    evaluate_trading_performance(buy_points, sell_points, original_data)

    x_last = 1000
    plot_buy_sell_points(original_data, historical_states, buy_points, sell_points, x_last)

if __name__ == "__main__":
    main()
