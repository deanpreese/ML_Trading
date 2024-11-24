import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import matplotlib as mpl
import pickle
from sklearn.model_selection import train_test_split


def print_model_details(hmm_model, rets):
    """Prints model score, hidden state details, and transition matrix."""
    
    print("\nModel Score:", hmm_model.score(rets))
    Z = hmm_model.predict(rets)
    states = pd.unique(Z)
    
    print(f"Percentage of hidden state 1 = {sum(Z) / len(Z):.6f}")
    print("Transition matrix:")
    print(hmm_model.transmat_)
    
    print("Means and vars of each hidden state:")
    for i in range(hmm_model.n_components):
        print(f"{i}th hidden state")
        print("mean =", hmm_model.means_[i])
        print("var =", np.diag(hmm_model.covars_[i]))
    print("\n")
    return Z


def plot_hidden_states(xchanges, Z, hmm_model, title="Hidden States vs Cumulative Changes"):
    """Plots hidden states against cumulative changes."""
    cmap = mpl.colormaps['tab10']
    fig, axs = plt.subplots(hmm_model.n_components, sharex=True, sharey=True, figsize=(10, 8))
    colours = cmap(np.linspace(0, 1, hmm_model.n_components))
    
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = Z == i
        ax.plot(xchanges['Time'][mask], xchanges['CumulativeChanges'][mask], ".-", c=colour)
        ax.set_title(f"{i}th hidden state", fontsize=12)
        ax.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_model(hmm_model, filename='hmm_model.pkl'):
    """Saves the HMM model to a file."""
    with open(filename, 'wb') as file:
        pickle.dump(hmm_model, file)
    print(f"Model saved to {filename}")


def load_model(filename='hmm_model.pkl'):
    """Loads the HMM model from a file."""
    with open(filename, 'rb') as file:
        hmm_model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return hmm_model


def predict_hidden_states(hmm_model, new_data):
    """Predicts hidden states for new data."""
    rets = np.column_stack([new_data])
    return hmm_model.predict(rets)


def predict_current_state(hmm_model, data):
    """Predicts the current hidden state and its probabilities."""
    rets = np.column_stack([data])
    state_probs = hmm_model.predict_proba(rets)[-1]
    current_state = np.argmax(state_probs)
    return current_state, state_probs


def main():
    datafile = [
        'data/NewModel_3070_oos.csv',
        'data/NewModel_3070.csv',
        'data/NewModel_ALL_oos.csv',
        'data/NewModel_ALL.csv',
        'data/NewModel_span3_3070_oos.csv',
        'data/NewModel_span3_3070.csv',
    ]
    
    file_path = datafile[3]
    
    data_r = pd.read_csv(file_path).tail(5000)
    data = data_r.drop(columns=['outputC'])
    changes = data['output']

    cumulative_changes = data['SeqClose']
    changes2 = list(enumerate(cumulative_changes))
    xchanges = pd.DataFrame(changes2, columns=['Time', 'CumulativeChanges'])    

    changes_train, changes_test = train_test_split(changes, test_size=0.3, random_state=42)
    
    rets_train = np.column_stack([changes_train])
    hmm_model = GaussianHMM(n_components=5, covariance_type="diag", 
                            n_iter=1000, tol=0.001, random_state=42, verbose=True)
    hmm_model.fit(rets_train)
    
    Z_train = print_model_details(hmm_model, rets_train)
    Z_test = predict_hidden_states(hmm_model, changes_test)
    
    # Calculate cumulative changes for the test set
    cumulative_changes_test = np.cumsum(changes_test)
    xchanges_test = pd.DataFrame(list(enumerate(cumulative_changes_test)), columns=['Time', 'CumulativeChanges'])
    
    # Predict the current state
    current_state, state_probs = predict_current_state(hmm_model, changes)
    print(f"Current state: {current_state}")
    
    # Dynamically map states
    state_mapping = {i: f"{i+1}th" for i in range(hmm_model.n_components)}
    mapped_state = state_mapping.get(current_state, "Unknown")
    print(f"The model is currently in the {mapped_state} state.")
    
    print("State probabilities:")
    for state, prob in zip(state_mapping.values(), state_probs):
        print(f"{state}: {prob:.2f}")
    
    # Plot hidden states for the testing set
    plot_hidden_states(xchanges_test, Z_test, hmm_model, title="Hidden States vs Cumulative Changes (Testing Data)")


if __name__ == "__main__":
    main()
