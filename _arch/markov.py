import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import matplotlib as mpl
import pickle
from sklearn.model_selection import train_test_split

def load_data(file_path):
    try:
        data_r = pd.read_csv(file_path).tail(5000)
        data = data_r.drop(columns=['outputC'])
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def process_data(data):
    changes = data['output']
    cumulative_changes = np.cumsum(changes)
    changes2 = list(enumerate(cumulative_changes))
    xchanges = pd.DataFrame(changes2, columns=['Time', 'CumulativeChanges'])
    return xchanges, changes

def fit_hmm_model(data, n_components=3):
    rets = np.column_stack([data])
    hmm_model = GaussianHMM(n_components=n_components, covariance_type="full", 
                            n_iter=1000, tol=0.001, random_state=99, verbose=True)
    hmm_model.fit(rets)
    return hmm_model, rets

def print_model_details(hmm_model, rets):
    print("Model Score:", hmm_model.score(rets))
    Z = hmm_model.predict(rets)
    states = pd.unique(Z)
    
    print('Percentage of hidden state 1 = %f' % (sum(Z) / len(Z)))
    print("Transition matrix")
    print(hmm_model.transmat_)
    
    print("Means and vars of each hidden state")
    for i in range(hmm_model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", hmm_model.means_[i])
        print("var = ", np.diag(hmm_model.covars_[i]))
    
    return Z

def plot_hidden_states(xchanges, Z, hmm_model, title="Hidden States vs Cumulative Changes"):
    cmap = mpl.colormaps['tab10']
    fig, axs = plt.subplots(hmm_model.n_components, sharex=True, sharey=True, figsize=(10, 8))
    colours = cmap(np.linspace(0, 1, hmm_model.n_components))
    
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = Z == i
        ax.plot(xchanges['Time'][mask], xchanges['CumulativeChanges'][mask], "--", c=colour)
        ax.plot(xchanges['Time'][mask], xchanges['CumulativeChanges'].iloc[mask], ".-", c=colour)
        ax.set_title(f"{i}th hidden state", fontsize=12)
        ax.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_model(hmm_model, filename='hmm_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(hmm_model, file)
    print(f"Model saved to {filename}")

def load_model(filename='hmm_model.pkl'):
    with open(filename, 'rb') as file:
        hmm_model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return hmm_model

def predict_hidden_states(hmm_model, new_data):
    rets = np.column_stack([new_data])
    hidden_states = hmm_model.predict(rets)
    return hidden_states

def predict_current_state(hmm_model, data):
    rets = np.column_stack([data])
    state_probs = hmm_model.predict_proba(rets)[-1]
    current_state = np.argmax(state_probs)
    return current_state, state_probs

def main():
    file_path = "data/buildSeqInd_Lucky13_5M_ALL.csv"
    #file_path = 'data/buildSeqInd_Lucky13_F.csv'
    #file_path = 'data/buildSeqInd_Lucky13.csv'
    data = load_data(file_path)
    if data is not None:
        xchanges, changes = process_data(data)
        
        # Split the data into training and testing sets
        changes_train, changes_test = train_test_split(changes, test_size=0.3, random_state=42)
        
        # Train the model on the training set
        hmm_model, rets_train = fit_hmm_model(changes_train, 5)
        
        # Print model details for the training set
        Z_train = print_model_details(hmm_model, rets_train)
        
        # Plot hidden states for the training set
        #xchanges_train = xchanges.iloc[:len(changes_train)]
        #plot_hidden_states(xchanges_train, Z_train, hmm_model, title="Hidden States vs Cumulative Changes (Training Data)")
        
        # Save the model
        #save_model(hmm_model)
        
        # Load the model and make predictions on the testing data
        #loaded_model = load_model()
        
        loaded_model = hmm_model
        
        # Predict hidden states for the entire testing set at once
        Z_test = predict_hidden_states(loaded_model, changes_test)
        
        # Calculate cumulative changes for the testing set
        cumulative_changes_test = np.cumsum(changes_test)
        xchanges_test = pd.DataFrame(list(enumerate(cumulative_changes_test)), columns=['Time', 'CumulativeChanges'])
        
      
        print( " ")
        print( Z_test)
        print( " ")
        
        # Determine the current state and its probabilities
        current_state, state_probs = predict_current_state(loaded_model, changes)
        print(f"Current state: {current_state}")
        state_mapping = {0: 'Zero', 1: '1st', 2: '2nd', 3: "3rd", 4: '4th', 5: '5th', 6: '6th' }  # Example mapping, adjust as needed
        mapped_state = state_mapping[current_state]

        print(f"The model is currently in the {mapped_state} state.")
        print('State probabilities:')
        for state, prob in zip(state_mapping.values(), state_probs):
            print(f'{state}: {prob:.2f}')

        print( " ")
        
        
        # Plot hidden states for the testing set
        #plot_hidden_states(xchanges_test, Z_test, loaded_model, title="Hidden States vs Cumulative Changes (Testing Data)")


if __name__ == "__main__":
    main()
