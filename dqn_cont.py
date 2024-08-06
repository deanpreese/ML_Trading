import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
im01port rando7
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_errorthcontroldis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure TensorFlow uses the Metal device (M1 GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
    except RuntimeError as e:
        logging.error(e)

# Define the DQN model
class DQN(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_dim)
    
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output_layer(x)

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        logging.info("Updated target model weights.")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        logging.debug(f"Remembered state-action pair: state={state}, action={action}, reward={reward}, next_state={next_state}, done={done}")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.uniform(-1, 1)  # Random action for exploration
            logging.debug(f"Exploration action: {action}")
        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.model(state)
            action = q_values[0, 0].numpy()
            logging.debug(f"Exploitation action: {action}")
        return action
    
    def replay(self, batch_size):
        start_time = time.time()
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for state, action, reward, next_state, done in minibatch:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            
            target = self.model(state).numpy()
            if done:
                target[0, 0] = reward
            else:
                t = self.target_model(next_state).numpy()
                target[0, 0] = reward + self.gamma * np.amax(t)
            
            with tf.GradientTape() as tape:
                q_values = self.model(state)
                loss = tf.keras.losses.MSE(target, q_values)
                losses.append(loss.numpy())
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        avg_loss = np.mean(losses)
        elapsed_time = time.time() - start_time
        logging.info(f"Replayed a batch of size {batch_size}. Avg loss: {avg_loss:.4f}. Current epsilon: {self.epsilon:.4f}. Replay time: {elapsed_time:.2f}s")

    def load(self, name):
        self.model.load_weights(name)
        logging.info(f"Loaded model weights from {name}.")
    
    def save(self, name):
        self.model.save_weights(name)
        logging.info(f"Saved model weights to {name}.")

# Load data from CSV
def load_data(filename):
    logging.info(f"Loading data from {filename}...")
    data = pd.read_csv(filename)
    X = data.drop(columns=['output', 'outputC'])
    features = X.values
    continuous_output = data['output'].values
    logging.info("Data loaded successfully.")
    logging.info(f"Features shape: {features.shape}, Continuous output shape: {continuous_output.shape}")
    return features, continuous_output

# Split data into training and validation sets
def train_val_split(features, continuous_output, val_ratio=0.3):
    train_features, val_features, train_output, val_output = train_test_split(
        features, continuous_output, test_size=val_ratio, random_state=42)
    return train_features, val_features, train_output, val_output

# Get initial state
def get_initial_state(features):
    state = features[0]
    logging.debug(f"Initial state: {state}")
    return state  # Example: return the first row of features

# Take action and return next state and reward
def take_action(action, features, continuous_output, index):
    next_index = index + 1
    if next_index >= len(features):
        next_index = len(features) - 1
    
    next_state = features[next_index]
    
    # Reward based on the accuracy of the prediction
    predicted_value = action  # Assume the action is the predicted continuous value
    actual_value = continuous_output[index]
    reward = -abs(predicted_value - actual_value)

    logging.debug(f"Action taken: {action}, Actual value: {actual_value}, Reward: {reward}, Next state: {next_state}")
    
    return next_state, reward

# Validate the model on the validation set and calculate performance metrics
def validate(agent, val_features, val_output):
    predictions = []
    for i in range(len(val_features)):
        state = val_features[i]
        action = agent.act(state)
        predictions.append(action)
    
    mse = mean_squared_error(val_output, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(val_output, predictions)
    return mse, rmse, mae

# Initialize DQN agents
def initialize_agents(state_dim, action_dim, num_agents=2, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    agents = [DQNAgent(state_dim, action_dim, lr, gamma, epsilon, epsilon_min, epsilon_decay) for _ in range(num_agents)]
    logging.info(f"Initialized {num_agents} DQN agents.")
    return agents

# Training loop
def train_agents(agents, train_features, train_output, val_features, val_output, batch_size=64, save_interval=10, max_episodes=1000, patience=10):
    max_time_steps = len(train_features) - 1
    best_val_loss = float('inf')
    patience_counter = 0

    for episode in range(max_episodes):
        logging.info(f"Starting episode {episode + 1}...")
        start_time = time.time()
        state = get_initial_state(train_features)
        for time_step in range(max_time_steps):
            actions = [agent.act(state) for agent in agents]
            action = np.mean(actions)  # Ensemble prediction by averaging actions
            next_state, reward = take_action(action, train_features, train_output, time_step)
            for agent in agents:
                agent.remember(state, action, reward, next_state, done=(time_step == max_time_steps - 1))
            state = next_state
            for agent in agents:
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
        for agent in agents:
            agent.update_target_model()
        
        # Validate the model
        val_metrics = [validate(agent, val_features, val_output) for agent in agents]
        val_mse = np.mean([m[0] for m in val_metrics])
        val_rmse = np.mean([m[1] for m in val_metrics])
        val_mae = np.mean([m[2] for m in val_metrics])
        logging.info(f"Validation MSE after episode {episode + 1}: {val_mse:.4f}")
        logging.info(f"Validation RMSE after episode {episode + 1}: {val_rmse:.4f}")
        logging.info(f"Validation MAE after episode {episode + 1}: {val_mae:.4f}")
        
        # Check for early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            for i, agent in enumerate(agents):
                agent.save(f"continuous_model_best_{i}.h5")
            logging.info(f"New best model saved at episode {episode + 1}.")
        else:
            patience_counter += 1
            logging.info(f"No improvement in validation loss for {patience_counter} consecutive episodes.")

        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

        if episode % save_interval == 0:
            for i, agent in enumerate(agents):
                agent.save(f"continuous_model_{i}.h5")
            logging.info(f"Saved models at episode {episode + 1}.")
        
        elapsed_time = time.time() - start_time
        logging.info(f"Completed episode {episode + 1} in {elapsed_time:.2f}s.")

    logging.info("Training completed.")

# Final evaluation
def evaluate_final_model(agents, val_features, val_output):
    final_metrics = [validate(agent, val_features, val_output) for agent in agents]
    final_mse = np.mean([m[0] for m in final_metrics])
    final_rmse = np.mean([m[1] for m in final_metrics])
    final_mae = np.mean([m[2] for m in final_metrics])
    logging.info(f"Final Validation MSE: {final_mse:.4f}")
    logging.info(f"Final Validation RMSE: {final_rmse:.4f}")
    logging.info(f"Final Validation MAE: {final_mae:.4f}")

def main():
    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/ndata_diff_lucky13_3070_oos.csv', 
        'data/ndata_diff_lucky13_3070.csv', #3
        'data/ndata_lag_3070_oos.csv', 
        'data/ndata_lag_3070.csv', #5
    ]
    
    logging.info("Starting main process...")
    features, continuous_output = load_data(datafile[1])
    train_features, val_features, train_output, val_output = train_val_split(features, continuous_output)
    state_dim = train_features.shape[1]
    continuous_action_dim = 1  # For continuous output

    # Initialize ensemble of agents
    agents = initialize_agents(state_dim, continuous_action_dim)

    # Train agents
    train_agents(agents, train_features, train_output, val_features, val_output)

    # Evaluate final model
    evaluate_final_model(agents, val_features, val_output)

if __name__ == "__main__":
    main()
