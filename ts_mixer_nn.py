import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Ensure reproducibility
torch.manual_seed(42)

# TSMixer Model
class TSMixer(nn.Module):
    def __init__(self, num_features, seq_length, hidden_dim=64, num_layers=2):
        super(TSMixer, self).__init__()
        
        self.feature_mixer = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features)
        )
        
        self.time_mixer = nn.Sequential(
            nn.Linear(seq_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_length)
        )
        
        self.fc_out = nn.Linear(num_features * seq_length, 1)

    def forward(self, x):
        # Mix features
        x = self.feature_mixer(x)
        
        # Mix time steps
        x = x.transpose(1, 2)
        x = self.time_mixer(x)
        x = x.transpose(1, 2)
        
        # Flatten and output
        x = x.flatten(start_dim=1)
        output = self.fc_out(x)
        return output

# Data Preprocessing
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Assuming 'output' is the target and other columns are features
    X = data.drop(columns=['output', 'outputC'])  # Dropping outputC as per your context
    y = data['output']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape data for sequence modeling
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))  # [batch_size, seq_length, num_features]

    return X_scaled, y.values

# Training Function with Early Stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=200, model_save_path="tsmixer_model.pth", patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    model.train()
    
    for epoch in range(epochs):
        if early_stop:
            print("Early stopping")
            break
            
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.float())
            loss = criterion(outputs.squeeze(), y_batch.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_loss = validate_model(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

# Validation Function
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch.float())
            loss = criterion(outputs.squeeze(), y_batch.float())
            val_loss += loss.item()
    model.train()
    return val_loss / len(val_loader)

# Metrics Calculation
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.float())
            y_true.extend(y_batch.numpy())
            y_pred.extend(outputs.squeeze().numpy())
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    wins = 0
    losses = 0
    
    for i in range(len(y_true)):
        if (y_pred[i] > 0 and y_true[i] > 0) or (y_pred[i] < 0 and y_true[i] < 0):
            wins += 1
        elif (y_pred[i] > 0 and y_true[i] < 0) or (y_pred[i] < 0 and y_true[i] > 0):
            losses += 1
        elif (y_pred[i] == 0 and y_true[i] == 0):
            wins += 1
        elif (y_pred[i] == 0 and y_true[i] != 0):
            losses += 1
    
    total_samples = wins + losses
    win_percentage = (wins / total_samples) * 100    
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print(f"Wins: {wins}, Losses: {losses}, Win Percentage: {win_percentage:.2f}%")
    
    return y_true, y_pred

# Plotting Function
def plot_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True Values", color="blue")
    plt.plot(y_pred, label="Predictions", color="red")
    plt.title("True Values vs Predictions")
    plt.xlabel("Samples")
    plt.ylabel("Output")
    plt.legend()
    plt.show()

# Main Function
def main():
    # Load and preprocess data
    data_file = 'data/Lucky13_3070.csv'
    #data_file = 'data/Lucky13_D.csv'
    X, y = load_and_preprocess_data(data_file)
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Create DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    num_features = X_train.shape[2]
    seq_length = X_train.shape[1]
    model = TSMixer(num_features=num_features, seq_length=seq_length)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model with validation, early stopping, and saving
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=200, patience=10)
    
    # Load the best model
    model.load_state_dict(torch.load("tsmixer_model.pth", weights_only=True))
    print("Best model loaded for evaluation.")
    
    # Evaluate the model on test set
    y_true, y_pred = evaluate_model(model, test_loader)
    
    # Plot the results
    plot_results(y_true, y_pred)

if __name__ == "__main__":
    main()
