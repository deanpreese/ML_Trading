import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TSMixerKANModel(nn.Module):
    def __init__(self, input_dim, hidden_units=32, output_dim=1):
        super(TSMixerKANModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(1, hidden_units, batch_first=True) for _ in range(input_dim)
        ])
        self.feature_mixing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, input_dim),
                nn.ReLU()
            ) for _ in range(input_dim)
        ])
        self.time_mixing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU()
            ) for _ in range(input_dim)
        ])
        self.attention = nn.MultiheadAttention(hidden_units, num_heads=1, batch_first=True)
        self.final_dense = nn.Linear(hidden_units, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.input_dim, 1)  # Reshape for LSTM
        
        univariate_outputs = []
        for i in range(self.input_dim):
            lstm_out, _ = self.lstm_layers[i](x[:, i:i+1, :])  # Apply LSTM per feature
            mixed_out = self.feature_mixing_layers[i](lstm_out.squeeze(1))  # Feature mixing
            mixed_out = self.time_mixing_layers[i](mixed_out)  # Time mixing
            univariate_outputs.append(mixed_out)
        
        combined_output = torch.stack(univariate_outputs, dim=1)  # Combine outputs
        attention_output, _ = self.attention(combined_output, combined_output, combined_output)
        
        # Flatten the output
        flattened_output = attention_output.reshape(batch_size, -1)
        
        # Debugging: print the shapes
        print(f"Flattened output shape: {flattened_output.shape}")
        print(f"Expected input shape for final dense layer: {self.final_dense.in_features}")
        
        # Adjust final_dense layer if necessary
        if flattened_output.shape[1] != self.final_dense.in_features:
            self.final_dense = nn.Linear(flattened_output.shape[1], self.output_dim)
        
        output = self.final_dense(flattened_output)
        return output

        


def train_model(model, dataloader, criterion, optimizer, num_epochs=100, device='cpu'):
    model.to(device)
    history = {'train_loss': []}
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        history['train_loss'].append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return history

def evaluate_model(model, X_test, y_test, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        y_pred = model(X_test).cpu().numpy()
    
    mse = mean_squared_error(y_test.cpu().numpy(), y_pred)
    mae = mean_absolute_error(y_test.cpu().numpy(), y_pred)
    r2 = r2_score(y_test.cpu().numpy(), y_pred)
    
    # Win/Loss calculation
    wins = np.sum((y_pred > 0) & (y_test.cpu().numpy() > 0)) + np.sum((y_pred < 0) & (y_test.cpu().numpy() < 0))
    losses = np.sum((y_pred > 0) & (y_test.cpu().numpy() < 0)) + np.sum((y_pred < 0) & (y_test.cpu().numpy() > 0))
    total_samples = wins + losses
    win_percentage = (wins / total_samples) * 100 if total_samples > 0 else 0
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Total Wins': wins,
        'Total Losses': losses,
        'Win Percentage': win_percentage,
        'Samples': total_samples
    }
    
    return metrics

def plot_training_history(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/ndata_diff_lucky13_3070_oos.csv', 
        'data/ndata_diff_lucky13_3070.csv', #3
        'data/ndata_lucky_13_lag_3070_oos.csv', 
        'data/ndata_lucky13_lag_3070.csv', #5
        'new_model_Z_lucky13_3070_oos.csv',
        'new_model_Z_lucky13_3070.csv', #7,
        'data/Lucky13_3070_oos_3.csv',   
        'data/Lucky13_3070_3.csv',  #8
        'data/Lucky13_3070_oos_5.csv',   
        'data/Lucky13_3070_5.csv',  #10
    ]

    file_path = datafile[1]
    df = pd.read_csv(file_path)
    df = df.drop(columns=['outputC'])
    X = df.drop(columns=['output']).values
    y = df['output'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, criterion, and optimizer
    model = TSMixerKANModel(input_dim=input_dim, hidden_units=32, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    history = train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Test MSE: {metrics['MSE']}, Test MAE: {metrics['MAE']}, R2: {metrics['R2']}")
    print(f"Total Wins: {metrics['Total Wins']}, Total Losses: {metrics['Total Losses']}, Win Percentage: {metrics['Win Percentage']:.2f}%")
    print(f"Number of Samples: {metrics['Samples']}")

if __name__ == "__main__":
    main()
