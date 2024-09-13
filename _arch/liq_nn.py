import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define the Liquid Neural Network class
class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidNeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Initialize dynamic weights to match input size
        self.dynamic_weight = torch.randn(input_size, requires_grad=True)
    
    def forward(self, x):
        # Apply dynamic adaptation to the weights
        dynamic_layer = torch.tanh(self.fc1(x * self.dynamic_weight))
        output = self.fc2(dynamic_layer)
        return output
    
    def adapt(self, x):
        # Update the dynamic weights based on input
        with torch.no_grad():
            self.dynamic_weight += 0.01 * torch.mean(x, dim=0)

# Function to load data from a CSV file
def load_data(filepath):
    df = pd.read_csv(filepath)
    features = df.drop(columns=['output'])
    target = df['output']
    return features.values, target.values

# Function to train the model with early stopping and aggressive monitoring
def train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs=100, patience=10):
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Convert to torch tensors
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Adapt the model
        model.adapt(inputs)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(X_val, dtype=torch.float32)
            val_targets = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()
        
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Plotting the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
def evaluate_model(model, X_test, y_test, margin=0.01):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        outputs = model(inputs)
        
        mse = mean_squared_error(targets.numpy(), outputs.numpy())
        r2 = r2_score(targets.numpy(), outputs.numpy())
        
        wins = 0
        losses = 0
        
        for i in range(len(targets)):
            
            predict = outputs[i].item()
            target = targets[i].item()
            
            if (predict > 0 and target > 0) or (predict < 0 and target < 0):
                wins += 1
            elif (predict > 0 and target < 0) or (predict < 0 and target > 0):
                losses += 1
            elif (predict == 0 and target == 0):
                wins += 1
            elif (predict == 0 and target != 0):
                losses += 1
        
        total_samples = wins + losses
        win_percentage = (wins / total_samples) * 100
        
        print(f"Total: {total_samples}, Wins: {wins}, Losses: {losses}, Win Percentage: {win_percentage:.2f}%")
        
    return mse, r2


# Main function
def main():
    # Configuration
    hidden_size = 512
    output_size = 1
    learning_rate = 0.0005
    epochs = 100
    patience = 10
    filepath = 'data/Lucky13_3070.csv'
    
    # Load data
    X, y = load_data(filepath)
    
    df = pd.read_csv(filepath)
    features = df.drop(columns=['output'])
    #SDLR310,SDBB91,SDKC91,SDKC9,ROC,ATR54,ATR53,ATR52,ATR51,ATR5,ATR21,ATR2,RSI,STOK1,output,outputC
    #features = df[['RSI', 'ATR53', 'STOK1']]
    target = df['output']
    
    X = features.values
    Y = target.values
    
    input_size = X.shape[1]  # Dynamically set input_size based on data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize model, criterion, and optimizer
    model = LiquidNeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs=epochs, patience=patience)
    
    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error on Test Data: {mse:.4f}')
    print(f'R2: {r2:.4f}')
    print(" ")

if __name__ == '__main__':
    main()
