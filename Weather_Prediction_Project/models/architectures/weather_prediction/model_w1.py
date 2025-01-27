import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os

class Modelw1(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 16, 4], output_size=1, learning_rate=0.001):
        super(Modelw1, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], output_size)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, x):
        return self.model(x)
    
    def train_model(self, train_loader, epochs=100):
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            epoch_loss /= len(train_loader.dataset)
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    def test_model(self, test_loader):
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.forward(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                total_loss += loss.item() * inputs.size(0)
        total_loss /= len(test_loader.dataset)
        print(f"Test Loss: {total_loss:.4f}")
        return total_loss
    
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
    
    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No weights found at {filepath}")
        self.load_state_dict(torch.load(filepath))
        self.eval()
        print(f"Weights loaded from {filepath}")
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            return probabilities

