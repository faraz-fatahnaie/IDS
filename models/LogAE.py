import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from utils import parse_data, save_dataframe  # Make sure you have the 'parse_data' function defined in 'utils.py'
from tqdm import tqdm
from pathlib import Path


# Define the Logarithmic Layer (LOGN)
# class LogarithmicLayer(nn.Module):
#     def __init__(self, input_size):
#         super(LogarithmicLayer, self).__init__()
#         self.log_weight = nn.Parameter(torch.ones(input_size))  # Logarithmic weight
#         self.bias = nn.Parameter(torch.zeros(input_size))  # Bias
#
#     def forward(self, x):
#         log_transformed = self.log_weight * x + self.bias
#         return log_transformed

class LogarithmicLayer(nn.Module):
    def __init__(self, input_size):
        super(LogarithmicLayer, self).__init__()

        # Initialize the logarithmic weight and bias
        self.log_weight = nn.Parameter(torch.ones(input_size))  # Logarithmic weight
        self.bias = nn.Parameter(torch.zeros(input_size))  # Bias

    def forward(self, x):
        # Apply logarithmic transformation with W = e^(3*W)
        log_weight_exp = torch.exp(3 * self.log_weight)
        log_transformed = torch.log(x + 1) / log_weight_exp - self.bias
        return log_transformed


# Define the Exponential Layer
class ExponentialLayer(nn.Module):
    def __init__(self, input_size):
        super(ExponentialLayer, self).__init__()
        self.exp_weight = nn.Parameter(torch.ones(input_size))  # Exponential weight
        self.bias = nn.Parameter(torch.zeros(input_size))  # Bias

    def forward(self, x):
        exp_transformed = torch.exp(self.exp_weight * x + self.bias)
        return exp_transformed


# Define the Logarithmic Autoencoder (LogAE) model
class LogAE(nn.Module):
    def __init__(self, input_size, hidden_dims=(30, 15, 30)):
        super(LogAE, self).__init__()

        self.log_layer = LogarithmicLayer(input_size)
        self.exp_layer = ExponentialLayer(input_size)

        # Autoencoder (AE)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dims[0], dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1], dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2], dtype=torch.float64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1], dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0], dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_size, dtype=torch.float64)
        )

    def forward(self, x):
        # Apply logarithmic transformation
        x = self.log_layer(x)

        # Encode data
        encoding = self.encoder(x)

        # Decode data
        decoding = self.decoder(encoding)

        x = self.exp_layer(decoding)

        return x


if __name__ == "__main__":
    # Example usage
    input_size = 196  # Adjust based on your input data size
    model = LogAE(input_size)

    # Define loss function as described (you can adjust the lambda value)
    lambda_value = 0.01
    criterion = nn.L1Loss()
    log_layer_parameters = list(model.log_layer.parameters())
    loss_params = log_layer_parameters[0]  # Logarithmic layer parameters

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.1,
        patience=5, threshold=100,
        min_lr=0.000001, verbose=True)

    # Load your data and create a DataLoader
    df = pd.read_csv('C:\\Users\\Faraz\\PycharmProjects\\IDS\dataset\\UNSW_NB15\\file\preprocessed\\train_binary.csv')
    X_train, y_train = parse_data(df, dataset_name='UNSW_NB15', mode='np', classification_mode='binary')
    print(f'train shape: x=>{X_train.shape}, y=>{y_train.shape}')
    train_dataset = TensorDataset(torch.tensor(X_train).reshape((-1, 196)), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
    reconstruct_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=False)

    # Training loop
    for epoch in range(100):
        total_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, batch_data in progress_bar:
            inputs, labels = batch_data
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, inputs) + lambda_value * torch.sum(torch.abs(loss_params))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_description(f'Epoch {epoch + 1}/{100}, Loss: {total_loss / (step + 1):.4f}')
        scheduler.step(total_loss)

    # After training, you can use the trained model for encoding and decoding
    reconstructed = []
    with torch.no_grad():
        progress_bar = tqdm(enumerate(reconstruct_loader), total=len(reconstruct_loader))
        for step, batch_data in progress_bar:
            inputs, labels = batch_data
            output = model(inputs)
            reconstructed.append(output.squeeze(0).numpy())

    data = np.concatenate((np.array(reconstructed), y_train), axis=1)
    reconstructed_df = pd.DataFrame(data=data, columns=df.columns)
    X, y = parse_data(reconstructed_df, dataset_name='UNSW_NB15', mode='np', classification_mode='binary')
    print(f'train shape: x=>{X.shape}, y=>{y.shape}')
    save_path = Path('C:\\Users\\Faraz\\PycharmProjects\\IDS\\dataset\\UNSW_NB15\\file\\preprocessed')
    save_dataframe(reconstructed_df, save_path, 'train', 'binary')
