import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import math

# Start the timer
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load and preprocess data
train_data = pd.read_csv("../train_p_t.csv").values
test_data = pd.read_csv("../test_p_t.csv").values

# Split features and target
train_x, train_y = train_data[:, :-1], train_data[:, -1]
test_X, test_y = test_data[:, :-1], test_data[:, -1]

# Scale data
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

train_x = x_scaler.fit_transform(train_x)
test_x = x_scaler.transform(test_X)
train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
test_y = y_scaler.fit_transform(test_y.reshape(-1, 1))

# Convert data to PyTorch tensors
train_x_tensor = torch.from_numpy(train_x).float()
train_y_tensor = torch.from_numpy(train_y).float()
test_x_tensor = torch.from_numpy(test_x).float()
test_y_tensor = torch.from_numpy(test_y).float()

# Prepare DataLoader
batch_size = 256

# Variable to store best RMSE and corresponding predictions
best_mape = float('inf')
best_test_y_pred = None

# MAPE Calculation
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Define models and training process in an Optuna objective function
def objective(trial):


    # Hyperparameter search space
    latent_dim = trial.suggest_int('latent_dim', 10, 100)
    latent_dim = trial.suggest_int('latent_dim', 10, 100)
    lr_vae = trial.suggest_loguniform('lr_vae', 1e-5, 1e-2)
    lr_gan = trial.suggest_loguniform('lr_gan', 1e-5, 1e-2)
    num_epochs_vae = trial.suggest_int('num_epochs_vae', 200, 1000)
    num_epochs_gan = trial.suggest_int('num_epochs_gan', 500, 5000)
    
    # New search space for number of layers and nodes in Generator
    num_layers = trial.suggest_int('num_layers', 2, 6)  # Between 2 and 5 layers
    num_nodes = trial.suggest_int('num_nodes', 64, 256)  # Number of nodes per layer

    # VAE Model
    class VAE(nn.Module):
        def __init__(self, config, latent_dim):
            super(VAE, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(*[nn.Sequential(nn.Linear(config[i - 1], config[i]), nn.ReLU()) for i in range(1, len(config))])
            self.fc_mu = nn.Linear(config[-1], latent_dim)
            self.fc_var = nn.Linear(config[-1], latent_dim)

            # Decoder
            self.decoder_input = nn.Linear(latent_dim, config[-1])
            self.decoder = nn.Sequential(*[nn.Sequential(nn.Linear(config[i], config[i - 1]), nn.ReLU()) for i in range(len(config) - 1, 1, -1)])
            self.decoder.add_module('output', nn.Sequential(nn.Linear(config[1], config[0]), nn.Sigmoid()))

        def encode(self, x):
            result = self.encoder(x)
            mu = self.fc_mu(result)
            logVar = self.fc_var(result)
            return mu, logVar

        def decode(self, x):
            return self.decoder(x)

        def reparameterize(self, mu, logVar):
            std = torch.exp(0.5 * logVar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def forward(self, x):
            mu, logVar = self.encode(x)
            z = self.reparameterize(mu, logVar)
            return self.decode(z), z, mu, logVar

    # Generator with dynamic number of layers and nodes
    class Generator(nn.Module):
        def __init__(self, input_size, latent_dim, num_layers, num_nodes):
            super(Generator, self).__init__()
            layers = [nn.Linear(input_size, num_nodes), nn.ReLU()]  # First layer with input size

            for _ in range(num_layers - 1):  # Add hidden layers dynamically
                layers.extend([nn.Linear(num_nodes, num_nodes), nn.ReLU()])
            
            layers.append(nn.Linear(num_nodes, 1))  # Output is a single value (CHF)
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Discriminator for WGAN
    class Discriminator(nn.Module):
        def __init__(self, input_size):
            super(Discriminator, self).__init__()
            self.fc1 = nn.Linear(input_size + 1, 128)  # 1 is the output CHF
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)  # Output a single value (Wasserstein distance)

        def forward(self, x, y):
            x = torch.cat([x, y], dim=1)  # Concatenate input and output (CHF)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)  # Linear output for Wasserstein distance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae_model = VAE([train_x.shape[1], 400, 400, 400, latent_dim], latent_dim).to(device)
    modelG = Generator(train_x.shape[1] + latent_dim, latent_dim, num_layers, num_nodes).to(device)
    modelD = Discriminator(train_x.shape[1] + latent_dim).to(device)

    optimizerG = torch.optim.Adam(modelG.parameters(), lr=lr_gan)
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=lr_gan)
    optimizerVAE = torch.optim.Adam(vae_model.parameters(), lr=lr_vae)

    # VAE training
    for epoch in range(num_epochs_vae):
        vae_model.train()
        total_loss = 0
        for (x,) in DataLoader(TensorDataset(train_x_tensor), batch_size=batch_size):
            x = x.to(device)
            optimizerVAE.zero_grad()
            reconstructed_x, z, mu, logVar = vae_model(x)
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(reconstructed_x, x) + kl_divergence
            loss.backward()
            optimizerVAE.step()
            total_loss += loss.item()

    vae_model.eval()
    _, train_z, _, _ = vae_model(train_x_tensor.to(device))
    _, test_z, _, _ = vae_model(test_x_tensor.to(device))

    # Concatenate latent features with original features
    train_x_combined = torch.cat([train_x_tensor, train_z.cpu().detach()], dim=1)
    test_x_combined = torch.cat([test_x_tensor, test_z.cpu().detach()], dim=1)

    # WGAN training loop
    lambda_gp = 10
    for epoch in range(num_epochs_gan):
        for x_batch, y_batch in DataLoader(TensorDataset(train_x_combined, train_y_tensor), batch_size=batch_size, shuffle=True):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Train Discriminator
            fake_y = modelG(x_batch)
            real_loss = modelD(x_batch, y_batch).mean()
            fake_loss = modelD(x_batch, fake_y).mean()

            # Calculate gradient penalty
            alpha = torch.rand(y_batch.size(0), 1).to(device)
            interpolated_y = (alpha * y_batch + (1 - alpha) * fake_y).requires_grad_(True)
            d_interpolated = modelD(x_batch, interpolated_y)
            grad_outputs = torch.ones_like(d_interpolated).to(device)
            gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated_y, grad_outputs=grad_outputs,
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            d_loss = fake_loss - real_loss + lambda_gp * gp
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # Train Generator
            fake_y = modelG(x_batch)
            g_loss = -modelD(x_batch, fake_y).mean()
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

    # Evaluate the model
    modelG.eval()
    with torch.no_grad():
        pred_test_y = modelG(test_x_combined.to(device))
        pred_test_y = pred_test_y.cpu().numpy()

    # Transform predictions and true values
    test_y_true = y_scaler.inverse_transform(test_y)  # Inverse transform for true values
    test_y_pred = y_scaler.inverse_transform(pred_test_y)  # Inverse transform for predicted values


    mape = mean_absolute_percentage_error(test_y_true, test_y_pred)
    # Calculate R2
    r2 = r2_score(test_y_true, test_y_pred)
    print(f"R-squared (R): {r2:.4f}")
    # Print MAPE and trial number
    print(f"Trial {trial.number}: MAPE = {mape:.4f}")
    return mape  # Return RMSE for Optuna optimization

# Running the optimization with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Best trial information
best_trial = study.best_trial
print(f'Best trial number: {best_trial.number}')
print(f'Best hyperparameters: {best_trial.params}')
print(f'Best pt MAPE: {best_trial.value}')

# Print the elapsed time
end_time = time.time()
print(f"Total time elapsed: {(end_time - start_time) / 60:.2f} minutes")

