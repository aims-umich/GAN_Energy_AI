import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.stats import entropy  # KL Divergence Calculation
import time

# Start the timer
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
train_data = pd.read_csv("train_1k.csv").values
test_data = pd.read_csv("test_all.csv").values

# Split features and target
train_x, train_y = train_data[:, :-1], train_data[:, -1]
test_x, test_y = test_data[:, :-1], test_data[:, -1]

# Scale data
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

train_x = x_scaler.fit_transform(train_x)
test_x = x_scaler.transform(test_x)
train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
test_y = y_scaler.fit_transform(test_y.reshape(-1, 1))

# Convert data to PyTorch tensors
train_x_tensor = torch.from_numpy(train_x).float()
train_y_tensor = torch.from_numpy(train_y).float()
test_x_tensor = torch.from_numpy(test_x).float()
test_y_tensor = torch.from_numpy(test_y).float()

# Define models and training process in an Optuna objective function
def objective(trial):

    # Hyperparameter search space
    latent_dim = trial.suggest_int('latent_dim', 10, 100)
    lr_gan = trial.suggest_loguniform('lr_gan', 1e-5, 1e-2)
    num_epochs_gan = trial.suggest_int('num_epochs_gan', 2000, 8000)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    num_nodes = trial.suggest_int('num_nodes', 8, 128)
    batch_size = trial.suggest_int('batch_size', 8, 256)

    # Generator with dynamic number of layers and nodes
    class Generator(nn.Module):
        def __init__(self, input_size, latent_dim, num_layers, num_nodes):
            super(Generator, self).__init__()
            layers = [nn.Linear(input_size, num_nodes), nn.ReLU()]

            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(num_nodes, num_nodes), nn.ReLU()])

            layers.append(nn.Linear(num_nodes, 1))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Discriminator with dynamic number of layers and nodes
    class Discriminator(nn.Module):
        def __init__(self, input_size, num_layers, num_nodes):
            super(Discriminator, self).__init__()
            layers = [nn.Linear(input_size + 1, num_nodes), nn.ReLU()]

            for _ in range(num_layers - 1):
                layers.extend([nn.Linear(num_nodes, num_nodes), nn.ReLU()])

            layers.append(nn.Linear(num_nodes, 1))
            self.model = nn.Sequential(*layers)

        def forward(self, x, y):
            x = torch.cat([x, y], dim=1)
            return self.model(x)

    # Initialize models and optimizers
    modelG = Generator(train_x_tensor.shape[1], latent_dim, num_layers, num_nodes).to(device)
    modelD = Discriminator(train_x_tensor.shape[1], num_layers, num_nodes).to(device)

    optimizerG = torch.optim.Adam(modelG.parameters(), lr=lr_gan)
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=lr_gan)

    # WGAN training loop
    lambda_gp = 10
    for epoch in range(num_epochs_gan):
        for x_batch, y_batch in DataLoader(TensorDataset(train_x_tensor, train_y_tensor), batch_size=batch_size, shuffle=True):
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
        pred_test_y = modelG(test_x_tensor.to(device)).cpu().numpy()

    # Transform predictions and true values back to original scale
    test_y_true = y_scaler.inverse_transform(test_y)  
    test_y_pred = y_scaler.inverse_transform(pred_test_y)

    # Calculate MAPE
    mape = mean_absolute_percentage_error(test_y_true, test_y_pred)
    
    # Calculate R2
    r2 = r2_score(test_y_true, test_y_pred)

    # KL Divergence Calculation
    num_bins = 50
    hist_true, bin_edges = np.histogram(test_y_true, bins=num_bins, density=True)
    hist_pred, _ = np.histogram(test_y_pred, bins=bin_edges, density=True)

    # Avoid zero probabilities
    hist_true += 1e-10
    hist_pred += 1e-10

    kl_div = entropy(hist_pred, hist_true)  # KL divergence calculation

    # Print results
    print(f"Trial {trial.number}: MAPE = {mape:.4f}, R2 = {r2:.4f}, KL Divergence = {kl_div:.4f}")

    return mape  # Optuna minimizes MAPE

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Best trial information
best_trial = study.best_trial
print(f'Best trial number: {best_trial.number}')
print(f'Best hyperparameters: {best_trial.params}')
print(f'Best 1k MAPE: {best_trial.value:.4f}')

# Print elapsed time
end_time = time.time()
print(f"Total time elapsed: {(end_time - start_time) / 60:.2f} minutes")
