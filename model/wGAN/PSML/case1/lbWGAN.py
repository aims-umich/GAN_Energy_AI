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

# Load data
data = pd.read_csv("../../../../data/CAISO_zone_1_.csv", index_col='time')


# Move "wind power", "solar power", "load power" to the end of the DataFrame
target_columns = ['wind_power', 'solar_power', 'load_power']
other_columns = [col for col in data.columns if col not in target_columns]

# Rearrange columns
data = data[other_columns + target_columns]

def smooth_dataset(df, window_size):
    df_smoothed = df.groupby(np.arange(len(df)) // window_size).mean()
    return df_smoothed

window_size = 5
data_smoothed = smooth_dataset(data, window_size)

x = data_smoothed.iloc[:, :8].values
y = data_smoothed.iloc[:, 8:11].values  # Assuming columns 9, 10, 11 are the three parameters

two_week = 1440 * 14 // window_size 
ten_days = 1440 * 10 // window_size

train_x, test_x = x[:two_week, :], x[two_week:two_week + ten_days, :]
train_y, test_y = y[:two_week, :], y[two_week:two_week + ten_days, :]

# Create a DataFrame for test_y
test_y_df = pd.DataFrame(test_y, columns=['wind_power', 'solar_power', 'load_power'])

# Save test_y to a CSV file
test_y_df.to_csv("/home/unabila/wgan/test_y.csv", index=False)

print("test_y saved to /home/unabila/wgan/test_y.csv")


# Scale data
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

train_x = x_scaler.fit_transform(train_x)
test_x = x_scaler.transform(test_x)

train_y = y_scaler.fit_transform(train_y)
test_y = y_scaler.transform(test_y)

# Define the VAE class
class VAE(nn.Module):
    def __init__(self, config, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        modules = [nn.Sequential(nn.Linear(config[i - 1], config[i]), nn.ReLU()) for i in range(1, len(config))]
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(config[-1], latent_dim)
        self.fc_var = nn.Linear(config[-1], latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, config[-1])
        modules = [nn.Sequential(nn.Linear(config[i], config[i - 1]), nn.ReLU()) for i in range(len(config) - 1, 1, -1)]
        modules.append(nn.Sequential(nn.Linear(config[1], config[0]), nn.Sigmoid()))
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logVar = self.fc_var(result)
        return mu, logVar

    def decode(self, x):
        result = self.decoder(x)
        return result

    def reparameterize(self, mu, logVar):
        std = torch.exp(0.5 * logVar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        output = self.decode(z)
        return output, z, mu, logVar

# Initialize and train the VAE
train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float()), batch_size=256, shuffle=False)
vae_model = VAE([8, 400, 400, 400, 10], latent_dim=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 250
learning_rate = 0.0001
vae_model = vae_model.to(device)
optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)

hist = np.zeros(num_epochs)
for epoch in range(num_epochs):
    total_loss = 0
    loss_ = []
    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        output, z, mu, logVar = vae_model(x)
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(output, x) + kl_divergence
        loss.backward()
        optimizer.step()
        loss_.append(loss.item())
    hist[epoch] = sum(loss_)
    print(f'[{epoch + 1}/{num_epochs}] Loss: {sum(loss_)}')

vae_model.eval()
_, VAE_train_x, _, _ = vae_model(torch.from_numpy(train_x).float().to(device))
_, VAE_test_x, _, _ = vae_model(torch.from_numpy(test_x).float().to(device))

def sliding_window(x, y, lookback, lookforward):
    x_ = []
    y_ = []
    y_gan = []
    for i in range(lookback, x.shape[0] - lookforward + 1):
        tmp_x = x[i - lookback: i, :]
        tmp_y = y[i + lookforward - 1]
        tmp_y_gan = y[i - lookback: i + lookforward]
        x_.append(tmp_x)
        y_.append(tmp_y)
        y_gan.append(tmp_y_gan)
    x_ = torch.from_numpy(np.array(x_)).float()
    y_ = torch.from_numpy(np.array(y_)).float()
    y_gan = torch.from_numpy(np.array(y_gan)).float()
    return x_, y_, y_gan

train_x = np.concatenate((train_x, VAE_train_x.cpu().detach().numpy()), axis=1)
test_x = np.concatenate((test_x, VAE_test_x.cpu().detach().numpy()), axis=1)

# Loop through lookback values
lookback_values = [2*60 // 5, 4*60 // 5, 6*60 // 5, 8*60 // 5, 10*60 // 5, 12*60 // 5]
#lookback_values = [3, 5]
lookforward = 144

results = []

for lookback in lookback_values:
    train_x_slide, train_y_slide, train_y_gan = sliding_window(train_x, train_y, lookback, lookforward)
    test_x_slide, test_y_slide, test_y_gan = sliding_window(test_x, test_y, lookback, lookforward)

    class Generator(nn.Module):
        def __init__(self, input_size):
            super(Generator, self).__init__()
            self.gru = nn.GRU(input_size, 64, num_layers=2, batch_first=True, dropout=0.1)
            self.linear_1 = nn.Linear(64, 3)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            out, _ = self.gru(x)
            out = self.dropout(out)
            out = self.linear_1(out[:, -1, :])
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(lookback+lookforward, 32, kernel_size=5, stride=1, padding='same')
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding='same')
            self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding='same')
            
            # Calculate the output size of the last conv layer 
            conv_output_size = 128*3 
            
            self.linear1 = nn.Linear(conv_output_size, 220)
            self.linear2 = nn.Linear(220, 220)
            self.linear3 = nn.Linear(220, 3)
            self.leaky = nn.LeakyReLU(0.01)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            conv1 = self.conv1(x)
            conv1 = self.leaky(conv1)
            conv2 = self.conv2(conv1)
            conv2 = self.leaky(conv2)
            conv3 = self.conv3(conv2)
            conv3 = self.leaky(conv3)
            
            # Flatten the tensor while keeping the batch dimension intact
            flatten_x = conv3.view(conv3.size(0), -1)
            
            out_1 = self.linear1(flatten_x)
            out_1 = self.leaky(out_1)
            out_2 = self.linear2(out_1)
            out_2 = self.relu(out_2)
            out = self.linear3(out_2)
            return out

    # Initialize and train the WGAN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    learning_rate = 0.0001
    num_epochs = 250 ##################
    critic_iterations = 5
    weight_clip = 0.01

    trainDataloader = DataLoader(TensorDataset(train_x_slide, train_y_gan), batch_size=batch_size, shuffle=False)

    modelG = Generator(train_x_slide.shape[2]).to(device)
    modelD = Discriminator().to(device)


    optimizerG = torch.optim.Adam(modelG.parameters(), lr=learning_rate, betas=(0.0, 0.9), weight_decay=1e-3)
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=learning_rate, betas=(0.0, 0.9), weight_decay=1e-3)

    histG = np.zeros(num_epochs)
    histD = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        loss_G = []
        loss_D = []
        for (x, y) in trainDataloader:
            x = x.to(device)
            y = y.to(device)
    
            fake_data = modelG(x)
            fake_data = torch.cat([y[:, :lookback + lookforward - 1, :], fake_data.reshape(-1, 1, 3)], axis=1)  # Adjust for 3 values
            critic_real = modelD(y)
            critic_fake = modelD(fake_data)
            lossD = -(torch.mean(critic_real) - torch.mean(critic_fake))
            modelD.zero_grad()
            lossD.backward(retain_graph=True)
            optimizerD.step()
    
            output_fake = modelD(fake_data)
            lossG = -torch.mean(output_fake)
            modelG.zero_grad()
            lossG.backward()
            optimizerG.step()
    
            loss_D.append(lossD.item())
            loss_G.append(lossG.item())
        histG[epoch] = sum(loss_G)
        histD[epoch] = sum(loss_D)
        print(f'[{epoch + 1}/{num_epochs}] LossD: {sum(loss_D)} LossG:{sum(loss_G)}')    


    modelG.eval()
    pred_y_test = modelG(test_x_slide.to(device))

    y_test_true = y_scaler.inverse_transform(test_y_slide.reshape(-1, 3))
    y_test_pred = y_scaler.inverse_transform(pred_y_test.cpu().detach().numpy())

    # Save predictions to CSV
    pd.DataFrame(y_test_pred).to_csv(f'pred_test_data_wgan64_lookback_{lookback}.csv', index=False)

    # Calculate metrics
    metrics = {}
    for i, param in enumerate(["wind power", "solar power", "load power"]):
        MSE = mean_squared_error(y_test_true[:, i], y_test_pred[:, i])
        RMSE = math.sqrt(MSE)
        MAE = mean_absolute_error(y_test_true[:, i], y_test_pred[:, i])

        # Store results in dictionary
        metrics[param] = {'RMSE': RMSE, 'MAE': MAE}

        # Plot predictions
        plt.figure(figsize=(12, 8))
        plt.scatter(range(len(y_test_true[:, i])), y_test_true[:, i], color='black', label=f'Actual {param}', s=10)
        plt.scatter(range(len(y_test_true[:, i])), y_test_pred[:, i], color='blue', label=f'Predict {param}', s=10)
        plt.title(f'WGAN prediction testing dataset ({param}) - Lookback {lookback} mins')
        plt.ylabel(param, fontsize=20)
        plt.xlabel('Minutes', fontsize=20)
        plt.legend(loc='upper right')
        plt.savefig(f'prediction_wgan_lookback_{lookback}_{param.replace(" ", "_")}.png')

    # Append results for the current lookback
    results.append({
        'Lookback': lookback,
        'RMSE_wind': metrics["wind power"]['RMSE'],
        'MAE_wind': metrics["wind power"]['MAE'],

        'RMSE_solar': metrics["solar power"]['RMSE'],
        'MAE_solar': metrics["solar power"]['MAE'],

        'RMSE_load': metrics["load power"]['RMSE'],
        'MAE_load': metrics["load power"]['MAE']
    })


# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('wgan12h_lookback_results.csv', index=False)

# Print the elapsed time
end_time = time.time()
print(f"Total time elapsed: {(end_time - start_time) / 60:.2f} minutes")
