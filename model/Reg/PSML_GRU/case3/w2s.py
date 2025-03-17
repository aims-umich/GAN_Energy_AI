import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import math

# Start the timer
start_time = time.time()

# Define the sliding window function
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

# Define the GRU model
class GRUmodel(nn.Module):
    def __init__(self, input_size):
        super(GRUmodel, self).__init__()
        self.gru = nn.GRU(input_size, 64, num_layers=3, batch_first=True, dropout=0.1)
        self.linear_1 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.dropout(x[:, -1, :])
        x = self.linear_1(x)
        return x

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
df_smoothed = smooth_dataset(data, window_size)
    
# Split features and targets
x = df_smoothed.iloc[:, :8].values
y = df_smoothed.iloc[:, 8:11].values

two_week = 1440 * 14 // window_size 
ten_days = 1440 * 10 // window_size

july = 1440 * 181 // window_size

train_x, test_x = x[:two_week, :], x[july:july + ten_days, :]
train_y, test_y = y[:two_week, :], y[july:july + ten_days, :]


# Scale data
x_scaler = MinMaxScaler(feature_range=(0, 1))
y_scaler = MinMaxScaler(feature_range=(0, 1))

train_x = x_scaler.fit_transform(train_x)
test_x = x_scaler.transform(test_x)

train_y = y_scaler.fit_transform(train_y)
test_y = y_scaler.transform(test_y)

# Lookback values and lookforward
#lookback_values = [3, 5]
lookback_values = [12*60 // 5]
lookforward = 720 // 5

results = []

# Loop through lookback values
for lookback in lookback_values:
    # Prepare data with sliding window
    train_x_slide, train_y_slide, _ = sliding_window(train_x, train_y, lookback, lookforward)
    test_x_slide, test_y_slide, _ = sliding_window(test_x, test_y, lookback, lookforward)

    # Move data to the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x_slide = train_x_slide.to(device)
    train_y_slide = train_y_slide.to(device)
    test_x_slide = test_x_slide.to(device)

    # Initialize the model, loss function, and optimizer
    modelG = GRUmodel(input_size=train_x_slide.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelG.parameters(), lr=0.001)

    # Training the model
    num_epochs = 300
    modelG.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = modelG(train_x_slide)
        loss = criterion(output, train_y_slide)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Lookback: {lookback} - Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    modelG.eval()
    pred_y_test = modelG(test_x_slide)

    y_test_true = y_scaler.inverse_transform(test_y_slide.cpu().numpy().reshape(-1, 3))
    y_test_pred = y_scaler.inverse_transform(pred_y_test.cpu().detach().numpy())

    pd.DataFrame(y_test_pred).to_csv(f'pred_test_data_w2s_{lookback}_lookback.csv', index=False)

    # Calculate metrics and plot
    metrics = {}
    for i, param in enumerate(["wind power", "solar power", "load power"]):
        MSE = mean_squared_error(y_test_true[:, i], y_test_pred[:, i])
        RMSE = math.sqrt(MSE)
        MAE = mean_absolute_error(y_test_true[:, i], y_test_pred[:, i])
        R2 = r2_score(y_test_true[:, i], y_test_pred[:, i])

        # Store results in dictionary
        metrics[param] = {'RMSE': RMSE, 'MAE': MAE, 'R2': R2}

        # Plot predictions
        plt.figure(figsize=(12, 8))
        plt.scatter(range(len(y_test_true[:, i])), y_test_true[:, i], color='black', label=f'Actual {param}', s=10)
        plt.scatter(range(len(y_test_true[:, i])), y_test_pred[:, i], color='blue', label=f'Predict {param}', s=10)
        plt.title(f'GAN prediction testing dataset ({param}) - Lookback {lookback} mins')
        plt.ylabel(param, fontsize=20)
        plt.xlabel('Minutes', fontsize=20)
        plt.legend(loc='upper right')
        plt.savefig(f'prediction_gru_lookfd_1w2s_{lookback}_{param.replace(" ", "_")}.png')

    # Append results for the current lookback
    results.append({
        'lookback': lookback,
        'RMSE_wind': metrics["wind power"]['RMSE'],
        'MAE_wind': metrics["wind power"]['MAE'],
        'R2_wind': metrics["wind power"]['R2'],
        'RMSE_solar': metrics["solar power"]['RMSE'],
        'MAE_solar': metrics["solar power"]['MAE'],
        'R2_solar': metrics["solar power"]['R2'],
        'RMSE_load': metrics["load power"]['RMSE'],
        'MAE_load': metrics["load power"]['MAE'],
        'R2_load': metrics["load power"]['R2']
    })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('gru_lookfor_w2s_results.csv', index=False)

# Print the elapsed time
end_time = time.time()
print(f"Total time elapsed: {(end_time - start_time) / 60:.2f} minutes")
