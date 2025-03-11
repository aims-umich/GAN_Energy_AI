# -*- coding: utf-8 -*-
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from itertools import product

# Define the regression model
def build_regression_model(input_dim, num_layers, num_nodes):
    model = Sequential()
    model.add(Dense(num_nodes, activation='relu', input_shape=(input_dim,)))
    
    for _ in range(1, num_layers):
        model.add(Dense(num_nodes, activation='relu'))
    
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

# Train the regression model
def train_regression_model(model, x_train, y_train, epochs, batch_size):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Plot comparison between true and predicted values
def plot_comparison(df1, df2, num_points_from_file2, item):
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)

    df1_sampled = df1.sample(n=num_points_from_file2, random_state=42)
    num_cols = len(df1_sampled.columns) - 1
    header = ["Diameter", "Length", "Pressure", "Mass flux", "Temperature", "CHF"]

    fig, axes = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))

    for i in range(num_cols):
        axes[i].scatter(df1_sampled.iloc[:, i], df1_sampled.iloc[:, -1], color='red', label='Real')
        axes[i].scatter(df2.iloc[:, i], df2.iloc[:, -1], color='blue', label='Predicted')
        axes[i].set_title(f'{header[i]} vs CHF')
        axes[i].set_xlabel(f'{header[i]}')
        axes[i].set_ylabel('CHF')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'comparison_plot_{item}.png')

# Define configurations
learning_rate = [1e-3]
num_layers = [3,4,6]
num_nodes = [200,250,300]
batch_size = [32, 64]
configs = list(product(learning_rate, num_layers, num_nodes, batch_size))
MS_list = []

# Load and preprocess data
train_data = pd.read_csv("/home/unabila/CondGan/1h/train_1h.csv").values
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
x_train = train_data[:, :-1]
y_train = train_data[:, -1]

test_data = pd.read_csv("/home/unabila/CondGan/1h/test_all.csv").values
Y_test = test_data[:, -1]
test_data = scaler.transform(test_data)
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

x_test_data = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)
test_data = scaler.inverse_transform(x_test_data)

# Train and evaluate the model for each configuration
for item in configs:
    model = build_regression_model(input_dim=x_train.shape[1], num_layers=item[1], num_nodes=item[2])
    epochs = 100  # Number of epochs to train
    trained_model = train_regression_model(model, x_train, y_train, epochs, batch_size=item[3])

    Y_pred = trained_model.predict(x_test)

    generated_data = np.concatenate((x_test, Y_pred), axis=1)
    generated_data = scaler.inverse_transform(generated_data)
    Y_pred = generated_data[:, -1]

    mape = mean_absolute_percentage_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    MS_list.append((mape, r2))

    generated_df = pd.DataFrame(generated_data, columns=['Diameter', 'Length', 'Pressure', 'Mass flux', 'Temperature', 'CHF'])
    generated_df.to_csv(f'generated_samples_{item}.csv', index=False)
    plot_comparison(test_data, generated_data, 4290, item)

# Store and save the results
gridres = pd.DataFrame(configs, columns=['learning_rate', 'num_layers', 'num_nodes', 'batch_size'])
results_df = pd.DataFrame(MS_list, columns=['MAPE', 'R2'])
final_result = pd.concat([gridres, results_df], axis=1)
final_result.to_csv('results_regression.csv', index=False)
gridres['Mape'] = [item[0] for item in MS_list]
gridres['R2'] = [item[1] for item in MS_list]
gridres = gridres.sort_values(by='R2', ascending=False)
print(gridres.head())
