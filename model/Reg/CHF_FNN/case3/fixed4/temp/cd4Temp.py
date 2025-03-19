# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from itertools import product

# Define the regression model with two outputs
def build_regression_model(input_dim, num_layers, num_nodes):
    model = Sequential()
    model.add(Dense(num_nodes, activation='relu', input_shape=(input_dim,)))
    
    for _ in range(1, num_layers):
        model.add(Dense(num_nodes, activation='relu'))
    
    model.add(Dense(2, activation='linear'))  # 2 output columns
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

# Train the regression model
def train_regression_model(model, x_train, y_train, epochs, batch_size):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Define configurations
learning_rate = [1e-3]
num_layers = [3, 4, 5]
num_nodes = [200, 250, 300]
batch_size = [128]
configs = list(product(learning_rate, num_layers, num_nodes, batch_size))
MS_list = []

# Load and preprocess data
train_data = pd.read_csv("xTemp.csv").values
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
x_train = train_data[:, :-2]  # All columns except last 2 as input
y_train = train_data[:, -2:]  # Last 2 columns as output

test_data = pd.read_csv("test_all.csv").values
Y_test = test_data[:, -2:]  # Last 2 columns as output
test_data = scaler.transform(test_data)
x_test = test_data[:, :-2]
y_test = test_data[:, -2:]

x_test_data = np.concatenate((x_test, y_test), axis=1)
test_data = scaler.inverse_transform(x_test_data)

# Train and evaluate the model for each configuration
for item in configs:
    model = build_regression_model(input_dim=x_train.shape[1], num_layers=item[1], num_nodes=item[2])
    epochs = 100  # Number of epochs to train
    trained_model = train_regression_model(model, x_train, y_train, epochs, batch_size=item[3])

    Y_pred = trained_model.predict(x_test)
    generated_data = np.concatenate((x_test, Y_pred), axis=1)
    generated_data = scaler.inverse_transform(generated_data)
    
    Y_pred_1 = generated_data[:, -2]  # First predicted column
    Y_pred_2 = generated_data[:, -1]  # Second predicted column
    
    mape_1 = mean_absolute_percentage_error(Y_test[:, 0], Y_pred_1)
    r2_1 = r2_score(Y_test[:, 0], Y_pred_1)
    
    mape_2 = mean_absolute_percentage_error(Y_test[:, 1], Y_pred_2)
    r2_2 = r2_score(Y_test[:, 1], Y_pred_2)
    
    MS_list.append((mape_1, r2_1, mape_2, r2_2))

# Store and save the results
gridres = pd.DataFrame(configs, columns=['learning_rate', 'num_layers', 'num_nodes', 'batch_size'])
results_df = pd.DataFrame(MS_list, columns=['MAPE_1', 'R2_1', 'MAPE_2', 'R2_2'])
final_result = pd.concat([gridres, results_df], axis=1)
final_result.to_csv('results_regressionTemp.csv', index=False)
gridres['MAPE_1'] = results_df['MAPE_1']
gridres['R2_1'] = results_df['R2_1']
gridres['MAPE_2'] = results_df['MAPE_2']
gridres['R2_2'] = results_df['R2_2']
gridres = gridres.sort_values(by='R2_1', ascending=False)
print(gridres.head())
