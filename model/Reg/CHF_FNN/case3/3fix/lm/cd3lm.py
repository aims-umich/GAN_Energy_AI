from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from itertools import product

# Define the regression model with three outputs
def build_regression_model(input_dim, num_layers, num_nodes):
    model = Sequential()
    model.add(Dense(num_nodes, activation='relu', input_shape=(input_dim,)))
    
    for _ in range(1, num_layers):
        model.add(Dense(num_nodes, activation='relu'))
    
    model.add(Dense(3, activation='linear'))  # 3 output columns
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
train_data = pd.read_csv("/home/unabila/CondGan/3fix/train_l_m.csv").values
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
xtrain = train_data[:, :-3]  # All columns except last 3 as input
ytrain = train_data[:, -3:]  # Last 3 columns as output

# Read test data
test_data = pd.read_csv("/home/unabila/CondGan/3fix/test_l_m.csv").values
Ytest = test_data[:, -3:]  # Last 3 columns as output
test_data = scaler.transform(test_data)  # Use the same scaler as for train data
xtest = test_data[:, :-3]
ytest = test_data[:, -3:]

# Train and evaluate the model for each configuration
for item in configs:
    model = build_regression_model(input_dim=xtrain.shape[1], num_layers=item[1], num_nodes=item[2])
    epochs = 100  # Number of epochs to train
    trained_model = train_regression_model(model, xtrain, ytrain, epochs, batch_size=item[3])

    Y_pred = trained_model.predict(xtest)
    generated_data = np.concatenate((xtest, Y_pred), axis=1)
    generated_data = scaler.inverse_transform(generated_data)
    Y_pred = generated_data[:, -3:]  # Focus on all three output columns

    # Calculate metrics only for the last output column (index 2)
    mape = mean_absolute_percentage_error(Ytest[:, 2], Y_pred[:, 2])
    r2 = r2_score(Ytest[:, 2], Y_pred[:, 2])
    MS_list.append((mape, r2))

# Store and save the results
gridres = pd.DataFrame(configs, columns=['learning_rate', 'num_layers', 'num_nodes', 'batch_size'])
results_df = pd.DataFrame(MS_list, columns=['MAPE_last_col', 'R2_last_col'])
final_result = pd.concat([gridres, results_df], axis=1)
final_result.to_csv('results_regressionLM.csv', index=False)

# Sort the final results by R2 score of the last column
final_result = final_result.sort_values(by='R2_last_col', ascending=False)
print(final_result.head())
