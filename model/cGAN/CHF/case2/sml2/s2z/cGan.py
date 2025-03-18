# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Evo2xxcfJqbTsUbOEfJmFtg-IvD4zJyV
"""

from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from itertools import product

def mean_absolute_percentage_error(y_true, y_pred):
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# define the standalone discriminator model
def build_generator(z_dim, x_dim,num_layers, num_nodes):
    z = Input(shape=(z_dim,))
    x = Input(shape=(x_dim,))

    merged_layer = Concatenate()([z, x])
    hidden = Dense(num_nodes, activation='relu')(merged_layer)
    for i in range(1, num_layers):
        hidden = Dense(num_nodes, activation='relu')(hidden)
    out_layer = Dense(1, activation='tanh')(hidden)

    model = Model(inputs=[z, x], outputs=out_layer)
    return model

# Define the discriminator model
def build_discriminator(x_dim,learning_rate, num_layers, num_nodes):
    x = Input(shape=(x_dim + 1,))

    hidden = Dense(num_nodes, activation='relu')(x)
    for i in range(1, num_layers):
        hidden = Dense(num_nodes, activation='relu')(hidden)
    out_layer = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=x, outputs=out_layer)

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['mean_absolute_error'])
    return model


def build_cgan(generator, discriminator):
    z_dim = generator.input_shape[0][1]
    x_dim = generator.input_shape[1][1]

    discriminator.trainable = False

    z = Input(shape=(z_dim,))
    x = Input(shape=(x_dim,))
    generated_y = generator([z, x])

    cgan_output = discriminator(Concatenate()([x, generated_y]))

    cgan = Model(inputs=[z, x], outputs=cgan_output)
    cgan.compile(loss='binary_crossentropy', optimizer=Adam())

    return cgan


def train_cgan(generator, discriminator, cgan, x_train, y_train, epochs, z_dim, batch_size):

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # Random indices for a batch
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_x, true_y = x_train[idx], y_train[idx]

        # Reshape true_y to have the same number of dimensions as true_x
        true_y = np.reshape(true_y, (-1, 1))

        # Stack true_x and true_y
        true_samples = np.hstack([true_x, true_y])

        # Train the discriminator with real samples
        d_loss_real = discriminator.train_on_batch(true_samples, valid)

        # Generate fake samples
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        gen_y = generator.predict([noise, true_x])

        # Stack true_x and the generated gen_y for fake samples
        fake_samples = np.hstack([true_x, gen_y])

        # Train the discriminator with fake samples
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake)

        # Train the generator (through combined cGAN model)
        g_loss = cgan.train_on_batch([noise, true_x], valid)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {np.mean([d_loss_real[0], d_loss_fake[0]])}] [G loss: {g_loss}]")


# Generate y_test
def generate_y_test(generator, x_test, z_dim):
    noise = np.random.normal(0, 1, (x_test.shape[0], z_dim))
    generated_y_test = generator.predict([noise, x_test])
    return generated_y_test

def plot_comparison(df1,df2, num_points_from_file2,item):
		# Convert numpy arrays to pandas DataFrames
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)

    # Take the same number of data points randomly from the first file as from the second file
    df1_sampled = df1.sample(n=num_points_from_file2, random_state=42)

    # Get the number of columns (excluding the last column)
    num_cols = len(df1_sampled.columns) - 1
    header = ["Diameter","Length","Pressure","Mass flux","Temperature","CHF"]

    # Create subplots
    fig, axes = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))

    # Iterate through each column (excluding the last column)
    for i in range(num_cols):
        # Scatter plot for each file's column against the last column
        axes[i].scatter(df1_sampled.iloc[:, i], df1_sampled.iloc[:, -1], color='red', label='Real')
        axes[i].scatter(df2.iloc[:, i], df2.iloc[:, -1], color='blue', label='Fake')

        # Customize each subplot
        axes[i].set_title(f'{header[i]} vs CHF')
        axes[i].set_xlabel(f'{header[i]}')
        axes[i].set_ylabel('CHF')
        axes[i].legend()

    # Customize the overall plot layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'comparison_plot_MS_{item}.png')
    
#######################################################################################################
# Define configurations
learning_rate = [5e-4,1e-3]
num_layers = [3]
num_nodes = [200]
batch_size = [64,128]
configs = list(product(learning_rate, num_layers, num_nodes, batch_size))
MS_list=[]

# Create an empty DataFrame to store results
gridres = pd.DataFrame(columns=['run', 'learning_rate', 'num_layers', 'num_nodes', 'Mape', 'R2'])
# Assuming x_train are your features and y_train are the labels you want to generate.
# Load real_data and input_data from CSV files
# Read train data
train_data = pd.read_csv("../Smolin.csv").values
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
xtrain = train_data[:, :-1]
ytrain = train_data[:, -1]

# Read test data
test_data = pd.read_csv("../Zenkevich.csv").values
Ytest = test_data[:, -1]
test_data = scaler.transform(test_data)  # Use the same scaler as for train data
xtest = test_data[:, :-1]
ytest = test_data[:, -1]

xtestData = np.concatenate((xtest, ytest.reshape(-1, 1)), axis=1)
test_data = scaler.inverse_transform(xtestData)
# Dimensions
z_dim = 100  # Latent vector size
x_dim = xtrain.shape[1]  # Feature size
for item in configs:
    # create the discriminator
    discriminator = build_discriminator(x_dim,learning_rate= item[0], num_layers= item[1], num_nodes= item[2])
    # create the generator
    generator = build_generator(z_dim, x_dim, num_layers= item[1], num_nodes= item[2])
    # create the gan
    cgan = build_cgan(generator, discriminator)
    # train model(g_model, d_model, gan_model, real_input_data, latent_dim, learning_rate, num_layers, num_nodes,..)
    epochs = 6000  # Number of epochs you want to train for
    train_cgan(generator, discriminator, cgan, xtrain, ytrain, epochs, z_dim, batch_size = item[3] )

    # generate data
    Ynn = generate_y_test(generator, xtest, z_dim)
    
    # Concatenate generated output with input_data along the columns axis
    generated_data = np.concatenate((xtest, Ynn), axis=1)
    generated_data = scaler.inverse_transform(generated_data)
    
    Ynn = generated_data[:, -1]
    
    mape = mean_absolute_percentage_error(Ytest, Ynn)
    r2 = r2_score(Ytest, Ynn)
    MS_list.append((mape,r2))

    # Save generated data to a CSV file
    columns = ['Diameter', 'length', 'Pressure', 'Mass flux', 'Temperature', 'CHF']
    generated_df = pd.DataFrame(generated_data, columns=columns)
    generated_df.to_csv(f'generated_samples_{item}.csv', index=False)
    #plot_comparison(test_data,generated_data,4636,item) 


gridres = pd.DataFrame(configs, columns = ['learning_rate', 'num_layers', 'num_nodes','batch_size'])
mean_values = [item[0] for item in MS_list]
r2 = [item[1] for item in MS_list]

# Create a DataFrame for the results
results_df = pd.DataFrame(MS_list, columns=['MAPE', 'R2'])

# Concatenate gridres and results_df along the columns axis
final_result = pd.concat([gridres, results_df], axis=1)

# Save the results to a CSV file
final_result.to_csv('resultsLr.csv', index=False)

# Assign mean and sigma values to separate columns in the DataFrame
gridres['Mape'] = mean_values
gridres['R2'] = r2

# Sort the DataFrame by 'Error' column in ascending order
gridres = gridres.sort_values(by='R2', ascending=False)

# Print the first few rows of the sorted DataFrame
print(gridres.head())
