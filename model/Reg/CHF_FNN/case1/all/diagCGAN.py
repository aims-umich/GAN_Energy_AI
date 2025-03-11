# Importing necessary libraries
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

# Function to calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to build the generator model
def build_generator(z_dim, x_dim, num_layers, num_nodes):
    # Inputs: random noise (z) and actual data (x)
    z = Input(shape=(z_dim,))
    x = Input(shape=(x_dim,))

    # Concatenate z and x to create merged layer
    merged_layer = Concatenate()([z, x])
    hidden = Dense(num_nodes, activation='relu')(merged_layer)

    # Add hidden layers based on the specified number of layers
    for _ in range(1, num_layers):
        hidden = Dense(num_nodes, activation='relu')(hidden)

    # Output layer with 'tanh' activation
    out_layer = Dense(1, activation='tanh')(hidden)

    # Build the generator model
    model = Model(inputs=[z, x], outputs=out_layer)
    return model

# Function to build the discriminator model
def build_discriminator(x_dim, learning_rate, num_layers, num_nodes):
    # Input: actual data combined with the generated data
    x = Input(shape=(x_dim + 1,))

    hidden = Dense(num_nodes, activation='relu')(x)

    # Add hidden layers based on the specified number of layers
    for _ in range(1, num_layers):
        hidden = Dense(num_nodes, activation='relu')(hidden)

    # Output layer with 'sigmoid' activation to classify as real or fake
    out_layer = Dense(1, activation='sigmoid')(hidden)

    # Build and compile the discriminator model
    model = Model(inputs=x, outputs=out_layer)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['mean_absolute_error'])
    return model

# Function to build the combined CGAN model
def build_cgan(generator, discriminator):
    # Get the input dimensions for z and x
    z_dim = generator.input_shape[0][1]
    x_dim = generator.input_shape[1][1]

    # Freeze the discriminator's weights when training CGAN
    discriminator.trainable = False

    # Inputs: random noise (z) and actual data (x)
    z = Input(shape=(z_dim,))
    x = Input(shape=(x_dim,))

    # Generate data using the generator
    generated_y = generator([z, x])

    # Discriminator evaluates the generated data
    cgan_output = discriminator(Concatenate()([x, generated_y]))

    # Build and compile the CGAN model
    cgan = Model(inputs=[z, x], outputs=cgan_output)
    cgan.compile(loss='binary_crossentropy', optimizer=Adam())
    return cgan

# Function to train the CGAN model
def train_cgan(generator, discriminator, cgan, x_train, y_train, epochs, z_dim, batch_size):
    # Labels for real and fake samples
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    d_losses = []  # Track discriminator loss
    g_losses = []  # Track generator loss

    # Training loop
    for epoch in range(epochs):
        # Randomly select a batch of real samples
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_x, true_y = x_train[idx], y_train[idx]
        true_y = np.reshape(true_y, (-1, 1))

        # Combine true_x and true_y for real samples
        true_samples = np.hstack([true_x, true_y])

        # Train the discriminator with real samples
        d_loss_real = discriminator.train_on_batch(true_samples, valid)

        # Generate fake samples using the generator
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        gen_y = generator.predict([noise, true_x])

        # Combine true_x and generated gen_y for fake samples
        fake_samples = np.hstack([true_x, gen_y])

        # Train the discriminator with fake samples
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake)

        # Train the generator via CGAN model
        g_loss = cgan.train_on_batch([noise, true_x], valid)
        
        d_losses.append(np.mean([d_loss_real[0], d_loss_fake[0]]))
        g_losses.append(g_loss)

        # Print the progress at every 100th epoch
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: [D loss: {np.mean([d_loss_real[0], d_loss_fake[0]])}] [G loss: {g_loss}]")
    return d_losses, g_losses

# Function to generate y_test using the trained generator
def generate_y_test(generator, x_test, z_dim):
    noise = np.random.normal(0, 1, (x_test.shape[0], z_dim))
    generated_y_test = generator.predict([noise, x_test])
    return generated_y_test

# Function to create a diagonal plot for true vs. predicted y_test
def plot_diagonal(true_y, predicted_y, item):
    # Create a scatter plot of true vs. predicted values
    plt.figure(figsize=(8, 8))
    plt.scatter(true_y, predicted_y, color='blue', label='Predicted vs True')

    # Plot a diagonal line for reference
    plt.plot([min(true_y), max(true_y)], [min(true_y), max(true_y)], color='red', linestyle='--', label='Ideal')

    # Add labels and title
    plt.xlabel('True y_test')
    plt.ylabel('Predicted y_test')
    plt.title(f'Diagonal Plot: True vs Predicted y_test (Config: {item})')
    plt.legend()

    # Save the plot
    plt.savefig(f'diagonal_plot_{item}.png')
    plt.close()

#######################################################################################################
# Configurations for hyperparameter tuning
learning_rate = [1e-4]
num_layers = [3]
num_nodes = [300]
batch_size = [128]
configs = list(product(learning_rate, num_layers, num_nodes, batch_size))
MS_list = []

# Create a DataFrame to store results
gridres = pd.DataFrame(columns=['learning_rate', 'num_layers', 'num_nodes', 'batch_size', 'MAPE', 'R2'])

# Load and preprocess the training data
train_data = pd.read_csv("/home/unabila/CondGan/all/train_all.csv").values
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
xtrain = train_data[:, :-1]
ytrain = train_data[:, -1]

# Read test data
test_data = pd.read_csv("/home/unabila/CondGan/all/test_all.csv").values
Ytest = test_data[:, -1]
test_data = scaler.transform(test_data)
xtest = test_data[:, :-1]
ytest = test_data[:, -1]

# Combine test features and labels for inverse transformation
xtestData = np.concatenate((xtest, ytest.reshape(-1, 1)), axis=1)
test_data = scaler.inverse_transform(xtestData)

# Latent vector size and feature size
z_dim = 100
x_dim = xtrain.shape[1]

# Loop over different hyperparameter configurations
for item in configs:
    # Build and train the models
    discriminator = build_discriminator(x_dim, learning_rate=item[0], num_layers=item[1], num_nodes=item[2])
    generator = build_generator(z_dim, x_dim, num_layers=item[1], num_nodes=item[2])
    cgan = build_cgan(generator, discriminator)
    epochs = 5000 # Number of training epochs
    d_losses, g_losses = train_cgan(generator, discriminator, cgan, xtrain, ytrain, epochs, z_dim, batch_size=item[3])

    # Generate and evaluate predictions
    Ynn = generate_y_test(generator, xtest, z_dim)
    generated_data = np.concatenate((xtest, Ynn), axis=1)
    generated_data = scaler.inverse_transform(generated_data)
    Ynn = generated_data[:, -1]
    
    # Plot the diagonal plot
    plot_diagonal(Ytest, Ynn, item)
    
    mape = mean_absolute_percentage_error(Ytest, Ynn)
    r2 = r2_score(Ytest, Ynn)
    MS_list.append((mape, r2))

    # Save generated data to CSV
    #columns = ['Diameter', 'Length', 'Pressure', 'Mass flux', 'Temperature', 'CHF']
    #generated_df = pd.DataFrame(generated_data, columns=columns)
    #generated_df.to_csv(f'generated_samples_{item}.csv', index=False)

# Store results in DataFrame and save to CSV
results_df = pd.DataFrame(MS_list, columns=['MAPE', 'R2'])
#final_result = pd.concat([gridres, results_df], axis=1)
#final_result.to_csv('resultsLR.csv', index=False)

# Sort and display the top results
gridres['MAPE'] = results_df['MAPE']
gridres['R2'] = results_df['R2']
gridres = gridres.sort_values(by='R2', ascending=False)
print(gridres.head())