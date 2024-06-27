import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to plot predictions
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10,7))
    
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c='b', label='Training Data')
    
    # Plot test data in green
    plt.scatter(test_data, test_labels, c='g', label='Testing Data')
    
    # Plot the predictions made on test data in red
    plt.scatter(test_data, predictions, c='r', label='Predictions')
    
    # Show the legend
    plt.legend(shadow=True)
    
    # Set the grids
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    
    # Add some text
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
    
    # Save the plot as a PNG file
    # plt.savefig('model_results.png', dpi=120)
    plt.savefig('model_results.png', dpi=120, format='png')
    # plt.show()  # Display the plot instead of saving it

# Function to calculate mean absolute error (MAE)
def mae(y_test, y_pred):
    """
    Calculates mean absolute error between y_test and y_preds.
    """
    return tf.metrics.mean_absolute_error(y_test, y_pred)

# Function to calculate mean squared error (MSE)
def mse(y_test, y_pred):
    """
    Calculates mean squared error between y_test and y_preds.
    """
    return tf.metrics.mean_squared_error(y_test, y_pred)

# Print TensorFlow version to ensure compatibility
print(tf.__version__)

# Create feature data (X) as a range from -100 to 100 with steps of 4
X = np.arange(-100, 100, 4)

# Create labels (y) as a range from -90 to 110 with steps of 4
y = np.arange(-90, 110, 4)

# Split data into training and testing sets
X_train = X[:40]  # First 40 examples (80% of the data)
y_train = y[:40] 

X_test = X[40:]  # Last 10 examples (20% of the data)
y_test = y[40:]

# Reshape the data to be compatible with the model
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Set random seed for reproducibility
tf.random.set_seed(42)

# Create a simple neural network model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)  # Single dense layer with one neuron
])

# Compile the model with MAE loss, SGD optimizer, and MAE as a metric
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

# Train the model on the training data for 50 epochs
model.fit(X_train, y_train, epochs=50)

# Make predictions on the test data
y_preds = model.predict(X_test)

# Calculate and print MAE and MSE for the predictions
mae_1 = np.round(float(mae(y_test, y_preds.squeeze()).numpy()), 2)
mse_1 = np.round(float(mse(y_test, y_preds.squeeze()).numpy()), 2)
print(f'\n Mean Absolute Error = {mae_1}, Mean Squared Error = {mse_1}')

# Plot the results
plot_predictions(X_train, y_train, X_test, y_test, y_preds)
