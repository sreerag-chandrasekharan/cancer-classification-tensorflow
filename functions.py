import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf

def load_data(file_path):
    # Load the data from the CSV file
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.iloc[:, :-1].values  # Convert features to NumPy array
    y = df.iloc[:, -1].values   # Convert target to NumPy array
    
    # Store the feature names
    feature_names = df.columns[:-1].tolist()
    
    return X, y, feature_names

def plot_histograms(X, X_norm):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram for raw data
    axes[0].hist(X.flatten(), bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title("Histogram of Raw Data")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")

    # Histogram for scaled data
    axes[1].hist(X_norm.flatten(), bins=30, color='salmon', edgecolor='black')
    axes[1].set_title("Histogram of Scaled Data")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def train_model(X_train, y_train, X_test, y_test, model):
    # Set up TensorBoard logging
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Train the model with TensorBoard callback
    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_test, y_test),  # Add test data for validation
        callbacks=[tensorboard_callback]
    )

    # Return the training history
    return history