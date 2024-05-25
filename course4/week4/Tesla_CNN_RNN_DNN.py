import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import shutil
import re
import datetime
import math
import pytz
import yfinance
import pandas as pd
from dateutil.relativedelta import relativedelta


timezone = pytz.timezone('America/New_York')

# Get the current time
# current_time = datetime.datetime.now(pytz.timezone('America/New_York'))

naive_datetime = datetime.datetime(2024, 5, 13, 0, 0, 0)

current_time = timezone.localize(naive_datetime)

start_time = current_time - relativedelta(years=+5)

# Convert it to seconds since the epoch
unix_epoch = int(math.floor(current_time.timestamp()))

print(unix_epoch)

# url = "https://query1.finance.yahoo.com/v7/finance/download/TSLA?period1=1557830052&period2="+str(unix_epoch)+"&interval=1d&events=history&includeAdjustedClose=true"

dst = "data"

file_name = "tesla.csv"

STOCK_CSV = "./"+dst+"/"+file_name

split_size = .7

def cleanup(extracted_dst):
    if os.path.exists(extracted_dst):
        shutil.rmtree(extracted_dst)
        print(extracted_dst + " deleted")

cleanup(dst)

# Define a function to download and extract the zip file
def fetch_and_extract_zip(ticker, start_time, end_time, filename, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    # Send a GET request to the URL
    data = yfinance.download(ticker, start=start_time, end=end_time)
    df = pd.DataFrame(data)
    df.to_csv(filename, index=True)

fetch_and_extract_zip("TSLA", start_time, current_time, STOCK_CSV, dst)

with open(STOCK_CSV, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i >= 10:
            break
        print(re.sub(r'\[|\]|\'|\"', '', str(row)))

# Initialize lists
time_step = []
close_price = []

# Open CSV file
with open(STOCK_CSV) as csvfile:

  # Initialize reader
  reader = csv.reader(csvfile, delimiter=',')

  # Skip the first line
  next(reader)

  # Append row and sunspot number to lists
  for row in reader:
    time_step.append(datetime.datetime.strptime(row[0], "%Y-%m-%d"))
    close_price.append(float(row[5]))

# Convert lists to numpy arrays
time = np.array(time_step)
series = np.array(close_price)

def plot_series(x, y, format="-", start=0, end=None,
                title=None, xlabel=None, ylabel=None, legend=None ):
    """
    Visualizes time series data

    Args:
      x (array of int) - contains values for the x-axis
      y (array of int or tuple of arrays) - contains the values for the y-axis
      format (string) - line style when plotting the graph
      label (string) - tag for the line
      start (int) - first time step to plot
      end (int) - last time step to plot
      title (string) - title of the plot
      xlabel (string) - label for the x-axis
      ylabel (string) - label for the y-axis
      legend (list of strings) - legend for the plot
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))

    # Check if there are more than two series to plot
    if type(y) is tuple:

      # Loop over the y elements
      for y_curr in y:

        # Plot the x and current y values
        plt.plot(x[start:end], y_curr[start:end], format)

    else:
      # Plot the x and y values
      plt.plot(x[start:end], y[start:end], format)

    # Label the x-axis
    plt.xlabel(xlabel)

    # Label the y-axis
    plt.ylabel(ylabel)

    # Set the legend
    if legend:
      plt.legend(legend)

    # Set the title
    plt.title(title)

    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()

# Preview the data
# plot_series(time, series, xlabel='Month', ylabel='Closing Share Price')

############################split dataset ###########################################

# Define the split time
split_time = int(split_size * len(time))

# Get the train set
time_train = time[:split_time]
x_train = series[:split_time]

# Get the validation set
time_valid = time[split_time:]
x_valid = series[split_time:]

####################################################################################

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    # dataset = dataset.shuffle(shuffle_buffer)

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)
    # dataset = dataset.batch(2).prefetch(1)


    return dataset

# Parameters
window_size = 30
batch_size = 32
shuffle_buffer_size = 1000

# Generate the dataset windows
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
# train_set = windowed_dataset(x_train[:10], 5, batch_size, 10)

for x, y in train_set:
    print("x = ", x.numpy())
    print("y = ", y.numpy())
    print()

print("pause")
############################learning rate training ###########################################

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Print the model summary
model.summary()

# Set the learning rate scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

# Initialize the optimizer
optimizer = tf.keras.optimizers.legacy.SGD(momentum=0.9)

# Set the training parameters
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer)

# Train the model
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

# Define the learning rate array
lrs = 1e-8 * (10 ** (np.arange(100) / 20))

# Set the figure size
plt.figure(figsize=(10, 6))

# Set the grid
plt.grid(True)

# Plot the loss in log scale
plt.semilogx(lrs, history.history["loss"])

# Increase the tickmarks size
plt.tick_params('both', length=10, width=1, which='both')

# Set the plot boundaries
plt.axis([1e-8, 1e-3, 0, 100])

###############################################################################################


# Reset states generated by Keras
tf.keras.backend.clear_session()

# Build the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Set the learning rate
learning_rate = 2e-6

# Set the optimizer
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9)

# Set the training parameters
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

# Train the model
history = model.fit(train_set,epochs=100)

def model_forecast(model, series, window_size, batch_size):
    """Uses an input model to generate predictions on data windows

    Args:
      model (TF Keras Model) - model that accepts data windows
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the window
      batch_size (int) - the batch size

    Returns:
      forecast (numpy array) - array containing predictions
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    # Get predictions on the entire dataset
    forecast = model.predict(dataset)

    return forecast

# Reduce the original series
forecast_series = series[split_time-window_size:-1]

# Use helper function to generate predictions
forecast = model_forecast(model, forecast_series, window_size, batch_size)

# Drop single dimensional axis
results = forecast.squeeze()

# Plot the results
# plot_series(time_valid, (x_valid, results))
plot_series(time_valid, (x_valid, results))

# Compute the MAE
print("final MAE",tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())


def future_forecast(model, last_window, steps_ahead):
    """Generates future predictions based on the last observed window."""
    future_predictions = []
    current_window = last_window

    for _ in range(steps_ahead):
        # Reshape window to fit model input
        input_data = np.expand_dims(current_window, axis=0)
        next_prediction = model.predict(input_data)
        next_value = next_prediction[0]
        future_predictions.append(next_value)

        # Update the window
        current_window = np.roll(current_window, -1)
        current_window[-1] = next_value

    return np.array(future_predictions)

last_window = series[-window_size:]

steps_ahead = 4

future_predictions = future_forecast(model, last_window, steps_ahead)

print(future_predictions)

