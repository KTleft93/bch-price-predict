import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Define the file path
csv_file = 'BCH-USD.csv'  # Replace 'your_file.csv' with the actual file path

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Extract the 'Close' column as the feature (X)
X = df['Close'].values.reshape(-1, 1)

# Shift the 'Close' column to obtain the future closing prices as the target variable (y)
y = df['Close'].shift(-1).values.reshape(-1, 1)

X = X[:-1]
y = y[:-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
# print('Mean Squared Error:', mse)

def prepare_feature_data(current_price, num_future_periods):

    future_features = np.linspace(current_price, current_price * 1.1, num_future_periods).reshape(-1, 1)

    # Return the feature data for future predictions
    return future_features


current_price = df['Close'].iloc[-1]

# Number of future periods to predict (e.g., next 3 months)
num_future_periods = 90

current_date = datetime.now()  # Assuming current date
future_dates = [current_date + timedelta(days=i) for i in range(1, num_future_periods + 1)] # Generate dates for the next 6 months

future_features = prepare_feature_data(current_price,num_future_periods)  # Prepare feature data for future predictions

# Use the trained model to predict prices for the next 3 months
predicted_prices = model.predict(future_features)

plt.figure(figsize=(10, 6))
plt.plot(future_dates, predicted_prices, marker='o', linestyle='-', color='b')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.title('Predicted Prices of BCH for Next 3 Months')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()