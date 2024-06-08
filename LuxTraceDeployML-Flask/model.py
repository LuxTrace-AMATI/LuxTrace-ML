import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ------------------------------------------------------------------------------

# Define the Anomaly Detection Model
class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(22, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load the Anomaly Detection Model
anomaly_model = AnomalyDetector()
anomaly_model.compile(optimizer='adam', loss='mae')
weights_model = tf.keras.models.load_model('autoencoder.h5')
anomaly_model.load_weights(weights_model)

# Preprocessing Function for Anomaly Detection
def preprocess_anomaly_data(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame(input_data)
    
    # Preprocess the data
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df['delivery_time'] = pd.to_datetime(df['delivery_time'])
    df['tracking_date'] = pd.to_datetime(df['tracking_date'])
    df['delivery_reception'] = pd.to_datetime(df['delivery_reception'])

    # Extract date and time features
    df['delivery_year'] = df['delivery_date'].dt.year
    df['delivery_month'] = df['delivery_date'].dt.month
    df['delivery_day'] = df['delivery_date'].dt.day
    df['delivery_hour'] = df['delivery_time'].dt.hour
    df['delivery_minute'] = df['delivery_time'].dt.minute
    df['tracking_year'] = df['tracking_date'].dt.year
    df['tracking_month'] = df['tracking_date'].dt.month
    df['tracking_day'] = df['tracking_date'].dt.day
    df['tracking_hour'] = df['tracking_date'].dt.hour
    df['tracking_minute'] = df['tracking_date'].dt.minute
    df['reception_year'] = df['delivery_reception'].dt.year
    df['reception_month'] = df['delivery_reception'].dt.month
    df['reception_day'] = df['delivery_reception'].dt.day
    df['reception_hour'] = df['delivery_reception'].dt.hour
    df['reception_minute'] = df['delivery_reception'].dt.minute

    # Select features to use
    features = ['delivery_product_amount', 'delivery_distance', 'delivery_duration',
                'delivery_year', 'delivery_month', 'delivery_day', 'delivery_hour', 'delivery_minute',
                'tracking_year', 'tracking_month', 'tracking_day', 'tracking_hour', 'tracking_minute',
                'reception_year', 'reception_month', 'reception_day', 'reception_hour', 'reception_minute']
    
    # Scale numeric features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df[features].values

def predict_anomaly(input_data):
    # Preprocess data
    processed_data = preprocess_anomaly_data(input_data)
    
    # Predict
    reconstructions = anomaly_model.predict(processed_data)
    reconstruction_errors = np.mean(np.abs(reconstructions - processed_data), axis=1)
    
    # Define a threshold for anomaly detection
    threshold = np.percentile(reconstruction_errors, 95)  # Example threshold
    anomalies = reconstruction_errors > threshold
    
    # Return results
    return anomalies.tolist()


# ----------------------------------------------------------------------------------------------

# Load the Demand Forecasting Model
forecasting_model = tf.keras.models.load_model('forecasting.h5')

# Preprocessing Function for Forecasting
def preprocess_forecasting_data(input_data):
    # Convert input data to DataFrame
    df = pd.DataFrame(input_data)
    
    # Convert 'delivery_date' column to datetime
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df.set_index('delivery_date', inplace=True)
    
    # Resample data by week and sum the product deliveries
    df = df.groupby(pd.Grouper(freq='W')).agg({'total_product_delivery':'sum'})
    
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def predict_forecasting(input_data):
    # Preprocess data
    processed_data, scaler = preprocess_forecasting_data(input_data)
    
    # Create sequences
    seq_length = 14
    X, _ = create_sequences(processed_data, seq_length)
    
    # Predict
    predictions = forecasting_model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.tolist()
