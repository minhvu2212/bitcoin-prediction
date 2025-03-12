# %% [markdown]
# # Bitcoin Price Prediction Using Machine Learning
# Nhập môn Học máy và Khai phá dữ liệu (IT3190)

# %% [markdown]
# ## 1. Import Libraries

# %%
# Data manipulation and analysis
import numpy as np
import pandas as pd
import math

# Machine learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

# ML Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# Deep learning
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Time series
import datetime as dt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 2. Data Loading and Exploration

# %%
# Load the data
print("Loading Bitcoin historical data...")
path = '/kaggle/input/bitcoin-historical-data/btcusd_1-min_data.csv'
df = pd.read_csv(path)

# Display basic information
print("\nDataset Information:")
print(f"Dataset shape: {df.shape}")
print("\nData Types:")
print(df.dtypes)

# %%
# Convert timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

# Set timestamp as index
df.set_index('Timestamp', inplace=True)

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# %%
# Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

# %%
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)
print(f"\nDataset shape after removing missing values: {df.shape}")

# %%
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# %% [markdown]
# ## 3. Data Visualization

# %%
# Plot Bitcoin price over time
plt.figure(figsize=(15, 6))
plt.plot(df['Close'], label='Bitcoin Close Price (USD)')
plt.title('Bitcoin Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Resample data to plot different time intervals
plt.figure(figsize=(15, 12))

plt.subplot(4, 1, 1)
plt.plot(df['Close'].resample('D').mean(), label='Daily Average')
plt.title('Daily Average Bitcoin Price')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(df['Close'].resample('W').mean(), label='Weekly Average')
plt.title('Weekly Average Bitcoin Price')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(df['Close'].resample('M').mean(), label='Monthly Average')
plt.title('Monthly Average Bitcoin Price')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(df['Volume'].resample('W').sum(), label='Weekly Volume')
plt.title('Weekly Trading Volume')
plt.legend()

plt.tight_layout()
plt.show()

# %%
# Create correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# %%
# Daily returns histogram
df_daily = df.resample('D').last()
df_daily['Returns'] = df_daily['Close'].pct_change() * 100
plt.figure(figsize=(12, 6))
plt.hist(df_daily['Returns'].dropna(), bins=50, alpha=0.75)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Returns (%)')
plt.ylabel('Frequency')
plt.axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Data Preprocessing

# %%
# Resample to daily data using appropriate aggregation
df_daily = df.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print(f"\nShape of daily aggregated data: {df_daily.shape}")

# %%
# Feature Engineering
def add_technical_indicators(df):
    # Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA14'] = df['Close'].rolling(window=14).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    
    # Price momentum
    df['Price_Momentum'] = df['Close'] - df['Close'].shift(7)
    
    # Volatility (standard deviation of returns)
    df['Volatility'] = df['Close'].pct_change().rolling(window=7).std() * 100
    
    # Price Rate of Change (ROC)
    df['ROC'] = ((df['Close'] - df['Close'].shift(7)) / df['Close'].shift(7)) * 100
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Target variables for classification
    df['Target_Next_Day'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df['Target_Next_Week'] = np.where(df['Close'].shift(-7) > df['Close'], 1, 0)
    
    return df

# Add technical indicators
df_daily = add_technical_indicators(df_daily)

# Drop NaN values after feature engineering
df_daily.dropna(inplace=True)
print(f"\nShape after adding technical indicators and dropping NaNs: {df_daily.shape}")

# %%
# Display the enhanced dataset
print("\nEnhanced Dataset Preview:")
print(df_daily.head())

# %%
# Visualize some technical indicators
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(df_daily['Close'], label='Close Price')
plt.plot(df_daily['MA7'], label='7-day MA')
plt.plot(df_daily['MA30'], label='30-day MA')
plt.title('Bitcoin Price with Moving Averages')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(df_daily['RSI'], label='RSI')
plt.axhline(y=70, color='r', linestyle='--')
plt.axhline(y=30, color='g', linestyle='--')
plt.title('Relative Strength Index (RSI)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(df_daily['MACD'], label='MACD')
plt.plot(df_daily['MACD_Signal'], label='Signal Line')
plt.title('Moving Average Convergence Divergence (MACD)')
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Preparing Data for Machine Learning

# %%
# Function to create sequences for time series forecasting
def create_sequences(data, target_col, window_size):
    """Create sequences for time series forecasting"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size].values)
        y.append(data.iloc[i+window_size][target_col])
    return np.array(X), np.array(y)

# Function to prepare data for regression/classification models
def prepare_data_for_ml(df, target_col, window_size=0, is_classification=False):
    """Prepare data for machine learning models"""
    # For time series models (using window_size)
    if window_size > 0:
        X, y = create_sequences(df, target_col, window_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # For traditional ML models
    else:
        # Features and target
        X = df.drop(['Open', 'High', 'Low', 'Close', 'Target_Next_Day', 'Target_Next_Week'], axis=1) 
        if is_classification:
            y = df[target_col].astype(int)
        else:
            y = df['Close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# %%
# Separate features and target for regression
X = df_daily.drop(['Open', 'High', 'Low', 'Close', 'Target_Next_Day', 'Target_Next_Week'], axis=1)
y_reg = df_daily['Close']

# Scale the data for regression models
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_reg_scaled = scaler_y.fit_transform(y_reg.values.reshape(-1, 1)).flatten()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reg_scaled, test_size=0.2, random_state=42)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# %%
# Prepare data for classification
X_class = df_daily.drop(['Open', 'High', 'Low', 'Close', 'Target_Next_Day', 'Target_Next_Week'], axis=1)
y_class = df_daily['Target_Next_Day']

# Split the data for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)

# %%
# For time series models (LSTM)
time_steps = 30  # Number of days to look back
X_ts, y_ts = create_sequences(df_daily[['Close'] + list(X.columns)], 'Close', time_steps)

# Scale the time series data
scaler_ts = MinMaxScaler()
X_ts_scaled = np.zeros_like(X_ts)
for i in range(X_ts.shape[0]):
    X_ts_scaled[i] = scaler_ts.fit_transform(X_ts[i])
y_ts_scaled = scaler_y.transform(y_ts.reshape(-1, 1)).flatten()

# Split the time series data
split_idx = int(0.8 * len(X_ts_scaled))
X_train_ts, X_test_ts = X_ts_scaled[:split_idx], X_ts_scaled[split_idx:]
y_train_ts, y_test_ts = y_ts_scaled[:split_idx], y_ts_scaled[split_idx:]

print(f"\nTime series training set shape: {X_train_ts.shape}")
print(f"Time series test set shape: {X_test_ts.shape}")

# %% [markdown]
# ## 6. Model Development and Training

# %% [markdown]
# ### 6.1 Linear Regression

# %%
# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_r2 = r2_score(y_test, y_pred_lr)

print("\n=== Linear Regression Results ===")
print(f"RMSE: {lr_rmse:.4f}")
print(f"MAE: {lr_mae:.4f}")
print(f"R^2: {lr_r2:.4f}")

# %% [markdown]
# ### A. Regression Models

# %%
def train_and_evaluate_regression_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple regression models"""
    results = {}
    
    # Decision Tree Regressor
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    results['Decision Tree'] = {
        'model': dt_model,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_dt)),
        'mae': mean_absolute_error(y_test, y_pred_dt),
        'r2': r2_score(y_test, y_pred_dt)
    }
    
    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['Random Forest'] = {
        'model': rf_model,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'r2': r2_score(y_test, y_pred_rf)
    }
    
    # Support Vector Regression
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)
    results['SVR'] = {
        'model': svr_model,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_svr)),
        'mae': mean_absolute_error(y_test, y_pred_svr),
        'r2': r2_score(y_test, y_pred_svr)
    }
    
    # K-Nearest Neighbors Regressor
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    results['KNN'] = {
        'model': knn_model,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_knn)),
        'mae': mean_absolute_error(y_test, y_pred_knn),
        'r2': r2_score(y_test, y_pred_knn)
    }
    
    return results

# Train and evaluate regression models
regression_results = train_and_evaluate_regression_models(X_train, X_test, y_train, y_test)

# Display regression results
print("\n=== Regression Model Results ===")
for model_name, result in regression_results.items():
    print(f"\n{model_name} Results:")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"MAE: {result['mae']:.4f}")
    print(f"R^2: {result['r2']:.4f}")

# %% [markdown]
# ### B. Classification Models

# %%
def train_and_evaluate_classification_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple classification models"""
    results = {}
    
    # Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    results['Decision Tree'] = {
        'model': dt_model,
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'precision': precision_score(y_test, y_pred_dt),
        'recall': recall_score(y_test, y_pred_dt),
        'f1': f1_score(y_test, y_pred_dt)
    }
    
    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf)
    }
    
    # Support Vector Machine Classifier
    svc_model = SVC(kernel='rbf', probability=True)
    svc_model.fit(X_train, y_train)
    y_pred_svc = svc_model.predict(X_test)
    results['SVC'] = {
        'model': svc_model,
        'accuracy': accuracy_score(y_test, y_pred_svc),
        'precision': precision_score(y_test, y_pred_svc),
        'recall': recall_score(y_test, y_pred_svc),
        'f1': f1_score(y_test, y_pred_svc)
    }
    
    # K-Nearest Neighbors Classifier
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    results['KNN'] = {
        'model': knn_model,
        'accuracy': accuracy_score(y_test, y_pred_knn),
        'precision': precision_score(y_test, y_pred_knn),
        'recall': recall_score(y_test, y_pred_knn),
        'f1': f1_score(y_test, y_pred_knn)
    }
    
    return results

# Train and evaluate classification models
classification_results = train_and_evaluate_classification_models(X_train_class, X_test_class, y_train_class, y_test_class)

# Display classification results
print("\n=== Classification Model Results (Price Direction Prediction) ===")
for model_name, result in classification_results.items():
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")

# %% [markdown]
# ### C. Time Series Models (LSTM)

# %%
# Reshape input data for LSTM [samples, time steps, features]
n_features = X_train_ts.shape[2]
X_train_ts_reshaped = X_train_ts.reshape(X_train_ts.shape[0], time_steps, n_features)
X_test_ts_reshaped = X_test_ts.reshape(X_test_ts.shape[0], time_steps, n_features)

# Define the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create and train the LSTM model
lstm_model = create_lstm_model((time_steps, n_features))
print("\nLSTM Model Summary:")
lstm_model.summary()

# %%
# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the LSTM model
lstm_history = lstm_model.fit(
    X_train_ts_reshaped, y_train_ts,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1
)

# %%
# Evaluate the LSTM model
lstm_test_loss = lstm_model.evaluate(X_test_ts_reshaped, y_test_ts, verbose=0)
lstm_predictions = lstm_model.predict(X_test_ts_reshaped)

# Convert predictions back to original scale
lstm_predictions = scaler_y.inverse_transform(lstm_predictions)
y_test_original = scaler_y.inverse_transform(y_test_ts.reshape(-1, 1))

# Calculate performance metrics
lstm_rmse = np.sqrt(mean_squared_error(y_test_original, lstm_predictions))
lstm_mae = mean_absolute_error(y_test_original, lstm_predictions)
lstm_r2 = r2_score(y_test_original, lstm_predictions)

print("\n=== LSTM Model Results ===")
print(f"Test Loss: {lstm_test_loss:.4f}")
print(f"RMSE: {lstm_rmse:.4f}")
print(f"MAE: {lstm_mae:.4f}")
print(f"R^2: {lstm_r2:.4f}")

# %% [markdown]
# ## 7. Model Evaluation and Comparison

# %%
# Plot training history for LSTM
plt.figure(figsize=(12, 6))
plt.plot(lstm_history.history['loss'], label='Training Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Plot predictions vs actual for LSTM
plt.figure(figsize=(15, 6))
plt.plot(y_test_original, label='Actual Bitcoin Price', color='blue')
plt.plot(lstm_predictions, label='LSTM Predictions', color='red', alpha=0.7)
plt.title('Bitcoin Price Prediction: Actual vs LSTM Predictions')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Get the best performing regression model
best_reg_model_name = min(regression_results, key=lambda k: regression_results[k]['rmse'])
best_reg_model = regression_results[best_reg_model_name]['model']

# Get feature importance for Random Forest (if it's the best model)
if best_reg_model_name == 'Random Forest':
    feature_importances = best_reg_model.feature_importances_
    feature_names = X.columns
    
    # Create feature importance dataframe and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance for Bitcoin Price Prediction')
    plt.tight_layout()
    plt.show()

# %%
# Compare performance of all regression models
plt.figure(figsize=(12, 6))
models = list(regression_results.keys()) + ['LSTM']
rmse_values = [regression_results[model]['rmse'] for model in regression_results] + [lstm_rmse]
colors = ['blue', 'green', 'red', 'purple', 'orange']

plt.bar(models, rmse_values, color=colors)
plt.title('RMSE Comparison Across Different Models')
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Compare performance of classification models
plt.figure(figsize=(12, 6))
classifiers = list(classification_results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_values = {
    metric: [classification_results[model][metric] for model in classifiers]
    for metric in metrics
}

x = np.arange(len(classifiers))
width = 0.2
multiplier = 0

fig, ax = plt.subplots(figsize=(12, 6))

for metric, values in metric_values.items():
    offset = width * multiplier
    ax.bar(x + offset, values, width, label=metric.capitalize())
    multiplier += 1

ax.set_title('Classification Performance Metrics Comparison')
ax.set_xticks(x + width * (len(metrics) - 1) / 2)
ax.set_xticklabels(classifiers)
ax.set_ylim(0, 1)
ax.set_ylabel('Score')
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Model Optimization with Cross-Validation

# %%
# Define a function for k-fold cross validation
def perform_kfold_cv(model, X, y, k=5):
    """Perform k-fold cross validation"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)
    
    return rmse_scores

# Perform cross-validation for the best regression model
if best_reg_model_name == 'Random Forest':
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_cv_scores = perform_kfold_cv(rf_model, X_scaled, y_reg_scaled)
    
    print(f"\nRandom Forest 5-Fold CV Results:")
    print(f"RMSE Scores: {rf_cv_scores}")
    print(f"Mean RMSE: {np.mean(rf_cv_scores):.4f}")
    print(f"Std Dev RMSE: {np.std(rf_cv_scores):.4f}")
    
    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_scaled, y_reg_scaled)
    
    print("\nBest Hyperparameters for Random Forest:")
    print(grid_search.best_params_)
    
    # Train optimized model
    best_rf_model = grid_search.best_estimator_
    best_rf_model.fit(X_train, y_train)
    y_pred_best_rf = best_rf_model.predict(X_test)
    
    # Evaluate optimized model
    best_rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))
    best_rf_mae = mean_absolute_error(y_test, y_pred_best_rf)
    best_rf_r2 = r2_score(y_test, y_pred_best_rf)
    
    print("\nOptimized Random Forest Results:")
    print(f"RMSE: {best_rf_rmse:.4f}")
    print(f"MAE: {best_rf_mae:.4f}")
    print(f"R^2: {best_rf_r2:.4f}")

# %% [markdown]
# ## 9. Predicting Future Bitcoin Prices

# %%
# Use the best model to make future predictions
best_model = best_rf_model if best_reg_model_name == 'Random Forest' else best_reg_model

# Function to create a prediction for the next day
def predict_next_day_price(model, last_data, scaler_x, scaler_y):
    """Predict the next day's price using the trained model"""
    # Scale the input data
    last_data_scaled = scaler_x.transform(last_data.reshape(1, -1))
    
    # Make prediction
    prediction_scaled = model.predict(last_data_scaled)
    
    # Inverse transform to get the actual price
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    
    return prediction[0][0]

# Get the last available data point
last_data = X_scaled[-1]

# Predict the next day's price
next_day_price = predict_next_day_price(best_model, last_data, scaler_X, scaler_y)
print(f"\nPredicted Bitcoin price for the next day: ${next_day_price:.2f}")

# %%
# For LSTM model, make future predictions
def predict_next_day_lstm(model, last_sequence, scaler):
    """Predict the next day's price using the LSTM model"""
    # Reshape the data for LSTM
    last_sequence_reshaped = last_sequence.reshape(1, time_steps, n_features)
    
    # Make prediction
    prediction_scaled = model.predict(last_sequence_reshaped)
    
    # Inverse transform to get the actual price
    prediction = scaler.inverse_transform(prediction_scaled)
    
    return prediction[0][0]

# Get the last sequence of data for LSTM prediction
last_sequence = X_test_ts_reshaped[-1]

# Predict the next day's price with LSTM
next_day_price_lstm = predict_next_day_lstm(lstm_model, last_sequence, scaler_y)
print(f"Predicted Bitcoin price for the next day (LSTM): ${next_day_price_lstm:.2f}")

# %% [markdown]
# ## 10. Multi-day Price Forecasting

# %%
# Function to make predictions for multiple days ahead
def forecast_future_prices(model, last_data, days=7, is_lstm=False):
    """Forecast Bitcoin prices for a number of days ahead"""
    forecasted_prices = []
    current_data = last_data.copy()
    
    for _ in range(days):
        if is_lstm:
            # Reshape for LSTM
            current_data_reshaped = current_data.reshape(1, time_steps, n_features)
            # Make prediction
            next_price = model.predict(current_data_reshaped)[0][0]
            # Update the sequence by removing first value and adding the prediction
            current_data = np.roll(current_data, -1, axis=0)
            current_data[-1, 0] = next_price  # Assuming 'Close' is at index 0
        else:
            # For traditional ML models
            next_price = model.predict(current_data.reshape(1, -1))[0]
            
        forecasted_prices.append(next_price)
    
    return forecasted_prices

# Get the best model based on evaluation
if lstm_rmse < best_rf_rmse:
    best_overall_model = lstm_model
    last_data_for_forecast = last_sequence
    is_lstm = True
    print("\nLSTM is the best performing model.")
else:
    best_overall_model = best_model
    last_data_for_forecast = last_data
    is_lstm = False
    print(f"\n{best_reg_model_name} is the best performing model.")

# Forecast prices for the next 7 days
forecast_days = 7
future_prices = forecast_future_prices(best_overall_model, last_data_for_forecast, forecast_days, is_lstm)

# Convert scaled predictions back to original values if necessary
if not is_lstm:
    future_prices = scaler_y.inverse_transform(np.array(future_prices).reshape(-1, 1)).flatten()

# Create date range for forecast
last_date = df_daily.index[-1]
future_dates = [last_date + dt.timedelta(days=i+1) for i in range(forecast_days)]

# Display the forecasted prices
print(f"\nForecasted Bitcoin prices for the next {forecast_days} days:")
for date, price in zip(future_dates, future_prices):
    print(f"{date.date()}: ${price:.2f}")

# %%
# Plot forecasted prices
plt.figure(figsize=(15, 6))

# Plot recent historical prices
recent_data = df_daily['Close'].iloc[-30:]
plt.plot(recent_data.index, recent_data.values, label='Historical Prices', color='blue')

# Plot forecasted prices
plt.plot(future_dates, future_prices, 'o-', label='Forecasted Prices', color='red')

plt.title('Bitcoin Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Trading Strategy Based on Price Direction Prediction

# %%
# Get the best classification model
best_class_model_name = max(classification_results, key=lambda k: classification_results[k]['accuracy'])
best_class_model = classification_results[best_class_model_name]['model']

print(f"\nBest classification model: {best_class_model_name}")
print(f"Accuracy: {classification_results[best_class_model_name]['accuracy']:.4f}")

# Define a simple trading strategy
def backtest_trading_strategy(model, X_test, y_test, starting_capital=10000):
    """Backtest a simple trading strategy using the classification model"""
    capital = starting_capital
    btc_holdings = 0
    transaction_log = []
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get actual prices (convert y_test to original scale)
    actual_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    for i in range(len(y_pred)-1):
        current_price = actual_prices[i]
        next_price = actual_prices[i+1]
        
        # Strategy: Buy if predicted price will go up, sell if predicted to go down
        if y_pred[i] == 1 and btc_holdings == 0:  # Predicted price increase & no holdings
            # Buy Bitcoin
            btc_to_buy = capital / current_price
            btc_holdings = btc_to_buy
            capital = 0
            transaction_log.append(f"Day {i}: Buy {btc_to_buy:.6f} BTC at ${current_price:.2f}")
        elif y_pred[i] == 0 and btc_holdings > 0:  # Predicted price decrease & have holdings
            # Sell Bitcoin
            capital = btc_holdings * current_price
            transaction_log.append(f"Day {i}: Sell {btc_holdings:.6f} BTC at ${current_price:.2f}, Capital: ${capital:.2f}")
            btc_holdings = 0
    
    # Final evaluation
    if btc_holdings > 0:
        final_capital = btc_holdings * actual_prices[-1]
        transaction_log.append(f"Final: Convert {btc_holdings:.6f} BTC to ${final_capital:.2f}")
    else:
        final_capital = capital
    
    profit_loss = final_capital - starting_capital
    profit_loss_percentage = (profit_loss / starting_capital) * 100
    
    return {
        'initial_capital': starting_capital,
        'final_capital': final_capital,
        'profit_loss': profit_loss,
        'profit_loss_percentage': profit_loss_percentage,
        'transaction_log': transaction_log
    }

# Backtest the trading strategy
strategy_results = backtest_trading_strategy(best_class_model, X_test_class, y_test, 10000)

print("\n=== Trading Strategy Backtest Results ===")
print(f"Initial Capital: ${strategy_results['initial_capital']:.2f}")
print(f"Final Capital: ${strategy_results['final_capital']:.2f}")
print(f"Profit/Loss: ${strategy_results['profit_loss']:.2f} ({strategy_results['profit_loss_percentage']:.2f}%)")

print("\nTransaction Log (first 5 and last 5 transactions):")
for log in strategy_results['transaction_log'][:5]:
    print(log)
print("...")
for log in strategy_results['transaction_log'][-5:]:
    print(log)

# %% [markdown]
# ## 12. Conclusion and Future Improvements

# %%
print("\n=== Conclusion ===")
print("In this project, we've successfully built and evaluated various machine learning models for Bitcoin price prediction.")

# Display the best models for different tasks
print("\nBest Models for Different Prediction Tasks:")
print(f"1. Price Regression: {best_reg_model_name if lstm_rmse > best_rf_rmse else 'LSTM'}")
print(f"2. Price Direction Classification: {best_class_model_name}")

print("\nKey Findings:")
print("1. Technical indicators provide valuable features for Bitcoin price prediction")
print("2. Time series models like LSTM can capture temporal dependencies in Bitcoin prices")
print("3. The cryptocurrency market's volatility makes precise prediction challenging")
print(f"4. Our trading strategy based on {best_class_model_name} showed {'promising' if strategy_results['profit_loss'] > 0 else 'limited'} results")

print("\nFuture Improvements:")
print("1. Incorporate sentiment analysis from social media and news")
print("2. Include macroeconomic indicators and market data")
print("3. Experiment with more advanced deep learning architectures")
print("4. Develop more sophisticated trading strategies")
print("5. Implement online learning to adapt to changing market conditions")

# %%
# Final chart showing prediction vs actual prices
plt.figure(figsize=(15, 8))

# Create date range for test data visualization
test_dates = df_daily.index[-len(y_test):]

# Plot actual prices
actual_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))
plt.plot(test_dates, actual_prices, label='Actual Prices', color='blue')

# Plot LSTM predictions
plt.plot(test_dates, lstm_predictions[:len(test_dates)], label='LSTM Predictions', color='red', alpha=0.7)

# Plot best regression model predictions
best_reg_predictions = best_model.predict(X_test)
best_reg_predictions = scaler_y.inverse_transform(best_reg_predictions.reshape(-1, 1))
plt.plot(test_dates, best_reg_predictions[:len(test_dates)], label=f'{best_reg_model_name} Predictions', color='green', alpha=0.7)

# Plot forecasted prices
plt.plot(future_dates, future_prices, 'o-', label='Future Forecast', color='purple')

plt.title('Bitcoin Price Prediction Summary')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nThank you for exploring Bitcoin price prediction with us!")