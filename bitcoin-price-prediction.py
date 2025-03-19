# %% [markdown]
# # Dự đoán giá Bitcoin theo giờ
# Nhập môn Học máy và Khai phá dữ liệu (IT3190)

# %% [markdown]
# ## 1. Import các thư viện cần thiết

# %%
# Import thư viện
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

# Các mô hình hồi quy từ slide
from sklearn.linear_model import LinearRegression, Ridge  # Lecture 3
from sklearn.tree import DecisionTreeRegressor  # Lecture 8
from sklearn.ensemble import RandomForestRegressor  # Lecture 8
from sklearn.neighbors import KNeighborsRegressor  # Lecture 7
from sklearn.svm import SVR  # Lecture 11

# %% [markdown]
# ## 2. Tải và tiền xử lý dữ liệu

# %%
# Tải dữ liệu
print("Đang tải dữ liệu lịch sử Bitcoin...")
df = pd.read_csv('btcusd_1-min_data.csv')
print(f"Kích thước dữ liệu: {df.shape}")

# %%
# Tiền xử lý dữ liệu
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Timestamp', inplace=True)
print("Xem 5 dòng đầu tiên:")
print(df.head())

# %%
# Kiểm tra giá trị thiếu
print("Kiểm tra giá trị thiếu:")
print(df.isnull().sum())
df.dropna(inplace=True)
print(f"Kích thước dữ liệu sau khi loại bỏ giá trị thiếu: {df.shape}")

# %% [markdown]
# ## 3. Tổng hợp dữ liệu theo giờ

# %%
# Lấy mẫu dữ liệu theo giờ thay vì theo phút
df_hourly = df.resample('H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print(f"Kích thước dữ liệu sau khi tổng hợp theo giờ: {df_hourly.shape}")
print("Dữ liệu theo giờ (5 dòng đầu):")
print(df_hourly.head())

# %% [markdown]
# ## 4. Thêm các chỉ báo kỹ thuật (đặc trưng)

# %%
def add_features(df):
    """Thêm các đặc trưng kỹ thuật vào DataFrame"""
    # Đường trung bình động theo giờ
    df['MA6'] = df['Close'].rolling(window=6).mean()  # MA 6 giờ
    df['MA12'] = df['Close'].rolling(window=12).mean()  # MA 12 giờ
    df['MA24'] = df['Close'].rolling(window=24).mean()  # MA 24 giờ (1 ngày)
    
    # Chỉ số sức mạnh tương đối (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD - chỉ báo động lượng
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std_dev * 2)
    df['BB_Lower'] = df['MA20'] - (std_dev * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['MA20']
    
    # Độ biến động (Volatility)
    df['Volatility'] = df['Close'].pct_change().rolling(window=12).std() * 100
    
    # Tỷ lệ thay đổi giá (Rate of Change)
    df['ROC6'] = ((df['Close'] - df['Close'].shift(6)) / df['Close'].shift(6)) * 100
    
    # Độ chênh lệch giá
    df['Close_Open_Diff'] = df['Close'] - df['Open']
    df['High_Low_Diff'] = df['High'] - df['Low']
    
    # Đặc trưng về thời gian
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    
    # Mã hóa giờ và ngày theo hình tròn (tránh bước nhảy từ 23 sang 0)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
    df['Day_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
    df['Day_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
    
    # Thêm biến mục tiêu cho việc dự đoán
    for i in range(1, 13):  # Tạo mục tiêu cho 12 giờ tới
        df[f'Next_{i}h_Price'] = df['Close'].shift(-i)
    
    return df

# %%
# Thêm các đặc trưng kỹ thuật vào dữ liệu
df_hourly = add_features(df_hourly)
df_hourly.dropna(inplace=True)
print(f"Kích thước dữ liệu sau khi thêm đặc trưng: {df_hourly.shape}")

# %% [markdown]
# ## 5. Trực quan hóa dữ liệu

# %%
# Vẽ biểu đồ giá Bitcoin theo giờ (500 giờ gần nhất)
plt.figure(figsize=(15, 6))
df_hourly['Close'].tail(500).plot()
plt.title('Biến động giá Bitcoin theo giờ (500 giờ gần nhất)')
plt.xlabel('Thời gian')
plt.ylabel('Giá (USD)')
plt.grid(True)
plt.show()

# %%
# Tạo ma trận tương quan
plt.figure(figsize=(14, 12))
# Chỉ chọn một số cột quan trọng để tránh ma trận quá lớn
correlation_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'MA6', 'MA12', 'MA24', 'RSI', 'MACD', 
                       'BB_Width', 'Volatility', 'Next_1h_Price']
correlation = df_hourly[correlation_columns].corr()
mask = np.triu(correlation)
sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Ma trận tương quan giữa các đặc trưng')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Chuẩn bị dữ liệu cho huấn luyện mô hình

# %%
# Chọn đặc trưng và biến mục tiêu
target_column = 'Next_1h_Price'  # Dự đoán giá cho giờ tiếp theo

# Loại bỏ các cột không sử dụng làm đặc trưng
feature_columns = [col for col in df_hourly.columns if col not in 
                   ['Open', 'High', 'Low', 'Close', 'Hour', 'DayOfWeek'] + 
                   [f'Next_{i}h_Price' for i in range(1, 13)]]

X = df_hourly[feature_columns]
y = df_hourly[target_column]

print(f"Số lượng đặc trưng: {len(feature_columns)}")
print(f"Các đặc trưng: {feature_columns}")

# %%
# Chuẩn hóa dữ liệu
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Chia dữ liệu thành tập huấn luyện và kiểm tra (không xáo trộn dữ liệu chuỗi thời gian)
test_size = 0.2
train_size = int(len(X_scaled) * (1 - test_size))

X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# %% [markdown]
# ## 7. Huấn luyện và đánh giá các mô hình hồi quy

# %%
def evaluate_model(model, X_train, X_test, y_train, y_test, scaler_y):
    """Đánh giá hiệu suất của một mô hình hồi quy"""
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    
    # Chuyển đổi về giá thực tế
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_actual = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Tính các chỉ số đánh giá
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_pred_actual': y_pred_actual,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# %%
# Huấn luyện mô hình Linear Regression (Lecture 3)
print("Huấn luyện mô hình Linear Regression...")
lr_model = LinearRegression()
lr_results = evaluate_model(lr_model, X_train, X_test, y_train, y_test, scaler_y)

print("Kết quả đánh giá:")
print(f"RMSE: ${lr_results['rmse']:.2f}")
print(f"MAE: ${lr_results['mae']:.2f}")
print(f"R^2: {lr_results['r2']:.4f}")

# %%
# Huấn luyện mô hình Ridge Regression (Lecture 3)
print("Huấn luyện mô hình Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_results = evaluate_model(ridge_model, X_train, X_test, y_train, y_test, scaler_y)

print("Kết quả đánh giá:")
print(f"RMSE: ${ridge_results['rmse']:.2f}")
print(f"MAE: ${ridge_results['mae']:.2f}")
print(f"R^2: {ridge_results['r2']:.4f}")

# %%
# Huấn luyện mô hình Decision Tree Regression (Lecture 8)
print("Huấn luyện mô hình Decision Tree...")
dt_model = DecisionTreeRegressor(random_state=42)
dt_results = evaluate_model(dt_model, X_train, X_test, y_train, y_test, scaler_y)

print("Kết quả đánh giá:")
print(f"RMSE: ${dt_results['rmse']:.2f}")
print(f"MAE: ${dt_results['mae']:.2f}")
print(f"R^2: {dt_results['r2']:.4f}")

# %%
# Huấn luyện mô hình Random Forest Regression (Lecture 8)
print("Huấn luyện mô hình Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_results = evaluate_model(rf_model, X_train, X_test, y_train, y_test, scaler_y)

print("Kết quả đánh giá:")
print(f"RMSE: ${rf_results['rmse']:.2f}")
print(f"MAE: ${rf_results['mae']:.2f}")
print(f"R^2: {rf_results['r2']:.4f}")

# %%
# Huấn luyện mô hình K-Nearest Neighbors Regression (Lecture 7)
print("Huấn luyện mô hình KNN...")
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_results = evaluate_model(knn_model, X_train, X_test, y_train, y_test, scaler_y)

print("Kết quả đánh giá:")
print(f"RMSE: ${knn_results['rmse']:.2f}")
print(f"MAE: ${knn_results['mae']:.2f}")
print(f"R^2: {knn_results['r2']:.4f}")

# %%
# Huấn luyện mô hình Support Vector Regression (Lecture 11)
print("Huấn luyện mô hình SVR...")
svr_model = SVR(kernel='rbf', C=1.0)
svr_results = evaluate_model(svr_model, X_train, X_test, y_train, y_test, scaler_y)

print("Kết quả đánh giá:")
print(f"RMSE: ${svr_results['rmse']:.2f}")
print(f"MAE: ${svr_results['mae']:.2f}")
print(f"R^2: {svr_results['r2']:.4f}")

# %% [markdown]
# ## 8. So sánh và lựa chọn mô hình tốt nhất

# %%
# So sánh hiệu suất của các mô hình
models = ['Linear Regression', 'Ridge Regression', 'Decision Tree', 
          'Random Forest', 'KNN', 'SVR']
results = [lr_results, ridge_results, dt_results, rf_results, knn_results, svr_results]

# Vẽ biểu đồ so sánh RMSE
plt.figure(figsize=(14, 6))
rmse_values = [result['rmse'] for result in results]
plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
plt.title('So sánh RMSE giữa các mô hình')
plt.xlabel('Mô hình')
plt.ylabel('RMSE (USD)')
plt.xticks(rotation=45)
for i, v in enumerate(rmse_values):
    plt.text(i, v + 5, f'${v:.2f}', ha='center')
plt.tight_layout()
plt.show()

# %%
# Vẽ biểu đồ so sánh R^2
plt.figure(figsize=(14, 6))
r2_values = [result['r2'] for result in results]
plt.bar(models, r2_values, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
plt.title('So sánh R² giữa các mô hình')
plt.xlabel('Mô hình')
plt.ylabel('R²')
plt.xticks(rotation=45)
for i, v in enumerate(r2_values):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.show()

# %%
# Xác định mô hình tốt nhất dựa trên RMSE (giá trị càng thấp càng tốt)
best_model_index = np.argmin(rmse_values)
best_model_name = models[best_model_index]
best_model = results[best_model_index]['model']
best_results = results[best_model_index]

print(f"\nMô hình có hiệu suất tốt nhất dựa trên RMSE: {best_model_name}")
print(f"RMSE: ${best_results['rmse']:.2f}")
print(f"MAE: ${best_results['mae']:.2f}")
print(f"R^2: {best_results['r2']:.4f}")

# %% [markdown]
# ## 9. Tối ưu hóa siêu tham số cho mô hình tốt nhất

# %%
# Tối ưu hóa siêu tham số cho mô hình Random Forest (nếu đó là mô hình tốt nhất)
if best_model_name == "Random Forest":
    print("\nĐang tối ưu hóa siêu tham số cho Random Forest...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42), 
        param_grid, 
        cv=3, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Siêu tham số tốt nhất: {grid_search.best_params_}")
    
    # Cập nhật mô hình tốt nhất với siêu tham số tối ưu
    best_model = grid_search.best_estimator_
    
    # Đánh giá lại mô hình tối ưu
    best_results = evaluate_model(best_model, X_train, X_test, y_train, y_test, scaler_y)
    
    print("Kết quả sau khi tối ưu hóa:")
    print(f"RMSE: ${best_results['rmse']:.2f}")
    print(f"MAE: ${best_results['mae']:.2f}")
    print(f"R^2: {best_results['r2']:.4f}")
    
    # Hiển thị tầm quan trọng của đặc trưng
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 đặc trưng quan trọng nhất')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 10. Dự đoán giá Bitcoin cho 12 giờ tới

# %%
def predict_next_hours(model, last_data, scaler_y, hours=12):
    """Dự đoán giá Bitcoin cho một số giờ tiếp theo"""
    forecasted_prices = []
    
    # Lấy dữ liệu mới nhất
    current_data = last_data.copy()
    
    for _ in range(hours):
        # Dự đoán giá cho giờ tiếp theo
        prediction_scaled = model.predict(current_data.reshape(1, -1))[0]
        
        # Chuyển đổi về giá thực tế
        prediction = scaler_y.inverse_transform([[prediction_scaled]])[0][0]
        forecasted_prices.append(prediction)
        
        # Trong thực tế, ta cần cập nhật các đặc trưng để dự đoán giờ tiếp theo
        # Đây là một cách đơn giản hóa
    
    return forecasted_prices

# %%
# Lấy dữ liệu mới nhất để dự đoán
last_data = X_scaled[-1]

# Dự đoán giá cho 12 giờ tới
future_prices = predict_next_hours(best_model, last_data, scaler_y, hours=12)

# Tạo dãy thời gian cho dự báo
last_time = df_hourly.index[-1]
future_times = [last_time + dt.timedelta(hours=i+1) for i in range(12)]

# Hiển thị kết quả dự báo
print("\nDự đoán giá Bitcoin trong 12 giờ tới:")
for time, price in zip(future_times, future_prices):
    print(f"{time.strftime('%Y-%m-%d %H:%M')}: ${price:.2f}")

# %%
# Trực quan hóa dự báo
plt.figure(figsize=(15, 6))

# Lấy dữ liệu lịch sử để vẽ biểu đồ
historical_times = df_hourly.index[-48:]  # 48 giờ cuối
historical_prices = df_hourly['Close'].iloc[-48:].values

# Vẽ dữ liệu lịch sử và dự báo
plt.plot(historical_times, historical_prices, 'b-', label='Giá lịch sử')
plt.plot(future_times, future_prices, 'r--', marker='o', label='Giá dự báo')

plt.title(f'Dự báo giá Bitcoin trong 12 giờ tới (sử dụng {best_model_name})')
plt.xlabel('Thời gian')
plt.ylabel('Giá (USD)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Kết luận

# %%
print("\n=== Kết luận ===")
print(f"Mô hình {best_model_name} đã được chọn là mô hình tốt nhất cho dự đoán giá Bitcoin theo giờ.")
print(f"RMSE của mô hình: ${best_results['rmse']:.2f}")
print(f"R² của mô hình: {best_results['r2']:.4f}")

print("\nCác mô hình đã sử dụng từ slide môn học:")
print("1. Linear Regression (Lecture 3): Mô hình hồi quy tuyến tính cơ bản")
print("2. Ridge Regression (Lecture 3): Hồi quy với điều chuẩn L2")
print("3. Decision Tree (Lecture 8): Cây quyết định cho hồi quy")
print("4. Random Forest (Lecture 8): Tập hợp nhiều cây quyết định")
print("5. KNN (Lecture 7): K-Nearest Neighbors Regression")
print("6. SVR (Lecture 11): Support Vector Regression")

if best_model_name == "Random Forest" and 'feature_importance' in locals():
    print(f"\nTop 3 đặc trưng quan trọng nhất:")
    for i in range(3):
        print(f"{i+1}. {feature_importance['Feature'].iloc[i]}: {feature_importance['Importance'].iloc[i]:.4f}")

print("\nLưu ý: Thị trường tiền điện tử rất biến động và dự đoán này chỉ nên được sử dụng như một tham khảo, không nên dùng làm cơ sở duy nhất cho quyết định đầu tư.")