# Dự đoán Giá Bitcoin Bằng Machine Learning

## Tổng quan dự án
Dự án này nhằm dự đoán biến động giá Bitcoin sử dụng các kỹ thuật toán machine learning khác nhau. Chúng tôi triển khai và so sánh nhiều thuật toán bao gồm Hồi quy tuyến tính (Linear Regression), Cây quyết định (Decision Trees), Rừng ngẫu nhiên (Random Forests), Máy vector hỗ trợ (SVM), K-láng giềng gần nhất (KNN) và mạng não LSTM (Long Short-Term Memory). Ngoài ra, dự án cũng áp dụng các dự đoán vào một chiến lược giao dịch giả lập.

## Dữ liệu
Dữ liệu lịch sử giá Bitcoin (OHLCV - Open, High, Low, Close, Volume) từ sàn giao dịch Bitstamp, chứa dữ liệu theo từng phút từ năm 2012. Để giảm nhiễu và yêu cầu tính toán, chúng tôi lấy mẫu lại dữ liệu theo ngày.

## Cấu trúc dự án
- `bitcoin_price_prediction.ipynb`: Notebook Jupyter chính chứa toàn bộ mã và phân tích
- `best_lstm_model.keras`: Mô hình LSTM đã huấn luyện để sử dụng sau này
- `requirements.txt`: Danh sách các thư viện cần thiết
- `data/`: Thư mục chứa dữ liệu (không bao gồm do dung lượng lớn)

## Tính năng
- **Phân tích & Tránh quan dữ liệu**: Khám phá dữ liệu lịch sử của Bitcoin với các biểu đồ trực quan
- **Xử lý dữ liệu & Tính năng**: Tạo các chỉ báo kỹ thuật (Moving Averages, RSI, MACD, ...)
- **Huấn luyện mô hình Machine Learning**: Triển khai các thuật toán ML khác nhau
- **Deep Learning**: Dự đoán chuỗi thời gian bằng mạng não LSTM
- **So sánh mô hình**: Đánh giá và so sánh hiệu suất của các mô hình
- **Chiến lược giao dịch**: Giả lập chiến lược giao dịch dựa trên dự đoán xu hướng giá
- **Dự đoán tương lai**: Dự đoán giá Bitcoin nhiều ngày

## Cài đặt và thiết lập

### Yêu cầu
- Python 3.7+
- Jupyter Notebook/Lab

### Cài đặt
1. Clone repository này:
2. Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```
3. Tải dữ liệu lịch sử Bitcoin và đặt vào thư mục `data/` (hoặc cập nhật đường dẫn trong notebook)

### Chạy dự án
1. Mở Jupyter notebook:
```
jupyter notebook bitcoin_price_prediction.ipynb
```
2. Chạy tất cả các cell theo thứ tự

## Phương pháp luận
1. **Tiền xử lý dữ liệu**: Lọc dữ liệu, lấy mẫu lại theo ngày
2. **Tính năng & Chỉ báo kỹ thuật**: Thêm các chỉ báo hỗ trợ dự đoán
3. **Huấn luyện mô hình**: Dùng nhiều kỹ thuật khác nhau
   - ML truyền thống: Hồi quy tuyến tính, Decision Trees, Random Forest, SVM, KNN
   - Deep Learning: Mạng LSTM
4. **Đánh giá hiệu suất**: Sử dụng RMSE, MAE, R² cho hồi quy và Accuracy, Precision, Recall, F1 cho phân loại
5. **Tối ưu hóa**: Tinh chỉnh siêu tham số bằng Grid Search với Cross-Validation
6. **Ứng dụng thực tế**: Giả lập quyết định giao dịch dựa trên dự đoán xu hướng giá

## Kết quả
- Mô hình LSTM và Random Forest đạt hiệu suất tốt nhất trong dự đoán giá
- Các mô hình phân loại có thể dự đoán xu hướng giá với độ chính xác tốt hơn ngẫu nhiên
- Các chỉ báo kỹ thuật cung cấp thông tin hữu ích cho việc dự đoán
- Tính biến động cao của thị trường tiền điện tử khiến dự đoán chính xác trở nên khó khăn
- Chiến lược giao dịch giả lập cho thấy tiềm năng áp dụng thực tế

## Cải tiến trong tương lai
- Kết hợp phân tích tâm lý từ mạng xã hội và tin tức
- Thêm các chỉ báo kinh tế vĩ mô và dữ liệu thị trường
- Thử nghiệm các mô hình deep learning tiến tiến hơn
- Xây dựng các chiến lược giao dịch phức tạp hơn
- Triển khai online learning để thích nghi động theo thị trường

## Yêu cầu
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras

## Tác giả
TANG MINH VU

## Lời cảm ơn
- Giáo viên hướng dẫn môn IT3190 (Nhập môn Học máy và Khai phá dữ liệu)
- Bitstamp cung cấp dữ liệu lịch sử

