# Bitcoin Price Prediction Using Machine Learning

## Project Overview
This project aims to predict Bitcoin price movements using various machine learning techniques. We implement and compare multiple algorithms including Linear Regression, Decision Trees, Random Forests, Support Vector Machines, K-Nearest Neighbors, and Long Short-Term Memory (LSTM) neural networks. The project also explores the practical application of these predictions through a simulated trading strategy.

## Dataset
We use historical Bitcoin price data (OHLCV - Open, High, Low, Close, Volume) from the Bitstamp exchange, containing minute-level data from 2012 onwards. For this project, we resample the data to daily intervals to reduce noise and computational requirements.

## Project Structure
- `bitcoin_price_prediction.ipynb`: Main Jupyter notebook containing all code and analysis
- `best_lstm_model.keras`: Saved LSTM model for future use
- `requirements.txt`: Required packages to run the code
- `data/`: Directory containing the dataset (not included due to size)

## Features
- **Data Analysis & Visualization**: Exploration of Bitcoin historical data with various visualizations
- **Feature Engineering**: Creation of technical indicators (Moving Averages, RSI, MACD, etc.)
- **Multiple ML Models**: Implementation of various machine learning approaches
- **Deep Learning**: LSTM networks for time series forecasting
- **Model Comparison**: Comprehensive evaluation and comparison of different models
- **Trading Strategy**: Simulation of a simple trading strategy based on price direction predictions
- **Future Forecasting**: Multi-day price predictions

## Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook/Lab

### Installation
1. Clone this repository
2. Install required packages:
```
pip install -r requirements.txt
```
3. Download the Bitcoin historical data and place it in the `data/` directory (or update the path in the notebook)

### Running the Project
1. Open the Jupyter notebook:
```
jupyter notebook bitcoin_price_prediction.ipynb
```
2. Run all cells sequentially

## Methodology
1. **Data Preprocessing**: Clean missing values, resample to daily frequency
2. **Feature Engineering**: Add technical indicators and derived features
3. **Model Training**: Train multiple models on historical data
   - Traditional ML: Linear Regression, Decision Trees, Random Forest, SVM, KNN
   - Deep Learning: LSTM neural networks
4. **Evaluation**: Compare model performance using RMSE, MAE, R² for regression and accuracy, precision, recall, F1 for classification
5. **Optimization**: Hyperparameter tuning using Grid Search with Cross-Validation
6. **Application**: Simulate trading decisions based on price direction predictions

## Results Summary
- LSTM and Random Forest models generally achieve the best performance for price regression
- Classification models can predict price direction with accuracy better than random guessing
- Technical indicators provide valuable features for prediction
- The cryptocurrency market's inherent volatility makes precise prediction challenging
- Trading strategy simulation shows potential for profitable application

## Future Improvements
- Incorporate sentiment analysis from social media and news
- Include macroeconomic indicators and market data
- Experiment with more advanced deep learning architectures
- Develop more sophisticated trading strategies
- Implement online learning to adapt to changing market conditions

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras

## Author
TANG MINH VU

## Acknowledgments
- Course instructors of IT3190 (Nhập môn Học máy và Khai phá dữ liệu)
- Bitstamp for providing the historical data