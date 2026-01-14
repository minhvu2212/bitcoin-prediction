# Bitcoin Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)](https://scikit-learn.org/)

A machine learning project for predicting Bitcoin prices on an hourly basis using various regression models. The system analyzes historical Bitcoin price data and technical indicators to forecast short-term price movements.

## Features

- **Data Processing**: Converts minute-level Bitcoin data to hourly aggregates
- **Technical Indicators**: Implements MA, RSI, MACD, Bollinger Bands, and more
- **Multiple ML Models**: Compares 6 different regression algorithms
- **12-Hour Forecasting**: Predicts Bitcoin prices for the next 12 hours
- **Performance Metrics**: Evaluates models using RMSE, MAE, R², and MAPE

## Models Implemented

| Model | Description |
|-------|-------------|
| Linear Regression | Basic linear model for baseline comparison |
| Ridge Regression | L2 regularization to prevent overfitting |
| Decision Tree | Non-linear tree-based regressor |
| Random Forest | Ensemble of decision trees |
| K-Nearest Neighbors | Distance-based prediction |
| Support Vector Regression | Kernel-based regression |

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/tang-vu/bitcoin-prediction.git
cd bitcoin-prediction

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

### Running the Python Script

```bash
python bitcoin-price-prediction.py
```

### Using the Jupyter Notebook

```bash
jupyter notebook bitcoin-price-prediction.ipynb
```

## Data

The project uses Bitcoin price data in CSV format with the following columns:
- `Timestamp`: Unix timestamp
- `Open`: Opening price
- `High`: Highest price
- `Low`: Lowest price
- `Close`: Closing price
- `Volume`: Trading volume

### Data Source

You can download Bitcoin historical data from:
- [Kaggle Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- [CryptoDataDownload](https://www.cryptodatadownload.com/)

Place the data file as `btcusd_1-min_data.csv` in the project root.

## Technical Indicators

The following technical indicators are computed:

- **Moving Averages (MA)**: 6h, 12h, 24h periods
- **RSI (Relative Strength Index)**: 14-period momentum indicator
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands
- **Price Change**: Hour-over-hour percentage change
- **Volume MA**: Moving average of trading volume

## Model Performance

Based on testing with historical data:

| Model | RMSE (USD) | MAE (USD) | R² | MAPE |
|-------|------------|-----------|-----|------|
| **Linear Regression** | $321.63 | $157.52 | 0.9998 | 0.37% |
| Ridge Regression | $338.98 | $170.83 | 0.9998 | 0.40% |
| Decision Tree | $7,371.42 | $2,365.37 | 0.8869 | 3.58% |
| Random Forest | $7,400.27 | $2,284.47 | 0.8860 | 3.33% |
| KNN | $8,285.57 | $4,057.83 | 0.8571 | 8.76% |
| SVR | $11,978.55 | $5,660.74 | 0.7012 | 9.83% |

Linear Regression achieved the best performance for hourly price prediction.

## Project Structure

```
bitcoin-prediction/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── bitcoin-price-prediction.py      # Main Python script
├── bitcoin-price-prediction.ipynb   # Jupyter notebook version
├── best_lstm_model.keras            # Pre-trained model (optional)
├── .gitignore                       # Git ignore rules
└── btcusd_1-min_data.csv           # Data file (not included)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

⚠️ **Important**: Cryptocurrency markets are highly volatile and unpredictable. These predictions are for educational and research purposes only. Do not use them as the sole basis for investment decisions.

## Acknowledgments

- Historical Bitcoin data providers
- Scikit-learn team for the excellent ML library
- Open source community

---

**Star ⭐ this repository if you find it helpful!**