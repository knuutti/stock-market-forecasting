import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf


def plot_stl_decomposition(indexes, series, trend, seasonal, resid):
    mu = resid.mean()
    sigma = resid.std()
    threshold = 3 * sigma

    # Find Outliers
    outliers = resid[np.abs(resid) > threshold]

    plt.figure(figsize=(10, 10))
    plt.subplot(4, 1, 1)
    plt.plot(indexes, series, label='Original', color='blue')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 2)
    plt.plot(indexes, trend, label='Trend', color='orange')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 3)
    plt.plot(indexes, seasonal, label='Seasonal', color='green')
    plt.legend(loc='upper left')
    plt.subplot(4, 1, 4)
    plt.plot(indexes, resid, label='Residual', color='red')
    plt.scatter(indexes[outliers.index], outliers, color='black', label='Outliers')
    plt.axhline(y=threshold, color='gray', linestyle='--', label='Thresholds')
    plt.axhline(y=-threshold, color='gray', linestyle='--')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    print(f"Number of outliers found: {len(outliers)}")
    print(f"Threshold: +/- {threshold:.4f}")


def plot_price_and_volume(indexes, series, volume):
    plt.figure(figsize=(10, 5))
    plt.plot(indexes, series, label='Open', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(indexes, volume, label='Volume', color='orange')
    trend_volume = volume.rolling(window=60).mean()
    plt.plot(indexes, trend_volume, label='Trend (60-day MA)', color='green')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend(loc='upper left')
    plt.show()


def plot_autocorrelation_analysis(data, column_name, title_suffix, lags=10, acf_lags=30):
    # Create a 1-10 day lagged dataset
    lagged = pd.DataFrame({column_name: data})
    for i in range(1, lags + 1):
        lagged[f'lag_{i}'] = lagged[column_name].shift(i)
    lagged = lagged.dropna()

    # Compute the correlation matrix
    corr = lagged.corr()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
    axes[0, 0].set_title(f'Heatmap of Correlation Matrix of {title_suffix}')

    # Lag plot
    pd.plotting.lag_plot(data, ax=axes[0, 1])
    axes[0, 1].set_title(f'Lag Plot of {title_suffix}')

    # Autocorrelation plot
    pd.plotting.autocorrelation_plot(data, ax=axes[1, 0])
    axes[1, 0].set_title(f'Autocorrelation Plot of {title_suffix}')

    # ACF (from statsmodels)
    plot_acf(data, lags=acf_lags, ax=axes[1, 1])
    axes[1, 1].set_title(f'Autocorrelation Function (ACF) of {title_suffix}')

    plt.tight_layout()
    plt.show()


def plot_returns(indexes, returns):
    plt.figure(figsize=(10, 5))
    plt.plot(indexes, returns, label='Returns', color='purple')
    # plt.title('Stock Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.show()


def plot_arima_forecast(train, test, forecast):
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label='Train', color='blue')
    plt.plot(test.index, test, label='Test', color='orange')
    plt.plot(test.index, forecast, label='Forecast', color='green')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('ARIMA Model Forecast')
    plt.legend()
    plt.show()


def plot_stl_arima_forecast(train_series, test_series, total_forecast):
    plt.figure(figsize=(10, 5))
    plt.plot(train_series, label='Train', color='blue')
    plt.plot(test_series, label='Actual Test', color='orange')
    plt.plot(total_forecast, label='True Forecast (STL + ARIMA)', color='green')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('STL + ARIMA Forecast')
    plt.legend()
    plt.show()


def plot_prediction(past_data, true_future, prediction, model_name, horizon_label):
    x_past = np.arange(0, len(past_data))

    true_vals = true_future
    pred_vals = prediction.flatten()
    x_future = np.arange(len(past_data) - 1, len(past_data) + len(true_vals))

    true_vals = np.insert(true_vals, 0, past_data[-1])
    pred_vals = np.insert(pred_vals, 0, past_data[-1])

    plt.figure(figsize=(10, 5))
    plt.plot(x_past, past_data, label='Past 63 Days', color='black')
    plt.plot(x_future, true_vals, label='Actual Prices', color='blue')
    plt.plot(x_future, pred_vals, label='Predicted Prices', color='red')
    plt.title(f'{horizon_label} Stock Price Prediction with {model_name}')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
