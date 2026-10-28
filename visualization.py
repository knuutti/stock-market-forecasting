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


def plot_forecast_comparison(past_dates, past_data, future_dates, true_future, forecasts, horizon_label, save_path=None):
    # forecasts: {model_name: forecast over the horizon}
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']

    # Start every future line at the last observed value so that it joins the history
    x_future = np.insert(np.asarray(future_dates), 0, past_dates[-1])

    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    ax.plot(past_dates, past_data, color='#0b0b0b', linewidth=2, label='Actual')
    ax.plot(x_future, np.insert(true_future, 0, past_data[-1]), color='#0b0b0b', linewidth=2)
    for (model_name, forecast), color in zip(forecasts.items(), colors):
        ax.plot(x_future, np.insert(np.ravel(forecast), 0, past_data[-1]), color=color, linewidth=2, label=model_name)
    ax.axvline(past_dates[-1], color='#898781', linestyle='--', linewidth=1)
    ax.annotate('Forecast origin', xy=(past_dates[-1], 1), xycoords=('data', 'axes fraction'),
                xytext=(-6, -6), textcoords='offset points', ha='right', va='top', fontsize=12, color='#52514e')

    ax.set_title(f'IBM opening price: {horizon_label} forecasts from {np.datetime_as_string(past_dates[-1], unit="D")}', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Opening price (USD)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(color='#e5e4e0', linewidth=0.8)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.legend(loc='best', fontsize=12, frameon=False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=100, facecolor='white')
    plt.show()
