import warnings

import numpy as np
import pandas as pd
import pmdarima as pm
from joblib import Parallel, delayed
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL


def fit_arima_forecast(train, test, m=7):
    auto_model = pm.auto_arima(train,
                               start_p=1, start_q=1,
                               test='adf',       # Use 'adf' test to find the optimal 'd'
                               max_p=5, max_q=5, # Maximum p and q
                               m=m,              # Seasonality
                               d=None,           # Let the test determine 'd'
                               seasonal=True,    # Seasonality
                               start_P=0,
                               D=0,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)

    # Best order
    best_order = auto_model.order
    print(f"Best Order: {best_order}")

    # Fit ARIMA model
    model = ARIMA(train, order=best_order)
    model_fit = model.fit()
    # Forecast
    forecast = model_fit.forecast(steps=len(test))

    # Calculate the RMSE, MAE and R2 Score
    rmse = root_mean_squared_error(test, forecast)
    mae = mean_absolute_error(test, forecast)
    r2 = r2_score(test, forecast)

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

    return forecast, model_fit, best_order


def fit_stl_arima_forecast(train_series, test_series, period=252, seasonal=13):
    stl_train = STL(train_series, period=period, seasonal=seasonal, robust=True)
    res_train = stl_train.fit()
    n_periods = len(test_series)

    # ARIMA on Residuals
    auto_model_resid = pm.auto_arima(res_train.resid,
                                     start_p=1, start_q=1,
                                     test='adf',
                                     max_p=5, max_q=5,
                                     m=7, seasonal=True,
                                     d=None, D=0,
                                     trace=False,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     stepwise=True)

    best_order_resid = auto_model_resid.order
    print(f"Best Order for Residuals: {best_order_resid}")

    model_resid = ARIMA(res_train.resid, order=best_order_resid)
    model_resid_fit = model_resid.fit()

    resid_forecast = pd.Series(model_resid_fit.forecast(steps=n_periods),
                                 index=test_series.index)

    # ARIMA on Trend
    auto_model_trend = pm.auto_arima(res_train.trend,
                                     start_p=1, start_q=1,
                                     d=1,
                                     max_p=3, max_q=3,
                                     seasonal=False, # No seasonality in the trend
                                     trace=False,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     stepwise=True)

    best_order_trend = auto_model_trend.order
    print(f"Best Order for Trend: {best_order_trend}")

    model_trend = ARIMA(res_train.trend, order=best_order_trend)
    model_trend_fit = model_trend.fit()

    trend_forecast = pd.Series(model_trend_fit.forecast(steps=n_periods),
                                 index=test_series.index)

    # Seasonal Forecasting by repeating the last observed seasonal cycle
    last_season = res_train.seasonal[-period:]
    num_reps = int(np.ceil(n_periods / period))
    seasonal_forecast_tiled = np.tile(last_season, num_reps)
    # Trim to the exact length of the test set
    seasonal_forecast = pd.Series(seasonal_forecast_tiled[:n_periods],
                                    index=test_series.index)

    # Combine forecasts
    total_forecast = trend_forecast + seasonal_forecast + resid_forecast

    # Calculate the RMSE, MAE and R2 Score for the hybrid model
    rmse_hybrid = root_mean_squared_error(test_series, total_forecast)
    mae_hybrid = mean_absolute_error(test_series, total_forecast)
    r2_hybrid = r2_score(test_series, total_forecast)

    print(f"RMSE (Hybrid): {rmse_hybrid}")
    print(f"MAE (Hybrid): {mae_hybrid}")
    print(f"R2 Score (Hybrid): {r2_hybrid}")

    return total_forecast, trend_forecast, seasonal_forecast, resid_forecast, best_order_resid, best_order_trend


def select_stl_arima_orders(train_series, period=252, seasonal=13):
    # Choose the ARIMA orders once, on data before the test period only
    res_train = STL(train_series, period=period, seasonal=seasonal, robust=True).fit()
    order_resid = pm.auto_arima(res_train.resid,
                                start_p=1, start_q=1,
                                test='adf',
                                max_p=5, max_q=5,
                                m=7, seasonal=True,
                                d=None, D=0,
                                trace=False,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True).order
    order_trend = pm.auto_arima(res_train.trend,
                                start_p=1, start_q=1,
                                d=1,
                                max_p=3, max_q=3,
                                seasonal=False,
                                trace=False,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True).order
    print(f"Best Order for Residuals: {order_resid}")
    print(f"Best Order for Trend: {order_trend}")
    return order_resid, order_trend


def arima_forecast_with_fallback(series, order, n_future):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return ARIMA(series, order=order).fit().forecast(steps=n_future), False
        except np.linalg.LinAlgError:
            # The default fit occasionally fails numerically; refit without the stationarity constraints
            model = ARIMA(series, order=order, enforce_stationarity=False, enforce_invertibility=False)
            return model.fit().forecast(steps=n_future), True


def stl_arima_forecast_from_origin(history, n_future, order_resid, order_trend, period=252, seasonal=13):
    # STL is two-sided, so it is refitted on the history up to the origin only
    res = STL(history, period=period, seasonal=seasonal, robust=True).fit()
    resid_forecast, resid_fallback = arima_forecast_with_fallback(res.resid, order_resid, n_future)
    trend_forecast, trend_fallback = arima_forecast_with_fallback(res.trend, order_trend, n_future)
    # Seasonal forecast repeats the last observed seasonal cycle
    seasonal_forecast = np.asarray(res.seasonal)[-period:][:n_future]
    forecast = np.asarray(trend_forecast) + seasonal_forecast + np.asarray(resid_forecast)
    return forecast, resid_fallback or trend_fallback


def rolling_stl_arima_forecast(series, origins, n_future, period=252, seasonal=13, n_jobs=-1):
    series = np.asarray(series, dtype=float)
    order_resid, order_trend = select_stl_arima_orders(series[:origins[0] + 1], period, seasonal)
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(stl_arima_forecast_from_origin)(series[:t + 1], n_future, order_resid, order_trend, period, seasonal)
        for t in origins
    )
    forecasts, fallbacks = zip(*outputs)
    print(f"Origins refitted without stationarity constraints: {sum(fallbacks)} of {len(origins)}")
    return np.array(forecasts)
