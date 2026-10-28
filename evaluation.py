import numpy as np
import pandas as pd


def naive_forecast(series, origins, n_future):
    # Repeat the last observed value over the whole horizon
    last_values = np.asarray(series)[origins]
    return np.repeat(last_values[:, None], n_future, axis=1)


def get_actuals(series, origins, n_future):
    series = np.asarray(series)
    return np.array([series[t + 1 : t + 1 + n_future] for t in origins])


def forecast_metrics(actuals, predictions):
    # Pooled over all test windows and horizon steps
    errors = np.asarray(actuals) - np.asarray(predictions)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    return rmse, mae


def results_table(results):
    # results: {model_name: {horizon_label: (rmse, mae)}}
    rows = {}
    for model_name, horizons in results.items():
        row = {}
        for horizon_label, (rmse, mae) in horizons.items():
            row[(horizon_label, 'RMSE')] = rmse
            row[(horizon_label, 'MAE')] = mae
        rows[model_name] = row
    return pd.DataFrame.from_dict(rows, orient='index').round(3)
