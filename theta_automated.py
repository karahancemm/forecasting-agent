import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Theta
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from matplotlib import pyplot as plt

# --- Eval Metrics --- #
def eval_metrics(y_true, y_pred, insample):
    mae = mean_absolute_error(y_true, y_pred)
    mse = root_mean_squared_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true.replace(0, np.nan))).dropna().mean()*100
    naive_mae = insample.diff().abs().dropna().mean() 
    mase = (np.abs(y_true - y_pred).mean()) / naive_mae
    return {'MAE': mae, 'RMSE': mse, 'MAPE%': mape, 'MASE': mase}

def search_theta(unique_id, train, test, season_length, decomposition_type):
    train_df = pd.DataFrame({'unique_id': unique_id, 'ds': train.index, 'y': train.values})
    best = None
    results = []
    for s_length in season_length:
        for decomp_type in decomposition_type:
            model = StatsForecast(models = [Theta(season_length = s_length, decomposition_type = decomp_type)], freq = 'D', n_jobs = 1)
            model.fit(train_df)
            forecast = model.predict(h = len(test))
            forecast = forecast['Theta'].values
            metrics = eval_metrics(test, forecast, train)
            results.append({'model_name': 'Theta', 'season_length': s_length, 'decomposition_type': decomp_type, 'AIC': None, 'BIC': None, 'ljung_pvalue': None, **metrics,
            'forecast': forecast})
    return None, results


















