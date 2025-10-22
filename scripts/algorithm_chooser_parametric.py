import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.graphics.gofplots import qqplot
from arima_automated import fit_sarima_with_diagnostics, search_sarima_orders
from ets_automated import ets_with_diagostics, search_ets
from itertools import product
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)



def rolling_backtest(y, X, model_search_fn, search_kwargs,
                     horizon_days, origins, name):
    current_kwargs = {k: (v.copy() if isinstance(v, list) else v)
                      for k, v in search_kwargs.items()}
    cache = {}
    fold_scores = []

    for idx, origin in enumerate(origins, start=1):
        y_train = y.iloc[:origin]
        y_test = y.iloc[origin:origin + horizon_days]
        if len(y_test) < horizon_days:
            continue

        X_train = X.iloc[:origin] if X is not None else None
        X_test = X.iloc[origin:origin + horizon_days] if X is not None else None

        key = (origin, tuple(sorted(current_kwargs.items())))
        if key in cache:
            best_model, all_models = cache[key]
        else:
            best_model, all_models = model_search_fn(
                train=y_train,
                exog_train=X_train,
                test=y_test,
                exog_test=X_test,
                **current_kwargs
            )
            cache[key] = (best_model, all_models)

        if best_model is None:
            continue

        for model in all_models:
            fold_scores.append({
                'fold': idx,
                'origin': y.index[origin],
                'horizon': horizon_days,
                'family': name,
                'MASE': model.get('MASE'),
                'MAE': model.get('MAE'),
                'RMSE': model.get('RMSE'),
                'BIC': model.get('BIC'),                         # may be None
                'LB_min': model.get('ljung_pvalue'),             # may be None
                'order': model.get('order'),                     # SARIMA only
                'seasonal_order': model.get('seasonal_order'),   # SARIMA only
                'trend': model.get('trend'),                     # ETS only
                'seasonal': model.get('seasonal'),               # ETS only
                'seasonal_periods': model.get('seasonal_periods')# ETS only
            })

        # keep your SARIMA warm-start; skip for others
        if name == 'SARIMA' and best_model.get('order') is not None:
            order = best_model['order']
            current_kwargs['p_values'] = [order[0]]
            current_kwargs['d_values'] = [order[1]]
            current_kwargs['q_values'] = [order[2]]
            current_kwargs['seasonal_order'] = [best_model['seasonal_order']]
            current_kwargs['trend_options'] = [best_model['trend']]

    return pd.DataFrame(fold_scores)



