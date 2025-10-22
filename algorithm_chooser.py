import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from arima_automated import fit_sarima_with_diagnostics, search_sarima_orders
from ets_automated import ets_with_diagostics, search_ets
from theta_automated import search_theta
from itertools import product
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _clone_kwargs(kwargs):
    return {k: (v.copy() if isinstance(v, list) else v) for k, v in kwargs.items()}

def _freeze_kwargs(kwargs):
    def freeze(value):
        if isinstance(value, list):
            return tuple(freeze(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted(k, freeze(v)) for k, v in value.items())
        return value
    return tuple(sorted((k, freeze(v)) for k, v in kwargs.items()))

def rolling_backtest(unique_id, y, y_exog, model_search_fn, search_kwargs,
                     horizon_days, origins, name):
    """
    y: full Series (daily demand)
    X: exogenous DataFrame (can be None)
    model_search_fn: e.g. search_sarima_orders or search_ets
    search_kwargs: dict with the grids you already pass
    horizon_days: forecast horizon per fold (e.g. 28 for 4 weeks)
    origins: list of int endpoints for each training window (indices into y)
             e.g. [len(y) - 3*horizon, len(y) - 2*horizon, len(y) - horizon]
    name: str label ("SARIMA", "ETS", ...) so you can tag results
    """
    current_kwargs = _clone_kwargs(search_kwargs)
    cache = {}
    fold_scores = []

    for idx, origin in enumerate(origins, start=1):
        y_train = y.iloc[:origin]
        y_test = y.iloc[origin:origin + horizon_days]
        if len(y_test) < horizon_days:
            continue
        
        if y_exog is not None:
            fold_exog_train = y_exog.loc[y_train.index]
            fold_exog_test = y_exog.loc[y_test.index]
        else:
            fold_exog_train = fold_exog_test = None

        key = origin, _freeze_kwargs(current_kwargs)
        if key in cache:
            best_model, all_models = cache[key]
        else:
            if name == 'SARIMAX':
                best_model, all_models = model_search_fn(unique_id, y_train, fold_exog_train, y_test, fold_exog_test, **current_kwargs)
            elif name == 'ETS':
                best_model, all_models = model_search_fn(unique_id, y_train, fold_exog_train, y_test, fold_exog_test, **current_kwargs)
            elif name == 'Theta':
                best_model, all_models = model_search_fn(unique_id, y_train, y_test, **current_kwargs)

            cache[key] = (best_model, all_models)

        """if best_model is None:
            continue"""

        for model in all_models:
            fold_scores.append({
                'fold': idx,
                'origin': y.index[origin],
                'horizon': horizon_days,
                'family': name,
                'MASE': model.get('MASE', None),
                'MAE': model.get('MAE', None),
                'RMSE': model.get('RMSE', None),
                'BIC': model.get('BIC', None),
                'LB_min': model.get('ljung_pvalue', None),
                'order': model.get('order', None),
                'seasonal_order': model.get('seasonal_order', None),
                'trend': model.get('trend', None),
                'seasonal': model.get('seasonal', None),
                'seasonal_periods': model.get('seasonal_periods', None),
                'season_length': model.get('season_length', None),
                'decomposition_type': model.get('decomposition_type', None)
            })

        if name == 'SARIMA' and best_model.get('order') is not None:
            order = best_model['order']
            current_kwargs['p_values'] = [order[0]]
            current_kwargs['d_values'] = [order[1]]
            current_kwargs['q_values'] = [order[2]]
            current_kwargs['seasonal_order'] = [best_model['seasonal_order']]
            current_kwargs['trend_options'] = [best_model['trend']]

    return pd.DataFrame(fold_scores)

# --- Folder --- #
folder = '/Users/cemkarahan/Desktop/Python_Projects/stock_opt/'

# --- Read File --- #
df = pd.read_csv(folder + 'supply_chain_dataset1.csv')

# --- Field Definitions --- #
ts_columns = ['Date', 'Units_Sold']
sku = ['SKU_ID']
region = ['Region']
warehouse = ['Warehouse_ID']
stock_management = ['Units_Sold', 'Inventory_Level', 'Supplier_Lead_Time_Days', 'Reorder_Point', 'Order_Quantity', 'Unit_Cost', 'Unit_Price']
promotion = ['Promotion_Flag']

# --- Base Data Frame for TSA --- #
base_df = df[sku + ts_columns + promotion]
base_df = base_df[base_df['SKU_ID'] == 'SKU_1']
unique_id = base_df['SKU_ID'].iat[0]
base_df = base_df[ts_columns + promotion]

# --- Formatting Data --- #
base_df['Date'] = pd.to_datetime(base_df['Date'], errors = 'coerce')
base_df.set_index('Date', inplace = True)
base_df.sort_index(inplace = True)
base_df = base_df.resample('D').agg({'Units_Sold': 'sum', 'Promotion_Flag': 'max'})
print(base_df)

# --- Train Test Split --- #
y_diff = base_df['Units_Sold'].diff().dropna()
weeks_for_test = 6
test_size = weeks_for_test * 7
y = base_df['Units_Sold']
train, test = y.iloc[:-test_size], y.iloc[-test_size:]

"""base_df['Promotion_Flag_1'] = base_df['Promotion_Flag'].shift(1).fillna(0)
base_df['Promotion_Flag_2'] = base_df['Promotion_Flag'].shift(2).fillna(0)
y_exog = base_df[['Promotion_Flag', 'Promotion_Flag_1', 'Promotion_Flag_2']]"""
y_exog = base_df[['Promotion_Flag']]

train_exog, test_exog = y_exog.iloc[:-test_size], y_exog[-test_size:]

# --- Fold Setup --- #
n = len(train)
horizon_days = len(test)  # 4 weeks forecast per fold
origins = [n - 3*horizon_days, n - 2*horizon_days, n - horizon_days]
origins = [o for o in origins if o > 0]  # keep valid splits

# --- Eval Metrics --- #
def eval_metrics(y_true, y_pred, insample):
    mae = mean_absolute_error(y_true, y_pred)
    mse = root_mean_squared_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true.replace(0, np.nan))).dropna().mean()*100
    naive_mae = insample.diff().abs().dropna().mean() 
    mase = (np.abs(y_true - y_pred).mean()) / naive_mae
    return {'MAE': mae, 'RMSE': mse, 'MAPE%': mape, 'MASE': mase}

# --- SARIMA Preperation --- #
p_values = [0, 1, 2] #, 3, 4, 5]
d_values = [0, 1]
q_values = [0, 1, 2] #, 3] #, 4, 5, 6, 7, 8]
s_values = [0, 7, 14] #, 30]
trend_options = ['n', 't']
seasonal_order = [(p, d, q, s) for p, d, q, s in product(p_values, d_values, q_values, s_values)]
seasonal_order = list(set(seasonal_order))
lags = [10, 20]
sarima_kwargs = {'p_values' : p_values, 'd_values': d_values, 'q_values': q_values, 'seasonal_order': seasonal_order, 'trend_options': trend_options, 'ljung_lags': lags}

# --- SARIMA --- #
sarima_backtest = rolling_backtest(unique_id, train, y_exog, search_sarima_orders, sarima_kwargs, horizon_days, origins, 'SARIMAX')

# -- ETS Prep -- ##
trend_options = ['add', 'mul', None]
seasonal_options = ['add', 'mul', None]
period_options = [7, 14] #, 30]
lags = [10, 20]
ets_kwargs = {'trend_options': trend_options, 'seasonal_options': seasonal_options, 'period_options': period_options, 'ljung_lags': lags}

# --- ETS --- #
ets_backtest = rolling_backtest(unique_id, train, y_exog, search_ets, ets_kwargs, horizon_days, origins, 'ETS')
    
# --- Theta --- #
decomposition_options = ['additive', 'mmultiplicative', 'none', 'log-additive']
season_options = [1, 7, 30, 90]
theta_kwargs = {'season_length': season_options, 'decomposition_type': decomposition_options}

theta_backtest = rolling_backtest(unique_id, train, None, search_theta, theta_kwargs, horizon_days, origins, 'Theta')

# --- Combining All --- #
all_folds = pd.concat([sarima_backtest, ets_backtest, theta_backtest], ignore_index = True)
all_folds.to_csv(folder + 'all_models.csv')

#summary = (all_folds.groupby('family')[['MASE', 'MAE', 'RMSE', 'LB_p10', 'LB_p20', 'LB_min']].agg(['mean', 'std']))
summary = (all_folds.groupby('family')[['MASE', 'MAE', 'RMSE', 'LB_min']].agg(['mean', 'std']))
candidates = all_folds.sort_values(['MASE', 'LB_min'], ascending = [True, False]).reset_index(drop = True)

# optionally filter for white-noise residuals:
mask = (~candidates['family'].isin(['SARIMAX', 'ETS'])) | (candidates['LB_min'] >= 0.05)
candidates = candidates[mask]
winners = (candidates.sort_values(['family', 'fold', 'MASE']).groupby(['family', 'fold'], as_index = False).head(1))
print(candidates)
winners.to_csv(folder + 'models.csv')
print(winners)