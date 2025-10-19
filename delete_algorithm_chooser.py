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

def rolling_backtest(y, X, model_search_fn, search_kwargs,
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
    fold_scores = []
    for idx, origin in enumerate(origins, start=1):
        y_train = y.iloc[:origin]
        y_test = y.iloc[origin:origin + horizon_days]
        if len(y_test) < horizon_days:
            break  # no more complete folds

        X_train = X.iloc[:origin] if X is not None else None
        X_test = X.iloc[origin:origin + horizon_days] if X is not None else None

        best_model, all_models = model_search_fn(
            train=y_train,
            exog_train=X_train,
            test=y_test,
            exog_test=X_test,
            **search_kwargs
        )
        if best_model is None:
            continue

        metrics = {
            'fold': idx,
            'origin': y.index[origin],
            'horizon': horizon_days,
            'family': name,
            'MASE': best_model['MASE'],
            'MAE': best_model['MAE'],
            'RMSE': best_model['RMSE'],
            #'LB_p10': best_model['LB_p10'],
            #'LB_p20': best_model['LB_p20'],
            #'LB_min': min(best_model['LB_p10'], best_model['LB_p20']),
            'LB_min': best_model['ljung_pvalue'],
            'order': best_model.get('order'),
            'seasonal_order': best_model.get('seasonal_order'),
            'trend': best_model.get('trend'),
            'seasonal': best_model.get('seasonal'),
            'seasonal_periods': best_model.get('seasonal_periods'),
        }
        fold_scores.append(metrics)

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
base_df = base_df[ts_columns + promotion]

# --- Formatting Data --- #
base_df['Date'] = pd.to_datetime(base_df['Date'], errors = 'coerce')
base_df.set_index('Date', inplace = True)
base_df.sort_index(inplace = True)
base_df = base_df.resample('D').agg({'Units_Sold': 'sum', 'Promotion_Flag': 'max'})
print(base_df)

# --- Train Test --- #
y_diff = base_df['Units_Sold'].diff().dropna()
weeks_for_test = 6
test_size = weeks_for_test * 7
y = base_df['Units_Sold']
train, test = y.iloc[:-test_size], y.iloc[-test_size:]

base_df['Promotion_Flag_1'] = base_df['Promotion_Flag'].shift(1).fillna(0)
base_df['Promotion_Flag_2'] = base_df['Promotion_Flag'].shift(2).fillna(0)
y_exog = base_df[['Promotion_Flag', 'Promotion_Flag_1', 'Promotion_Flag_2']]
y_exog = base_df[['Promotion_Flag']]

train_exog, test_exog = y_exog.iloc[:-test_size], y_exog[-test_size:]

# --- Fold Setup --- #
horizon_days = 28  # 4 weeks forecast per fold
origins = [
    len(y) - 3 * horizon_days,
    len(y) - 2 * horizon_days,
    len(y) - 1 * horizon_days,
]
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
q_values = [0, 1, 2, 3] #, 4, 5, 6, 7, 8]
s_values = [0, 7, 14] #, 30]
trend_options = ['n', 't']
seasonal_order = [(p, d, q, s) for p, d, q, s in product(p_values, d_values, q_values, s_values)]
seasonal_order = list(set(seasonal_order))
lags = [10, 20]
sarima_kwargs = {'p_values' : p_values, 'd_values': d_values, 'q_values': q_values, 's_values': seasonal_order, 'trend_options': trend_options, 'ljung_lags': lags}

# --- SARIMA --- #
sarima_backtest = rolling_backtest(train, train_exog, search_sarima_orders, sarima_kwargs, horizon_days, origins, 'SARIMA')
"""best_result, all_results = search_sarima_orders(train, train_exog, test, test_exog, p_values, d_values, q_values, s_values, trend_options, [10, 20])

if not all_results:
    raise RuntimeError('No SARIMA models converged')

leaderboard_sarimax = pd.DataFrame({'hyperparameter_1': [r['hyperparameter_1'] for r in all_results],
                                'hyperparameter_2': [r['hyperparameter_2'] for r in all_results],
                                'hyperparameter_3': [r['hyperparameter_3'] for r in all_results],
                                'AIC': [r['AIC'] for r in all_results],
                                'BIC': [r['BIC'] for r in all_results],
                                'MAE': [r['MAE'] for r in all_results],
                                'RMSE': [r['RMSE'] for r in all_results],
                                'MAPE%': [r['MAPE%'] for r in all_results],
                                'MASE': [r['MASE'] for r in all_results],
                                'LjungBox_p': [r['ljung_pvalue'] for r in all_results]}).sort_values(['MASE', 'BIC']).reset_index(drop = True)
leaderboard_sarimax['Algorithm'] = 'SARIMA'"""

# -- ETS Prep -- ##
trend_options = ['add', 'mul', None]
seasonal_options = ['add', 'mul', None]
period_options = [7, 14] #, 30]
lags = [10, 20]
ets_kwargs = {'trend_options': trend_options, 'seasonal_options': seasonal_options, 'period_options': period_options, 'ljung_lags': lags}

# --- ETS --- #
ets_backtest = rolling_backtest(train, train_exog, search_ets, ets_kwargs, horizon_days, origins, 'ETS')

all_folds = pd.concat([sarima_backtest, ets_backtest], ignore_index = True)
#summary = (all_folds.groupby('family')[['MASE', 'MAE', 'RMSE', 'LB_p10', 'LB_p20', 'LB_min']].agg(['mean', 'std']))
summary = (all_folds.groupby('family')[['MASE', 'MAE', 'RMSE', 'LB_min']].agg(['mean', 'std']))
print(summary)

candidates = (
    all_folds
    .groupby('family')[['MASE', 'LB_min']]
    .mean()
    .sort_values(['MASE'])
)
# optionally filter for white-noise residuals:
candidates = candidates[candidates['LB_min'] >= 0.05]
print(candidates)

exit()

# --- OLD CODE, DELETE MAYBE --- #

best_result, all_results = search_ets(train, train_exog, test_exog, test, trend_options, seasonal_options, period_options, [10, 20])
if not all_results:
    raise RuntimeError('No ETS models converged')

leaderboard_ets = pd.DataFrame({'hyperparameter_1': [r['hyperparameter_1'] for r in all_results],
                                'hyperparameter_2': [r['hyperparameter_2'] for r in all_results],
                                'AIC': [r['AIC'] for r in all_results],
                                'BIC': [r['BIC'] for r in all_results],
                                'MAE': [r['MAE'] for r in all_results],
                                'RMSE': [r['RMSE'] for r in all_results],
                                'MAPE%': [r['MAPE%'] for r in all_results],
                                'MASE': [r['MASE'] for r in all_results],
                                'LjungBox_p': [r['ljung_pvalue'] for r in all_results]}).sort_values(['MASE', 'BIC']).reset_index(drop = True)
leaderboard_ets['Algorithm'] = 'ETS'

comarison_columns = ['hyperparameter_1', 'hyperparameter_2', 'hyperparameter_3', 'MASE', 'BIC', 'LjungBox_p', 'Algorithm']

all_columns = leaderboard_sarimax.columns.union(leaderboard_ets.columns)
for i in all_columns:
    if i not in leaderboard_sarimax.columns:
        leaderboard_sarimax[i] = None
for i in all_columns:
    if i not in leaderboard_ets.columns:
        leaderboard_ets[i] = None

comparison_sarimax = leaderboard_sarimax[comarison_columns]
comparison_ets = leaderboard_ets[comarison_columns]

comparison_df = pd.concat([comparison_sarimax, comparison_ets])
comparison_df = comparison_df.sort_values(by = ['MASE', 'BIC'])

print(comparison_df)

