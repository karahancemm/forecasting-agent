import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

"""# --- Folder --- #
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
test_size = int(len(y_diff)*0.15)
y = base_df['Units_Sold']
train, test = y.iloc[:-test_size], y.iloc[-test_size:]

base_df['Promotion_Flag_1'] = base_df['Promotion_Flag'].shift(1).fillna(0)
base_df['Promotion_Flag_2'] = base_df['Promotion_Flag'].shift(2).fillna(0)
y_exog = base_df[['Promotion_Flag', 'Promotion_Flag_1', 'Promotion_Flag_2']]
y_exog = base_df[['Promotion_Flag']]

train_exog, test_exog = y_exog.iloc[:-test_size], y_exog.iloc[-test_size:]"""

# --- Eval Metrics --- #
def eval_metrics(y_true, y_pred, insample):
    mae = mean_absolute_error(y_true, y_pred)
    mse = root_mean_squared_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true.replace(0, np.nan))).dropna().mean()*100
    naive_mae = insample.diff().abs().dropna().mean() 
    mase = (np.abs(y_true - y_pred).mean()) / naive_mae
    return {'MAE': mae, 'RMSE': mse, 'MAPE%': mape, 'MASE': mase}

def ets_with_diagostics(train, exog_train, exog_test, test, error, trend, seasonal, period, ljung_lags):
    seasonal_periods = period if seasonal is not None else None
    try:
        ets = ETSModel(train,error = error, trend = trend, seasonal = seasonal, seasonal_periods = seasonal_periods).fit(optimized = True)
    except Exception as e:
        print('exception: ', e)
    forecast = ets.forecast(steps = len(test))
    metrics = eval_metrics(test, forecast, train)
    ets_fitted = ets.fittedvalues
    residuals_ets = train - ets_fitted
    reg_model = sm.OLS(residuals_ets, sm.add_constant(exog_train['Promotion_Flag'])).fit()
    train_adjustment = reg_model.predict(sm.add_constant(exog_train['Promotion_Flag']))
    test_adjustment = reg_model.predict(sm.add_constant(exog_test['Promotion_Flag']))
    train_hybrid = ets_fitted + train_adjustment
    forecast_hybrid = forecast + test_adjustment
    residuals = (test - forecast_hybrid)
    ljung_10 = acorr_ljungbox(residuals, lags = ljung_lags[0], return_df = True).iloc[-1]
    ljung_20 = acorr_ljungbox(residuals, lags = ljung_lags[-1], return_df = True).iloc[-1]
    ljung_pvalue = min(ljung_10['lb_pvalue'], ljung_20['lb_pvalue'])

    return {'trend': trend, 'seasonal': seasonal, 'seasonal_periods': seasonal_periods, 'AIC': ets.aic, 'BIC': ets.bic, 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE'], 
            'MAPE%': metrics['MAPE%'], 'MASE': metrics['MASE'], 'ljung_pvalue': ljung_pvalue, 'forecast': forecast_hybrid, 'model': ets}

def search_ets(unique_id, train, exog_train, test, exog_test, error_options, trend_options, seasonal_options, period_options, ljung_lags):
    best = None
    results = []
    for error in error_options:
      for trend in trend_options:
          for seasonal in seasonal_options:
              for period in period_options:
                  if seasonal is None and period != period_options[0]:
                      continue
                  with warnings.catch_warnings():
                      warnings.filterwarnings(
                      "ignore",
                      message="overflow encountered in matmul",
                      category=RuntimeWarning,
                      module="statsmodels.tsa.holtwinters.model",
                  )
                      model = ets_with_diagostics(train, exog_train, exog_test, test, error, trend, seasonal, period, ljung_lags)
                  if model is None:
                      continue
                  results.append(model)
                  if best is None or (model['MASE'], model['BIC']) < (best['MASE'], best['BIC']):
                      best = model
    return best, results

"""
trend_options = ['add', 'mul', None]
seasonal_options = ['add', 'mul', None]
period_options = [7, 14, 30]

best_result, all_results = search_ets(train, train_exog, test_exog, test, trend_options, seasonal_options, period_options, 10)
if not all_results:
    raise RuntimeError('No ETS models converged')

leaderboard = pd.DataFrame({'trend': [r['trend'] for r in all_results],
                                'seasonal': [r['seasonal'] for r in all_results],
                                'AIC': [r['AIC'] for r in all_results],
                                'BIC': [r['BIC'] for r in all_results],
                                'MAE': [r['MAE'] for r in all_results],
                                'RMSE': [r['RMSE'] for r in all_results],
                                'MAPE%': [r['MAPE%'] for r in all_results],
                                'MASE': [r['MASE'] for r in all_results],
                                'LjungBox_p': [r['ljung_pvalue'] for r in all_results]}).sort_values(['MASE', 'BIC']).reset_index(drop = True)

print('Top ARIMA candidates (sorted by MASE then BIC):')
print(leaderboard.head(5))

best_model = best_result['model']
forecast = best_result['forecast']

print("Best trend option: ", best_result['trend'])
print('Best seasonal option: ', best_result['seasonal'])

for key, value in best_result.items():
    if key not in ('model', 'forecast'):
        print(f"{key}: {value}")


residuals = best_model.resid
residuals.plot(title='ARIMA residuals')
plt.axhline(0, color='black', linewidth=0.8)
plt.show()

plot_acf(residuals, lags=30)
plt.show()
plot_pacf(residuals, lags=30)
plt.show()

ljung = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print(ljung)

residuals.hist(bins=20)
plt.title('Residual histogram')
plt.show()

qqplot(residuals, line='s')
plt.title('Residual QQ plot')
plt.show()

comparison_df = pd.DataFrame({'test': test.values, 'arima_forecast': forecast}, index=test.index)
print(comparison_df)
print(f"test sum: {np.sum(comparison_df['test'])}, forecast sum: {np.sum(comparison_df['arima_forecast'])}")
"""