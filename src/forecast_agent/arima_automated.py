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

train_exog, test_exog = y_exog.iloc[:-test_size], y_exog[-test_size:]
"""

# --- Eval Metrics --- #
def eval_metrics(y_true, y_pred, insample):
    mae = mean_absolute_error(y_true, y_pred)
    mse = root_mean_squared_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true.replace(0, np.nan))).dropna().mean()*100
    naive_mae = insample.diff().abs().dropna().mean() 
    mase = (np.abs(y_true - y_pred).mean()) / naive_mae
    return {'MAE': mae, 'RMSE': mse, 'MAPE%': mape, 'MASE': mase}

# --- ARIMA --- #
def fit_sarima_with_diagnostics(train, exog_train, test, exog_test, order, seasonal_order, trend, ljung_lags):

    if seasonal_order[-1] == 0 and sum(seasonal_order[:-1]):
        return None
    
    try:
        model = SARIMAX(train, order = order, seasonal_order = seasonal_order, trend = trend, exog = exog_train, enforce_stationarity = False, enforce_invertibility = False, freq = 'D').fit(disp = False)
    except Exception as e:
        print(e)
        return None

    forecast = model.forecast(steps = len(test), exog = exog_test)
    metrics = eval_metrics(test, forecast, train)
    ljung_10 = acorr_ljungbox(model.resid, lags = ljung_lags[0], return_df = True).iloc[-1]
    ljung_20 = acorr_ljungbox(model.resid, lags = ljung_lags[-1], return_df = True).iloc[-1]
    ljung_pvalue = min(ljung_10['lb_pvalue'], ljung_20['lb_pvalue'])

    return {'order': order, 'seasonal_order': seasonal_order, 'trend': trend, 'AIC': model.aic, 'BIC': model.bic, 'MAE': metrics['MAE'], 
            'RMSE': metrics['RMSE'], 'MAPE%': metrics['MAPE%'], 'MASE': metrics['MASE'], 'ljung_pvalue': ljung_pvalue, 'forecast': forecast, 'model': model}

def search_sarima_orders(unique_id, train, exog_train, test, exog_test, p_values, d_values, q_values, seasonal_order, trend_options, ljung_lags):
    best = None
    results = []
    for trend in trend_options:
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    if p == 0 and d == 0 and q == 0:
                        continue
                    for s in seasonal_order:
                        if s[-1] == 0 or len(train) < 3 * s[-1] or sum(s[:-1]) != 0:
                            continue
                        res = fit_sarima_with_diagnostics(train, exog_train, test, exog_test, order = (p, d, q), seasonal_order= s, trend= trend, ljung_lags= ljung_lags)
                        if res is None:
                            continue
                        elif res['ljung_pvalue'] < 0.05:
                            res = None
                            continue
                        results.append(res)
                        if best is None or (res['MASE'], res['BIC']) < (best['MASE'], best['BIC']):
                            best = res

    return best, results

"""def main():

if __name__ == '__main__':
    main()"""

"""
p_values = [0, 1, 2, 3, 4, 5]
d_values = [0, 1]
q_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
s_values = [0, 7, 14, 30]
trend_options = ['n', 't']

best_result, all_results = search_sarima_orders(train, train_exog, test, test_exog, p_values, d_values, q_values, s_values, trend_options, 10)

if not all_results:
    raise RuntimeError('No ARIMA models converged')

leaderboard = pd.DataFrame({'order': [r['order'] for r in all_results],
                                'trend': [r['trend'] for r in all_results],
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

print("Best order:", best_result['order'])

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

print('Evaluation metrics:', eval_metrics(test, forecast, train))"""