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
weeks_for_test = 4
test_size = weeks_for_test * 7
y = base_df['Units_Sold']
train, test = y.iloc[:-test_size], y.iloc[-test_size:]

#base_df['Promotion_Flag_1'] = base_df['Promotion_Flag'].shift(1).fillna(0)
#base_df['Promotion_Flag_2'] = base_df['Promotion_Flag'].shift(2).fillna(0)
#y_exog = base_df[['Promotion_Flag', 'Promotion_Flag_1', 'Promotion_Flag_2']]
y_exog = base_df[['Promotion_Flag']]

train_exog, test_exog = y_exog.iloc[:-test_size], y_exog[-test_size:]

# --- SARIMAX --- #
order = [1, 1, 2]
seasonal_order = [0, 0, 0, 14]
trend = 't'
model = SARIMAX(train, order = order, seasonal_order = seasonal_order, trend = trend, exog = train_exog, enforce_stationarity = False, enforce_invertibility = False, freq = 'D').fit(disp = False)

results = pd.DataFrame({'model': ['ARIMA'], 
                        'AIC': [model.aic], 
                        'BIC': [model.bic]})

residuals = model.resid
residuals.plot(title= 'ARIMA RESIDS')
plt.show()

plot_acf(residuals, lags = 30)
plt.show()
plot_pacf(residuals, lags = 30)
plt.show()

ljung = acorr_ljungbox(residuals, lags = [10, 20], return_df = True)
print(ljung)

residuals.hist(bins = 20)
plt.title('Residual histogram')

qqplot(residuals, line = 's')
plt.title('Residual QQ plot')
plt.show()

sarimax_test_values = model.forecast(steps = len(test), exog = test_exog)

#print(model.summary())
comparison_df = pd.DataFrame({'test': test.values, 'arima_forecast': sarimax_test_values}, index = test.index)
print(comparison_df)
print(f'test sum: {np.sum(comparison_df['test'])}, forecast sum: {np.sum(comparison_df['arima_forecast'])}')


# --- Eval Metrics --- #
def eval_metrics(y_true, y_pred, insample):
    mae = mean_absolute_error(y_true, y_pred)
    mse = root_mean_squared_error(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true.replace(0, np.nan))).dropna().mean()*100
    naive_mae = insample.diff().abs().dropna().mean() 
    mase = (np.abs(y_true - y_pred).mean()) / naive_mae
    return {'MAE': mae, 'RMSE': mse, 'MAPE%': mape, 'MASE': mase}

print(eval_metrics(test, sarimax_test_values, train))

comparison_df.plot(
    y=['test', 'arima_forecast'],
    figsize=(10, 5),
    ylabel='Units Sold',
    title='Actual vs ETS + Promo Forecast'
)
plt.legend(['Actual', 'ETS forecast'])
plt.tight_layout()
plt.show()