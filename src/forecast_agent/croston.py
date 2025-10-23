import pandas as pd
import numpy as np
from train_test import train_test_split
from eval import eval_metrics
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- Folders --- #
folder = 'C:/Users/ckarahan/Desktop/Python Projects/Forecast-Agent/data/raw/'
folder_models = 'C:/Users/ckarahan/Desktop/Python Projects/Forecast-Agent/models/'

# --- Read File --- #
df = pd.read_csv(folder + 'supply_chain_dataset1.csv')
unique_id = 'SKU_2' # df['SKU_ID'].iloc[0]
train, test, train_exog, test_exog, y_exog = train_test_split(df, unique_id)

def search_croston(train, test, alpha_options, lags):
    best = None
    results = []
    non_zero = np.where(train > 0)[0]
    if len(non_zero) < 1:
        return None
    for alpha in alpha_options:
        def croston_with_eval(train, non_zero, test, alpha):
            z = train.iloc[non_zero]
            p = np.diff(np.concatenate(([non_zero[0]], non_zero)))
            z_hat = np.zeros_like(z, dtype = float)
            p_hat = np.zeros_like(p, dtype = float)
            z_hat[0], p_hat[0] = z[0], p[0]

            for t in range(1, len(z)):
                z_hat[t] = alpha*z[t] + (1-alpha)*z_hat[t-1]
                p_hat[t] = alpha*p[t] + (1-alpha)*p_hat[t-1]

            if p_hat[-1] == 0:
                croston_level = 0
            else:
                croston_level = z_hat[-1] / p_hat[-1]
            forecast = pd.series(croston_level, index = test.index)
            residuals = (test - forecast)
            ljung_10 = acorr_ljungbox(residuals, lags = lags[0], return_df = True).iloc[-1]
            ljung_20 = acorr_ljungbox(residuals, lags = lags[-1], return_df = True).iloc[-1]
            ljung_pvalue = min(ljung_10['lb_pvalue'], ljung_20['lb_pvalue'])
            metrics = eval_metrics(test, forecast, train)
            return {'alpha': alpha, 'MAE': metrics['MAE'], 'RMSE': metrics['RMSE'], 'MAPE%': metrics['MAPE%'], 'MASE': metrics['MASE'], 'ljung_pvalue': ljung_pvalue}
        model = croston_with_eval(train, non_zero, test, alpha)
        model.append(results)
        if best is None or (model['MASE'], model['BIC']) < (best['MASE'], best['BIC']):
            best = model
    return best, results    
                                    
                                    
            

    

# making zero demand
rng = np.random.default_rng(seed = 42)
missind_idx = rng.choice(train.index, size = 40, replace = False)
train.loc[missind_idx] = np.nan


# --- Croston --- #
alpha = 0.2
non_zero = np.where(train > 0)[0]

z = train.iloc[non_zero]

p = np.diff(np.concatenate(([non_zero[0]], non_zero)))

z_hat = np.zeros_like(z, dtype = float)
p_hat = np.zeros_like(p, dtype = float)
z_hat[0], p_hat[0] = z[0], p[0]

for t in range(1, len(z)):
    z_hat[t] = alpha*z[t] + (1 - alpha)*z_hat[t-1]
    p_hat[t] = alpha*p[t] + (1 - alpha)*p_hat[t-1]

if p_hat[-1] == 0:
    croston_level = 0.0
else:
    croston_level = z_hat[-1] / p_hat[-1]

forecast = pd.Series(croston_level, index = test.index)
print(forecast)



