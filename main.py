import numpy as np
import pandas as pd
from utils.data_loader import DataLoader
import statsmodels.api as sm
from tqdm import tqdm

def corr_process(df):
    try:
        df_copy = df.copy()
        df_copy.dropna(inplace=True)
        
        if len(df_copy) < 2:
            return np.nan, np.nan

        x = df_copy['ret']
        y = df_copy['signal']
        weights = df_copy['weight']
        
        weighted_mean_x = np.average(x, weights=weights)
        weighted_mean_y = np.average(y, weights=weights)
        weighted_cov = np.sum(weights * (x - weighted_mean_x) * (y - weighted_mean_y))
        weighted_std_x = np.sqrt(np.sum(weights * (x - weighted_mean_x) ** 2))
        weighted_std_y = np.sqrt(np.sum(weights * (y - weighted_mean_y) ** 2))
        if weighted_std_x * weighted_std_y == 0:
            weighted_correlation = np.nan
        else:
            weighted_correlation = weighted_cov / (weighted_std_x * weighted_std_y)
        
        correlation = x.corr(y, method='pearson')

    except Exception:
        weighted_correlation = np.nan
        correlation = np.nan
        
    return correlation, weighted_correlation

def error_factor(pb, con_np_rolling, bValue, bValue_lagged, master_dates, master_tickers):
    Y = pb - 1
    X1 = con_np_rolling / bValue
    X2 = bValue_lagged / bValue

    MISV_FY1 = pd.DataFrame(index=master_dates, columns=master_tickers)
    for date in tqdm(master_dates, desc='Fitting: '):
        y_series = Y.loc[date]
        x1_series = X1.loc[date]
        x2_series = X2.loc[date]
        
        df_regression = pd.concat([y_series, x1_series, x2_series], axis=1)
        df_regression.columns = ['y', 'x1', 'x2']
        df_regression.dropna(inplace=True)
        
        if len(df_regression) < 3: 
            continue
        
        y_reg = df_regression['y']
        X_reg = df_regression[['x1', 'x2']]
        
        model = sm.OLS(y_reg, X_reg).fit(method='qr')

        y_fitted = model.fittedvalues
        pb_fitted = y_fitted + 1
        pb_truth = pb.loc[date, pb_fitted.index]

        misv_fy1 = (pb_truth / pb_fitted) - 1
        MISV_FY1.loc[date] = misv_fy1


    MISV_FY1_rank = MISV_FY1.rank(axis=1, ascending=False)
    return MISV_FY1_rank

def get_corr(price, factors, negMarketValue):
    returns = {}
    horizons = [5, 10, 20, 60]

    for n in horizons:
        future_price = price.shift(-n)
        returns[n] = (future_price / price) - 1

    for n in horizons:
        returns_n = returns[n]
        daily_ic = []
        daily_weighted_ic = []
        for date in factors.index:
            df_today = pd.DataFrame({
            'signal': factors.loc[date],
            'ret': returns_n.loc[date],
            'weight': negMarketValue.loc[date]
            })

            ic, weighted_ic = corr_process(df_today)
            daily_ic.append(ic)
            daily_weighted_ic.append(weighted_ic)

        ic_series = pd.Series(daily_ic, index=factors.index)
        weighted_ic_series = pd.Series(daily_weighted_ic, index=factors.index)
        
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std != 0 else np.nan
        
        weighted_ic_mean = weighted_ic_series.mean()
        weighted_ic_std = weighted_ic_series.std()
        weighted_icir = weighted_ic_mean / weighted_ic_std if weighted_ic_std != 0 else np.nan
        
        print(f"** {n}日 因子表现报告 **")
        print(f"IC均值 (Mean IC): {ic_mean:.4f}")
        print(f"IC标准差 (IC Std): {ic_std:.4f}")
        print(f"信息比率 (ICIR): {icir:.4f}")
        print("-" * 20)
        print(f"加权IC均值 (Weighted Mean IC): {weighted_ic_mean:.4f}")
        print(f"加权IC标准差 (Weighted IC Std): {weighted_ic_std:.4f}")
        print(f"加权信息比率 (Weighted ICIR): {weighted_icir:.4f}")

def main():
    folder_path = 'data/'
    data_loader = DataLoader(folder_path)
    bValue, bValue_lagged, con_np_rolling, _, pb, negMarketValue, price, master_dates, master_tickers = data_loader.load()
    misv_fy1 = error_factor(pb, con_np_rolling, bValue, bValue_lagged, master_dates, master_tickers)
    get_corr(price, misv_fy1, negMarketValue)


if __name__ == '__main__':
    main()




