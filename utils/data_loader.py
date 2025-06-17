import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys


class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

        bValue = pd.read_csv(f'{self.folder_path}bValue.csv')
        self.bValue = bValue.rename(columns={'TShEquity': 'bValue'})
        self.con_np_rolling = pd.read_csv(f'{self.folder_path}con_np_rolling.csv')
        self.marketValue = pd.read_csv(f'{self.folder_path}marketValue.csv')
        pb = pd.read_csv(f'{self.folder_path}PB.csv')
        self.pb = pb.rename(columns={'PB': 'pb'})

        # load data for IC calculation
        negMarketValue = pd.read_csv(f'{self.folder_path}negMarketValue.csv')
        self.negMarketValue = negMarketValue.rename(columns={'tradeDate': 'date'})
        price = pd.read_csv(f'{self.folder_path}twap09310940price.csv')
        self.price = price.rename(columns={'twap': 'price'})

        print("All data loaded successfully!")

    def data_pivot(self):
        self.bValue['date'] = self.bValue['date'].astype(str)
        self.con_np_rolling['date'] = self.con_np_rolling['date'].astype(str)
        self.marketValue['date'] = self.marketValue['date'].astype(str)
        self.pb['date'] = self.pb['date'].astype(str)
        self.price['date'] = self.price['date'].astype(str)
        
        self.bValue['date'] = pd.to_datetime(self.bValue['date'])
        self.bValue_lagged = self.bValue.shift(1)
        self.con_np_rolling['date'] = pd.to_datetime(self.con_np_rolling['date'])
        self.marketValue['date'] = pd.to_datetime(self.marketValue['date'])
        self.pb['date'] = pd.to_datetime(self.pb['date'])
        self.negMarketValue['date'] = pd.to_datetime(self.negMarketValue['date'])
        self.price['date'] = pd.to_datetime(self.price['date'])

        self.bValue = self.bValue.pivot(index='date', columns='ticker', values='bValue')
        self.bValue_lagged = self.bValue_lagged.pivot(index='date', columns='ticker', values='bValue')
        self.con_np_rolling = self.con_np_rolling.pivot(index='date', columns='ticker', values='con_np_roll')
        self.marketValue = self.marketValue.pivot(index='date', columns='ticker', values='marketValue')
        self.pb = self.pb.pivot(index='date', columns='ticker', values='pb')
        self.negMarketValue = self.negMarketValue.pivot(index='date', columns='ticker', values='negMarketValue')
        self.price = self.price.pivot(index='date', columns='ticker', values='price')

    def data_alignment(self):
        '''
        Must be executed after data_pivot method !!!
        '''
        # master date index list
        start_date = '2021-01-01'
        end_date = '2024-05-30'
        self.master_dates = self.marketValue.loc[start_date:end_date].index
        print(f'standard time index composed, number of trade dates: {len(self.master_dates)}')
        print(f'trade date range: {self.master_dates.min()} ~ {self.master_dates.max()}')

        # master tickers list
        all_tickers = set()
        df_list_to_align = [self.bValue, self.con_np_rolling, self.marketValue, self.pb, self.negMarketValue, self.price]
        for df in df_list_to_align:
            all_tickers.update(df.columns)
        self.master_tickers = sorted(list(all_tickers))
        print(f'standard tickers list composed, number of tickers: {len(self.master_tickers)}')

        # forward fill the bValue
        self.bValue = self.bValue.reindex(index=self.master_dates, columns=self.master_tickers)
        self.bValue_lagged = self.bValue_lagged.reindex(index=self.master_dates, columns=self.master_tickers)
        self.bValue = self.bValue.ffill()
        self.bValue_lagged = self.bValue_lagged.ffill()
        
        # market value neutralization
        self.con_np_rolling = self.con_np_rolling.reindex(index=self.master_dates, columns=self.master_tickers)
        self.con_np_rolling = self.neutralization()

        self.marketValue = self.marketValue.reindex(index=self.master_dates, columns=self.master_tickers)
        self.pb = self.pb.reindex(index=self.master_dates, columns=self.master_tickers)
        self.negMarketValue = self.negMarketValue.reindex(index=self.master_dates, columns=self.master_tickers)
        self.price = self.price.reindex(index=self.master_dates, columns=self.master_tickers)

    def neutralization(self):
        con_np_neutralized = pd.DataFrame(index=self.master_dates, columns=self.master_tickers)
        for date in self.master_dates:
            con_series = self.con_np_rolling.loc[date]
            mv_series = self.marketValue.loc[date]

            df_cross_section = pd.concat([con_series, mv_series], axis=1)
            df_cross_section.columns = ['con', 'mv']
            df_cross_section.dropna(inplace=True)
            if len(df_cross_section) < 2:
                continue
            
            Y = df_cross_section['con']
            X = sm.add_constant(df_cross_section['mv'])
            model = sm.OLS(Y, X)
            results = model.fit()
            con_np_neutralized.loc[date] = results.resid

        return con_np_neutralized.astype(np.float64)
        

    def load(self):
        '''
        Main logic of data preprocessing logits:

            load -> pivot method -> alignment (time and tickers)
        
        The following is the complete pipeline
        '''
        self.data_pivot()
        self.data_alignment()
        print('------------------------------------------------------------------------------')
        return self.bValue, self.bValue_lagged, self.con_np_rolling, self.marketValue, self.pb, self.negMarketValue, self.price, self.master_dates, self.master_tickers