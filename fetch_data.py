#from Algorithmic_Stock_Trading_Public.get_yfinance_stock import get_year_stock
#from Algorithmic_Stock_Trading_Public.get_sp500_stocks_data import *
import os
import pandas as pd
import datetime as dt
import yfinance as yf

#plt.style.use('ggplot') 

def load_symbols():
    data = pd.read_csv("global_index_tickers_yf.csv")
    return data["Symbol"]

def save_stock(ticker='TSLA', years=1):
    # Specify date range
    start = dt.datetime(2021 - years, 1, 1)
    end = dt.datetime.now()

    # Get Stock Data in date range
    tickerData = yf.Ticker(ticker)
    df = tickerData.history(period='1d', start=start, end=end)

    # Select columns from raw data
    columns = ['Close']# 'Open', 'Volume', 'High', 'Low', 
    df = df[columns] 
    df = df.interpolate()
    df.to_csv('global_index_close_dfs/%s.csv'%(ticker))
    #print(df.tail())
    #return df

def save_from_symbols(folder="global_index_close_dfs", years=10):
    # Requires: a csv with column symbols containing ticker values
    # Inputs:   The name of an output folder 
    # Outputs:  All daily close price for symbols as csv for n years
    ticks = load_symbols()
    N = len(ticks)
    for i, symbol in enumerate(ticks):
        if i % int(N/10) == 0: print(i*100/N,'% complete')
        if not os.path.exists('global_index_close_dfs/%s.csv'%(symbol)):
            print('Fetching price data for %s'%(symbol))
            try:
                save_stock(ticker=symbol, years=years)
            except KeyError as e:
                print('\t KeyError while fetching:', e)
            
        else:
            print('Already have %s' % (symbol))

def combine_csv_folder(folder="global_index_close_dfs"):
    tickers = load_symbols()

    main_df = pd.DataFrame()

    for i, tick in enumerate(tickers):
        if os.path.exists('%s/%s.csv'%(folder,tick)):
            df = pd.read_csv('%s/%s.csv'%(folder,tick))
            df.set_index('Date', inplace=True)
            df.rename(columns = {'Close': tick}, inplace=True)

            if main_df.empty: main_df = df
            else: main_df = main_df.join(df, how='outer')

            if i % 10 == 0: print(i*100/len(tickers))
        else:print('Data for %s not found' % (tick))
        
    main_df = main_df.interpolate(axis="columns")
    main_df.to_csv('global_idx_close_price.csv')
    print(main_df.tail())

if __name__ == "__main__":
    #save_from_symbols()
    combine_csv_folder()
