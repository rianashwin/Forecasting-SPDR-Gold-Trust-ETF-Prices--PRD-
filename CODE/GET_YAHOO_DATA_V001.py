from __main__ import *

#######################################################
# Define functions
#######################################################

def create_placeholder_df(start_date, end_date):
    '''
    arg: start and end dates to pull data
    outputs: single dataframe indexed by date, placeholder for merging in next step
    '''    
    
    placeholder_df_index = pd.date_range(start_date, end_date)
    placeholder_df = pd.DataFrame({'Date':placeholder_df_index})
    
    return placeholder_df

def compile_data(start_date, end_date, placeholder_df, ticker_list):
    '''
    arg: sheet of tickers to extract from Yahoo Finance
    outputs: single dataframe indexed by date, with one column for adjusted closing prices for each ticker
    '''

    print(placeholder_df.shape)

    symbols_list = ticker_list['Ticker Code'].tolist()
    ticker_names = ticker_list['Description'].tolist()

    for this_tickers_symbol, this_tickers_name in zip(symbols_list,ticker_names):

        print("\n", this_tickers_symbol, this_tickers_name)
        
        this_tickers_raw_json = YahooFinancials(this_tickers_symbol)
        this_tickers_raw_json = this_tickers_raw_json.get_historical_price_data(start_date, end_date, 'daily')
        this_tickers_raw_json = pd.DataFrame(this_tickers_raw_json[this_tickers_symbol]['prices'])[['formatted_date','adjclose']]
        
        this_tickers_raw_json['formatted_date'] = pd.to_datetime(this_tickers_raw_json['formatted_date'])
        this_tickers_raw_json = this_tickers_raw_json.rename(columns={'adjclose':this_tickers_name})        
        
        placeholder_df = placeholder_df.merge(this_tickers_raw_json,left_on='Date',right_on='formatted_date',how='left')
        placeholder_df.drop('formatted_date',axis=1,inplace=True)
        print(this_tickers_raw_json.shape,placeholder_df.shape)
        
    placeholder_df.to_csv(r'.\DATA\RAW_DATA.csv'.format(this_tickers_name),index=False)

    print("\nData extracted from {} to {}..".format(start_date, end_date))
