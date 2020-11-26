"""
Script to extract data from Yahoo! Finance
"""

from __main__ import *

#######################################################
# Define functions
#######################################################

def create_placeholder_df(start_date, end_date):
    """
    Creates placholder dataframe with one row per each date within specified range

    Parameters
    ----------
    start_date : string
        Date to start pulling data from
    end_date : string
        Date to end pulling data

    Returns
    -------
    placeholder_df
        A dataframe with one row per each date within specified range
    """  
    
    placeholder_df_index = pd.date_range(start_date, end_date)
    placeholder_df = pd.DataFrame({'Date':placeholder_df_index})
    
    return placeholder_df

def compile_data(start_date, end_date, placeholder_df, ticker_list):
    """
    Extracts data from Yahoo! Finance and outputs to csv

    Parameters
    ----------
    start_date : string
        Date to start pulling data from
    end_date : string
        Date to end pulling data
    placeholder_df : dataframe
        A dataframe with one row per each date within specified range
    ticker_list: dataframe
        A dataframe containing ticker symbol and description

    Returns
    -------
    None
    """  

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
        
    #placeholder_df.to_csv(r'.\DATA\RAW_DATA.csv'.format(this_tickers_name),index=False)

    print("\nData extracted from {} to {}..".format(start_date, end_date))

    return placeholder_df


def delete_db_rows(this_db_configs):
    """
    Deletes rows in db table

    Parameters
    ----------
    this_db_configs : json
        DB login credentials

    Returns
    -------
    None
    """    
    conn = None
    try:
        # connect to the PostgreSQL database
        conn = psycopg2.connect(user = this_db_configs['user'],
                                        password = this_db_configs['password'],
                                        host = this_db_configs['host'],
                                        port = this_db_configs['port'],
                                        database = this_db_configs['database'])
        # create a new cursor
        cursor = conn.cursor()
        # execute the read statement
        postgres_delete_query = """ DELETE FROM "FACT_YAHOO_DATA" """
        cursor.execute(postgres_delete_query)
        cursor.close()
        conn.commit()
        
        print("\nAll rows in db table have been deleted successfully")
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()



def insert_yahoo_data_to_table(this_df,this_db_configs):
    """
    Writes yahoo data dataframe to db table

    Parameters
    ----------
    this_df : dataframe
        The raw data which has been compiled from Yahoo!Finance
    this_db_configs : json
        DB login credentials

    Returns
    -------
    None
    """    
    conn = None
    try:
        #https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table 
        engine = create_engine('postgresql+psycopg2://{}:{}@{}:{}/{}'.format(this_db_configs['user'],this_db_configs['password'],this_db_configs['host'],this_db_configs['port'],this_db_configs['database']))

        this_df.head(0).to_sql(r'"FACT_YAHOO_DATA"', engine, if_exists='replace',index=False) #truncates the table

        conn = engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        this_df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        contents = output.getvalue()
        cur.copy_from(output, r'"FACT_YAHOO_DATA"', null="") # null values become ''
        conn.commit()

        print("\nSuccessfully saved Yahoo! Finance data to db table")        

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()