import os
import sys
from qr_lib.format import format_df
import pandas as pd

def load_data(symbol: str, start: str, end: str, source: str, path: str):
    '''
    Loads data from store
    
        Parameters:
            symbol (str): symbol to load. (i.e. USDJPY)
            start (str): start time to filter from in format of '1970-01-01 01:00:00:
            end (str): end time to filter from in format of '1970-01-01 01:00:00:
            source (str): source of data. Only forextester and dukascopy are supported.
        
        Returns:
            out (pd.DataFrame): dataframe with loaded data containing OHLC. 
                There may be additioanl attributes depending on the type of data.
    '''

    if path is None:
      path = os.path.dirname(sys.modules['__main__'].__file__)

    cached_path = f'{path}/{source}_{symbol}_{start}_{end}_formatted.csv'
    data_path = f'{path}/{source}_{symbol}.csv'

    if os.path.exists(cached_path):
        df = pd.read_csv(cached_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df.sort_index()
    else:
        df = pd.read_csv(data_path)
        df = format_df(df, source)

    # Filter timeframe
    df = df.loc[(df.index >= start) & (df.index <= end)]
    
    # Save to cache
    df.to_csv(cached_path)

    return df

