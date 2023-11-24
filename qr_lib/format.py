
import pandas as pd


def format_df(df: pd.DataFrame, source: str):
    if source == "forextester":
        return format_df_forextester(df)
    elif source == "dukascopy":
        return format_df_dukascopy(df)
    
def format_df_forextester(df: pd.DataFrame):
    # forextester

    # <TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
    df['date'] = df['<DTYYYYMMDD>'].astype("string")

    # some 000200 (12:02:00 am) becomes 200 so we need to add padding
    df['time'] = df['<TIME>'].astype("string").str.zfill(6)

    df['datetime'] = pd.to_datetime(df['date'] + df['time'], format='%Y%m%d%H%M%S', errors='coerce')
    df = df.set_index('datetime')
    df = df.sort_index()

    # Convert UTC 10pm into monday 0 am so that it is fixed to market open time
    # need to revise the timezone again
    df = df.shift(2, freq='H')

    df['Open'] = df['<OPEN>']
    df['Close'] = df['<CLOSE>']
    df['High'] = df['<HIGH>']
    df['Low'] = df['<LOW>']
    df['Volume'] = df['<VOL>']

    df = df.drop(['date', 'time', '<DTYYYYMMDD>', '<TIME>', '<TICKER>', '<OPEN>', '<CLOSE>','<HIGH>', '<LOW>', '<VOL>'], axis=1)
    
    return df


def format_df_dukascopy(df: pd.DataFrame):
    '''
    Formats Dukascopy Forex data and drops unnecessary columns
    
        Parameters:
            df (pd.DataFrame): dataframe from CSV file. The CSV looks like below:
                # Gmt time,Open,High,Low,Close,Volume
                # 18.11.2020 00:00:00.000,104.174,104.174,104.147,104.150,271.15
        
        Returns:
            out (pd.DataFrame): dataframe with properly formatted datetime and OHLCV bar
    '''
    
    df['datetime'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
    df = df.set_index('datetime')
    df = df.drop(['Gmt time'], axis=1)

    return df