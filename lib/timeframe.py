
import pandas as pd


def resample_timeframe(df: pd.DataFrame, time: str, param: str = 'Close'):
    '''
    Resamples into specific timeframe and creates OHLC
    
        Parameters:
            df (pd.DataFrame): raw dataframe
            time (str): time interval to resample to
        
        Returns:
            out (pd.DataFrame): dataframe with a resampled timeframe according to time
    '''

    return df.resample(time)[param].ohlc().dropna()

def merge_timeframe(high_tf: pd.DataFrame, low_tf: pd.DataFrame, high_tf_time_delta: str):
    '''
    Merges high timeframe into low timeframe by index
    
        Parameters:
            high_tf (pd.DataFrame): dataframe for higher timeframe
                                   - index: The timestamp sorted in ascending order
            low_tf (pd.DataFrame): dataframe for lower timeframe
                                   - index: The timestamp sorted in ascending order
            high_tf_time_delta (str): the timedelta of high_tf (i.e. 30T, 1H, 1D).
        
        Returns:
            out (pd.DataFrame): dataframe with datetime index column, columns that appeared in both high_tf and low_tf.
                                if the colum names clash among the two dataframes, they become suffixed with '_y' for high_tf and '_x' for low_tf
    '''

    df1 = high_tf.copy(deep=True)
    df2 = low_tf.copy(deep=True)
    
    # high timeframe needs to shift one to prevent lookahead bias
    df1 = df1.shift()
    
    df1['my_datetime'] = df1.index
    df2['my_datetime'] = df2.index
    
    a = pd.merge_asof(df2, 
                      df1, 
                      on='my_datetime', 
                      tolerance = pd.Timedelta(high_tf_time_delta))
    
    a.index = a['my_datetime']
    a = a.sort_index()
    a = a.drop(columns=['my_datetime'])
    return a
