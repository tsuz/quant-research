
import pandas as pd


def append_fvg(df: pd.DataFrame, fvg_min: float = 0, low_col: str = 'low', high_col: str = 'high'):
    '''
    Appends Fair Value Gap to an existing dataframe
    
        Parameters:
            df (pd.DataFrame): dataframe to append fair value gap to
                - low (float): The lowest price for the candle within interval interval
                - high (float): The highest price for the candle within its interval
            fvg_min (float): The minimum difference required to be considered a large enough fair value gap
        
        Returns:
            df (pd.DataFrame): dataframe with the below colmns appended. The values need to be shifted by one to prevent lookahead bias
                - fvg_occurred (boolean): Whether the current close of the candle stick forms a FVG.
                - prev_fvg_bullish (boolean or NaN): Whether the most recent FVG was bullish. False if bearish. NaN if no FVG has existed before.
                - prev_fvg_high (number or NaN): The higher price of the most recent fair value gap
                - prev_fvg_low (number or NaN): The lower price of the most recent fair value gap
    '''
    fvg_occurred = 'fvg_occurred'
    prev_fvg = 'prev_fvg_bullish'
    prev_fvg_high = 'prev_fvg_high'
    prev_fvg_low = 'prev_fvg_low'
    
    bullish_fvg = (df[low_col] > df[high_col].shift(2) + fvg_min)
    bearish_fvg = (df[high_col] + fvg_min < df[low_col].shift(2))

    df[fvg_occurred] = False
    df.loc[bullish_fvg, prev_fvg] = True
    df.loc[bearish_fvg, prev_fvg] = False

    df.loc[(bullish_fvg | bearish_fvg), fvg_occurred] = True

    df.loc[df[prev_fvg] == True, prev_fvg_high] = df[low_col]
    df.loc[df[prev_fvg] == True, prev_fvg_low] = df[high_col].shift(2)

    df.loc[df[prev_fvg] == False, prev_fvg_high] = df[low_col].shift(2)
    df.loc[df[prev_fvg] == False, prev_fvg_low] = df[high_col]

    # fills NaN with previous values
    df[prev_fvg_high] = df[prev_fvg_high].fillna(method='ffill')
    df[prev_fvg_low] = df[prev_fvg_low].fillna(method='ffill')
    df[prev_fvg] = df[prev_fvg].fillna(method='ffill')

    return df
