
import pandas as pd


def append_fvg(df: pd.DataFrame, fvg_min: float = 0):
    '''
    Appends Fair Value Gap to an existing dataframe
    
        Parameters:
            df (pd.DataFrame): dataframe to append fair value gap to
                - low (float): The lowest price for the candle within interval interval
                - high (float): The highest price for the candle within its interval
            fvg_min (float): The minimum difference required to be considered a large enough fair value gap
        
        Returns:
            df (pd.DataFrame): dataframe with the below colmns appended. The values need to be shifted by one to prevent lookahead bias
                - prev_fvg_bullish (boolean or NaN): Whether the previous FVG was bullish. False if bearish. NaN if no FVG has existed before.
                - prev_fvg_high (number or NaN): The higher price of the most recent fair value gap
                - prev_fvg_low (number or NaN): The lower price of the most recent fair value gap
    '''

    prev_fvg = 'prev_fvg_bullish'
    prev_fvg_high = 'prev_fvg_high'
    prev_fvg_low = 'prev_fvg_low'
    
    bullish_fvg = (df['low'] > df['high'].shift(2) + fvg_min)
    bearish_fvg = (df['high'] + fvg_min < df['low'].shift(2))

    df.loc[bullish_fvg, prev_fvg] = True
    df.loc[bearish_fvg, prev_fvg] = False

    df.loc[df[prev_fvg] == True, prev_fvg_high] = df['low']
    df.loc[df[prev_fvg] == True, prev_fvg_low] = df['high'].shift(2)

    df.loc[df[prev_fvg] == False, prev_fvg_high] = df['low'].shift(2)
    df.loc[df[prev_fvg] == False, prev_fvg_low] = df['high']

    # fills NaN with previous values
    df[prev_fvg_high] = df[prev_fvg_high].fillna(method='ffill')
    df[prev_fvg_low] = df[prev_fvg_low].fillna(method='ffill')
    df[prev_fvg] = df[prev_fvg].fillna(method='ffill')

    return df
