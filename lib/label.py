import pandas as pd
import numpy as np
from typing import List, Union


def get_vertical_barriers(close: pd.DataFrame, structure_break_events: pd.DataFrame, time_delta: pd.Timedelta):
    '''
    Returns vertical barriers after the structure_break_events have occurred
    '''
    t1 = close.index.searchsorted(structure_break_events+time_delta) # find where the next timedelta is
    t1 = t1[t1 < close.shape[0]] # don't go over existing index
    t1 = pd.Series(close.index[t1], index=structure_break_events[:t1.shape[0]]) # NaNs at the end
    return t1

def apply_tripple_barrier(close: pd.Series, events: pd.DataFrame,
                                   pt_sl: List, molecule: np.ndarray) -> pd.DataFrame:
    '''
    Labeling observations using tripple-barrier method
    
        Parameters:
            close (pd.Series): close prices of bars
            events (pd.DataFrame): dataframe with columns:
                                   - t1: The timestamp of vertical barrier (if np.nan, there will not be
                                         a vertical barrier)
                                   - trgt: The unit width of the horizontal barriers
            pt_sl (list): list of two non-negative float values:
                          - pt_sl[0]: The factor that multiplies trgt to set the width of the upper barrier.
                                      If 0, there will not be an upper barrier.
                          - pt_sl[1]: The factor that multiplies trgt to set the width of the lower barrier.
                                      If 0, there will not be a lower barrier.
            molecule (np.ndarray):  subset of event indices that will be processed by a
                                    single thread (will be used later)
        
        Returns:
            out (pd.DataFrame): dataframe with columns [pt, sl, t1] corresponding to timestamps at which
                                each barrier was touched (if it happened)
    '''
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(data=[np.nan] * len(events.index), index=events.index)    # NaNs
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(data=[np.nan] * len(events.index), index=events.index)    # NaNs
    
    
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc: t1]                                       # path prices
        first_val = close[loc]
        sl_val = first_val + sl[loc]
        pt_val = first_val + pt[loc]
        out.loc[loc, 'sl'] = df0[df0 <= sl_val].index.min()        # earlisest stop loss
        out.loc[loc, 'pt'] = df0[df0 >= pt_val].index.min()        # earlisest profit taking
    return out


def get_tripple_barrier_first_touch_events(
    close: pd.Series, structure_break_events: np.ndarray, pt_sl: float, return_unit_width: pd.Series, minimum_unit_return: float,
    numThreads: int = 1, vertical_barriers: Union[pd.Series, bool] = False, side: pd.Series = None
) -> pd.DataFrame:
    '''
    Getting times of the first barrier touch
    
        Parameters:
            close (pd.Series): close prices of bars
            structure_break_events (np.ndarray): np.ndarray of timestamps that seed every barrier (they can be generated
                                  by CUSUM filter for example)
            pt_sl (float): non-negative float that sets the width of the two barriers (if 0 then no barrier)
            return_unit_width (pd.Series): s series of targets expressed in terms of absolute returns
            minimum_unit_return (float): minimum target return required for running a triple barrier search
            numThreads (int): number of threads to use concurrently
            t1 (pd.Series): series with the timestamps of the vertical barriers (pass False
                            to disable vertical barriers)
            side (pd.Series) (optional): metalabels containing sides of bets
        
        Returns:
            events (pd.DataFrame): dataframe with columns:
                                       - t1: timestamp of the first barrier touch
                                       - trgt: target that was used to generate the horizontal barriers
                                       - side (optional): side of bets
    '''
    return_unit_width = return_unit_width.loc[return_unit_width.index.intersection(structure_break_events)]
    return_unit_width = return_unit_width[return_unit_width > minimum_unit_return]

    if vertical_barriers is False:
        vertical_barriers = pd.Series(pd.NaT, index=structure_break_events)
    if side is None:
        side_, pt_sl_ = pd.Series(np.array([1.] * len(return_unit_width.index)), index=return_unit_width.index), [pt_sl[0], pt_sl[0]]
    else:
        side_, pt_sl_ = side.loc[return_unit_width.index.intersection(side.index)], pt_sl[:2]
    
    events = pd.concat({'t1': vertical_barriers, 'trgt': return_unit_width, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0 = apply_tripple_barrier(close, events, pt_sl_, events.index)
    events['t1'] = df0.dropna(how='all').min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events


# including metalabeling possibility & modified to generate 0 labels
def get_bins(close: pd.Series, events: pd.DataFrame, t1: Union[pd.Series, bool] = False) -> pd.DataFrame:
    '''
    Generating labels with possibility of knowing the side (metalabeling)
    
        Parameters:
            close (pd.Series): close prices of bars
            events (pd.DataFrame): dataframe returned by 'get_events' with columns:
                                   - index: event starttime
                                   - t1: event endtime
                                   - trgt: event target
                                   - side (optional): position side
            t1 (pd.Series): series with the timestamps of the vertical barriers (pass False
                            to disable vertical barriers)
        
        Returns:
            out (pd.DataFrame): dataframe with columns:
                                       - ret: return realized at the time of the first touched barrier
                                       - bin: if metalabeling ('side' in events), then {0, 1} (take the bet or pass)
                                              if no metalabeling, then {-1, 1} (buy or sell)
    '''
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    out['start_price'] = px.loc[events_.index]
    out['touched_price'] = px.loc[events_['t1'].values].values
    if 'side' in events_:
        out['ret'] *= events_['side']
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0
    else:
        if t1 is not None:
            vertical_first_touch_idx = events_[events_['t1'].isin(t1.values)].index
            out.loc[vertical_first_touch_idx, 'bin'] = 0
    return out


def valdidate_first_touched_events(df: pd.DataFrame) -> bool:
    return not df.isnull().values.any()
