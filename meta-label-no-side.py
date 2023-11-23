import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from backtest.lib.timeframe  import merge_timeframe, resample_timeframe
from lib.format import format_df
from lib.load_data import load_data
from lib.label import get_tripple_barrier_first_touch_events, get_bins, apply_tripple_barrier, get_vertical_barriers, valdidate_first_touched_events

# TA
import talib
import pandas_ta as ta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.metrics import RocCurveDisplay

symbol = 'GBPJPY'
start = '2019-01-01 00:00:00'
end = '2023-07-28 00:00:00'
tf = '1H'
target = 0.3

vertical_barrier_delta = pd.Timedelta(days=1)

df = load_data(symbol, start, end)

def create_indicator(tf: pd.DataFrame):
    resampled = tf.copy(deep=True)
    # generate moving average
    resampled['ma_20'] = resampled['close'].rolling(20, min_periods=1).mean()
    resampled['ma_200'] = resampled['close'].rolling(200, min_periods=1).mean()

    # generate body size
    resampled['body_abs_len'] = np.abs(resampled['close'] - resampled['open'])
    resampled['up_bar'] = np.where(resampled['close'] > resampled['open'], 1, 0)
    # generate volatility
    returns = np.log(resampled['close'] / resampled['close'].shift(1))
    returns.fillna(0, inplace=True)
    resampled['vol'] = returns.rolling(window=20).std() * np.sqrt(20)

    # generate autocorrelation
    resampled['auto_corr'] = returns.rolling(window=20, min_periods=1).corr(returns.shift(1))

    # add ATR
    resampled['atr'] = talib.MOM(resampled['close'].to_numpy(), timeperiod=24)

    # add supertrend
    sti = ta.supertrend(resampled['high'], resampled['low'], resampled['close'], length=14, multiplier=3)
    sti2 = ta.supertrend(resampled['high'], resampled['low'], resampled['close'], length=100, multiplier=3)
    sti3 = ta.supertrend(resampled['high'], resampled['low'], resampled['close'], length=3, multiplier=2)
    resampled['SUPERTd_14_3.0'] = sti['SUPERTd_14_3.0']
    resampled['SUPERTd_100_3.0'] = sti2['SUPERTd_100_3.0']
    resampled['SUPERTd_3_2.0'] = sti3['SUPERTd_3_2.0']

    # add AROON
    aroond, aroonu = talib.AROON(resampled['high'].to_numpy(), resampled['low'].to_numpy(), 20)
    resampled['aroond_20'] = aroond
    resampled['aroonu_20'] = aroonu
    
    aroond, aroonu = talib.AROON(resampled['high'].to_numpy(), resampled['low'].to_numpy(), 50)
    resampled['aroond_50'] = aroond
    resampled['aroonu_50'] = aroonu
    
    aroond, aroonu = talib.AROON(resampled['high'].to_numpy(), resampled['low'].to_numpy(), 200)
    resampled['aroond_200'] = aroond
    resampled['aroonu_200'] = aroonu

    # kama
    resampled['kama_20'] = talib.KAMA(resampled['close'].to_numpy(), timeperiod=20)
    resampled['kama_50'] = talib.KAMA(resampled['close'].to_numpy(), timeperiod=50)
    resampled['kama_200'] = talib.KAMA(resampled['close'].to_numpy(), timeperiod=200)

    # rsi
    resampled['rsi_20'] = talib.RSI(resampled['close'].to_numpy(), timeperiod=20)
    resampled['rsi_50'] = talib.RSI(resampled['close'].to_numpy(), timeperiod=50)
    resampled['rsi_200'] = talib.RSI(resampled['close'].to_numpy(), timeperiod=200)

    # stochrsi
    resampled['stochrsi_20_k'], resampled['stochrsi_20_d'] = talib.STOCHRSI(resampled['close'].to_numpy(), timeperiod=20)
    resampled['stochrsi_50_k'], resampled['stochrsi_50_d'] = talib.STOCHRSI(resampled['close'].to_numpy(), timeperiod=50)
    resampled['stochrsi_200_k'], resampled['stochrsi_200_d'] = talib.STOCHRSI(resampled['close'].to_numpy(), timeperiod=200)

    # generate ADX
    resampled['adx_plus_14'] = talib.PLUS_DI(resampled['high'].to_numpy(), resampled['low'].to_numpy(), resampled['close'].to_numpy(), 14)
    resampled['adx_minux_14'] = talib.MINUS_DI(resampled['high'].to_numpy(), resampled['low'].to_numpy(), resampled['close'].to_numpy(), 14)

    return resampled


def get_structure_break_events(tf: pd.DataFrame, threshold: float):
    tf['body_abs_len_prev'] = tf['body_abs_len'].shift()
    tf = tf.dropna()
    return tf[(
        (tf['body_abs_len_prev'] < threshold) &
        (tf['body_abs_len'] >= threshold)
    )].index

def get_upside_bars_ma(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df['close'] < df['ma']) & (df.shift(-1)['close'] > df.shift(-1)['ma'])]

def get_downside_bars_ma(df: pd.DataFrame) -> np.ndarray:
    return df[(df['close'] > df['ma']) & (df.shift(-1)['close'] < df.shift(-1)['ma'])]

def build_sides(up_timestamps: pd.DataFrame, down_timestamps: pd.DataFrame) -> pd.Series:
    side_index = up_timestamps.index.union(down_timestamps.index)
    side_data = []
    for idx in side_index:
        if idx in up_timestamps.index:
            side_data.append(1)
        else:
            side_data.append(-1)
    side = pd.Series(data=side_data, index=side_index)
    return side


close = df['Close']

resampled = resample_timeframe(df, tf)

indicator_unsafe = create_indicator(resampled)
indicator_unsafe = indicator_unsafe.dropna()

merged_df = merge_timeframe(indicator_unsafe, df, tf)
merged_df = merged_df.dropna()
print(merged_df.columns)
# up_timestamps, down_timestamps = get_upside_bars_ma(indicator), get_downside_bars_ma(indicator)

threshold = indicator_unsafe['body_abs_len'].quantile(0.9)
structure_break_events = get_structure_break_events(merged_df, threshold)

df_intersection = df.loc[df.index.intersection(structure_break_events)] # ensures structure break event exist in a lower tf

structure_break_events = df_intersection.index

# side = build_sides(up_timestamps, down_timestamps)

profit_target = pd.Series(target, index=structure_break_events)

vertical_barriers = get_vertical_barriers(
    close=close,
    structure_break_events=structure_break_events,
    time_delta=vertical_barrier_delta)

first_touched_events = get_tripple_barrier_first_touch_events(
    close=close, 
    structure_break_events=structure_break_events, 
    pt_sl=[1, 1], 
    return_unit_width=profit_target,
    minimum_unit_return=0.0001,
    vertical_barriers=vertical_barriers,)

# print(first_touched_events.to_string())

first_touched_events = first_touched_events.dropna()

if not valdidate_first_touched_events(first_touched_events):
    raise Exception('First touched events include NaNs')

labels = get_bins(close=close, events=first_touched_events, t1=vertical_barriers)

print(labels)

first_touched_events_indicator = merged_df.loc[merged_df.index.intersection(first_touched_events.index)]

print(first_touched_events_indicator.index.difference(labels.index))

def print_results(rf: RandomForestClassifier, X_test: np.ndarray,
                  y_test: np.ndarray, y_pred: np.ndarray) -> None:
    print(f'RF accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'RF precision: {precision_score(y_test, y_pred, average="micro")}')
    print(f'RF recall: {recall_score(y_test, y_pred, average="micro")}')

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    print(f'AUC metrics: {auc(fpr, tpr)}')

    # roc_auc = auc(fpr, tpr)
    # display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                    #   estimator_name='example estimator')
    # display.plot()
    # plt.show()

X = pd.DataFrame({
    # 'SUPERTd_3_2.0': first_touched_events_indicator['SUPERTd_3_2.0'],
    # 'SUPERTd_14_3.0': first_touched_events_indicator['SUPERTd_14_3.0'],
    # 'SUPERTd_100_3.0': first_touched_events_indicator['SUPERTd_100_3.0'],
    'aroon_20': first_touched_events_indicator['aroonu_20'] - first_touched_events_indicator['aroond_20'],
    'aroon_50': first_touched_events_indicator['aroonu_50'] - first_touched_events_indicator['aroond_50'],
    'aroon_200': first_touched_events_indicator['aroonu_200'] - first_touched_events_indicator['aroond_200'],
    'ma_200_diff': first_touched_events_indicator['close'] - first_touched_events_indicator['ma_200'],
    'ma_20_diff': first_touched_events_indicator['close'] - first_touched_events_indicator['ma_20'],
    'adx_plus_14': first_touched_events_indicator['adx_plus_14'],
    'adx_minux_14': first_touched_events_indicator['adx_minux_14'],
    'adx_diff_14': first_touched_events_indicator['adx_plus_14'] - first_touched_events_indicator['adx_minux_14'],
    'kama_20': first_touched_events_indicator['close'] - first_touched_events_indicator['kama_20'],
    'kama_50': first_touched_events_indicator['close'] - first_touched_events_indicator['kama_50'],
    'kama_200': first_touched_events_indicator['close'] - first_touched_events_indicator['kama_200'],
    
    'rsi_20': first_touched_events_indicator['rsi_20'],
    'rsi_50': first_touched_events_indicator['rsi_50'],
    'rsi_200': first_touched_events_indicator['rsi_200'],
    
    # 'stochrsi_20': first_touched_events_indicator['stochrsi_20_k'] - first_touched_events_indicator['stochrsi_20_d'] ,
    # 'stochrsi_50': first_touched_events_indicator['stochrsi_50_k'] - first_touched_events_indicator['stochrsi_50_d'],
    # 'stochrsi_200': first_touched_events_indicator['stochrsi_200_k'] - first_touched_events_indicator['stochrsi_200_d'],
    
    # 'stochrsi_20_k': first_touched_events_indicator['stochrsi_20_k'],
    # 'stochrsi_50_k': first_touched_events_indicator['stochrsi_50_k'],
    # 'stochrsi_200_k': first_touched_events_indicator['stochrsi_200_k'],

    # 'up_bar': first_touched_events_indicator['up_bar'],

    # 'vol': first_touched_events_indicator['vol'],
    # 'autocorr': first_touched_events_indicator['auto_corr'],
    # 'ma': first_touched_events_indicator['ma'],
    #  'side': first_touched_events['side']
}).dropna()

# X = X[~X.index.duplicated(keep='first')]

# X, y = first_touched_events['side'].values.reshape(-1, 1), labels['bin'].values.astype(int)
y = labels['bin'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

feature_scores = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(feature_scores)
print(f"train count {len(X_train)}")

y_pred = rf.predict(X_test)

print_results(rf, X_test, y_test, y_pred)

X_test['y_pred'] = y_pred.tolist()
X_test['y_true'] = y_test.tolist()
X_test['gain'] = -1
X_test.loc[X_test['y_pred'] == X_test['y_true'], 'gain'] = 1
X_test['gain_cum'] = X_test['gain'].cumsum()

gain_counts = X_test['gain'].value_counts()
win_rate = gain_counts[1] / len(X_test.index)

print(X_test.to_string())
print(f'predicted win rate {win_rate}')
# print(y_pred)