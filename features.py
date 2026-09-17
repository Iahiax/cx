# features.py
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import ForceIndexIndicator, VolumeWeightedAveragePrice

def add_features(df):
    """هندسة الخصائص الأسطورية: التقلب الضمني، السرعة، والسيولة العميقة"""
    
    # 1. مؤشرات الزخم والترند الأساسية
    df['RSI_14'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd.macd()
    df['MACD_Diff'] = macd.macd_diff()
    
    df['ATR_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['ADX_14'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    
    df['EMA_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['EMA_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['EMA_Cross'] = df['EMA_9'] - df['EMA_21']
    
    # 2. مؤشرات السيولة وحجم التداول (VWAP & Force Index)
    try:
        vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14)
        df['VWAP'] = vwap.volume_weighted_average_price()
    except:
        df['VWAP'] = df['close']
        
    force_index = ForceIndexIndicator(close=df['close'], volume=df['volume'], window=13)
    df['Force_Index'] = force_index.force_index()

    # 3. محرك السرعة اللحظية (Price Velocity لمعيار الضوضاء والأخبار الكاذبة)
    df['Price_Velocity'] = df['close'].diff().abs()
    
    # 4. سطح التقلب (Volatility Surface Dynamics)
    df['Volatility_Surface'] = df['ATR_14'].pct_change(3)

    # الهدف (Target)
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.dropna(inplace=True)
    
    # التطبيع الاحترافي (Normalization)
    features_to_scale = df.columns.drop(['open', 'high', 'low', 'close', 'volume', 'Target'])
    for col in features_to_scale:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
            
    return df
