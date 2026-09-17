# features.py
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def add_features(df):
    """إضافة مؤشرات فنية متقدمة وهندسة الخصائص لرفع دقة الذكاء الاصطناعي"""
    
    # المؤشرات الأساسية
    df['RSI_14'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd.macd()
    df['MACD_Diff'] = macd.macd_diff()
    
    df['ATR_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['ADX_14'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    
    # مؤشرات إضافية للتفوق الاتجاهي
    df['EMA_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['EMA_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['EMA_Cross'] = df['EMA_9'] - df['EMA_21'] # الفرق بين المتوسطات كإشارة اتجاه قوية
    
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    
    # الهدف (Target): الشمعة القادمة صاعدة (1) أم هابطة (0)
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # تنظيف البيانات
    df.dropna(inplace=True)
    
    # التطبيع الاحترافي (Normalization) لجعل القيم بين 0 و 1
    features_to_scale = df.columns.drop(['open', 'high', 'low', 'close', 'volume', 'Target'])
    for col in features_to_scale:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
            
    return df
