# features.py
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

def add_features(df):
    """إضافة المؤشرات الفنية ومعالجة البيانات باستخدام مكتبة ta المستقرة"""
    # 1. حساب المؤشرات
    df['RSI_14'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd.macd()
    
    df['ATR_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    df['ADX_14'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    
    # 2. تحديد الهدف (Target) للتدريب - الشمعة القادمة صاعدة (1) أم هابطة (0)
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 3. تنظيف البيانات من الـ NaN
    df.dropna(inplace=True)
    
    # 4. التطبيع (Normalization - خطوة احترافية):
    # الشبكات العصبية تتطلب أن تكون البيانات بين 0 و 1
    features_to_scale = df.columns.drop(['open', 'high', 'low', 'close', 'volume', 'Target'])
    for col in features_to_scale:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
            
    return df
