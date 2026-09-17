# features.py
import pandas as pd
import pandas_ta as ta

def add_features(df):
    """إضافة المؤشرات الفنية ومعالجة البيانات ليفهمها الذكاء الاصطناعي"""
    # 1. حساب المؤشرات
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.bbands(length=20, append=True)
    
    # 2. تحديد الهدف (Target) للتدريب - الشمعة القادمة صاعدة (1) أم هابطة (0)
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # 3. تنظيف البيانات من الـ NaN
    df.dropna(inplace=True)
    
    # 4. التطبيع (Normalization - خطوة احترافية):
    # الشبكات العصبية (Transformer) تتطلب أن تكون البيانات بين 0 و 1
    features_to_scale = df.columns.drop(['open', 'high', 'low', 'close', 'volume', 'Target'])
    for col in features_to_scale:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
            
    return df
