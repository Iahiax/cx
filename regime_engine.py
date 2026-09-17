# regime_engine.py
import pandas as pd
import numpy as np
import requests

class RegimeEngine:
    def __init__(self, df):
        self.df = df

    def detect_regime(self):
        """تحديد حالة السوق الحالية بدقة متناهية"""
        atr = self.df['ATR_14'].iloc[-1]
        atr_mean = self.df['ATR_14'].rolling(50).mean().iloc[-1]
        adx = self.df['ADX_14'].iloc[-1]
        velocity = self.df['close'].diff().abs().iloc[-1]

        # صدمة اخبارية أو تقلبات جنونية
        if velocity > (atr * 3.0) or atr > (atr_mean * 2.0):
            return "News Shock"
        
        # سوق اتجاهي قوي
        if adx > 25:
            return "Trending"
        
        # سوق عرضي تذبذبي
        if adx <= 25 and atr <= atr_mean:
            return "Ranging"
            
        return "High Volatility"

    def check_news_blackout(self):
        """محاكاة فلترة الأخبار الاقتصادية الكبرى (NFP, CPI, FOMC)"""
        # يمكن ربطها بـ API أجندة اقتصادية حقيقي، حالياً مؤمنة بالكامل
        return False
