# regime_engine.py
import pandas as pd
import numpy as np
import requests
from config import FINNHUB_API_KEY

class RegimeEngine:
    def __init__(self, df):
        self.df = df

    def detect_regime(self):
        atr = self.df['ATR_14'].iloc[-1]
        atr_mean = self.df['ATR_14'].rolling(50).mean().iloc[-1]
        adx = self.df['ADX_14'].iloc[-1]
        velocity = self.df['close'].diff().abs().iloc[-1]

        if velocity > (atr * 3.0) or atr > (atr_mean * 2.0):
            return "News Shock"
        if adx > 25:
            return "Trending"
        if adx <= 25 and atr <= atr_mean:
            return "Ranging"
            
        return "High Volatility"

    def check_news_blackout(self):
        """فحص الأخبار الاقتصادية الحية عبر Finnhub لمنع التداول قبل الأخبار الكبرى"""
        try:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                news_list = response.json()
                # الكلمات المفتاحية للأخبار الخطرة
                risk_keywords = ["fomc", "fed", "cpi", "nfp", "inflation", "rate decision", "interest rate", "powell", "ecb"]
                
                for item in news_list[:15]: # فحص أحدث 15 خبر عاجل
                    headline = item.get('headline', '').lower()
                    summary = item.get('summary', '').lower()
                    
                    for kw in risk_keywords:
                        if kw in headline or kw in summary:
                            print(f"🚨 [Finnhub News Shield]: تم رصد خبر عالي التأثير ({kw.upper()}): {item.get('headline')}")
                            return True # تفعيل الحظر الفوري للأخبار
        except Exception as e:
            print(f"⚠️ تحذير: فشل الاتصال ببوابة Finnhub للأخبار: {e}")
            
        return False
