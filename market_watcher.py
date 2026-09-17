# market_watcher.py
import pandas as pd
import numpy as np

class MarketWatcher:
    def __init__(self, df):
        self.df = df

    def detect_market_structure(self):
        """اكتشاف هيكل السوق وكسر الهيكل BOS ومناطق السيولة"""
        recent = self.df.tail(200).copy()
        
        # حساب الفجوات السعرية FVG (Fair Value Gaps)
        recent['FVG_Bullish'] = (recent['low'].shift(-1) > recent['high'].shift(1))
        recent['FVG_Bearish'] = (recent['high'].shift(-1) < recent['low'].shift(1))
        
        # كشف سحب السيولة Liquidity Sweep
        recent['High_20'] = recent['high'].rolling(20).max()
        recent['Low_20'] = recent['low'].rolling(20).min()
        
        sweep_high = (recent['high'].iloc[-1] > recent['High_20'].iloc[-2]) and (recent['close'].iloc[-1] < recent['High_20'].iloc[-2])
        sweep_low = (recent['low'].iloc[-1] < recent['Low_20'].iloc[-2]) and (recent['close'].iloc[-1] > recent['Low_20'].iloc[-2])
        
        structure_info = {
            "fvg_bullish": bool(recent['FVG_Bullish'].iloc[-1]),
            "fvg_bearish": bool(recent['FVG_Bearish'].iloc[-1]),
            "sweep_high": bool(sweep_high),
            "sweep_low": bool(sweep_low)
        }
        return structure_info
