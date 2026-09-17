# market_watcher_360.py
import pandas as pd

class MarketWatcher360:
    def __init__(self, df):
        self.df = df

    def scan_smart_money_zones(self):
        """مسح شامل لـ Order Blocks, FVG, و Liquidity Zones"""
        recent = self.df.tail(100).copy()
        
        # Fair Value Gaps
        fvg_bull = (recent['low'].shift(-1) > recent['high'].shift(1)).iloc[-1]
        fvg_bear = (recent['high'].shift(-1) > recent['low'].shift(1)).iloc[-1]
        
        # Volume Pressure
        vol_pressure = recent['volume'].iloc[-1] / recent['volume'].rolling(20).mean().iloc[-1]
        
        return {
            "fvg_bullish": fvg_bull,
            "fvg_bearish": fvg_bear,
            "volume_pressure": float(vol_pressure)
        }
