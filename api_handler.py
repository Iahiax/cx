# api_handler.py
import requests
import json
import pandas as pd
from config import *

def login():
    url = f"{SERVER}/api/v1/session"
    headers = {"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"}
    payload = {"identifier": EMAIL, "password": API_KEY_PASSWORD, "encryptedPassword": False}
    
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if r.status_code == 200:
            return r.headers.get("CST"), r.headers.get("X-SECURITY-TOKEN")
        else:
            print(f"❌ خطأ في تسجيل الدخول: {r.text}")
    except Exception as e:
        print(f"⚠️ فشل الاتصال بالخادم: {e}")
    return None, None

def get_market_data(cst, xst, target_epic=EPIC):
    url = f"{SERVER}/api/v1/prices/{target_epic}?resolution={RESOLUTION}&max={MAX_DATA_POINTS}"
    headers = {"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": xst}
    
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()
        
        if 'prices' not in data: 
            return None
            
        prices = [{'time': p['snapshotTime'], 'open': p['openPrice']['bid'], 
                   'high': p['highPrice']['bid'], 'low': p['lowPrice']['bid'], 
                   'close': p['closePrice']['bid'], 'volume': p['lastTradedVolume']} 
                  for p in data['prices']]
        
        df = pd.DataFrame(prices)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        print(f"⚠️ خطأ في سحب بيانات {target_epic}: {e}")
        return None

def execute_order(cst, xst, direction, size=1000, stop_distance=None, profit_distance=None):
    url = f"{SERVER}/api/v1/positions"
    headers = {"X-CAP-API-KEY": API_KEY, "CST": cst, "X-SECURITY-TOKEN": xst, "Content-Type": "application/json"}
    
    # تصحيح وتقييد المسافات ضمن النطاق الآمن والمقبول لـ Capital.com (أقل من الحد الأقصى المسموح)
    if stop_distance:
        stop_distance = min(float(stop_distance), 0.0050) # تقييد وقف الخسارة ليكون آمناً ومقبولاً
        stop_distance = max(stop_distance, 0.0005)      # ألا يقل عن الحد الأدنى
        
    if profit_distance:
        profit_distance = min(float(profit_distance), 0.0100)
        profit_distance = max(profit_distance, 0.0010)

    payload = {
        "epic": EPIC, 
        "direction": direction, 
        "size": size,
        "guaranteedStop": False,
        "leverage": LEVERAGE
    }
    
    if stop_distance:
        payload["stopDistance"] = round(stop_distance, 4)
    if profit_distance:
        payload["profitDistance"] = round(profit_distance, 4)
    
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if r.status_code == 200:
            print(f"🚀 [DEMO EXECUTION] تم تنفيذ صفقة {direction} بنجاح تام على حساب الـ Demo!")
            return True
        else:
            print(f"❌ فشل التنفيذ على الديمو: {r.text}")
            return False
    except Exception as e:
        print(f"⚠️ خطأ في إرسال الأمر: {e}")
        return False
