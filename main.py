# main.py
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import logging
import numpy as np
import pandas as pd
from datetime import datetime

from api_handler import login, get_market_data, execute_order
from features import add_features
from environments import ProTradingEnv
from strategies import StrategyEngine
from market_watcher_360 import MarketWatcher360
from regime_engine import RegimeEngine
from signal_evaluator import SignalEvaluator
from journal import TradeJournal
from config import LOG_FILE, LEVERAGE

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')
journal = TradeJournal()
evaluator = SignalEvaluator()

COUNCIL_MEMBERS = {
    "TransformerForecaster": 0.35,  
    "MultiAgentTrendSwarm": 0.30,   
    "OnlineSVMTrend": 0.15,         
    "VolatilityProphet": 0.20       
}

def run_comprehensive_backtest(df, engines):
    """محرك باكتيست شامل لتحليل نسبة النجاح التاريخية وإعطاء تقرير مفصل"""
    print("\n" + "="*50)
    print("📊 [Comprehensive Backtest Engine]: بدء تقرير الاختبار الخلفي الشامل...")
    wins = 0
    losses = 0
    total_profit_pips = 0.0
    
    test_df = df.tail(200) # فحص آخر 200 شمعة لاختبار دقة أقوى
    for i in range(len(test_df) - 1):
        row_features = test_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Target']).iloc[i].values.astype(np.float32)
        actual_target = test_df['Target'].iloc[i]
        
        consensus = 0.0
        for member, weight in COUNCIL_MEMBERS.items():
            vote = engines[member].predict(row_features)
            consensus += (vote * weight)
            
        predicted_dir = 1 if consensus > 0 else 0
        if predicted_dir == actual_target:
            wins += 1
            total_profit_pips += 12.5
        else:
            losses += 1
            total_profit_pips -= 10.0
            
    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"📈 --- 📋 تقرير أداء الباكتيست الشامل ---")
    print(f"   🔹 إجمالي الصفقات المختبرة : {total_trades}")
    print(f"   ✅ الصفقات الرابحة         : {wins}")
    print(f"   ❌ الصفقات الخاسرة         : {losses}")
    print(f"   🎯 نسبة النجاح (Win Rate)  : {win_rate:.2f}%")
    print(f"   💰 الأرباح الافتراضية     : {total_profit_pips:+.1f} نقطة")
    print("="*50 + "\n")
    
    return win_rate

def run_trading_cycle():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 👑 [Gen-5 Sovereign] بدء دورة تداول جديدة...")
    
    cst, xst = login()
    if not cst:
        print("❌ فشل الاتصال بالمنصة، إعادة المحاولة بعد قليل...")
        return

    raw_df = get_market_data(cst, xst)
    if raw_df is None or raw_df.empty:
        print("❌ لم يتم استلام بيانات السوق.")
        return

    df_temp = add_features(raw_df)
    
    # 1. التدريب الذاتي المستمر
    print("⚡ [1/3] التدريب الذاتي المستمر لعقول المجلس...")
    env = ProTradingEnv(df_temp)
    engines = {}
    for member in COUNCIL_MEMBERS.keys():
        engine = StrategyEngine(member)
        engine.train(df=df_temp, env=env)
        engines[member] = engine

    # 2. تشغيل الباكتيست الشامل وعرض التقرير
    win_rate = run_comprehensive_backtest(df_temp, engines)

    # 3. اتخاذ القرار والتنفيذ بكامل رأس المال المتاح مع رافعة 100:1
    print("🚀 [3/3] تقييم السوق وتنفيذ الصفقة الحية (Full Capital Mode)...")
    watcher360 = MarketWatcher360(raw_df)
    smc_data = watcher360.scan_smart_money_zones()
    
    regime_eng = RegimeEngine(df_temp)
    market_regime = regime_eng.detect_regime()
    
    latest_features = env.features[-1].astype(np.float32)
    weighted_consensus = 0.0
    for member, weight in COUNCIL_MEMBERS.items():
        raw_vote = engines[member].predict(latest_features)
        weighted_consensus += (raw_vote * weight)

    current_atr = df_temp['ATR_14'].iloc[-1]
    stop_dist = current_atr * 1.5
    profit_dist = current_atr * 3.0
    
    # تضخيم حجم العقود لاستغلال كامل رأس المال برافعة 100:1
    trade_size = max(5000, int(abs(weighted_consensus) * 50000))

    CONFIDENCE_THRESHOLD = 0.10 # عتبة مرنة لاقتناص كافة الفرص المتاحة

    if weighted_consensus >= CONFIDENCE_THRESHOLD:
        direction = "BUY"
        print(f"🚀 [FULL CAPITAL BUY] تنفيذ شراء بكامل الرأس مال - الحجم: {trade_size} - رافعة: {LEVERAGE}:1 - إجماع: {weighted_consensus:.2f}")
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            journal.log_trade(direction, trade_size, weighted_consensus, market_regime, f"WinRate: {win_rate:.1f}%", "EXECUTED")
            
    elif weighted_consensus <= -CONFIDENCE_THRESHOLD:
        direction = "SELL"
        print(f"📉 [FULL CAPITAL SELL] تنفيذ بيع بكامل الرأس مال - الحجم: {trade_size} - رافعة: {LEVERAGE}:1 - إجماع: {weighted_consensus:.2f}")
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            journal.log_trade(direction, trade_size, weighted_consensus, market_regime, f"WinRate: {win_rate:.1f}%", "EXECUTED")
    else:
        print("⚖️ قرار (CASH): السوق هادئ، بانتظار الفرصة التالية...")

if __name__ == "__main__":
    print("🔥 تم تفعيل نظام التداول الآلي المستمر 24/7 (Infinite Loop Mode)...")
    while True:
        try:
            run_trading_cycle()
        except Exception as e:
            print(f"⚠️ حدث خطأ غير متوقع في الدورة: {e}")
        
        print("⏳ الانتظار لمدة 60 ثانية للدورة القادمة...")
        time.sleep(60)
