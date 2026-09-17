# main.py
import os
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

def run_backtest_simulation(df, engines):
    """محرك الاختبار الخلفي (Backtest) للتحقق من كفاءة الاستراتيجية تاريخياً"""
    print("📊 [Backtest Engine]: بدء الاختبار الخلفي على البيانات التاريخية...")
    wins = 0
    losses = 0
    
    # محاكاة على آخر 100 شمعة تاريخية
    test_df = df.tail(100)
    for i in range(len(test_df) - 1):
        row_features = test_df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Target']).iloc[i].values.astype(np.float32)
        actual_target = test_df['Target'].iloc[i] # الاتجاه الحقيقي للشمعة التالية
        
        consensus = 0.0
        for member, weight in COUNCIL_MEMBERS.items():
            vote = engines[member].predict(row_features)
            consensus += (vote * weight)
            
        predicted_dir = 1 if consensus > 0 else 0
        if predicted_dir == actual_target:
            wins += 1
        else:
            losses += 1
            
    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    print(f"📈 [Backtest Results]: نسبة النجاح التاريخية = {win_rate:.2f}% (صحيحة: {wins}, خاطئة: {losses})")
    
    # السماح بالتداول إذا كانت نسبة النجاح أعلى من 50%
    return win_rate >= 50.0

def run_gen5_sovereign_bot():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 👑 [Gen-5 Sovereign] بدء دورة التدريب، الباكتيست والتداول الآلي")
    
    cst, xst = login()
    if not cst: return

    raw_df = get_market_data(cst, xst)
    if raw_df is None or raw_df.empty: return

    df_temp = add_features(raw_df)
    env = ProTradingEnv(df_temp)
    
    # 1. تدريب النماذج ذاتياً
    print("⚡ [1/3] بدء التدريب الذاتي المستمر لعقول المجلس...")
    engines = {}
    for member in COUNCIL_MEMBERS.keys():
        engine = StrategyEngine(member)
        engine.train(df=df_temp, env=env)
        engines[member] = engine
    print("✅ تم الانتهاء من التدريب الذاتي بنجاح.")

    # 2. تشغيل الباكتيست (Backtest)
    print("🧪 [2/3] تشغيل نظام الاختبار الخلفي (Backtest)...")
    passed_backtest = run_backtest_simulation(df_temp, engines)
    
    if not passed_backtest:
        print("❌ [Backtest Failed]: نسبة النجاح التاريخية أقل من الحد المقبول. تم تعليق التداول الحي لحماية رأس المال.")
        return
    print("✅ [Backtest Passed]: اجتازت الاستراتيجية الاختبار الخلفي بنجاح تام!")

    # 3. اتخاذ القرار والتداول الحي على حساب الـ Demo
    print("🚀 [3/3] الانتقال للتداول الحي على حساب الديمو (Demo Execution)...")
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
    profit_dist = current_atr * 2.5
    trade_size = max(1000, int(abs(weighted_consensus) * 20000))

    if weighted_consensus >= 0.18:
        direction = "BUY"
        print(f"🚀 [DEMO BUY] تنفيذ صفقة شراء - الحجم: {trade_size} - رافعة: {LEVERAGE}:1")
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            journal.log_trade(direction, trade_size, weighted_consensus, market_regime, "Backtest Passed", "EXECUTED")
            
    elif weighted_consensus <= -0.18:
        direction = "SELL"
        print(f"📉 [DEMO SELL] تنفيذ صفقة بيع - الحجم: {trade_size} - رافعة: {LEVERAGE}:1")
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            journal.log_trade(direction, trade_size, weighted_consensus, market_regime, "Backtest Passed", "EXECUTED")
    else:
        print("⚖️ قرار (CASH): الاستقرار خارج السوق لعدم وجود إتجاه واضح.")

if __name__ == "__main__":
    run_gen5_sovereign_bot()
