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
from datetime import datetime

from api_handler import login, get_market_data, execute_order
from features import add_features
from environments import ProTradingEnv
from strategies import StrategyEngine
from config import LOG_FILE, LEVERAGE

# استدعاء الأنظمة الـ 12 الجديدة
from market_watcher_360 import MarketWatcher360
from regime_engine import RegimeEngine
from online_learner import OnlineLearner
from signal_evaluator import SignalEvaluator
from auto_maintenance import AutoMaintenance
from journal import TradeJournal

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')
journal = TradeJournal()
maintenance = AutoMaintenance()
learner = OnlineLearner()
evaluator = SignalEvaluator()

COUNCIL_MEMBERS = {
    "TransformerForecaster": 0.35,  
    "MultiAgentTrendSwarm": 0.30,   
    "OnlineSVMTrend": 0.15,         
    "VolatilityProphet": 0.20       
}

def run_gen5_sovereign_bot():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 👑 [Gen-5 Sovereign] بدء دورة التشغيل الذاتي الخارقة (Demo - {LEVERAGE}:1)")
    
    # 1. صيانة أسبوعية تلقائية إذا لزم الأمر
    maintenance.check_auto_rebuild()
    maintenance.run_daily_self_analysis()

    cst, xst = login()
    if not cst: return

    raw_df = get_market_data(cst, xst)
    if raw_df is None or raw_df.empty: return

    # 2. فحص الانحرافات والشموع الشاذة (Market Anomaly Detector)
    df_temp = add_features(raw_df)
    is_anomaly, anomaly_msg = learner.detect_anomaly(df_temp)
    if is_anomaly:
        print(anomaly_msg)
        journal.log_trade("NONE", 0, 0, "Anomaly", anomaly_msg, "ABORTED")
        return

    # 3. Market Watcher 360 & Regime Detection
    watcher360 = MarketWatcher360(raw_df)
    smc_data = watcher360.scan_smart_money_zones()
    
    regime_eng = RegimeEngine(df_temp)
    market_regime = regime_eng.detect_regime()
    
    print(f"📊 النظام السوقي: {market_regime} | تحليل الأموال الذكية (SMC): {smc_data}")

    env = ProTradingEnv(df_temp)
    engines = {}
    for member in COUNCIL_MEMBERS.keys():
        engine = StrategyEngine(member)
        engine.train(df=df_temp, env=env)
        engines[member] = engine
        
    latest_features = env.features[-1].astype(np.float32)
    weighted_consensus = 0.0
    for member, weight in COUNCIL_MEMBERS.items():
        raw_vote = engines[member].predict(latest_features)
        weighted_consensus += (raw_vote * weight)

    # 4. تقييم جودة الإشارة (Signal Quality Scoring)
    quality_score, score_msg = evaluator.evaluate_signal_quality(weighted_consensus, market_regime, is_anomaly, smc_data)
    print(f"🎯 {score_msg}")

    if quality_score < 40.0:
        print("⚠️ جودة الإشارة منخفضة جداً. تم إلغاء الصفقة لحماية رأس المال.")
        return

    current_atr = df_temp['ATR_14'].iloc[-1]
    stop_dist = current_atr * 1.5
    profit_dist = current_atr * 2.5
    
    base_size = int(abs(weighted_consensus) * 20000)
    trade_size = max(1000, base_size)

    if weighted_consensus >= 0.18:
        direction = "BUY"
        print(f"🚀 [GEN-5 BUY] تنفيذ صفقة شراء بذكاء اصطناعي ذاتي التعلّم - الحجم: {trade_size}")
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            journal.log_trade(direction, trade_size, weighted_consensus, market_regime, f"Score: {quality_score}", "EXECUTED")
            
    elif weighted_consensus <= -0.18:
        direction = "SELL"
        print(f"📉 [GEN-5 SELL] تنفيذ صفقة بيع بذكاء اصطناعي ذاتي التعلّم - الحجم: {trade_size}")
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            journal.log_trade(direction, trade_size, weighted_consensus, market_regime, f"Score: {quality_score}", "EXECUTED")
    else:
        print("⚖️ قرار (CASH): الاستقرار خارج السوق.")

if __name__ == "__main__":
    run_gen5_sovereign_bot()
