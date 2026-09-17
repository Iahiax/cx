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
from market_watcher import MarketWatcher
from regime_engine import RegimeEngine
from meta_model import MetaModel
from journal import TradeJournal
from config import LOG_FILE, LEVERAGE

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')
journal = TradeJournal()

COUNCIL_MEMBERS = {
    "TransformerForecaster": 0.35,  
    "MultiAgentTrendSwarm": 0.30,   
    "OnlineSVMTrend": 0.15,         
    "VolatilityProphet": 0.20       
}

def run_ultimate_ai_council():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 👑 [الكيان السيادي] بدء دورة التحليل الشاملة (Demo - Leverage {LEVERAGE}:1)")
    
    # التحقق من إغلاق السوق (نهاية الأسبوع أو الإغلاق اليومي)
    now = datetime.now()
    if now.weekday() == 4 and now.hour >= 22: # الجمعة متأخر
        print("🛡️ نظام إيقاف الحماية: اقتراب إغلاق السوق الأسبوعي. منع فتح صفقات جديدة.")
        return

    cst, xst = login()
    if not cst: return

    raw_df = get_market_data(cst, xst)
    if raw_df is None or raw_df.empty: return

    # 1. Market Watcher & Regime Detection
    watcher = MarketWatcher(raw_df)
    structure = watcher.detect_market_structure()
    
    df = add_features(raw_df)
    regime_eng = RegimeEngine(df)
    market_regime = regime_eng.detect_regime()
    
    if regime_eng.check_news_blackout():
        print("🚨 [News Blackout]: توقف تام بسبب صدور بيانات اقتصادية خطرة.")
        return

    print(f"📊 حالة السوق الحالية (Regime): {market_regime} | بنية السيولة: {structure}")

    env = ProTradingEnv(df)
    engines = {}
    for member in COUNCIL_MEMBERS.keys():
        engine = StrategyEngine(member)
        engine.train(df=df, env=env)
        engines[member] = engine
        
    latest_features = env.features[-1].astype(np.float32)
    weighted_consensus = 0.0
    
    for member, weight in COUNCIL_MEMBERS.items():
        raw_vote = engines[member].predict(latest_features)
        weighted_consensus += (raw_vote * weight)

    print(f"🧠 الإجماع العصبي الأولي: {weighted_consensus:+.3f}")

    # 2. Meta-Model Evaluation
    meta = MetaModel()
    approved, size_multiplier, meta_msg = meta.evaluate_signal(weighted_consensus, market_regime, structure)
    print(f"🛡️ قرار Meta-Model: {meta_msg}")

    if not approved:
        journal.log_trade("NONE", 0, weighted_consensus, market_regime, meta_msg, "REJECTED")
        return

    current_atr = df['ATR_14'].iloc[-1]
    stop_dist = current_atr * 1.5
    profit_dist = current_atr * 2.5
    
    base_size = int(abs(weighted_consensus) * 20000)
    trade_size = max(1000, int(base_size * size_multiplier))

    CONFIDENCE_THRESHOLD = 0.18
    if weighted_consensus >= CONFIDENCE_THRESHOLD:
        direction = "BUY"
        print(f"🚀 [BUY] تنفيذ صفقة شراء متكيفة - الحجم: {trade_size}")
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            journal.log_trade(direction, trade_size, weighted_consensus, market_regime, meta_msg, "EXECUTED")
            
    elif weighted_consensus <= -CONFIDENCE_THRESHOLD:
        direction = "SELL"
        print(f"📉 [SELL] تنفيذ صفقة بيع متكيفة - الحجم: {trade_size}")
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            journal.log_trade(direction, trade_size, weighted_consensus, market_regime, meta_msg, "EXECUTED")
    else:
        print("⚖️ قرار (CASH): بقاء خارج السوق لعدم وجود وضوح كافٍ.")

if __name__ == "__main__":
    run_ultimate_ai_council()
