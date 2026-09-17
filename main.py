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

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

# تشكيل المجلس الأسطوري المكون من 4 عقول عصبية متقدمة لزوج EUR/USD
COUNCIL_MEMBERS = {
    "TransformerForecaster": 0.35,  # التنبؤ العميق
    "MultiAgentTrendSwarm": 0.30,   # سرب تعلم الآلة
    "OnlineSVMTrend": 0.15,         # الخبير الإحصائي
    "VolatilityProphet": 0.20       # نبي التقلبات والسيولة
}

def run_ultimate_ai_council():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 👑 بدء تشغيل الكيان الأسطوري لزوج EUR/USD (Demo - Leverage {LEVERAGE}:1)")
    
    cst, xst = login()
    if not cst: return

    print("📥 سحب بيانات EUR/USD اللحظية...")
    raw_df = get_market_data(cst, xst)
    if raw_df is None: return

    # 1. درع تخدير الأخبار الكاذبة (Noise Spike Shield)
    last_velocity = raw_df['close'].diff().abs().iloc[-1]
    current_atr = raw_df['high'].iloc[-1] - raw_df['low'].iloc[-1]
    if current_atr == 0: current_atr = 0.0005

    if last_velocity > (current_atr * 3.5):
        print(f"🚨 [Noise Spike Shield]: رصد حركة سعرية جنونية مفاجئة ({last_velocity:.4f}). تم تفعيل درع التخدير وإلغاء التداول تفادياً للأخبار المزيفة!")
        return

    # 2. صيد فخاخ الحيتان (Stop-Hunt / Trap Detector)
    last_volume = raw_df['volume'].iloc[-1]
    avg_volume = raw_df['volume'].rolling(20).mean().iloc[-1]
    is_price_breaking_low = raw_df['close'].iloc[-1] < raw_df['low'].rolling(5).min().iloc[-2]
    
    if is_price_breaking_low and last_volume < (avg_volume * 0.7):
        print("💡 [The Trap Detector]: تم كشف فخ صانع السوق! كسر وهمي للقاع بحجم تداول ضعيف جداً. انعكاس فوري نحو الشراء مع الحيتان!")
        direction = "BUY"
        trade_size = 2000
        stop_dist = current_atr * 1.2
        profit_dist = current_atr * 2.5
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            logging.info("Trap Detector executed BUY trade on EUR/USD.")
        return

    df = add_features(raw_df)
    env = ProTradingEnv(df)
    
    engines = {}
    print("⚡ إيقاظ المجلس العصبي الأسطوري...")
    for member in COUNCIL_MEMBERS.keys():
        engine = StrategyEngine(member)
        engine.train(df=df, env=env)
        engines[member] = engine
        
    print("\n⚖️ [جلسة التصويت العصبي - Confidence-Weighted Voting]")
    latest_features = env.features[-1].astype(np.float32)
    weighted_consensus = 0.0
    
    for member, weight in COUNCIL_MEMBERS.items():
        raw_vote = engines[member].predict(latest_features)
        weighted_vote = raw_vote * weight
        weighted_consensus += weighted_vote
        
        dir_text = "🟢 LONG" if raw_vote > 0 else "🔴 SHORT"
        print(f"   🤖 {member:<25} | الصوت: {dir_text} | القوة: {weight*100}%")

    print("-" * 50)
    print(f"🧠 الإجماع العصبي النهائي لزوج EUR/USD (Fusion Score): {weighted_consensus:+.3f}")
    print("-" * 50)
    
    CONFIDENCE_THRESHOLD = 0.18 
    stop_dist = current_atr * 1.5
    profit_dist = current_atr * 2.5
    trade_size = max(1000, int(abs(weighted_consensus) * 20000))
    
    if weighted_consensus >= CONFIDENCE_THRESHOLD:
        direction = "BUY"
        msg = f"🚀 [EUR/USD BUY] تنفيذ صفقة شراء - حجم العقد: {trade_size} - رافعة: {LEVERAGE}:1 - إجماع: {weighted_consensus:.2f}"
        print(msg)
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            logging.info(msg)
            
    elif weighted_consensus <= -CONFIDENCE_THRESHOLD:
        direction = "SELL"
        msg = f"📉 [EUR/USD SELL] تنفيذ صفقة بيع - حجم العقد: {trade_size} - رافعة: {LEVERAGE}:1 - إجماع: {weighted_consensus:.2f}"
        print(msg)
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            logging.info(msg)
            
    else:
        msg = f"⚖️ قرار (CASH) - السوق عرضي لزوج EUR/USD. البقاء خارج السوق. الإجماع: {weighted_consensus:.2f}"
        print(msg)
        logging.info(msg)

if __name__ == "__main__":
    run_ultimate_ai_council()
