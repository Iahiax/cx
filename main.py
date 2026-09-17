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
from config import LOG_FILE, LEVERAGE, EPIC_DXY, EPIC_GOLD

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

# تشكيل المجلس الأسطوري المكون من 4 عقول عصبية متقدمة
COUNCIL_MEMBERS = {
    "TransformerForecaster": 0.35,  # التنبؤ العميق
    "MultiAgentTrendSwarm": 0.30,   # سرب تعلم الآلة
    "OnlineSVMTrend": 0.15,         # الخبير الإحصائي
    "VolatilityProphet": 0.20       # نبي التقلبات والسيولة
}

def run_ultimate_ai_council():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 👑 بدء تشغيل كيان الذكاء الاصطناعي الأسطوري (Demo - Leverage {LEVERAGE}:1)")
    
    cst, xst = login()
    if not cst: return

    print("📥 سحب البيانات الكلية والتقاطعية (EUR/USD, DXY, Gold)...")
    raw_df = get_market_data(cst, xst)
    df_dxy = get_market_data(cst, xst, target_epic=EPIC_DXY)
    
    if raw_df is None: return

    # 1. درع تخدير الأخبار الكاذبة (Noise Spike Shield عبر قياس سرعة السعر العالية)
    last_velocity = raw_df['close'].diff().abs().iloc[-1]
    current_atr = raw_df['high'].iloc[-1] - raw_df['low'].iloc[-1]
    if current_atr == 0: current_atr = 0.0005

    if last_velocity > (current_atr * 3.5):
        print(f"🚨 [Noise Spike Shield]: رصد حركة سعرية جنونية مفاجئة ({last_velocity:.4f}). تم تفعيل درع التخدير وإلغاء التداول تفادياً للأخبار المزيفة!")
        return

    # 2. فحص الارتباط التقاطعي مع مؤشر الدولار (Cross-Asset Macro Correlation)
    if df_dxy is not None and not df_dxy.empty:
        dxy_change = df_dxy['close'].iloc[-1] - df_dxy['close'].iloc[-2]
        eur_change = raw_df['close'].iloc[-1] - raw_df['close'].iloc[-2]
        # إذا كان مؤشر الدولار واليورو يتحركان في نفس الاتجاه بشكل شاذ، فهذا تضارب تقاطعي
        if dxy_change > 0 and eur_change > 0:
            print("⚠️ [Cross-Asset Filter]: تضارب بين حركة مؤشر الدولار وزوج اليورو. سيتم خفض الحذر أو تقييد الصفقة.")

    # 3. صيد فخاخ الحيتان (Stop-Hunt / Trap Detector)
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
            logging.info("Trap Detector executed BUY trade.")
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
    print(f"🧠 الإجماع العصبي النهائي للكيان (Fusion Score): {weighted_consensus:+.3f}")
    print("-" * 50)
    
    CONFIDENCE_THRESHOLD = 0.18 
    stop_dist = current_atr * 1.5
    profit_dist = current_atr * 2.5
    trade_size = max(1000, int(abs(weighted_consensus) * 20000))
    
    if weighted_consensus >= CONFIDENCE_THRESHOLD:
        direction = "BUY"
        msg = f"🚀 [LEGENDARY BUY] تنفيذ صفقة شراء - حجم العقد: {trade_size} - رافعة: {LEVERAGE}:1 - إجماع: {weighted_consensus:.2f}"
        print(msg)
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            logging.info(msg)
            
    elif weighted_consensus <= -CONFIDENCE_THRESHOLD:
        direction = "SELL"
        msg = f"📉 [LEGENDARY SELL] تنفيذ صفقة بيع - حجم العقد: {trade_size} - رافعة: {LEVERAGE}:1 - إجماع: {weighted_consensus:.2f}"
        print(msg)
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            logging.info(msg)
            
    else:
        msg = f"⚖️ قرار (CASH) - السوق في منطقة غائمة. البقاء خارج السوق. الإجماع: {weighted_consensus:.2f}"
        print(msg)
        logging.info(msg)

if __name__ == "__main__":
    run_ultimate_ai_council()
