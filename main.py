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

COUNCIL_MEMBERS = {
    "TransformerForecaster": 0.35,  
    "MultiAgentTrendSwarm": 0.30,   
    "OnlineSVMTrend": 0.15,         
    "VolatilityProphet": 0.20       
}

def run_ultimate_ai_council():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 👑 [1/6] بدء تشغيل الكيان الأسطوري لزوج EUR/USD (Demo - Leverage {LEVERAGE}:1)")
    
    # 1. الاتصال
    print("🔄 جاري الاتصال بمنصة Capital.com...")
    cst, xst = login()
    if not cst:
        print("❌ فشل الاتصال بالمنصة.")
        return

    # 2. سحب البيانات
    print("📥 [2/6] جاري سحب الشموع التاريخية واللحظية لـ EUR/USD (1000 شمعة)...")
    raw_df = get_market_data(cst, xst)
    if raw_df is None or raw_df.empty:
        print("❌ لم يتم استلام أي بيانات من المنصة.")
        return
    print(f"✅ تم بنجاح سحب {len(raw_df)} شمعة تاريخية.")

    # 3. درع الأخبار والفخاخ
    print("🛡️ [3/6] فحص درع تخدير الأخبار وكشف فخاخ الحيتان...")
    last_velocity = raw_df['close'].diff().abs().iloc[-1]
    current_atr = raw_df['high'].iloc[-1] - raw_df['low'].iloc[-1]
    if current_atr == 0: current_atr = 0.0005

    if last_velocity > (current_atr * 3.5):
        print(f"🚨 [Noise Spike Shield]: رصد حركة سعرية جنونية مفاجئة ({last_velocity:.4f}). تم تفعيل درع التخدير وإلغاء التداول!")
        return

    # 4. هندسة الخصائص
    print("⚙️ [4/6] تطبيق هندسة الخصائص، مؤشرات السيولة، والتطبيع الاحترافي...")
    df = add_features(raw_df)
    env = ProTradingEnv(df)
    print("✅ تمت هندسة الخصائص وتجهيز البيئة بنجاح.")

    # 5. تدريب العقول الأربعة مع تتبع التقدم
    print("⚡ [5/6] بدء تدريب عقول المجلس العصبي الأربعة...")
    engines = {}
    total_members = len(COUNCIL_MEMBERS)
    for i, (member, weight) in enumerate(COUNCIL_MEMBERS.items(), 1):
        print(f"   ⏳ [{i}/{total_members}] جاري تدريب العقل: {member}...")
        engine = StrategyEngine(member)
        engine.train(df=df, env=env)
        engines[member] = engine
        print(f"   ✅ تم تدريب واستقرار العقل: {member}")

    # 6. التصويت واتخاذ القرار
    print("\n⚖️ [6/6] انعقاد جلسة التصويت العصبي وزنه حسب الثقة...")
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
