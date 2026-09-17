# main.py
import os

# 1. إجبار النظام على استخدام CPU ومنع البحث عن GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 2. الحل السحري لمنع تعارض مكتبات الحساب بين TensorFlow و PyTorch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 3. منع تصادم خيوط المعالج (Threads) على مستوى نظام التشغيل
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# 4. تقييد استهلاك PyTorch لمسارات المعالج
import torch
torch.set_num_threads(1)

# 5. تقييد استهلاك TensorFlow لمسارات المعالج
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
from config import LOG_FILE

# إعداد نظام تسجيل الأحداث اليومية (Logging)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

# تشكيل المجلس الأعلى (الاسم : نسبة قوة التصويت)
COUNCIL_MEMBERS = {
    "TransformerForecaster": 0.45,
    "MultiAgentTrendSwarm": 0.35, 
    "OnlineSVMTrend": 0.20,      
}

def run_ultimate_ai_council():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 👑 بدء تشغيل مجلس AI Trend Council")
    
    # 1. الاتصال وتجهيز البيانات
    cst, xst = login()
    if not cst: return

    print("📥 سحب البيانات، التطبيع، وهندسة الخصائص...")
    raw_df = get_market_data(cst, xst)
    if raw_df is None: return
    
    df = add_features(raw_df)
    env = ProTradingEnv(df)
    
    # 2. تدريب/إيقاظ النماذج
    engines = {}
    print("⚡ تفعيل شبكات الذكاء الاصطناعي...")
    for member in COUNCIL_MEMBERS.keys():
        engine = StrategyEngine(member)
        engine.train(df=df, env=env)
        engines[member] = engine
        
    # 3. جلسة التصويت اللحظي
    print("\n⚖️ [جلسة التصويت - Confidence-Weighted Voting]")
    latest_features = env.features[-1].astype(np.float32)
    weighted_consensus = 0.0
    
    for member, weight in COUNCIL_MEMBERS.items():
        raw_vote = engines[member].predict(latest_features)
        weighted_vote = raw_vote * weight
        weighted_consensus += weighted_vote
        
        dir_text = "🟢 LONG" if raw_vote > 0 else "🔴 SHORT"
        print(f"   🤖 {member:<25} | الصوت: {dir_text} | قوة القرار: {weight*100}%")

    # 4. إصدار القرار النهائي
    print("-" * 50)
    print(f"🧠 الإجماع النهائي للمجلس (AI Fusion Score): {weighted_consensus:+.3f}")
    print("-" * 50)
    
    CONFIDENCE_THRESHOLD = 0.25 # الحد الأدنى للتداول
    
    if weighted_consensus >= CONFIDENCE_THRESHOLD:
        direction = "BUY"
        trade_size = max(1000, int(abs(weighted_consensus) * 10000))
        msg = f"🚀 قرار (LONG) - حجم العقد: {trade_size} - قوة الإجماع: {weighted_consensus:.2f}"
        print(msg)
        if execute_order(cst, xst, direction, trade_size):
            logging.info(msg)
            
    elif weighted_consensus <= -CONFIDENCE_THRESHOLD:
        direction = "SELL"
        trade_size = max(1000, int(abs(weighted_consensus) * 10000))
        msg = f"📉 قرار (SHORT) - حجم العقد: {trade_size} - قوة الإجماع: {weighted_consensus:.2f}"
        print(msg)
        if execute_order(cst, xst, direction, trade_size):
            logging.info(msg)
            
    else:
        msg = f"⚖️ قرار (CASH) - لا يوجد إتفاق واضح. البقاء خارج السوق. قوة الإجماع: {weighted_consensus:.2f}"
        print(msg)
        logging.info(msg)

if __name__ == "__main__":
    run_ultimate_ai_council()
