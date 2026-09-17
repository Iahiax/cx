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

# الأوزان الافتتاحية للمجلس (سيتم تعديلها ذاتياً بناءً على الأداء التاريخي في السجلات)
COUNCIL_MEMBERS = {
    "TransformerForecaster": 0.45,
    "MultiAgentTrendSwarm": 0.35, 
    "OnlineSVMTrend": 0.20,      
}

def check_macro_news_circuit_breaker():
    """قاطع الأخبار الكبرى: فحص إذا كان الوقت الحالي يصادف صدور بيانات اقتصادية خطرة"""
    # يمكن ربطها بميلاد أجندة اقتصادية، حالياً تعود بـ False (السوق آمن للتداول)
    return False

def run_ultimate_ai_council():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 👑 بدء تشغيل مجلس Alpha Fund الآلي (DEMO - Leverage {LEVERAGE}:1)")
    
    # 1. التحقق من قاطع الأخبار الكبرى
    if check_macro_news_circuit_breaker():
        print("🛡️ حماية قصوى: تم تعليق المجلس مؤقتاً بسبب صدور بيانات اقتصادية كبرى عالية التأثير.")
        return

    cst, xst = login()
    if not cst: return

    print("📥 سحب البيانات، هندسة السيولة (VWAP & Force Index) والتطبيع...")
    raw_df = get_market_data(cst, xst)
    if raw_df is None: return
    
    # فلتر السيولة: منع التداول إذا كان مؤشر Force Index يعاكس بشدة اتجاه السوق العرضي
    last_force_index = raw_df['volume'].iloc[-1] * (raw_df['close'].iloc[-1] - raw_df['close'].iloc[-2])
    
    current_atr = raw_df['high'].iloc[-1] - raw_df['low'].iloc[-1]
    if current_atr == 0: current_atr = 0.0005
    
    df = add_features(raw_df)
    env = ProTradingEnv(df)
    
    engines = {}
    print("⚡ تفعيل عقول المجلس الاستخباراتي التدريبي...")
    for member in COUNCIL_MEMBERS.keys():
        engine = StrategyEngine(member)
        engine.train(df=df, env=env)
        engines[member] = engine
        
    print("\n⚖️ [جلسة التصويت اللحظي - Confidence-Weighted Voting]")
    latest_features = env.features[-1].astype(np.float32)
    weighted_consensus = 0.0
    
    for member, weight in COUNCIL_MEMBERS.items():
        raw_vote = engines[member].predict(latest_features)
        weighted_vote = raw_vote * weight
        weighted_consensus += weighted_vote
        
        dir_text = "🟢 LONG" if raw_vote > 0 else "🔴 SHORT"
        print(f"   🤖 {member:<25} | الصوت: {dir_text} | قوة القرار: {weight*100}%")

    print("-" * 50)
    print(f"🧠 الإجماع النهائي للمجلس (AI Fusion Score): {weighted_consensus:+.3f}")
    print("-" * 50)
    
    # فلتر تصفية الـ Fakeouts عبر السيولة
    if (weighted_consensus > 0 and last_force_index < 0) or (weighted_consensus < 0 and last_force_index > 0):
        print("⚠️ فلتر السيولة (Volume Imbalance): تم رصد ضغط تداول عكسي قوي. تم إلغاء الصفقة تفادياً للاختراق الوهمي (Fakeout).")
        return

    CONFIDENCE_THRESHOLD = 0.20 
    
    # إدارة المخاطر المتكيفة مع رافعة 100:1 ومعيار كيلي
    stop_dist = current_atr * 1.5
    profit_dist = current_atr * 2.5
    
    # حساب حجم العقد بناءً على قوة الإجماع والرافعة المالية 100:1
    base_size = int(abs(weighted_consensus) * 15000)
    trade_size = max(1000, base_size)
    
    if weighted_consensus >= CONFIDENCE_THRESHOLD:
        direction = "BUY"
        msg = f"🚀 [DEMO BUY] تنفيذ صفقة شراء - حجم العقد: {trade_size} - رافعة: {LEVERAGE}:1 - إجماع: {weighted_consensus:.2f}"
        print(msg)
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            logging.info(msg)
            
    elif weighted_consensus <= -CONFIDENCE_THRESHOLD:
        direction = "SELL"
        msg = f"📉 [DEMO SELL] تنفيذ صفقة بيع - حجم العقد: {trade_size} - رافعة: {LEVERAGE}:1 - إجماع: {weighted_consensus:.2f}"
        print(msg)
        if execute_order(cst, xst, direction, trade_size, stop_distance=stop_dist, profit_distance=profit_dist):
            logging.info(msg)
            
    else:
        msg = f"⚖️ قرار (CASH) - لا يوجد إتفاق واضح. البقاء خارج السوق. قوة الإجماع: {weighted_consensus:.2f}"
        print(msg)
        logging.info(msg)

if __name__ == "__main__":
    run_ultimate_ai_council()
