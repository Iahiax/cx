# online_learner.py
import numpy as np
import os
import joblib
from sklearn.linear_model import SGDClassifier # نموذج تدريجي تدريبي ذاتي

class OnlineLearner:
    def __init__(self, model_path="online_model.pkl"):
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = SGDClassifier(loss='log_loss', random_state=42)

    def partial_fit(self, X, y):
        """التعلّم المستمر من الصفقات والبيانات اللحظية الجديدة"""
        self.model.partial_fit(X, y, classes=np.array([0, 1]))
        joblib.dump(self.model, self.model_path)

    def detect_anomaly(self, df):
        """نظام اكتشاف الانحرافات والشموع غير الطبيعية (Market Anomaly & Manipulation)"""
        last_candle = df.iloc[-1]
        avg_volume = df['volume'].rolling(30).mean().iloc[-1]
        atr = df['ATR_14'].iloc[-1]
        
        # كشف تلاعب أو ضغط سيولة مفاجئ (Smart Money Manipulation & Stop Hunt)
        is_volume_anomaly = last_candle['volume'] > (avg_volume * 3.5)
        is_price_anomaly = abs(last_candle['close'] - last_candle['open']) > (atr * 2.5)
        
        if is_volume_anomaly and is_price_anomaly:
            return True, "🚨 [Anomaly/Manipulation]: تم رصد تلاعب مؤسسي أو سحب سيولة (Stop Hunt) غير طبيعي!"
        return False, "السوق طبيعي"
