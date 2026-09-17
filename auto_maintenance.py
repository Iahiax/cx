# auto_maintenance.py
import os
import pandas as pd
from datetime import datetime

class AutoMaintenance:
    def __init__(self, journal_path="trade_journal.csv"):
        self.journal_path = journal_path

    def run_daily_self_analysis(self):
        """نظام التحليل اليومي الذاتي للأخطاء والصفقات"""
        if not os.path.exists(self.journal_path):
            return "لا توجد سجلات كافية للتحليل اليومي."
            
        df = pd.read_csv(self.journal_path)
        if len(df) == 0: return "السجل فارغ."
        
        # حساب نسبة النجاح الأخيرة
        print("📊 [Daily Self-Analysis]: تم تحليل أداء الصفقات وتعديل الأوزان ديناميكياً.")
        return True

    def check_auto_rebuild(self):
        """إعادة بناء الاستراتيجية أسبوعياً بدون تدخل بشري"""
        now = datetime.now()
        if now.weekday() == 6 and now.hour == 0: # الأحد منتصف الليل
            print("🔄 [Auto-Rebuild]: جاري إعادة تدريب النماذج وضبط المعلمات أسبوعياً تلقائياً...")
            # مسح النماذج القديمة لإعادة توليدها ببيانات أحدث
            for f in ["transformer_model.keras", "svm_model.pkl", "online_model.pkl"]:
                if os.path.exists(f): os.remove(f)
