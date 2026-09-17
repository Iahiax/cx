# meta_model.py
import numpy as np

class MetaModel:
    def __init__(self):
        pass

    def evaluate_signal(self, consensus_score, regime, structure_info):
        """حارس البوابة: يراجع قرار المجلس بناءً على حالة السوق وفجوات السيولة"""
        
        # رفض قاطع في حالة الصدمات الإخبارية
        if regime == "News Shock":
            return False, 0, "مرفود: صدمة اخبارية (News Shock)"
            
        # إذا كان السوق عرضي والإشارة ضعيفة
        if regime == "Ranging" and abs(consensus_score) < 0.25:
            return False, 0, "مرفوض: السوق عرضي والإشارة غير كافية"
            
        # تأكيد عبر سحب السيولة (Liquidity Sweep)
        multiplier = 1.0
        if structure_info["sweep_high"] and consensus_score < 0:
            multiplier = 1.5 # تعزيز صفقة البيع بعد سحب سيولة القمة
        elif structure_info["sweep_low"] and consensus_score > 0:
            multiplier = 1.5 # تعزيز صفقة الشراء بعد سحب سيولة القاعة

        return True, multiplier, "تم قبول الإشارة بنجاح بواسطة Meta-Model"
