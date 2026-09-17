# signal_evaluator.py
class SignalEvaluator:
    def __init__(self):
        pass

    def evaluate_signal_quality(self, consensus, regime, anomaly_detected, smc_data):
        """نظام تقييم الإشارة ومنع الإشارات الضعيفة"""
        if anomaly_detected:
            return 0.0, "مرفوض تماماً بسبب شذوذ السوق"
            
        score = abs(consensus) * 100
        
        # تعزيز الدرجة إذا توافقت مع بنية الأموال الذكية والترند
        if regime == "Trending":
            score *= 1.25
        if smc_data["volume_pressure"] > 1.5:
            score *= 1.15
            
        return score, f"درجة جودة الإشارة: {score:.1f}%"
