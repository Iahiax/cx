# config.py
# إعدادات حساب Capital.com (تأكد من تحديثها)
EMAIL = "yahia.x@outlook.sa"
API_KEY = "ut2RpxSbx6fiDdHv"
API_KEY_PASSWORD = "Yahia@1411"
DEMO = True

# إعدادات السيرفر والسوق
SERVER = "https://demo-api-capital.backend-capital.com" if DEMO else "https://api-capital.backend-capital.com"
EPIC = "CS.D.EURUSD.CFD.IP"


MAX_DATA_POINTS = 1000

# 🔄 تم التعديل هنا: تغيير الفريم الزمني إلى الدقيقة الواحدة
RESOLUTION = "MINUTE" 

# إعدادات النظام
LOG_FILE = "trading_log.txt"
