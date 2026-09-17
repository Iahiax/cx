pip install requests pandas numpy pandas-ta gym stable-baselines3 scikit-learn tensorflow joblib

python main.py
CUDA_VISIBLE_DEVICES="" python3 main.py


python3 -c '
with open("config.py", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()
cleaned = "".join(c for c in content if ord(c) < 128)
with open("config.py", "w", encoding="utf-8") as f:
    f.write(cleaned)
print("✅ تم تنظيف ملف config.py وإزالة أي رموز مخفية بنجاح!")
'
