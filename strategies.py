# strategies.py
import os
import numpy as np
import joblib

# إيقاف رسائل تحذير TensorFlow المزعجة في الـ Terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from sklearn.svm import SVC
from stable_baselines3 import PPO, A2C

class StrategyEngine:
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.model = None

    # ==========================================
    # 🧠 Transformer Trend Forecaster
    # ==========================================
    def build_transformer_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # 🟢 التعديل الجوهري هنا: تحديد أسماء المدخلات صراحة (query, value) لتفادي خطأ Keras
        attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(query=inputs, value=inputs)
        attention_output = Dropout(0.2)(attention_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        ffn_output = Dense(64, activation="relu")(out1)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        ffn_output = Dropout(0.2)(ffn_output)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        pooling = GlobalAveragePooling1D()(out2)
        outputs = Dense(1, activation="sigmoid")(pooling) # يخرج بين 0 و 1
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train_TransformerForecaster(self, df):
        print(f"🌌 تدريب: {self.strategy_name}...")
        X = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Target']).values[:-1]
        y = df['Target'].values[:-1]
        X = X.reshape((X.shape[0], 1, X.shape[1])) 
        
        # 🟢 التعديل الثاني: استخدام صيغة .keras الحديثة بدلاً من .h5 القديمة
        model_path = "transformer_model.keras"
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self.build_transformer_model((1, X.shape[2]))
            
        self.model.fit(X, y, epochs=15, batch_size=32, verbose=0)
        self.model.save(model_path)

    def predict_TransformerForecaster(self, features):
        X = features.reshape((1, 1, len(features)))
        prediction = self.model.predict(X, verbose=0)[0][0]
        return (prediction * 2) - 1.0 # تحويل الناتج ليكون بين -1 و 1

    # ==========================================
    # 🤖 Multi-Agent Trend Swarm
    # ==========================================
    def train_MultiAgentTrendSwarm(self, env):
        print(f"🐝 تدريب: {self.strategy_name}...")
        self.agents = {}
        self.agents['PPO'] = PPO("MlpPolicy", env, verbose=0)
        self.agents['PPO'].learn(total_timesteps=5000)
        
        self.agents['A2C'] = A2C("MlpPolicy", env, verbose=0)
        self.agents['A2C'].learn(total_timesteps=5000)

    def predict_MultiAgentTrendSwarm(self, features):
        action_ppo, _ = self.agents['PPO'].predict(features)
        action_a2c, _ = self.agents['A2C'].predict(features)
        return (action_ppo[0] + action_a2c[0]) / 2.0

    # ==========================================
    # 📈 Online SVM Trend (الخبير الرياضي)
    # ==========================================
    def train_OnlineSVMTrend(self, df):
        print(f"⚙️ تدريب: {self.strategy_name}...")
        X = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Target']).values[:-1]
        y = df['Target'].values[:-1]
        
        self.model = SVC(kernel='rbf', probability=True)
        self.model.fit(X, y)
        joblib.dump(self.model, "svm_model.pkl")

    def predict_OnlineSVMTrend(self, features):
        pred = self.model.predict(features.reshape(1, -1))[0]
        return 1.0 if pred == 1 else -1.0 

    # ==========================================
    # 🧬 محول التشغيل الذكي (Router)
    # ==========================================
    def train(self, df=None, env=None):
        method_name = f"train_{self.strategy_name}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            if "Swarm" in self.strategy_name or "RL" in self.strategy_name:
                method(env)
            else:
                method(df)
        else:
            print(f"⚠️ دالة {method_name} مفقودة.")

    def predict(self, features):
        method_name = f"predict_{self.strategy_name}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(features)
        return 0.0
