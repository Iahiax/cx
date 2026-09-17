# strategies.py
import os
import numpy as np
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

class StrategyEngine:
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.model = None
        self.transformer_path = f"transformer_{strategy_name}.keras"
        self.ml_path = f"ml_{strategy_name}.pkl"

    def build_transformer_model(self, input_shape):
        inputs = Input(shape=input_shape)
        attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(query=inputs, value=inputs)
        attention_output = Dropout(0.2)(attention_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        ffn_output = Dense(64, activation="relu")(out1)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        ffn_output = Dropout(0.2)(ffn_output)
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        pooling = GlobalAveragePooling1D()(out2)
        outputs = Dense(1, activation="sigmoid")(pooling)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train(self, df=None, env=None):
        """التدريب الأولي أو التحديث الذاتي المستمر"""
        X = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Target']).values[:-1]
        y = df['Target'].values[:-1]

        if "Transformer" in self.strategy_name:
            X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
            if os.path.exists(self.transformer_path):
                self.model = tf.keras.models.load_model(self.transformer_path)
                # التدريب الذاتي المستمر بـ Epochs قليلة على البيانات الجديدة
                self.model.fit(X_reshaped, y, epochs=2, batch_size=32, verbose=0)
            else:
                self.model = self.build_transformer_model((1, X.shape[2]))
                self.model.fit(X_reshaped, y, epochs=10, batch_size=32, verbose=0)
            self.model.save(self.transformer_path)

        elif "Swarm" in self.strategy_name:
            if os.path.exists(self.ml_path):
                self.agent_rf, self.agent_gb = joblib.load(self.ml_path)
            else:
                self.agent_rf = RandomForestClassifier(n_estimators=100, random_state=42)
                self.agent_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
            # التحديث الذاتي التدريجي
            self.agent_rf.fit(X, y)
            self.agent_gb.fit(X, y)
            joblib.dump((self.agent_rf, self.agent_gb), self.ml_path)

        else: # SVM أو التقلبات
            if os.path.exists(self.ml_path):
                self.model = joblib.load(self.ml_path)
            else:
                self.model = SVC(kernel='rbf', probability=True)
            
            self.model.fit(X, y)
            joblib.dump(self.model, self.ml_path)

        print(f"🔄 [Autonomous Self-Training]: تم تحديث وتدريب العقل ({self.strategy_name}) ذاتياً بنجاح.")

    def predict(self, features):
        if "Transformer" in self.strategy_name:
            X = features.reshape((1, 1, len(features)))
            prediction = self.model.predict(X, verbose=0)[0][0]
            return (prediction * 2) - 1.0 
        elif "Swarm" in self.strategy_name:
            pred_rf = self.agent_rf.predict(features.reshape(1, -1))[0]
            pred_gb = self.agent_gb.predict(features.reshape(1, -1))[0]
            return ((1.0 if pred_rf == 1 else -1.0) + (1.0 if pred_gb == 1 else -1.0)) / 2.0
        else:
            pred = self.model.predict(features.reshape(1, -1))[0]
            return 1.0 if pred == 1 else -1.0
