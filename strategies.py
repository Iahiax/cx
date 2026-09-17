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

class StrategyEngine:
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.model = None

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

    def train_TransformerForecaster(self, df):
        print(f"🌌 تدريب عقل التنبؤ العميق: {self.strategy_name}...")
        X = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Target']).values[:-1]
        y = df['Target'].values[:-1]
        X = X.reshape((X.shape[0], 1, X.shape[1])) 
        
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
        return (prediction * 2) - 1.0 

    def train_MultiAgentTrendSwarm(self, df):
        print(f"🐝 تدريب عقل السرب (ML Ensemble): {self.strategy_name}...")
        X = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Target']).values[:-1]
        y = df['Target'].values[:-1]
        
        self.agent_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.agent_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        self.agent_rf.fit(X, y)
        self.agent_gb.fit(X, y)

    def predict_MultiAgentTrendSwarm(self, features):
        f = features.reshape(1, -1)
        pred_rf = self.agent_rf.predict(f)[0]
        pred_gb = self.agent_gb.predict(f)[0]
        vote_rf = 1.0 if pred_rf == 1 else -1.0
        vote_gb = 1.0 if pred_gb == 1 else -1.0
        return (vote_rf + vote_gb) / 2.0

    def train_OnlineSVMTrend(self, df):
        print(f"⚙️ تدريب عقل الخبير الرياضي (SVM): {self.strategy_name}...")
        X = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Target']).values[:-1]
        y = df['Target'].values[:-1]
        
        self.model = SVC(kernel='rbf', probability=True)
        self.model.fit(X, y)
        joblib.dump(self.model, "svm_model.pkl")

    def predict_OnlineSVMTrend(self, features):
        pred = self.model.predict(features.reshape(1, -1))[0]
        return 1.0 if pred == 1 else -1.0 

    # عقل نبي التقلبات الرابع (Volatility Prophet)
    def train_VolatilityProphet(self, df):
        print(f"🔮 تدريب نبي التقلبات والسيولة العصبية: {self.strategy_name}...")
        X = df[['ATR_14', 'Volatility_Surface', 'Price_Velocity']].values[:-1]
        y = df['Target'].values[:-1]
        self.vol_model = SVC(kernel='poly', degree=3, probability=True)
        self.vol_model.fit(X, y)

    def predict_VolatilityProphet(self, features):
        # استخلاص خصائص التقلب المحددة فقط
        vol_features = np.array([features[3], features[-1], features[-2]]) # ATR, Velocity, Vol_Surface تقريبياً
        pred = self.vol_model.predict(vol_features.reshape(1, -1))[0]
        return 1.0 if pred == 1 else -1.0

    def train(self, df=None, env=None):
        method_name = f"train_{self.strategy_name}"
        if hasattr(self, method_name):
            getattr(self, method_name)(df)

    def predict(self, features):
        method_name = f"predict_{self.strategy_name}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(features)
        return 0.0
