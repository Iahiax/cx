# environments.py
import gym
from gym import spaces
import numpy as np

class ProTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000.0, transaction_cost=0.0001):
        super(ProTradingEnv, self).__init__()
        self.df = df
        
        # المدخلات هي المؤشرات فقط (بدون الأسعار المجردة والهدف)
        self.features = df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'Target'], errors='ignore').values
        self.prices = df['close'].values
        
        # حجم الصفقات من -1 للبيع الكامل إلى 1 للشراء الكامل
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32)
        
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.current_position = 0.0
        return self.features[self.current_step].astype(np.float32)

    def step(self, action):
        action_val = np.clip(action[0], -1.0, 1.0)
        current_price = self.prices[self.current_step]
        prev_price = self.prices[self.current_step - 1] if self.current_step > 0 else current_price
        
        # حساب الأرباح/الخسائر وتكلفة التداول
        step_pnl = self.current_position * (current_price - prev_price) * 100000
        trade_size = abs(action_val - self.current_position)
        cost = trade_size * self.transaction_cost * 100000
        
        self.balance += (step_pnl - cost)
        self.current_position = action_val
        
        # المكافأة الحقيقية معدلة بالمخاطرة
        reward = step_pnl - cost
        if self.balance < self.max_balance:
            drawdown = self.max_balance - self.balance
            reward -= (drawdown * 0.05) # عقاب التراجع
        else:
            self.max_balance = self.balance
            
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self.features[self.current_step] if not done else self.features[-1]
        
        return obs.astype(np.float32), reward, done, {}
