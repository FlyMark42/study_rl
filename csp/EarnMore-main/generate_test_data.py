# -*- coding: utf-8 -*-
"""
生成测试训练数据，用于验证监控功能
"""
import os
import pandas as pd
import numpy as np
from monitor import TrainingMonitor

# 创建测试实验目录
test_exp = "exp_001_sac_base_lr_0.0001_batch_size_64_seed_10"
exp_dir = os.path.join("experiments", test_exp)
logs_dir = os.path.join(exp_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# 生成模拟训练数据
np.random.seed(42)
episodes = 100

data = []
base_actor_loss = 0.5
base_critic_loss = 1.0
base_reward = 0.01

for i in range(1, episodes + 1):
    # 模拟训练指标的变化趋势
    actor_loss = base_actor_loss * np.exp(-i * 0.01) + np.random.normal(0, 0.05)
    critic_loss = base_critic_loss * np.exp(-i * 0.008) + np.random.normal(0, 0.08)
    reward = base_reward * (1 + i * 0.01) + np.random.normal(0, 0.02)
    
    # 每10个episode记录一次验证ASR
    validation_asr = None
    if i % 10 == 0:
        validation_asr = 0.3 + i * 0.005 + np.random.normal(0, 0.02)
        validation_asr = max(0.1, min(0.9, validation_asr))  # 限制在合理范围
    
    record = {
        'episode': i,
        'timestamp': f"2025-09-10 10:{i:02d}:00",
        'actor_loss': max(0.01, actor_loss),
        'critic_loss': max(0.01, critic_loss),
        'reward': reward,
        'learning_rate': 0.0001
    }
    
    if validation_asr is not None:
        record['validation_asr'] = validation_asr
    
    data.append(record)

# 保存测试数据
df = pd.DataFrame(data)
csv_path = os.path.join(logs_dir, "training_metrics.csv")
df.to_csv(csv_path, index=False)

print("Test training data generated: {}".format(csv_path))
print("Episodes: {}".format(len(df)))