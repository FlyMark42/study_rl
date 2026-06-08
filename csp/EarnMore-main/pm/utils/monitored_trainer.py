# -*- coding: utf-8 -*-
"""
训练监控包装器
该文件应该放在pm/utils目录下，用于集成训练监控功能到现有训练流程
"""
import os
import sys
import json
from datetime import datetime

# 添加根目录到路径以导入monitor模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, root_dir)

from monitor import TrainingMonitor

class MonitoredTrainer:
    """带监控功能的训练器包装类"""

    def __init__(self, experiment_name, enable_monitoring=True):
        self.experiment_name = experiment_name
        self.enable_monitoring = enable_monitoring
        self.monitor = TrainingMonitor() if enable_monitoring else None
        self.episode_count = 0

        print("MonitoredTrainer initialized for experiment: {}".format(experiment_name))
        print("Monitoring enabled: {}".format(enable_monitoring))

    def log_episode_metrics(self, metrics_dict):
        """记录单个episode的训练指标"""
        if not self.enable_monitoring:
            return

        # 使用传入的episode编号，如果没有就用计数器
        episode_num = metrics_dict.get('episode', self.episode_count + 1)
        self.episode_count = max(self.episode_count, episode_num)

        # 记录基础训练指标
        self.monitor.record_metrics(self.experiment_name, episode_num, metrics_dict)

        # 每20个episode显示一次指标摘要
        if episode_num % 20 == 0:
            print("Monitor: Episode {} metrics logged".format(episode_num))

    def log_value_function_metrics(self, episode, agent):
        """记录价值函数相关指标"""
        if not self.enable_monitoring:
            return

        try:
            value_metrics = {}

            # 尝试从agent获取价值函数指标
            if hasattr(agent, 'get_q_values') and callable(agent.get_q_values):
                q_values = agent.get_q_values()
                if q_values is not None:
                    import torch
                    if isinstance(q_values, torch.Tensor):
                        q_values = q_values.detach().cpu().numpy()
                    value_metrics['q_values_mean'] = float(q_values.mean())
                    value_metrics['q_values_std'] = float(q_values.std())
                    value_metrics['q_values_min'] = float(q_values.min())
                    value_metrics['q_values_max'] = float(q_values.max())

            # 尝试从agent获取TD误差
            if hasattr(agent, 'last_td_error') and agent.last_td_error is not None:
                import torch
                td_error = agent.last_td_error
                if isinstance(td_error, torch.Tensor):
                    td_error = td_error.detach().cpu().numpy()
                value_metrics['td_error_mean'] = float(td_error.mean())
                value_metrics['td_error_std'] = float(td_error.std())
                value_metrics['td_error_abs_mean'] = float(abs(td_error).mean())

            # 尝试获取策略熵
            if hasattr(agent, 'last_policy_entropy') and agent.last_policy_entropy is not None:
                entropy = agent.last_policy_entropy
                if hasattr(entropy, 'item'):
                    entropy = entropy.item()
                value_metrics['policy_entropy'] = float(entropy)

            # 尝试获取网络权重范数
            if hasattr(agent, 'act') and hasattr(agent.act, 'parameters'):
                import torch
                total_norm = 0
                for param in agent.act.parameters():
                    if param.grad is not None:
                        param_norm = param.data.norm(2)
                        total_norm += param_norm.item() ** 2
                value_metrics['q_network_weights_norm'] = float(total_norm ** (1. / 2))

            if value_metrics:
                self.monitor.record_value_function_metrics(self.experiment_name, episode, value_metrics)

        except Exception as e:
            print("Warning: Could not collect value function metrics: {}".format(e))

    def should_stop_early(self, min_episodes=50):
        """检查是否应该早停"""
        if not self.enable_monitoring or self.episode_count < min_episodes:
            return False

        # 每10个episode检查一次早停条件
        if self.episode_count % 10 == 0:
            try:
                return self.monitor.early_stop_check(self.experiment_name)
            except Exception as e:
                print("Warning: Early stop check failed: {}".format(e))
                return False

        return False

    def get_training_status(self):
        """获取当前训练状态"""
        if not self.enable_monitoring:
            return "Monitoring disabled"

        self.monitor.status(self.experiment_name)

    def analyze_training_stability(self):
        """分析训练稳定性"""
        if not self.enable_monitoring:
            return "Monitoring disabled"

        self.monitor.analyze_stability(self.experiment_name)

def create_monitored_trainer(config_path, enable_monitoring=True):
    """从配置文件创建监控训练器"""
    if not os.path.exists(config_path):
        print("Config file not found: {}".format(config_path))
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    experiment_name = config.get('experiment_name', 'unknown_experiment')
    
    return MonitoredTrainer(experiment_name, enable_monitoring)