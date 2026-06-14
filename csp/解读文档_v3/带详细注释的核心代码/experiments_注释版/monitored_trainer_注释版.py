# -*- coding: utf-8 -*-
"""
==========================================================================================
训练监控桥接器 MonitoredTrainer —— 逐行详细中文注释版（学习用）
对应原文件：EarnMore-main_v20260610/experiments/monitored_trainer.py
==========================================================================================

【这份文件做什么】
它是 tools/train.py（训练主循环）与 experiments/monitor.py（监控/早停系统）之间的"桥"。
当你用 `python tools/train.py --monitor` 启动训练时，train.py 会创建一个 MonitoredTrainer，
并在主循环里调用它的三个接口，把训练指标喂给监控器、并询问是否该早停。

【它在 train.py 里怎么被用】（见 train_新版_注释版.py）
  monitor = MonitoredTrainer(exp_name, enable_monitoring=True)   # --monitor 时创建
  ...
  monitor.log_episode_metrics(enhanced_train_metrics)            # 训练每 episode：记录指标
  monitor.log_value_function_metrics(episode, agent)             # 记录 Q值/TD误差/熵 等
  if monitor.should_stop_early(...): break                       # 验证后：判断是否早停

【内部实际用的监控器】
  TrainingMonitor_Alpha（alpha_monitor.py）—— 是 TrainingMonitor（monitor.py）的子类，
  把通用监控特化到本项目的 ASR/ARR 指标上。
==========================================================================================
"""
import os
import sys
import json
from datetime import datetime

# 把项目根目录加入 import 路径，确保能找到 experiments.monitor 等模块
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..')
sys.path.insert(0, root_dir)

from experiments.monitor import TrainingMonitor   # 基类监控器（这里 import 但实际用下面的 Alpha 子类）


class MonitoredTrainer:
    """带监控功能的训练器包装类（train.py 与 monitor 之间的桥）。"""

    def __init__(self, experiment_name, enable_monitoring=True):
        # 用 ASR/ARR 特化版监控器（TrainingMonitor 的子类）
        from experiments.alpha_monitor import TrainingMonitor_Alpha
        self.experiment_name = experiment_name        # 实验名（= exp_xxx，用于定位 logs 目录）
        self.enable_monitoring = enable_monitoring     # 总开关
        self.monitor = TrainingMonitor_Alpha() if enable_monitoring else None
        self.episode_count = 0                         # 已记录的 episode 数

        print("MonitoredTrainer initialized for experiment: {}".format(experiment_name))
        print("Monitoring enabled: {}".format(enable_monitoring))

    def log_episode_metrics(self, metrics_dict):
        """记录单个 episode 的训练指标（写进 exp 的 logs/training_metrics.csv）。"""
        if not self.enable_monitoring:
            return

        # 优先用传入的 episode 编号；没有就用内部计数器 +1
        episode_num = metrics_dict.get('episode', self.episode_count + 1)
        self.episode_count = max(self.episode_count, episode_num)

        # 委托给底层监控器把这条指标落盘（CSV）
        self.monitor.record_metrics(self.experiment_name, episode_num, metrics_dict)

        # 每 20 个 episode 打印一次进度提示
        if episode_num % 20 == 0:
            print("Monitor: Episode {} metrics logged".format(episode_num))

    def log_value_function_metrics(self, episode, agent):
        """从 agent 上"尽力"抓取价值函数相关指标（Q值/TD误差/策略熵/权重范数），记录下来。
        用大量 hasattr 做"鸭子类型"探测——agent 有就抓、没有就跳过，保证对不同算法都不报错。"""
        if not self.enable_monitoring:
            return

        try:
            value_metrics = {}

            # ① Q 值统计（若 agent 提供 get_q_values）
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

            # ② TD 误差统计（若 agent 暂存了 last_td_error）—— 对应论文公式(5)的贝尔曼残差
            if hasattr(agent, 'last_td_error') and agent.last_td_error is not None:
                import torch
                td_error = agent.last_td_error
                if isinstance(td_error, torch.Tensor):
                    td_error = td_error.detach().cpu().numpy()
                value_metrics['td_error_mean'] = float(td_error.mean())
                value_metrics['td_error_std'] = float(td_error.std())
                value_metrics['td_error_abs_mean'] = float(abs(td_error).mean())

            # ③ 策略熵（若 agent 暂存了 last_policy_entropy）—— SAC 的熵，越大越爱探索
            if hasattr(agent, 'last_policy_entropy') and agent.last_policy_entropy is not None:
                entropy = agent.last_policy_entropy
                if hasattr(entropy, 'item'):
                    entropy = entropy.item()
                value_metrics['policy_entropy'] = float(entropy)

            # ④ Actor 网络权重的 L2 范数（监控权重是否爆炸/消失）
            if hasattr(agent, 'act') and hasattr(agent.act, 'parameters'):
                import torch
                total_norm = 0
                for param in agent.act.parameters():
                    if param.grad is not None:
                        param_norm = param.data.norm(2)
                        total_norm += param_norm.item() ** 2
                value_metrics['q_network_weights_norm'] = float(total_norm ** (1. / 2))

            # 有抓到任何指标就落盘（写 value_function_metrics.csv）
            if value_metrics:
                self.monitor.record_value_function_metrics(self.experiment_name, episode, value_metrics)

        except Exception as e:
            # 抓指标失败不能让训练崩溃，所以吞掉异常、只告警
            print("Warning: Could not collect value function metrics: {}".format(e))

    def should_stop_early(self, min_episodes=50, patience=5, metrics='ASR'):
        """判断是否该早停。委托给底层监控器的 early_stop_check。
        当前直接调用（调试用，快速出结果）；注释掉的下半段是"生产环境"逻辑：
        要求至少跑满 min_episodes、且每 10 个 episode 才检查一次。"""
        # —— 调试中：每次都检查，快速出结果 ——
        return self.monitor.early_stop_check(self.experiment_name, patience, metrics)

        # —— 生产环境逻辑（当前被注释）——
        # if not self.enable_monitoring or self.episode_count < min_episodes:
        #     return False
        # if self.episode_count % 10 == 0:
        #     try:
        #         return self.monitor.early_stop_check(self.experiment_name, patience, metrics)
        #     except Exception as e:
        #         print("Warning: Early stop check failed: {}".format(e))
        #         return False
        # return False


def create_monitored_trainer(config_path, enable_monitoring=True):
    """从一个 json 配置文件创建 MonitoredTrainer（取里面的 experiment_name）。一个便捷工厂函数。"""
    if not os.path.exists(config_path):
        print("Config file not found: {}".format(config_path))
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    experiment_name = config.get('experiment_name', 'unknown_experiment')
    return MonitoredTrainer(experiment_name, enable_monitoring)


# ==========================================================================================
# 【一句话总结】
#   MonitoredTrainer = train.py 主循环 ←→ TrainingMonitor_Alpha 之间的薄薄一层适配器：
#     训练每 episode → log_episode_metrics / log_value_function_metrics（写 CSV）
#     验证之后        → should_stop_early（基于 ASR/ARR 的 patience 早停）
#   它本身不做分析，只是"转发 + 容错"；真正的记录/分析/早停逻辑在 monitor.py / alpha_monitor.py。
# ==========================================================================================
