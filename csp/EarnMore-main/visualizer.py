# -*- coding: utf-8 -*-
# 可视化展示功能模块
# 本文件位置：项目根目录/visualizer.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime
import seaborn as sns
from typing import List, Dict, Optional

# 设置matplotlib使用英文字体
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False

class TrainingVisualizer:
    """训练过程可视化功能"""

    def __init__(self, experiment_path: str):
        """
        初始化可视化器

        Args:
            experiment_path: 实验目录路径
        """
        self.experiment_path = experiment_path
        self.logs_dir = os.path.join(experiment_path, "logs")
        self.plots_dir = os.path.join(experiment_path, "plots")

        # 确保plots目录存在
        os.makedirs(self.plots_dir, exist_ok=True)

        # 加载数据
        self.training_data = self._load_training_data()
        self.value_function_data = self._load_value_function_data()
        self.portfolio_data = self._load_portfolio_data()

    def _load_training_data(self) -> Optional[pd.DataFrame]:
        """加载训练指标数据"""
        csv_path = os.path.join(self.logs_dir, "training_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 区分训练和验证记录
            df['record_type'] = df.apply(
                lambda row: 'training' if pd.notna(row.get('train_obj_critics')) else 'validation',
                axis=1
            )
            return df
        return None

    def _load_value_function_data(self) -> Optional[pd.DataFrame]:
        """加载价值函数指标数据"""
        csv_path = os.path.join(self.logs_dir, "value_function_metrics.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return None

    def _load_portfolio_data(self) -> Optional[pd.DataFrame]:
        """加载每日portfolio数据"""
        csv_path = os.path.join(self.logs_dir, "portfolio_daily_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # 确保日期列是datetime类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        return None

    def plot_loss_curves(self):
        """绘制损失曲线图"""
        if self.training_data is None:
            print("训练数据不存在，无法绘制损失曲线")
            return

        # 筛选训练记录
        train_data = self.training_data[self.training_data['record_type'] == 'training'].copy()

        if train_data.empty:
            print("无训练记录数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Loss Curves', fontsize=14, fontweight='bold')

        # Actor Loss
        if 'train_obj_actors' in train_data.columns:
            axes[0, 0].plot(train_data['episode'], train_data['train_obj_actors'],
                           color='blue', linewidth=2, label='Actor Loss')
            axes[0, 0].set_title('Actor Loss')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

        # Critic Loss
        if 'train_obj_critics' in train_data.columns:
            axes[0, 1].plot(train_data['episode'], train_data['train_obj_critics'],
                           color='red', linewidth=2, label='Critic Loss')
            axes[0, 1].set_title('Critic Loss')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # Alpha Parameter (Temperature)
        if 'train_alphas' in train_data.columns:
            axes[1, 0].plot(train_data['episode'], train_data['train_alphas'],
                           color='green', linewidth=2, label='Alpha (Temperature)')
            axes[1, 0].set_title('Alpha Parameter')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Alpha Value')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Gradient Norms
        if 'train_gradient_norms' in train_data.columns:
            axes[1, 1].plot(train_data['episode'], train_data['train_gradient_norms'],
                           color='purple', linewidth=2, label='Gradient Norm')
            axes[1, 1].set_title('Gradient Norms')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Norm Value')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'loss_curves.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("损失曲线图已保存到：{}".format(os.path.join(self.plots_dir, 'loss_curves.png')))

    def plot_reward_curves(self):
        """绘制奖励曲线图"""
        if self.training_data is None:
            print("训练数据不存在，无法绘制奖励曲线")
            return

        train_data = self.training_data[self.training_data['record_type'] == 'training'].copy()
        val_data = self.training_data[self.training_data['record_type'] == 'validation'].copy()

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Training Reward
        if 'reward' in train_data.columns and not train_data['reward'].isna().all():
            ax.plot(train_data['episode'], train_data['reward'],
                   color='blue', linewidth=2, label='Training Reward', marker='o')

        # Validation ASR
        if 'validation_asr' in val_data.columns and not val_data['validation_asr'].isna().all():
            ax2 = ax.twinx()
            ax2.plot(val_data['episode'], val_data['validation_asr'],
                    color='red', linewidth=2, label='Validation ASR', marker='s')
            ax2.set_ylabel('Validation ASR', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

        ax.set_title('Training Reward and Validation Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        if 'validation_asr' in val_data.columns:
            ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'reward_curves.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("奖励曲线图已保存到：{}".format(os.path.join(self.plots_dir, 'reward_curves.png')))

    def plot_validation_curves(self):
        """绘制验证性能曲线"""
        if self.training_data is None:
            print("训练数据不存在，无法绘制验证曲线")
            return

        val_data = self.training_data[self.training_data['record_type'] == 'validation'].copy()

        if val_data.empty:
            print("无验证记录数据")
            return

        # 查找所有验证指标列
        val_columns = [col for col in val_data.columns if col.startswith('val_')]

        if not val_columns:
            print("无验证指标数据")
            return

        # 分组绘制不同类型的指标
        arr_cols = [col for col in val_columns if 'ARR' in col]
        sr_cols = [col for col in val_columns if 'SR' in col]
        cr_cols = [col for col in val_columns if 'CR' in col]
        mdd_cols = [col for col in val_columns if 'MDD' in col]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Validation Performance Curves', fontsize=14, fontweight='bold')

        colors = ['blue', 'red', 'green', 'orange']

        # ARR Performance
        if arr_cols:
            for i, col in enumerate(arr_cols):
                axes[0, 0].plot(val_data['episode'], val_data[col],
                               color=colors[i % len(colors)], linewidth=2,
                               label=col.replace('val_', ''), marker='o')
            axes[0, 0].set_title('Annual Return Rate (ARR%)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('ARR (%)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

        # Sharpe Ratio
        if sr_cols:
            for i, col in enumerate(sr_cols):
                axes[0, 1].plot(val_data['episode'], val_data[col],
                               color=colors[i % len(colors)], linewidth=2,
                               label=col.replace('val_', ''), marker='s')
            axes[0, 1].set_title('Sharpe Ratio (SR)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # Calmar Ratio
        if cr_cols:
            for i, col in enumerate(cr_cols):
                axes[1, 0].plot(val_data['episode'], val_data[col],
                               color=colors[i % len(colors)], linewidth=2,
                               label=col.replace('val_', ''), marker='^')
            axes[1, 0].set_title('Calmar Ratio (CR)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Calmar Ratio')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Maximum Drawdown
        if mdd_cols:
            for i, col in enumerate(mdd_cols):
                axes[1, 1].plot(val_data['episode'], val_data[col],
                               color=colors[i % len(colors)], linewidth=2,
                               label=col.replace('val_', ''), marker='v')
            axes[1, 1].set_title('Maximum Drawdown (MDD%)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('MDD (%)')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'validation_curves.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("验证性能曲线图已保存到：{}".format(os.path.join(self.plots_dir, 'validation_curves.png')))

    def plot_learning_curves(self):
        """绘制学习曲线"""
        if self.training_data is None:
            print("训练数据不存在，无法绘制学习曲线")
            return

        train_data = self.training_data[self.training_data['record_type'] == 'training'].copy()

        if train_data.empty:
            print("无训练记录数据")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Learning Curves', fontsize=14, fontweight='bold')

        # Learning Rates
        lr_cols = ['train_act_lr', 'train_cri_lr', 'train_alpha_lr']
        lr_labels = ['Actor LR', 'Critic LR', 'Alpha LR']
        lr_colors = ['blue', 'red', 'green']

        for i, (col, label, color) in enumerate(zip(lr_cols, lr_labels, lr_colors)):
            if col in train_data.columns:
                axes[0, 0].plot(train_data['episode'], train_data[col],
                               color=color, linewidth=2, label=label)
        axes[0, 0].set_title('Learning Rates')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Learning Rate')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Exploration Rate (Alpha)
        if 'exploration_rate' in train_data.columns:
            axes[0, 1].plot(train_data['episode'], train_data['exploration_rate'],
                           color='purple', linewidth=2, label='Exploration Rate')
            axes[0, 1].set_title('Exploration Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Alpha Value')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # Gradient Norm
        if 'gradient_norm' in train_data.columns:
            axes[1, 0].plot(train_data['episode'], train_data['gradient_norm'],
                           color='orange', linewidth=2, label='Gradient Norm')
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Norm Value')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Special Algorithm Losses (Beta & Representation)
        if 'beta_loss' in train_data.columns or 'representation_loss' in train_data.columns:
            if 'beta_loss' in train_data.columns:
                axes[1, 1].plot(train_data['episode'], train_data['beta_loss'],
                               color='cyan', linewidth=2, label='Beta Loss')
            if 'representation_loss' in train_data.columns:
                axes[1, 1].plot(train_data['episode'], train_data['representation_loss'],
                               color='magenta', linewidth=2, label='Representation Loss')
            axes[1, 1].set_title('Special Algorithm Losses')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss Value')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'learning_curves.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("学习曲线图已保存到：{}".format(os.path.join(self.plots_dir, 'learning_curves.png')))

    def plot_value_function_curves(self):
        """绘制价值函数曲线"""
        if self.value_function_data is None:
            print("价值函数数据不存在，无法绘制价值函数曲线")
            return

        data = self.value_function_data

        if data.empty:
            print("价值函数数据为空")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Value Function Analysis', fontsize=14, fontweight='bold')

        # Q-Values Distribution
        if 'q_values_mean' in data.columns and 'q_values_std' in data.columns:
            mean_vals = data['q_values_mean']
            std_vals = data['q_values_std']
            episodes = data['episode']

            axes[0, 0].plot(episodes, mean_vals, color='blue', linewidth=2, label='Q-Values Mean')
            axes[0, 0].fill_between(episodes, mean_vals - std_vals, mean_vals + std_vals,
                                   alpha=0.3, color='blue', label='±1 Std')
            axes[0, 0].set_title('Q-Values Distribution')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Q-Value')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

        # TD Error Analysis
        if 'td_error_mean' in data.columns and 'td_error_std' in data.columns:
            td_mean = data['td_error_mean']
            td_std = data['td_error_std']
            episodes = data['episode']

            axes[0, 1].plot(episodes, td_mean, color='red', linewidth=2, label='TD Error Mean')
            axes[0, 1].plot(episodes, td_std, color='orange', linewidth=2, label='TD Error Std')
            axes[0, 1].set_title('TD Error Analysis')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('TD Error')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # Policy Entropy
        if 'policy_entropy' in data.columns and 'policy_entropy_std' in data.columns:
            entropy_vals = data['policy_entropy']
            entropy_std = data['policy_entropy_std']
            episodes = data['episode']

            axes[1, 0].plot(episodes, entropy_vals, color='green', linewidth=2, label='Policy Entropy')
            axes[1, 0].fill_between(episodes, entropy_vals - entropy_std, entropy_vals + entropy_std,
                                   alpha=0.3, color='green', label='±1 Std')
            axes[1, 0].set_title('Policy Entropy')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Target Q Difference and Network Weights
        if 'target_q_diff' in data.columns:
            axes[1, 1].plot(data['episode'], data['target_q_diff'],
                           color='purple', linewidth=2, label='Target Q Diff')

        if 'q_network_weights_norm' in data.columns:
            ax2 = axes[1, 1].twinx()
            ax2.plot(data['episode'], data['q_network_weights_norm'],
                    color='brown', linewidth=2, label='Q Network Weights Norm')
            ax2.set_ylabel('Weights Norm', color='brown')
            ax2.tick_params(axis='y', labelcolor='brown')
            ax2.legend(loc='upper right')

        axes[1, 1].set_title('Target Q Difference & Network Weights')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Target Q Diff', color='purple')
        axes[1, 1].tick_params(axis='y', labelcolor='purple')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'value_function_curves.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("价值函数曲线图已保存到：{}".format(os.path.join(self.plots_dir, 'value_function_curves.png')))

    def plot_portfolio_curves(self):
        """绘制每日portfolio数据曲线"""
        if self.portfolio_data is None:
            print("每日portfolio数据不存在，无法绘制portfolio曲线")
            return

        data = self.portfolio_data

        if data.empty:
            print("每日portfolio数据为空")
            return

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Daily Portfolio Analysis', fontsize=16, fontweight='bold')

        # 获取环境数量
        envs = data['env'].unique()
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        # 1. 每日收益率曲线
        for i, env in enumerate(envs):
            env_data = data[data['env'] == env].sort_values('step')
            if not env_data.empty and 'date' in env_data.columns:
                axes[0, 0].plot(env_data['date'], env_data['portfolio_return'],
                               color=colors[i % len(colors)], linewidth=2,
                               label=f'Environment {env}', alpha=0.8)
        axes[0, 0].set_title('Daily Portfolio Returns')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Return Rate')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 资产价值曲线
        for i, env in enumerate(envs):
            env_data = data[data['env'] == env].sort_values('step')
            if not env_data.empty and 'date' in env_data.columns:
                axes[0, 1].plot(env_data['date'], env_data['portfolio_value'],
                               color=colors[i % len(colors)], linewidth=2,
                               label=f'Environment {env}', alpha=0.8)
        axes[0, 1].set_title('Portfolio Value Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Portfolio Value')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 现金比例变化
        for i, env in enumerate(envs):
            env_data = data[data['env'] == env].sort_values('step')
            if not env_data.empty and 'cash_ratio' in env_data.columns and 'date' in env_data.columns:
                axes[1, 0].plot(env_data['date'], env_data['cash_ratio'],
                               color=colors[i % len(colors)], linewidth=2,
                               label=f'Environment {env}', alpha=0.8)
        axes[1, 0].set_title('Cash Ratio Over Time')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Cash Ratio')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. 股票权重分布热力图（选择一个环境的最新数据）
        if envs.size > 0:
            latest_env_data = data[data['env'] == envs[0]].sort_values('step').tail(20)

            # 提取股票权重列
            weight_cols = [col for col in latest_env_data.columns if col.startswith('stock_') and col.endswith('_weight')]

            if weight_cols:
                weight_matrix = latest_env_data[weight_cols].values.T

                im = axes[1, 1].imshow(weight_matrix, cmap='RdYlBu', aspect='auto', interpolation='nearest')
                axes[1, 1].set_title(f'Stock Weights Heatmap (Env {envs[0]}, Last 20 Steps)')
                axes[1, 1].set_xlabel('Time Steps')
                axes[1, 1].set_ylabel('Stock Index')

                # 添加颜色条
                cbar = plt.colorbar(im, ax=axes[1, 1])
                cbar.set_label('Weight')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Stock Weights Data',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Stock Weights Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'portfolio_curves.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("Portfolio曲线图已保存到：{}".format(os.path.join(self.plots_dir, 'portfolio_curves.png')))

    def generate_all_plots(self):
        """生成所有训练可视化图表"""
        print("开始生成训练可视化图表...")

        self.plot_loss_curves()
        self.plot_reward_curves()
        self.plot_validation_curves()
        self.plot_learning_curves()
        self.plot_value_function_curves()
        self.plot_portfolio_curves()

        print("所有训练可视化图表生成完成！")
        print("图表保存目录：{}".format(self.plots_dir))


class ExperimentComparator:
    """实验对比可视化功能"""

    def __init__(self, experiments_dir: str):
        """
        初始化实验对比器

        Args:
            experiments_dir: 实验根目录路径
        """
        self.experiments_dir = experiments_dir
        self.experiments_data = self._load_all_experiments()

    def _load_all_experiments(self) -> Dict[str, Dict]:
        """加载所有实验数据"""
        experiments = {}

        if not os.path.exists(self.experiments_dir):
            print("实验目录不存在：{}".format(self.experiments_dir))
            return experiments

        for exp_name in os.listdir(self.experiments_dir):
            exp_path = os.path.join(self.experiments_dir, exp_name)
            if os.path.isdir(exp_path):
                exp_data = self._load_single_experiment(exp_path)
                if exp_data:
                    experiments[exp_name] = exp_data

        print("已加载{}个实验数据".format(len(experiments)))
        return experiments

    def _load_single_experiment(self, exp_path: str) -> Optional[Dict]:
        """加载单个实验数据"""
        logs_dir = os.path.join(exp_path, "logs")

        data = {}

        # 加载训练指标
        training_csv = os.path.join(logs_dir, "training_metrics.csv")
        if os.path.exists(training_csv):
            df = pd.read_csv(training_csv)
            df['record_type'] = df.apply(
                lambda row: 'training' if pd.notna(row.get('train_obj_critics')) else 'validation',
                axis=1
            )
            data['training'] = df

        # 加载价值函数数据
        value_csv = os.path.join(logs_dir, "value_function_metrics.csv")
        if os.path.exists(value_csv):
            data['value_function'] = pd.read_csv(value_csv)

        # 加载配置信息
        config_path = os.path.join(exp_path, "config.py")
        if os.path.exists(config_path):
            data['config_path'] = config_path

        return data if data else None

    def plot_experiments_comparison(self, metric: str, save_dir: str = None):
        """对比多个实验的指定指标"""
        if not self.experiments_data:
            print("无实验数据可对比")
            return

        if save_dir is None:
            save_dir = os.path.join(self.experiments_dir, "comparison_plots")
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.experiments_data)))

        for i, (exp_name, exp_data) in enumerate(self.experiments_data.items()):
            if 'training' in exp_data:
                df = exp_data['training']

                # 根据指标类型选择数据
                if metric.startswith('val_'):
                    plot_data = df[df['record_type'] == 'validation']
                else:
                    plot_data = df[df['record_type'] == 'training']

                if metric in plot_data.columns and not plot_data[metric].isna().all():
                    ax.plot(plot_data['episode'], plot_data[metric],
                           color=colors[i], linewidth=2, label=exp_name, marker='o')

        ax.set_title('Experiments Comparison: {}'.format(metric), fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comparison_{}.png'.format(metric)),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("对比图表已保存：{}".format(os.path.join(save_dir, 'comparison_{}.png'.format(metric))))

    def generate_performance_ranking(self, save_dir: str = None):
        """生成性能排行榜"""
        if not self.experiments_data:
            print("无实验数据可排行")
            return

        if save_dir is None:
            save_dir = os.path.join(self.experiments_dir, "comparison_plots")
        os.makedirs(save_dir, exist_ok=True)

        # 收集最终性能数据
        performance_data = []

        for exp_name, exp_data in self.experiments_data.items():
            if 'training' in exp_data:
                df = exp_data['training']
                val_data = df[df['record_type'] == 'validation']

                if not val_data.empty:
                    # 获取最后一次验证的性能
                    last_val = val_data.iloc[-1]

                    perf_dict = {'experiment': exp_name}

                    # 收集所有验证指标
                    for col in val_data.columns:
                        if col.startswith('val_') and pd.notna(last_val.get(col)):
                            perf_dict[col] = last_val[col]

                    performance_data.append(perf_dict)

        if not performance_data:
            print("无性能数据可排行")
            return

        perf_df = pd.DataFrame(performance_data)

        # 选择主要指标进行排行
        main_metrics = ['val_ARR_env0', 'val_SR_env0', 'val_CR_env0']

        fig, axes = plt.subplots(1, len(main_metrics), figsize=(15, 6))

        for i, metric in enumerate(main_metrics):
            if metric in perf_df.columns:
                # 按指标值排序
                sorted_df = perf_df.sort_values(metric, ascending=False)

                axes[i].barh(range(len(sorted_df)), sorted_df[metric])
                axes[i].set_yticks(range(len(sorted_df)))
                axes[i].set_yticklabels([name[:20] + '...' if len(name) > 20 else name
                                       for name in sorted_df['experiment']])
                axes[i].set_title(metric.replace('val_', '').replace('_env0', ''))
                axes[i].grid(True, alpha=0.3)

                # 添加数值标签
                for j, v in enumerate(sorted_df[metric]):
                    axes[i].text(v, j, ' {:.3f}'.format(v), va='center')

        plt.suptitle('Experiments Performance Ranking', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_ranking.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("性能排行榜已保存：{}".format(os.path.join(save_dir, 'performance_ranking.png')))

        # 保存排行榜数据
        ranking_csv = os.path.join(save_dir, 'performance_ranking.csv')
        perf_df.to_csv(ranking_csv, index=False)
        print("排行榜数据已保存：{}".format(ranking_csv))

    def generate_hyperparameter_analysis(self, save_dir: str = None):
        """生成超参数分析图表"""
        if not self.experiments_data:
            print("无实验数据可分析")
            return

        if save_dir is None:
            save_dir = os.path.join(self.experiments_dir, "comparison_plots")
        os.makedirs(save_dir, exist_ok=True)

        # 解析实验名称中的超参数信息
        hyperparams_data = []

        for exp_name, exp_data in self.experiments_data.items():
            if 'training' in exp_data:
                df = exp_data['training']
                val_data = df[df['record_type'] == 'validation']

                if not val_data.empty:
                    last_val = val_data.iloc[-1]

                    # 从实验名称解析超参数
                    params = self._parse_hyperparams_from_name(exp_name)
                    params['experiment'] = exp_name

                    # 添加性能指标
                    if 'val_ARR_env0' in last_val:
                        params['final_arr'] = last_val['val_ARR_env0']
                    if 'val_SR_env0' in last_val:
                        params['final_sr'] = last_val['val_SR_env0']

                    hyperparams_data.append(params)

        if not hyperparams_data:
            print("无法解析超参数信息")
            return

        hp_df = pd.DataFrame(hyperparams_data)

        # 绘制超参数与性能的关系
        if 'lr' in hp_df.columns and 'final_arr' in hp_df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            scatter = ax.scatter(hp_df['lr'], hp_df['final_arr'],
                               c=hp_df.get('batch_size', 1), s=100, alpha=0.7, cmap='viridis')

            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Final ARR (%)')
            ax.set_title('Hyperparameter Analysis: Learning Rate vs Performance')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

            # 添加颜色条
            if 'batch_size' in hp_df.columns:
                cbar = plt.colorbar(scatter)
                cbar.set_label('Batch Size')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'hyperparameter_analysis.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("超参数分析图表已保存：{}".format(os.path.join(save_dir, 'hyperparameter_analysis.png')))

    def _parse_hyperparams_from_name(self, exp_name: str) -> Dict:
        """从实验名称解析超参数"""
        params = {}

        # 解析学习率
        if 'lr_' in exp_name:
            lr_part = exp_name.split('lr_')[1].split('_')[0]
            try:
                params['lr'] = float(lr_part)
            except ValueError:
                pass

        # 解析批次大小
        if 'batch_size_' in exp_name:
            batch_part = exp_name.split('batch_size_')[1].split('_')[0]
            try:
                params['batch_size'] = int(batch_part)
            except ValueError:
                pass

        # 解析随机种子
        if 'seed_' in exp_name:
            seed_part = exp_name.split('seed_')[1].split('_')[0]
            try:
                params['seed'] = int(seed_part)
            except ValueError:
                pass

        return params


def batch_plot_all_experiments(experiments_dir: str):
    """批量为所有实验生成可视化图表"""
    if not os.path.exists(experiments_dir):
        print("实验目录不存在：{}".format(experiments_dir))
        return

    experiment_count = 0
    success_count = 0

    print("开始批量生成所有实验的可视化图表...")
    print("扫描实验目录：{}".format(experiments_dir))

    for item in os.listdir(experiments_dir):
        item_path = os.path.join(experiments_dir, item)

        # 跳过非目录文件
        if not os.path.isdir(item_path):
            continue

        # 跳过特殊目录
        if item in ['configs', 'templates', 'comparison_plots']:
            continue

        # 检查是否是有效的实验目录（包含logs目录）
        logs_path = os.path.join(item_path, 'logs')
        if not os.path.exists(logs_path):
            print("跳过无效实验目录：{}（缺少logs目录）".format(item))
            continue

        experiment_count += 1
        print("\n[{}/X] 处理实验：{}".format(experiment_count, item))

        try:
            visualizer = TrainingVisualizer(item_path)
            visualizer.generate_all_plots()
            success_count += 1
            print("实验 {} 可视化完成[成功]".format(item))
        except Exception as e:
            print("实验 {} 可视化失败[错误]: {}".format(item, str(e)))

    print("\n=== 批量可视化完成 ===")
    print("总实验数：{}".format(experiment_count))
    print("成功生成：{}".format(success_count))
    print("失败数量：{}".format(experiment_count - success_count))

    if success_count > 0:
        print("所有成功实验的图表已保存到各自的plots目录中")


def main():
    """主函数：命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description='可视化展示功能模块')
    parser.add_argument('action', choices=['plot-training', 'plot-all-experiments', 'compare-experiments', 'plot-portfolio'],
                       help='操作类型')
    parser.add_argument('--experiment', type=str, help='实验目录路径')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='实验根目录路径')
    parser.add_argument('--metric', type=str, help='对比指标名称')

    args = parser.parse_args()

    if args.action == 'plot-training':
        if not args.experiment:
            print("错误：需要指定实验目录路径")
            return

        visualizer = TrainingVisualizer(args.experiment)
        visualizer.generate_all_plots()

    elif args.action == 'plot-all-experiments':
        batch_plot_all_experiments(args.experiments_dir)

    elif args.action == 'plot-portfolio':
        if not args.experiment:
            print("错误：需要指定实验目录路径")
            return

        visualizer = TrainingVisualizer(args.experiment)
        visualizer.plot_portfolio_curves()

    elif args.action == 'compare-experiments':
        comparator = ExperimentComparator(args.experiments_dir)

        if args.metric:
            comparator.plot_experiments_comparison(args.metric)
        else:
            # 生成完整对比分析
            comparator.generate_performance_ranking()
            comparator.generate_hyperparameter_analysis()

            # 对比主要指标
            main_metrics = ['val_ARR_env0', 'val_SR_env0', 'train_obj_critics', 'train_obj_actors']
            for metric in main_metrics:
                comparator.plot_experiments_comparison(metric)


if __name__ == '__main__':
    main()