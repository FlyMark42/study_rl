# -*- coding: utf-8 -*-
# 项目特别的可视化展示功能模块
# 本文件位置：项目根目录/visualizer.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime
import seaborn as sns
from typing import List, Dict, Optional
from visualizer import TrainingVisualizer
from visualizer import ExperimentComparator
# 设置matplotlib使用英文字体
# matplotlib.rcParams['font.family'] = 'DejaVu Sans'
# matplotlib.rcParams['font.size'] = 10
# matplotlib.rcParams['axes.unicode_minus'] = False

class ExperimentComparator_Alpha(ExperimentComparator):
    """实验对比可视化功能"""
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

        # 选择主要指标进行排行，# 对比主要指标,不同的项目自由选择，把csv文件里面的指标添加进去就可以了
        main_metrics = ['val_ARR_env0', 'val_SR_env0']

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

class TrainingVisualizer_Alpha(TrainingVisualizer):
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
        # Validation ARR 补充代码
        if 'validation_arr' in val_data.columns and not val_data['validation_arr'].isna().all():
            if not ('validation_asr' in val_data.columns and not val_data['validation_asr'].isna().all()):
                ax2 = ax.twinx()
                ax2.set_ylabel('Validation ARR', color='green')
                ax2.tick_params(axis='y', labelcolor='green')
            ax2.plot(val_data['episode'], val_data['validation_arr'],
                     color='green', linewidth=2, label='Validation ARR', marker='^')

        ax.set_title('Training Reward and Validation Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        if ('validation_asr' in val_data.columns and not val_data['validation_asr'].isna().all()) or \
           ('validation_arr' in val_data.columns and not val_data['validation_arr'].isna().all()):
            ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'reward_curves.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("奖励曲线图已保存到：{}".format(os.path.join(self.plots_dir, 'reward_curves.png')))

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
        if 'train_exploration_rate' in train_data.columns:
            axes[0, 1].plot(train_data['episode'], train_data['train_exploration_rate'],
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