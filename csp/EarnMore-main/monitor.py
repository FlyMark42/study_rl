# -*- coding: utf-8 -*-
"""
训练监控功能模块
该文件应该放在项目根目录下
"""
import os
import json
import pandas as pd
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime
from visualizer import TrainingVisualizer as AdvancedTrainingVisualizer, ExperimentComparator

# 设置英文字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

class TrainingMonitor:
    def __init__(self):
        pass
    
    def record_metrics(self, experiment_name, episode, metrics_dict):
        """记录训练指标到CSV文件"""
        exp_dir = os.path.join("experiments", experiment_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

        logs_dir = os.path.join(exp_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        csv_path = os.path.join(logs_dir, "training_metrics.csv")

        # 准备记录数据 - 扩展支持的指标类型
        record = {"episode": episode, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        # 基础训练指标
        standard_metrics = [
            'actor_loss', 'critic_loss', 'reward', 'exploration_rate',
            'learning_rate', 'gradient_norm', 'beta_loss', 'representation_loss',
            'validation_asr', 'validation_reward'
        ]

        # 添加标准指标
        for metric in standard_metrics:
            if metric in metrics_dict:
                record[metric] = metrics_dict[metric]

        # 添加其他自定义指标
        for key, value in metrics_dict.items():
            if key not in record and key not in ['episode', 'timestamp']:
                record[key] = value

        # 读取现有数据或创建新DataFrame
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            new_df = pd.DataFrame([record])

        new_df.to_csv(csv_path, index=False)
        print("Episode {}: Metrics recorded to {}".format(episode, csv_path))

    def record_value_function_metrics(self, experiment_name, episode, value_metrics):
        """记录价值函数相关指标"""
        exp_dir = os.path.join("experiments", experiment_name)
        logs_dir = os.path.join(exp_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        csv_path = os.path.join(logs_dir, "value_function_metrics.csv")

        # 价值函数指标记录
        record = {
            "episode": episode,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "q_values_mean": value_metrics.get('q_values_mean', 0.0),
            "q_values_std": value_metrics.get('q_values_std', 0.0),
            "q_values_min": value_metrics.get('q_values_min', 0.0),
            "q_values_max": value_metrics.get('q_values_max', 0.0),
            "td_error_mean": value_metrics.get('td_error_mean', 0.0),
            "td_error_std": value_metrics.get('td_error_std', 0.0),
            "td_error_abs_mean": value_metrics.get('td_error_abs_mean', 0.0),
            "policy_entropy": value_metrics.get('policy_entropy', 0.0),
            "policy_entropy_std": value_metrics.get('policy_entropy_std', 0.0),
            "target_q_diff": value_metrics.get('target_q_diff', 0.0),
            "value_estimation_bias": value_metrics.get('value_estimation_bias', 0.0),
            "q_network_weights_norm": value_metrics.get('q_network_weights_norm', 0.0)
        }

        # 读取现有数据或创建新DataFrame
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        else:
            new_df = pd.DataFrame([record])

        new_df.to_csv(csv_path, index=False)

        # 每20个episode显示价值函数监控信息
        if episode % 20 == 0:
            print("Episode {}: Value function metrics - Q_mean={:.4f}, TD_error_std={:.4f}, Entropy={:.4f}".format(
                episode, record['q_values_mean'], record['td_error_std'], record['policy_entropy']))

    def analyze_value_function_quality(self, experiment_name):
        """分析价值函数学习质量"""
        exp_dir = os.path.join("experiments", experiment_name)
        csv_path = os.path.join(exp_dir, "logs", "value_function_metrics.csv")

        if not os.path.exists(csv_path):
            print("No value function metrics found for experiment {}.".format(experiment_name))
            return

        df = pd.read_csv(csv_path)
        if len(df) < 10:
            print("Insufficient data for value function analysis (need at least 10 episodes).")
            return

        print("Value Function Quality Analysis:")

        # Q值分布分析
        recent_q_mean = df['q_values_mean'].tail(20).mean()
        recent_q_std = df['q_values_std'].tail(20).mean()
        q_stability = df['q_values_mean'].tail(20).std()

        print("Q-Value Distribution (last 20 episodes):")
        print("  Mean Q-value: {:.6f}".format(recent_q_mean))
        print("  Q-value diversity (std): {:.6f}".format(recent_q_std))
        print("  Q-value stability: {:.6f} ({})".format(
            q_stability,
            "Stable" if q_stability < 0.1 else "Moderate" if q_stability < 0.5 else "Unstable"
        ))

        # TD误差分析
        recent_td_mean = df['td_error_mean'].tail(20).mean()
        recent_td_std = df['td_error_std'].tail(20).mean()
        td_trend = df['td_error_abs_mean'].tail(20).iloc[-1] - df['td_error_abs_mean'].tail(20).iloc[0]

        print("TD Error Analysis:")
        print("  Mean TD error: {:.6f}".format(recent_td_mean))
        print("  TD error std: {:.6f}".format(recent_td_std))
        print("  TD error trend: {:.6f} ({})".format(
            td_trend,
            "Improving" if td_trend < -0.001 else "Stable" if abs(td_trend) < 0.001 else "Worsening"
        ))

        # 策略熵分析
        recent_entropy = df['policy_entropy'].tail(20).mean()
        entropy_trend = df['policy_entropy'].tail(20).iloc[-1] - df['policy_entropy'].tail(20).iloc[0]

        print("Policy Entropy Analysis:")
        print("  Current entropy: {:.6f}".format(recent_entropy))
        print("  Entropy trend: {:.6f} ({})".format(
            entropy_trend,
            "Exploring more" if entropy_trend > 0.001 else "Stable exploration" if abs(entropy_trend) < 0.001 else "Exploiting more"
        ))

        # 价值函数学习质量评估
        quality_score = 0
        if q_stability < 0.1:
            quality_score += 30
        elif q_stability < 0.5:
            quality_score += 20
        else:
            quality_score += 10

        if abs(td_trend) < 0.001 or td_trend < -0.001:
            quality_score += 30
        elif td_trend < 0.005:
            quality_score += 20
        else:
            quality_score += 10

        if 0.1 < recent_entropy < 2.0:
            quality_score += 25
        elif recent_entropy > 0.05:
            quality_score += 15
        else:
            quality_score += 5

        if recent_td_std < 0.5:
            quality_score += 15
        elif recent_td_std < 1.0:
            quality_score += 10
        else:
            quality_score += 5

        quality_level = "Excellent" if quality_score >= 80 else "Good" if quality_score >= 60 else "Fair" if quality_score >= 40 else "Poor"
        print("Overall Value Function Quality: {}/100 ({})".format(quality_score, quality_level))

        # 保存分析结果
        analysis_file = os.path.join(exp_dir, "logs", "value_function_analysis.json")
        analysis_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "q_value_stats": {
                "mean": float(recent_q_mean),
                "std": float(recent_q_std),
                "stability": float(q_stability)
            },
            "td_error_stats": {
                "mean": float(recent_td_mean),
                "std": float(recent_td_std),
                "trend": float(td_trend)
            },
            "policy_entropy_stats": {
                "mean": float(recent_entropy),
                "trend": float(entropy_trend)
            },
            "quality_score": quality_score,
            "quality_level": quality_level
        }

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)

        print("Value function analysis saved to: {}".format(analysis_file))
    
    def status(self, experiment_name):
        """查看当前训练状态和关键指标"""
        exp_dir = os.path.join("experiments", experiment_name)
        if not os.path.exists(exp_dir):
            print("Experiment {} not found.".format(experiment_name))
            return

        csv_path = os.path.join(exp_dir, "logs", "training_metrics.csv")
        if not os.path.exists(csv_path):
            print("No training metrics found for experiment {}.".format(experiment_name))
            return

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print("No data found in training metrics.")
            return

        # 获取最新的训练记录和验证记录
        train_df = df[df['train_obj_critics'].notna()]
        val_df = df[df['val_ARR_env0'].notna()]

        # 获取最新训练指标
        if len(train_df) > 0:
            latest_train = train_df.iloc[-1]
        else:
            latest_train = None

        # 获取最新验证指标
        if len(val_df) > 0:
            latest_val = val_df.iloc[-1]
        else:
            latest_val = None

        print("Experiment: {}".format(experiment_name))
        if latest_train is not None:
            print("Status: Running (Episode {})".format(int(latest_train['episode'])))
        else:
            print("Status: No training data available")

        if latest_train is not None:
            print("Latest training metrics:")

            # 显示训练损失指标
            if 'train_obj_critics' in latest_train.index and pd.notna(latest_train['train_obj_critics']):
                print("  train_obj_critics (critic loss): {:.6f}".format(latest_train['train_obj_critics']))

            if 'train_obj_actors' in latest_train.index and pd.notna(latest_train['train_obj_actors']):
                print("  train_obj_actors (actor loss): {:.6f}".format(latest_train['train_obj_actors']))

            if 'train_alphas' in latest_train.index and pd.notna(latest_train['train_alphas']):
                print("  train_alphas (alpha loss): {:.6f}".format(latest_train['train_alphas']))

            # 显示学习率
            learning_rates = ['train_act_lr', 'train_cri_lr', 'train_alpha_lr']
            for lr_col in learning_rates:
                if lr_col in latest_train.index and pd.notna(latest_train[lr_col]):
                    print("  {}: {:.2e}".format(lr_col, latest_train[lr_col]))

        if latest_val is not None:
            # 显示验证指标（选择主要的）
            print("Latest validation metrics:")
            validation_metrics = ['val_ARR_env0', 'val_SR_env0', 'val_CR_env0', 'val_MDD%_env0']
            for val_col in validation_metrics:
                if val_col in latest_val.index and pd.notna(latest_val[val_col]):
                    print("  {}: {:.6f}".format(val_col, latest_val[val_col]))

        # 分析最近的训练趋势（基于critic loss）
        if len(train_df) > 0 and 'train_obj_critics' in train_df.columns:
            recent_train_df = train_df.tail(10)
            critic_loss_data = recent_train_df['train_obj_critics'].dropna()

            if len(critic_loss_data) > 1:
                recent_variance = critic_loss_data.var()
                if pd.notna(recent_variance):
                    stability = "Good" if recent_variance < 1000 else "Moderate" if recent_variance < 10000 else "Poor"
                    print("Training stability: {} (critic loss variance: {:.2f})".format(stability, recent_variance))

        print("Total episodes completed: {}".format(len(df)))
        if latest_train is not None:
            print("Last training update: {}".format(latest_train['timestamp']))
        if latest_val is not None:
            print("Last validation update: {}".format(latest_val['timestamp']))
    
    def analyze_stability(self, experiment_name):
        """分析训练稳定性"""
        exp_dir = os.path.join("experiments", experiment_name)
        csv_path = os.path.join(exp_dir, "logs", "training_metrics.csv")

        if not os.path.exists(csv_path):
            print("No training metrics found for experiment {}.".format(experiment_name))
            return

        df = pd.read_csv(csv_path)

        # 只选择训练记录 (包含train_*字段有值的行)
        train_df = df[df['train_obj_critics'].notna()].copy()

        if len(train_df) < 3:
            print("Insufficient training data for stability analysis (need at least 3 training episodes).")
            return

        print("Training Stability Analysis for Real Data:")

        # 计算稳定性指标
        stability_data = []
        window_size = min(max(3, len(train_df) // 3), 10)  # 动态调整窗口大小，最小3，最大10

        for i in range(window_size, len(train_df) + 1):
            window_df = train_df.iloc[i-window_size:i]
            episode = window_df.iloc[-1]['episode']

            stability_record = {"episode": int(episode), "window_size": window_size}

            # 分析训练损失指标
            loss_metrics = ['train_obj_critics', 'train_obj_actors', 'train_alphas']
            for loss_col in loss_metrics:
                if loss_col in window_df.columns:
                    values = window_df[loss_col].dropna()
                    if len(values) > 1:
                        variance = values.var()
                        mean_val = values.mean()
                        std_val = values.std()
                        cv = std_val / (abs(mean_val) + 1e-8) if abs(mean_val) > 1e-8 else 0

                        # 计算趋势
                        x = np.arange(len(values))
                        if len(values) > 2:
                            slope = np.polyfit(x, values, 1)[0]
                            correlation = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0
                        else:
                            slope = 0
                            correlation = 0

                        # 稳定性评分（损失应该趋于稳定或下降）
                        stability_score = max(0, 100 - cv * 50 - abs(slope) * 10)

                        stability_record.update({
                            "{}_variance".format(loss_col): variance,
                            "{}_mean".format(loss_col): mean_val,
                            "{}_cv".format(loss_col): cv,
                            "{}_trend".format(loss_col): slope,
                            "{}_correlation".format(loss_col): correlation,
                            "{}_stability_score".format(loss_col): stability_score
                        })

            # 分析学习率稳定性
            lr_metrics = ['train_act_lr', 'train_cri_lr', 'train_alpha_lr']
            for lr_col in lr_metrics:
                if lr_col in window_df.columns:
                    values = window_df[lr_col].dropna()
                    if len(values) > 1:
                        cv = values.std() / (values.mean() + 1e-12)
                        stability_record["{}_cv".format(lr_col)] = cv

            # 分析验证指标稳定性（重点关注env0）
            val_metrics = ['val_ARR_env0', 'val_SR_env0', 'val_CR_env0', 'val_MDD%_env0']
            for val_col in val_metrics:
                if val_col in window_df.columns:
                    values = window_df[val_col].dropna()
                    if len(values) > 1:
                        variance = values.var()
                        mean_val = values.mean()
                        cv = values.std() / (abs(mean_val) + 1e-8) if abs(mean_val) > 1e-8 else 0

                        # 计算趋势（验证指标应该有改善趋势）
                        x = np.arange(len(values))
                        slope = np.polyfit(x, values, 1)[0] if len(values) > 2 else 0

                        stability_record.update({
                            "{}_variance".format(val_col): variance,
                            "{}_mean".format(val_col): mean_val,
                            "{}_cv".format(val_col): cv,
                            "{}_trend".format(val_col): slope
                        })

            stability_data.append(stability_record)

        # 保存稳定性分析结果
        stability_df = pd.DataFrame(stability_data)
        stability_path = os.path.join(exp_dir, "logs", "stability_analysis.csv")
        stability_df.to_csv(stability_path, index=False)

        if len(stability_df) > 0:
            latest = stability_df.iloc[-1]

            # 训练损失稳定性分析
            print("Training Loss Stability (last {} episodes):".format(window_size))

            loss_scores = []
            for loss_col in ['train_obj_critics', 'train_obj_actors', 'train_alphas']:
                cv_key = "{}_cv".format(loss_col)
                trend_key = "{}_trend".format(loss_col)
                score_key = "{}_stability_score".format(loss_col)

                if cv_key in latest and trend_key in latest:
                    cv = latest[cv_key]
                    trend = latest[trend_key]
                    score = latest.get(score_key, 0)

                    # 处理NaN值
                    if np.isnan(cv) or np.isnan(trend) or np.isnan(score):
                        print("  {}: Insufficient data for analysis".format(loss_col))
                        continue

                    print("  {}: CV={:.4f}, trend={:.2f}, stability={:.1f}/100".format(
                        loss_col, cv, trend, score))

                    loss_scores.append(score)

            if loss_scores:
                avg_loss_stability = np.mean(loss_scores)
                loss_level = "Excellent" if avg_loss_stability >= 80 else "Good" if avg_loss_stability >= 60 else "Fair" if avg_loss_stability >= 40 else "Poor"
                print("  Average Loss Stability: {:.1f}/100 ({})".format(avg_loss_stability, loss_level))
            else:
                avg_loss_stability = 0
                print("  Average Loss Stability: Insufficient data")

            # 学习率稳定性分析
            print("Learning Rate Stability:")
            lr_stable = True
            for lr_col in ['train_act_lr', 'train_cri_lr', 'train_alpha_lr']:
                cv_key = "{}_cv".format(lr_col)
                if cv_key in latest:
                    cv = latest[cv_key]
                    print("  {}: CV={:.6f}".format(lr_col, cv))
                    if cv > 0.1:  # 学习率变异系数过大
                        lr_stable = False

            print("  Overall LR Stability: {}".format("Stable" if lr_stable else "Unstable"))

            # 验证指标稳定性分析
            print("Validation Metrics Stability (env0):")
            val_performance = {}

            for val_col in ['val_ARR_env0', 'val_SR_env0', 'val_CR_env0']:
                cv_key = "{}_cv".format(val_col)
                trend_key = "{}_trend".format(val_col)

                if cv_key in latest and trend_key in latest:
                    cv = latest[cv_key]
                    trend = latest[trend_key]

                    # 对于这些指标，正趋势是好的
                    trend_quality = "Improving" if trend > 0.001 else "Stable" if abs(trend) < 0.001 else "Declining"
                    stability_quality = "Low variance" if cv < 0.1 else "Moderate variance" if cv < 0.3 else "High variance"

                    print("  {}: CV={:.4f} ({}), trend={:.4f} ({})".format(
                        val_col, cv, stability_quality, trend, trend_quality))

                    val_performance[val_col] = {"cv": cv, "trend": trend}

            # 训练建议
            print("Training Recommendations:")

            recommendations = []
            if avg_loss_stability >= 70:
                recommendations.append("+ Loss functions are converging well")
            elif avg_loss_stability >= 50:
                recommendations.append("! Loss stability is moderate - monitor closely")
            else:
                recommendations.append("- Loss functions are unstable - consider adjusting learning rates")

            if lr_stable:
                recommendations.append("+ Learning rates are stable")
            else:
                recommendations.append("! Learning rate instability detected")

            # 检查验证指标趋势
            improving_metrics = sum(1 for val_info in val_performance.values() if val_info["trend"] > 0.001)
            if improving_metrics >= 2:
                recommendations.append("+ Validation metrics show improvement")
            elif improving_metrics >= 1:
                recommendations.append("~ Some validation metrics are improving")
            else:
                recommendations.append("! Validation metrics need attention")

            for rec in recommendations:
                print("  {}".format(rec))

            # 总体评估
            if not np.isnan(avg_loss_stability):
                overall_score = (avg_loss_stability + (50 if lr_stable else 20) + improving_metrics * 15) / 2
                overall_level = "Excellent" if overall_score >= 80 else "Good" if overall_score >= 60 else "Fair" if overall_score >= 40 else "Needs Attention"
                print("Overall Training Health: {:.1f}/100 ({})".format(overall_score, overall_level))
            else:
                print("Overall Training Health: Insufficient data for evaluation")

        print("Stability analysis saved to: {}".format(stability_path))
    
    def early_stop_check(self, experiment_name):
        """检查是否建议早停"""
        exp_dir = os.path.join("experiments", experiment_name)
        csv_path = os.path.join(exp_dir, "logs", "training_metrics.csv")

        if not os.path.exists(csv_path):
            print("No training metrics found for experiment {}.".format(experiment_name))
            return False

        df = pd.read_csv(csv_path)
        if len(df) < 20:
            print("Insufficient data for early stopping analysis (need at least 20 episodes).")
            return False

        print("Enhanced Early Stopping Analysis:")

        # 多维度早停分析
        recent_episodes = min(30, len(df))
        recent_df = df.tail(recent_episodes)

        # 1. 验证ASR停滞分析
        asr_stagnant_episodes = 0
        asr_improvement_rate = 0
        asr_current_value = 0

        if 'validation_asr' in recent_df.columns:
            asr_values = recent_df['validation_asr'].dropna()
            if len(asr_values) >= 15:
                asr_current_value = asr_values.iloc[-1]

                # 检查最近15个episode的ASR改善
                recent_15_asr = asr_values.tail(15)
                asr_improvement = recent_15_asr.iloc[-1] - recent_15_asr.iloc[0]
                asr_improvement_rate = asr_improvement / len(recent_15_asr)

                if asr_improvement < 0.001:  # 改善小于0.1%
                    asr_stagnant_episodes = 15

                # 检查是否有明显下降趋势
                if len(asr_values) >= 20:
                    recent_20_trend = np.polyfit(range(len(asr_values.tail(20))), asr_values.tail(20), 1)[0]
                    if recent_20_trend < -0.0001:  # 明显下降趋势
                        asr_stagnant_episodes = max(asr_stagnant_episodes, 20)

        print("Validation ASR Analysis:")
        print("  Current ASR: {:.6f}".format(asr_current_value))
        print("  Improvement rate: {:.6f} per episode".format(asr_improvement_rate))
        print("  Stagnant episodes: {}".format(asr_stagnant_episodes))

        # 2. 损失收敛分析
        loss_converged = False
        loss_stable_count = 0
        loss_details = {}

        loss_cols = [col for col in recent_df.columns if 'loss' in col.lower()]
        for col in loss_cols:
            loss_values = recent_df[col].dropna()
            if len(loss_values) >= 15:
                # 计算最近15个episode的统计
                recent_loss = loss_values.tail(15)
                loss_std = recent_loss.std()
                loss_mean = recent_loss.mean()
                loss_cv = loss_std / (abs(loss_mean) + 1e-8)

                # 趋势分析
                loss_trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]

                # 收敛判断
                is_stable = loss_cv < 0.05  # 变异系数小于5%
                is_decreasing = loss_trend < -0.001

                loss_details[col] = {
                    'cv': loss_cv,
                    'trend': loss_trend,
                    'stable': is_stable,
                    'decreasing': is_decreasing
                }

                if is_stable or is_decreasing:
                    loss_stable_count += 1

        if loss_stable_count >= len(loss_cols) * 0.6:  # 60%以上的loss指标稳定
            loss_converged = True

        print("Loss Convergence Analysis:")
        for loss_name, details in loss_details.items():
            print("  {}: CV={:.4f}, trend={:.6f}, stable={}, decreasing={}".format(
                loss_name, details['cv'], details['trend'], details['stable'], details['decreasing']))
        print("  Overall convergence: {}".format("Yes" if loss_converged else "No"))

        # 3. 训练稳定性检查
        stability_score = 0
        if 'reward' in recent_df.columns:
            reward_values = recent_df['reward'].dropna()
            if len(reward_values) >= 15:
                reward_cv = reward_values.tail(15).std() / (abs(reward_values.tail(15).mean()) + 1e-8)
                if reward_cv < 0.1:
                    stability_score += 30
                elif reward_cv < 0.3:
                    stability_score += 20
                else:
                    stability_score += 10

        # 梯度稳定性
        if 'gradient_norm' in recent_df.columns:
            grad_values = recent_df['gradient_norm'].dropna()
            if len(grad_values) >= 10:
                grad_cv = grad_values.tail(10).std() / (grad_values.tail(10).mean() + 1e-8)
                if grad_cv < 0.2:
                    stability_score += 20
                elif grad_cv < 0.5:
                    stability_score += 10

        print("Training Stability Score: {}/50".format(stability_score))

        # 4. 综合早停决策
        decision_factors = []
        confidence_factors = []

        # ASR停滞权重
        if asr_stagnant_episodes >= 20:
            decision_factors.append("ASR stagnant for {} episodes".format(asr_stagnant_episodes))
            confidence_factors.append(0.4)
        elif asr_stagnant_episodes >= 15:
            decision_factors.append("ASR stagnant for {} episodes".format(asr_stagnant_episodes))
            confidence_factors.append(0.3)

        # 损失收敛权重
        if loss_converged:
            decision_factors.append("Loss functions converged")
            confidence_factors.append(0.25)

        # 稳定性权重
        if stability_score >= 40:
            decision_factors.append("High training stability")
            confidence_factors.append(0.2)
        elif stability_score < 20:
            decision_factors.append("Poor training stability")
            confidence_factors.append(-0.1)  # 负权重，不建议停止

        # 计算综合置信度
        total_confidence = sum(confidence_factors) if confidence_factors else 0
        should_stop = total_confidence >= 0.6

        # 决策输出
        if should_stop:
            if total_confidence >= 0.8:
                recommendation = "Strong recommendation: STOP training"
                confidence_level = min(95, int(total_confidence * 100))
            else:
                recommendation = "Moderate recommendation: Consider STOPPING training"
                confidence_level = min(85, int(total_confidence * 100))
        else:
            if total_confidence <= 0.2:
                recommendation = "Continue training - insufficient evidence for stopping"
                confidence_level = max(20, int((1 - total_confidence) * 100))
            else:
                recommendation = "Continue training but monitor closely"
                confidence_level = max(30, int((0.8 - total_confidence) * 100))

        print("Decision Factors:")
        for factor in decision_factors:
            print("  - {}".format(factor))

        print("Final Recommendation: {}".format(recommendation))
        print("Confidence: {}%".format(confidence_level))

        # 记录决策历史
        log_path = os.path.join(exp_dir, "logs", "early_stop_log.txt")
        with open(log_path, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write("[{}] Episode {}: {} ({}% confidence)\n".format(
                timestamp, int(df.iloc[-1]['episode']), recommendation, confidence_level))
            f.write("  Factors: {}\n".format(", ".join(decision_factors)))
            f.write("  ASR: {:.6f}, Stagnant: {} episodes\n".format(asr_current_value, asr_stagnant_episodes))
            f.write("  Loss converged: {}, Stability: {}/50\n\n".format(loss_converged, stability_score))

        # 保存决策分析结果
        decision_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "episode": int(df.iloc[-1]['episode']),
            "should_stop": should_stop,
            "confidence": confidence_level,
            "recommendation": recommendation,
            "asr_analysis": {
                "current_value": float(asr_current_value),
                "stagnant_episodes": int(asr_stagnant_episodes),
                "improvement_rate": float(asr_improvement_rate)
            },
            "loss_analysis": {
                "converged": loss_converged,
                "details": {k: {key: float(val) if isinstance(val, (int, float)) else val
                               for key, val in v.items()} for k, v in loss_details.items()}
            },
            "stability_score": int(stability_score),
            "decision_factors": decision_factors
        }

        decision_file = os.path.join(exp_dir, "logs", "early_stop_analysis.json")
        with open(decision_file, 'w', encoding='utf-8') as f:
            json.dump(decision_result, f, indent=2, ensure_ascii=False)

        print("Early stopping analysis saved to: {}".format(decision_file))
        return should_stop

class TrainingVisualizer:
    def __init__(self):
        # 设置英文字体
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_training_curves(self, experiment_name):
        """绘制训练过程曲线"""
        exp_dir = os.path.join("experiments", experiment_name)
        csv_path = os.path.join(exp_dir, "logs", "training_metrics.csv")

        if not os.path.exists(csv_path):
            print("No training metrics found for experiment {}.".format(experiment_name))
            return

        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print("No data found in training metrics.")
            return

        # 创建图表目录
        plots_dir = os.path.join(exp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 根据实际数据列绘制不同类型的图表
        # 1. 训练损失图表
        loss_columns = ['train_obj_critics', 'train_obj_actors', 'train_alphas']
        available_loss_cols = [col for col in loss_columns if col in df.columns]

        if available_loss_cols:
            plt.figure(figsize=(10, 6))
            for loss_col in available_loss_cols:
                loss_data = df[loss_col].dropna()
                if len(loss_data) > 0:
                    episodes = df.loc[loss_data.index, 'episode']
                    plt.plot(episodes, loss_data, label=loss_col, linewidth=2)

            plt.xlabel('Episode')
            plt.ylabel('Loss Value')
            plt.title('Training Loss Curves - {}'.format(experiment_name))
            plt.legend()
            plt.grid(True, alpha=0.3)

            loss_plot_path = os.path.join(plots_dir, 'training_loss_curves.png')
            plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print("- {}".format(loss_plot_path))

        # 2. 学习率图表
        lr_columns = ['train_act_lr', 'train_cri_lr', 'train_alpha_lr']
        available_lr_cols = [col for col in lr_columns if col in df.columns]

        if available_lr_cols:
            plt.figure(figsize=(10, 6))
            for lr_col in available_lr_cols:
                lr_data = df[lr_col].dropna()
                if len(lr_data) > 0:
                    episodes = df.loc[lr_data.index, 'episode']
                    plt.plot(episodes, lr_data, label=lr_col, linewidth=2)

            plt.xlabel('Episode')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Curves - {}'.format(experiment_name))
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # 学习率通常用对数尺度

            lr_plot_path = os.path.join(plots_dir, 'learning_rate_curves.png')
            plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print("- {}".format(lr_plot_path))

        # 3. 验证指标图表 - 分环境展示
        for env_id in range(4):  # env0 to env3
            env_metrics = {
                'ARR%': 'val_ARR%_env{}'.format(env_id),
                'SR': 'val_SR_env{}'.format(env_id),
                'CR': 'val_CR_env{}'.format(env_id),
                'MDD%': 'val_MDD%_env{}'.format(env_id),
                'VOL': 'val_VOL_env{}'.format(env_id),
                'SOR': 'val_SOR_env{}'.format(env_id)
            }

            available_metrics = {name: col for name, col in env_metrics.items() if col in df.columns}

            if available_metrics:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                for idx, (metric_name, col_name) in enumerate(available_metrics.items()):
                    if idx >= 6:  # 最多6个子图
                        break

                    metric_data = df[col_name].dropna()
                    if len(metric_data) > 0:
                        episodes = df.loc[metric_data.index, 'episode']
                        axes[idx].plot(episodes, metric_data, linewidth=2, color='blue')
                        axes[idx].set_title('{} (env{})'.format(metric_name, env_id))
                        axes[idx].set_xlabel('Episode')
                        axes[idx].set_ylabel(metric_name)
                        axes[idx].grid(True, alpha=0.3)

                # 隐藏多余的子图
                for idx in range(len(available_metrics), 6):
                    axes[idx].set_visible(False)

                plt.tight_layout()
                env_plot_path = os.path.join(plots_dir, 'validation_metrics_env{}.png'.format(env_id))
                plt.savefig(env_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print("- {}".format(env_plot_path))

        # 4. 综合对比图 - 所有环境的主要指标对比
        main_metrics = ['val_ARR%', 'val_SR', 'val_CR', 'val_MDD%']
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, base_metric in enumerate(main_metrics):
            ax = axes[idx]
            for env_id in range(4):
                col_name = '{}_env{}'.format(base_metric, env_id)
                if col_name in df.columns:
                    metric_data = df[col_name].dropna()
                    if len(metric_data) > 0:
                        episodes = df.loc[metric_data.index, 'episode']
                        ax.plot(episodes, metric_data, label='env{}'.format(env_id), linewidth=2)

            ax.set_title('{} Comparison Across Environments'.format(base_metric))
            ax.set_xlabel('Episode')
            ax.set_ylabel(base_metric)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_plot_path = os.path.join(plots_dir, 'validation_metrics_comparison.png')
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print("- {}".format(comparison_plot_path))

        print("Training plots generated in: {}".format(plots_dir))

def main():
    parser = argparse.ArgumentParser(description='训练监控工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # status命令
    status_parser = subparsers.add_parser('status', help='查看训练状态')
    status_parser.add_argument('--experiment', required=True, help='实验名称')

    # analyze-stability命令
    stability_parser = subparsers.add_parser('analyze-stability', help='分析训练稳定性')
    stability_parser.add_argument('--experiment', required=True, help='实验名称')

    # analyze-value-function命令 - 新增
    value_parser = subparsers.add_parser('analyze-value-function', help='分析价值函数学习质量')
    value_parser.add_argument('--experiment', required=True, help='实验名称')

    # early-stop-check命令
    early_stop_parser = subparsers.add_parser('early-stop-check', help='检查早停建议')
    early_stop_parser.add_argument('--experiment', required=True, help='实验名称')

    # plot-training命令
    plot_parser = subparsers.add_parser('plot-training', help='绘制训练曲线')
    plot_parser.add_argument('--experiment', required=True, help='实验名称')

    # comprehensive-analysis命令 - 新增综合分析
    comp_parser = subparsers.add_parser('comprehensive-analysis', help='执行综合训练分析')
    comp_parser.add_argument('--experiment', required=True, help='实验名称')

    args = parser.parse_args()

    if args.command == 'status':
        monitor = TrainingMonitor()
        monitor.status(args.experiment)
    elif args.command == 'analyze-stability':
        monitor = TrainingMonitor()
        monitor.analyze_stability(args.experiment)
    elif args.command == 'analyze-value-function':
        monitor = TrainingMonitor()
        monitor.analyze_value_function_quality(args.experiment)
    elif args.command == 'early-stop-check':
        monitor = TrainingMonitor()
        monitor.early_stop_check(args.experiment)
    elif args.command == 'plot-training':
        experiment_path = os.path.join("experiments", args.experiment)
        if not os.path.exists(experiment_path):
            print("实验目录不存在：{}".format(experiment_path))
            return

        visualizer = AdvancedTrainingVisualizer(experiment_path)
        visualizer.generate_all_plots()
    elif args.command == 'comprehensive-analysis':
        # 综合分析：执行所有分析功能
        experiment_path = os.path.join("experiments", args.experiment)
        if not os.path.exists(experiment_path):
            print("实验目录不存在：{}".format(experiment_path))
            return

        monitor = TrainingMonitor()
        visualizer = AdvancedTrainingVisualizer(experiment_path)

        print("=== 执行综合训练分析 ===")
        print("1. 基础状态检查...")
        monitor.status(args.experiment)

        print("\n2. 训练稳定性分析...")
        monitor.analyze_stability(args.experiment)

        print("\n3. 价值函数质量分析...")
        monitor.analyze_value_function_quality(args.experiment)

        print("\n4. 早停决策检查...")
        monitor.early_stop_check(args.experiment)

        print("\n5. 生成训练图表...")
        visualizer.generate_all_plots()

        print("\n=== 综合分析完成 ===")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()