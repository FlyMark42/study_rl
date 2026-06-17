# -*- coding: utf-8 -*-
"""
本项目特别训练监控功能模块
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

def _get_exp_dir(experiment_name):
    return os.path.join("experiments", experiment_name)

try:
    from experiments.visualizer import TrainingVisualizer as AdvancedTrainingVisualizer, ExperimentComparator
    from experiments.monitor import TrainingMonitor
except ImportError:
    from visualizer import TrainingVisualizer as AdvancedTrainingVisualizer, ExperimentComparator
    from monitor import TrainingMonitor

# 设置英文字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

class TrainingMonitor_Alpha(TrainingMonitor):
    def __init__(self):
        super().__init__()
        pass
    def early_stop_check(self, experiment_name, patience=5, metrics='ASR'):
        """
        检查是否建议早停

        参数:
            experiment_name: 实验名称
            patience: 耐心值（停滞多少个episode后建议停止）
            metrics: 优化指标，'ASR'或'ARR'
        """
        exp_dir = _get_exp_dir(experiment_name)
        csv_path = os.path.join(exp_dir, "logs", "training_metrics.csv")

        if not os.path.exists(csv_path):
            print("No training metrics found for experiment {}.".format(experiment_name))
            return False

        df = pd.read_csv(csv_path)
        if len(df) < patience * 4:
            print("Insufficient data for early stopping analysis (need at least {} episodes).".format(patience * 4))
            return False

        print("Enhanced Early Stopping Analysis (Optimization Target: {})".format(metrics))

        # 根据metrics选择对应的验证字段
        if metrics == 'ASR':
            validation_col = 'validation_asr'
            metric_name = 'ASR'
        elif metrics == 'ARR':
            validation_col = 'validation_arr'
            metric_name = 'ARR'
        else:
            print("Error: metrics must be 'ASR' or 'ARR', got '{}'".format(metrics))
            return False

        # 多维度早停分析
        recent_episodes = min(30, len(df))
        recent_df = df.tail(recent_episodes)

        # 1. 验证指标停滞分析
        metric_stagnant_episodes = 0
        metric_improvement_rate = 0
        metric_current_value = 0

        if validation_col in recent_df.columns:
            metric_values = recent_df[validation_col].dropna()
            if len(metric_values) >= patience:
                metric_current_value = metric_values.iloc[-1]

                # 检查最近patience个episode的指标改善
                recent_N_metric = metric_values.tail(patience)
                metric_improvement = recent_N_metric.iloc[-1] - recent_N_metric.iloc[0]
                metric_improvement_rate = metric_improvement / len(recent_N_metric)

                if metric_improvement < 0.001:  # 改善小于0.1%
                    metric_stagnant_episodes = patience

                # 检查是否有明显下降趋势
                if len(metric_values) >= patience * 2 and patience * 2 >= 2:
                    recent_trend = np.polyfit(range(len(metric_values.tail(patience * 2))), metric_values.tail(patience * 2), 1)[0]
                    if recent_trend < -0.0001:  # 明显下降趋势
                        metric_stagnant_episodes = max(metric_stagnant_episodes, patience * 2)

        print("Validation {} Analysis:".format(metric_name))
        print("  Current {}: {:.6f}".format(metric_name, metric_current_value))
        print("  Improvement rate: {:.6f} per episode".format(metric_improvement_rate))
        print("  Stagnant episodes: {}".format(metric_stagnant_episodes))

        # 2. 损失收敛分析
        loss_converged = False
        loss_stable_count = 0
        loss_details = {}

        loss_cols = [col for col in recent_df.columns if 'loss' in col.lower()]
        for col in loss_cols:
            loss_values = recent_df[col].dropna()
            if len(loss_values) >= patience:
                # 计算最近patience个episode的统计
                recent_loss = loss_values.tail(patience)
                loss_std = recent_loss.std()
                loss_mean = recent_loss.mean()
                loss_cv = loss_std / (abs(loss_mean) + 1e-8)

                # 趋势分析
                if len(recent_loss) >= 2:
                    loss_trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]
                else:
                    loss_trend = 0.0

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
            if len(reward_values) >= patience:
                reward_cv = reward_values.tail(patience).std() / (abs(reward_values.tail(patience).mean()) + 1e-8)
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

        # 验证指标停滞权重
        if metric_stagnant_episodes >= patience * 2:
            decision_factors.append("{} stagnant for {} episodes".format(metric_name, metric_stagnant_episodes))
            confidence_factors.append(0.4)
        elif metric_stagnant_episodes >= patience:
            decision_factors.append("{} stagnant for {} episodes".format(metric_name, metric_stagnant_episodes))
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
            f.write("  {}: {:.6f}, Stagnant: {} episodes\n".format(metric_name, metric_current_value, metric_stagnant_episodes))
            f.write("  Loss converged: {}, Stability: {}/50\n\n".format(loss_converged, stability_score))

        # 保存决策分析结果
        # 转换loss_details中的所有值为JSON可序列化类型
        serializable_loss_details = {}
        for k, v in loss_details.items():
            serializable_loss_details[k] = {}
            for key, val in v.items():
                if isinstance(val, (np.integer, int)):
                    serializable_loss_details[k][key] = int(val)
                elif isinstance(val, (np.floating, float)):
                    serializable_loss_details[k][key] = float(val)
                elif isinstance(val, (np.bool_, bool)):
                    serializable_loss_details[k][key] = bool(val)
                else:
                    serializable_loss_details[k][key] = str(val)

        decision_result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "episode": int(df.iloc[-1]['episode']),
            "should_stop": bool(should_stop),
            "confidence": int(confidence_level),
            "recommendation": str(recommendation),
            "optimization_target": str(metrics),
            "metric_analysis": {
                "metric_name": str(metric_name),
                "current_value": float(metric_current_value),
                "stagnant_episodes": int(metric_stagnant_episodes),
                "improvement_rate": float(metric_improvement_rate)
            },
            "loss_analysis": {
                "converged": bool(loss_converged),
                "details": serializable_loss_details
            },
            "stability_score": int(stability_score),
            "decision_factors": decision_factors
        }

        decision_file = os.path.join(exp_dir, "logs", "early_stop_analysis.json")
        with open(decision_file, 'w', encoding='utf-8') as f:
            json.dump(decision_result, f, indent=2, ensure_ascii=False)

        print("Early stopping analysis saved to: {}".format(decision_file))
        return should_stop
    def status(self, experiment_name, metrics='ASR'):
        """
        查看当前训练状态和关键指标

        参数:
            experiment_name: 实验名称
            metrics: 优化指标，'ASR'或'ARR'
        """
        exp_dir = _get_exp_dir(experiment_name)
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

        # 根据metrics选择验证字段
        if metrics == 'ASR':
            val_key_col = 'val_SR_env0'
            metric_name = 'ASR'
        elif metrics == 'ARR':
            val_key_col = 'val_ARR_env0'
            metric_name = 'ARR'
        else:
            print("Error: metrics must be 'ASR' or 'ARR', got '{}'".format(metrics))
            return

        # 获取最新的训练记录和验证记录
        train_df = df[df['train_obj_critics'].notna()]
        val_df = df[df[val_key_col].notna()]

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

        print("Experiment: {} (Optimization Target: {})".format(experiment_name, metric_name))
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
            # 显示验证指标（根据优化目标动态选择）
            print("Latest validation metrics:")
            if metrics == 'ASR':
                validation_metrics = ['val_SR_env0', 'val_VOL_env0', 'val_MDD%_env0', 'val_SR_env0', 'val_CR_env0']
            else:  # ARR
                validation_metrics = ['val_ARR_env0', 'val_VOL_env0', 'val_MDD%_env0', 'val_SR_env0', 'val_CR_env0']

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
        exp_dir = _get_exp_dir(experiment_name)
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

            # 分析梯度稳定性
            grad_metrics = ['gradient_norm']
            for grad_col in grad_metrics:
                if grad_col in window_df.columns:
                    values = window_df[grad_col].dropna()
                    if len(values) > 1:
                        variance = values.var()
                        mean_val = values.mean()
                        std_val = values.std()
                        cv = std_val / (abs(mean_val) + 1e-8) if abs(mean_val) > 1e-8 else 0

                        # 计算趋势
                        x = np.arange(len(values))
                        if len(values) > 2:
                            slope = np.polyfit(x, values, 1)[0]
                        else:
                            slope = 0

                        # 梯度稳定性评分（梯度应该趋于稳定，CV不应太大）
                        gradient_stability_score = max(0, 100 - cv * 100 - abs(slope) * 5)

                        stability_record.update({
                            "{}_variance".format(grad_col): variance,
                            "{}_mean".format(grad_col): mean_val,
                            "{}_cv".format(grad_col): cv,
                            "{}_trend".format(grad_col): slope,
                            "{}_stability_score".format(grad_col): gradient_stability_score
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
                    if cv > 0.1:
                        lr_stable = False

            print("  Overall LR Stability: {}".format("Stable" if lr_stable else "Unstable"))

            # 梯度稳定性分析
            print("Gradient Stability:")
            gradient_stable = True
            gradient_score = 0
            for grad_col in ['gradient_norm']:
                cv_key = "{}_cv".format(grad_col)
                trend_key = "{}_trend".format(grad_col)
                score_key = "{}_stability_score".format(grad_col)

                if cv_key in latest and trend_key in latest:
                    cv = latest[cv_key]
                    trend = latest[trend_key]
                    score = latest.get(score_key, 0)

                    if not np.isnan(cv) and not np.isnan(trend) and not np.isnan(score):
                        print("  {}: CV={:.4f}, trend={:.6f}, stability={:.1f}/100".format(
                            grad_col, cv, trend, score))
                        gradient_score = score

                        if cv > 0.5:
                            gradient_stable = False
                    else:
                        print("  {}: Insufficient data for analysis".format(grad_col))

            if gradient_score > 0:
                grad_level = "Excellent" if gradient_score >= 80 else "Good" if gradient_score >= 60 else "Fair" if gradient_score >= 40 else "Poor"
                print("  Overall Gradient Stability: {}/100 ({})".format(gradient_score, grad_level))
            else:
                print("  Overall Gradient Stability: Insufficient data")

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

            if gradient_score >= 60:
                recommendations.append("+ Gradients are stable")
            elif gradient_score >= 40:
                recommendations.append("! Gradient stability is moderate")
            elif gradient_score > 0:
                recommendations.append("- Gradients are unstable - may indicate training issues")

            for rec in recommendations:
                print("  {}".format(rec))

            # 总体评估
            if not np.isnan(avg_loss_stability):
                overall_score = avg_loss_stability * 0.5 + (25 if lr_stable else 10) + gradient_score * 0.25
                overall_level = "Excellent" if overall_score >= 80 else "Good" if overall_score >= 60 else "Fair" if overall_score >= 40 else "Needs Attention"
                print("Overall Training Health: {:.1f}/100 ({})".format(overall_score, overall_level))
            else:
                print("Overall Training Health: Insufficient data for evaluation")

        print("Stability analysis saved to: {}".format(stability_path))