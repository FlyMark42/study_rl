# -*- coding: utf-8 -*-
# 可视化功能测试脚本
# 本文件位置：项目根目录/test_visualizer.py

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from experiments.visualizer import TrainingVisualizer, ExperimentComparator

def create_mock_experiment_data(exp_name, base_path="experiments"):
    """创建模拟实验数据"""
    exp_path = os.path.join(base_path, exp_name)
    logs_path = os.path.join(exp_path, "logs")
    os.makedirs(logs_path, exist_ok=True)

    # 创建模拟的training_metrics.csv
    episodes = list(range(1, 11))  # 10个episode
    timestamps = [datetime.now() - timedelta(hours=10-i) for i in range(10)]

    # 训练数据（奇数episode）
    train_data = []
    for i in range(0, 10, 2):  # episode 1, 3, 5, 7, 9
        episode = episodes[i]
        train_data.append({
            'episode': episode,
            'timestamp': timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
            'reward': -500 + np.random.normal(0, 50),
            'exploration_rate': 0.4 * np.exp(-episode * 0.1) + np.random.normal(0, 0.05),
            'gradient_norm': 3.0 + np.random.normal(0, 0.5),
            'train_obj_critics': 500 + np.random.normal(0, 50),
            'train_obj_actors': -4 + np.random.normal(0, 1),
            'train_alphas': 0.4 * np.exp(-episode * 0.1),
            'train_act_lr': 3e-7 + np.random.normal(0, 1e-8),
            'train_cri_lr': 3e-7 + np.random.normal(0, 1e-8),
            'train_alpha_lr': 3e-7 + np.random.normal(0, 1e-8),
            'train_gradient_norms': 3.0 + np.random.normal(0, 0.5),
            'actor_learning_rate': 5e-7 + np.random.normal(0, 1e-8),
            'critic_learning_rate': 5e-7 + np.random.normal(0, 1e-8),
            'beta_loss': np.random.uniform(0.01, 0.1),
            'representation_loss': np.random.uniform(0.1, 0.5),
        })

    # 验证数据（偶数episode）
    val_data = []
    for i in range(1, 10, 2):  # episode 2, 4, 6, 8, 10
        episode = episodes[i]
        base_arr = 10 + episode * 0.5 + np.random.normal(0, 2)
        val_data.append({
            'episode': episode,
            'timestamp': timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
            'validation_asr': base_arr * 0.1,
            'validation_reward': base_arr * 10,
            'val_ARR%_env0': base_arr,
            'val_SR_env0': base_arr * 0.08,
            'val_CR_env0': base_arr * 0.3,
            'val_MDD%_env0': base_arr * 0.25,
            'val_VOL_env0': 0.005 + np.random.normal(0, 0.001),
            'val_DD_env0': 0.003 + np.random.normal(0, 0.0005),
            'val_SOR_env0': base_arr * 0.25,
            'val_ARR%_env1': base_arr * 0.9,
            'val_SR_env1': base_arr * 0.07,
            'val_CR_env1': base_arr * 0.28,
            'val_MDD%_env1': base_arr * 0.27,
            'val_VOL_env1': 0.0045 + np.random.normal(0, 0.001),
            'val_DD_env1': 0.0025 + np.random.normal(0, 0.0005),
            'val_SOR_env1': base_arr * 0.22,
        })

    # 合并数据并保存
    all_data = train_data + val_data
    all_data.sort(key=lambda x: x['episode'])

    training_df = pd.DataFrame(all_data)
    training_csv = os.path.join(logs_path, "training_metrics.csv")
    training_df.to_csv(training_csv, index=False)

    # 创建模拟的value_function_metrics.csv
    value_data = []
    for episode in episodes[::2]:  # 只在训练episode记录
        value_data.append({
            'episode': episode,
            'timestamp': (datetime.now() - timedelta(hours=10-episode)).strftime("%Y-%m-%d %H:%M:%S"),
            'q_values_mean': -2.5 + np.random.normal(0, 0.5),
            'q_values_std': 1.2 + np.random.normal(0, 0.2),
            'q_values_min': -5.0 + np.random.normal(0, 1),
            'q_values_max': 1.0 + np.random.normal(0, 0.5),
            'td_error_mean': 0.1 + np.random.normal(0, 0.02),
            'td_error_std': 0.05 + np.random.normal(0, 0.01),
            'td_error_abs_mean': 0.12 + np.random.normal(0, 0.02),
            'policy_entropy': 1.5 * np.exp(-episode * 0.05) + np.random.normal(0, 0.1),
            'policy_entropy_std': 0.3 + np.random.normal(0, 0.05),
            'target_q_diff': 0.5 + np.random.normal(0, 0.1),
            'value_estimation_bias': 0.02 + np.random.normal(0, 0.005),
            'q_network_weights_norm': 18 + episode * 0.1 + np.random.normal(0, 0.5),
        })

    value_df = pd.DataFrame(value_data)
    value_csv = os.path.join(logs_path, "value_function_metrics.csv")
    value_df.to_csv(value_csv, index=False)

    print("已创建模拟实验：{}".format(exp_name))
    return exp_path

def test_all_visualizations():
    """测试所有可视化功能"""
    print("=== 开始测试可视化功能 ===\n")

    # 1. 创建多个模拟实验
    experiments = [
        "test_exp_lr_0.001_batch_32_seed_1234",
        "test_exp_lr_0.0005_batch_64_seed_1234",
        "test_exp_lr_0.0001_batch_16_seed_1234",
        "test_exp_lr_0.001_batch_32_seed_5678",
    ]

    print("1. 创建模拟实验数据...")
    for exp_name in experiments:
        create_mock_experiment_data(exp_name)

    print("\n2. 测试单个实验可视化...")
    # 测试第一个实验的可视化
    test_exp_path = os.path.join("experiments", experiments[0])
    visualizer = TrainingVisualizer(test_exp_path)

    print("  - 生成损失曲线图...")
    visualizer.plot_loss_curves()

    print("  - 生成奖励曲线图...")
    visualizer.plot_reward_curves()

    print("  - 生成验证性能曲线...")
    visualizer.plot_validation_curves()

    print("  - 生成学习曲线图...")
    visualizer.plot_learning_curves()

    print("  - 生成价值函数曲线...")
    visualizer.plot_value_function_curves()

    print("  - 生成所有图表...")
    visualizer.generate_all_plots()

    print("\n3. 测试实验对比可视化...")
    comparator = ExperimentComparator("experiments")

    print("  - 生成性能排行榜...")
    comparator.generate_performance_ranking()

    print("  - 生成超参数分析...")
    comparator.generate_hyperparameter_analysis()

    print("  - 生成指标对比图...")
    metrics_to_compare = [
        'val_ARR%_env0', 'val_SR_env0', 'val_CR_env0',
        'train_obj_critics', 'train_obj_actors', 'exploration_rate',
        'gradient_norm'
    ]

    for metric in metrics_to_compare:
        print("    对比指标: {}".format(metric))
        comparator.plot_experiments_comparison(metric)

    print("\n4. 测试所有指标图形输出...")
    test_metrics = {
        '基础训练指标': ['train_obj_critics', 'train_obj_actors', 'train_alphas'],
        '学习率指标': ['train_act_lr', 'train_cri_lr', 'train_alpha_lr'],
        '探索和梯度': ['exploration_rate', 'gradient_norm'],
        '验证性能指标': ['val_ARR%_env0', 'val_SR_env0', 'val_CR_env0', 'val_MDD%_env0'],
        '价值函数指标': ['q_values_mean', 'td_error_mean', 'policy_entropy'],
        '掩码算法指标': ['beta_loss', 'representation_loss']
    }

    for category, metrics in test_metrics.items():
        print("  测试{}:".format(category))
        for metric in metrics:
            print("    - {}".format(metric))

    print("\n=== 可视化功能测试完成 ===")
    print("所有图表已保存到相应的实验目录和对比目录中")

    # 清理测试数据
    cleanup = input("\n是否删除测试实验数据? (y/n): ")
    if cleanup.lower() == 'y':
        import shutil
        for exp_name in experiments:
            exp_path = os.path.join("experiments", exp_name)
            if os.path.exists(exp_path):
                shutil.rmtree(exp_path)
                print("已删除测试实验：{}".format(exp_name))

        comparison_path = os.path.join("experiments", "comparison_plots")
        if os.path.exists(comparison_path):
            shutil.rmtree(comparison_path)
            print("已删除对比图表目录")

if __name__ == '__main__':
    test_all_visualizations()