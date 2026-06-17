# -*- coding: utf-8 -*-
"""
最优模型查找工具
该文件应该放在experiments目录下
功能：两阶段筛选 - 先评估稳定性，再按验证集性能排序找到最优模型
"""
import os
import json
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

try:

    from alpha_monitor import TrainingMonitor_Alpha
except ImportError:
    from experiments.alpha_monitor import TrainingMonitor_Alpha

g_episodeCount=1#至少要达到这个次数才去检查
g_patience=1#至少要达到这个次数才去检查
g_metrics='ASR'  # 优化目标：'ASR' 或 'ARR'，根据实际情况设置
class StableBestModelFinder:
    def __init__(self, experiments_dir="experiments"):
        """
        初始化最优模型查找器

        参数:
            experiments_dir: 实验目录路径
        """
        self.experiments_dir = experiments_dir
        self.monitor = TrainingMonitor_Alpha()
        self.all_experiments = []
        self.stable_models = []
        self.performance_data = []

        print("初始化最优模型查找器，实验目录：{}".format(self.experiments_dir))

    def scan_all_experiments(self):
        """扫描所有实验目录"""
        print("开始扫描实验目录：{}".format(self.experiments_dir))

        if not os.path.exists(self.experiments_dir):
            print("错误：实验目录不存在：{}".format(self.experiments_dir))
            return []

        experiments = []
        for item in os.listdir(self.experiments_dir):
            exp_path = os.path.join(self.experiments_dir, item)

            # 跳过非目录和特殊目录
            if not os.path.isdir(exp_path):
                continue
            if item in ['templates', 'configs', '__pycache__', 'comparison_plots', 'hyperopt_results']:
                continue

            # 检查是否有必要的文件
            status_file = os.path.join(exp_path, "status.json")
            metrics_file = os.path.join(exp_path, "logs", "training_metrics.csv")

            if os.path.exists(status_file) and os.path.exists(metrics_file):
                experiments.append({
                    'name': item,
                    'path': exp_path,
                    'status_file': status_file,
                    'metrics_file': metrics_file
                })
                print("发现实验：{}".format(item))

        self.all_experiments = experiments
        print("扫描完成，共发现 {} 个实验".format(len(experiments)))
        return experiments

    def evaluate_stability(self, exp_name, exp_path):
        """
        评估单个实验的稳定性

        参数:
            exp_name: 实验名称
            exp_path: 实验路径

        返回:
            稳定性评估结果字典
        """
        print("评估实验稳定性：{}".format(exp_name))

        # 检查实验状态
        status_file = os.path.join(exp_path, "status.json")
        with open(status_file, 'r', encoding='utf-8') as f:
            status_data = json.load(f)

        if status_data.get('status') != 'completed':
            print("  实验状态：{} - 跳过".format(status_data.get('status')))
            return None

        # 检查训练数据量
        metrics_file = os.path.join(exp_path, "logs", "training_metrics.csv")
        df = pd.read_csv(metrics_file)

        if len(df) < g_episodeCount:
            print("  训练数据不足（{}个episode），需要至少{}个 - 跳过".format(len(df),g_episodeCount))
            return None

        # 运行稳定性分析
        print("  运行稳定性分析...")
        self.monitor.analyze_stability(exp_name)

        # 读取稳定性分析结果
        stability_file = os.path.join(exp_path, "logs", "stability_analysis.csv")
        if not os.path.exists(stability_file):
            print("  稳定性分析文件不存在 - 跳过")
            return None

        stability_df = pd.read_csv(stability_file)
        if len(stability_df) == 0:
            print("  稳定性分析数据为空 - 跳过")
            return None

        latest_stability = stability_df.iloc[-1]

        # 计算训练健康分数
        training_health = self._calculate_training_health(latest_stability)

        # 运行早停分析
        print("  运行早停分析...")
        should_stop = self.monitor.early_stop_check(exp_name, g_patience, g_metrics)

        # 读取早停分析结果
        early_stop_file = os.path.join(exp_path, "logs", "early_stop_analysis.json")
        if not os.path.exists(early_stop_file):
            print("  早停分析文件不存在 - 跳过")
            return None

        with open(early_stop_file, 'r', encoding='utf-8') as f:
            early_stop_data = json.load(f)

        # 计算稳定性综合得分
        stability_score = self._calculate_stability_score(
            training_health,
            early_stop_data
        )

        result = {
            'experiment_name': exp_name,
            'experiment_path': exp_path,
            'training_health': training_health,
            'early_stop_confidence': early_stop_data.get('confidence', 0),
            'loss_converged': early_stop_data.get('loss_analysis', {}).get('converged', False),
            'metric_stagnant_episodes': early_stop_data.get('metric_analysis', {}).get('stagnant_episodes', 0),
            'stability_score': stability_score,
            'stability_data': latest_stability.to_dict(),
            'early_stop_data': early_stop_data
        }

        print("  稳定性得分：{:.1f}/100".format(stability_score))
        return result

    def _calculate_training_health(self, stability_record):
        """
        计算训练健康分数（基于稳定性分析）

        参数:
            stability_record: 稳定性分析记录（DataFrame行）

        返回:
            训练健康分数 (0-100)
        """
        score = 0

        # 损失稳定性评分
        loss_scores = []
        loss_metrics = ['train_obj_critics', 'train_obj_actors', 'train_alphas'] #我的项目使用
        # loss_metrics = ['train_critic_loss', 'train_obj_actors', 'train_alphas'] 别的项目使用

        for loss_col in loss_metrics:
            score_key = "{}_stability_score".format(loss_col)
            if score_key in stability_record and not np.isnan(stability_record[score_key]):
                loss_scores.append(stability_record[score_key])

        if loss_scores:
            avg_loss_stability = np.mean(loss_scores)
            score += avg_loss_stability * 0.5  # 50%权重

        # 学习率稳定性评分
        lr_stable = True
        lr_metrics = ['train_act_lr', 'train_cri_lr', 'train_alpha_lr'] #我的项目使用
        # lr_metrics = ['train_learning_rate', 'train_cri_lr', 'train_alpha_lr'] 别的项目使用
        for lr_col in lr_metrics:
            cv_key = "{}_cv".format(lr_col)
            if cv_key in stability_record and not np.isnan(stability_record[cv_key]):
                if stability_record[cv_key] > 0.1:
                    lr_stable = False
                    break

        score += 25 if lr_stable else 10  # 25分或10分

        # 梯度稳定性评分
        grad_col = 'gradient_norm'
        score_key = "{}_stability_score".format(grad_col)
        if score_key in stability_record and not np.isnan(stability_record[score_key]):
            gradient_stability = stability_record[score_key]
            score += gradient_stability * 0.25  # 25%权重

        return min(100, score)

    def _calculate_stability_score(self, training_health, early_stop_data):
        """
        计算稳定性综合得分

        参数:
            training_health: 训练健康分数
            early_stop_data: 早停分析数据

        返回:
            稳定性综合得分 (0-100)
        """
        score = 0

        # 训练健康分数权重40%
        score += training_health * 0.4

        # 早停置信度权重30%
        early_stop_confidence = early_stop_data.get('confidence', 0)
        score += early_stop_confidence * 0.3

        # 损失收敛状态20分
        loss_converged = early_stop_data.get('loss_analysis', {}).get('converged', False)
        score += 20 if loss_converged else 0

        # 稳定性分数权重10%
        stability_raw_score = early_stop_data.get('stability_score', 0)
        score += stability_raw_score * 0.1

        return min(100, score)

    def filter_stable_models(self, min_score=60):
        """
        筛选稳定模型

        参数:
            min_score: 最小稳定性得分阈值

        返回:
            稳定模型列表
        """
        print("开始筛选稳定模型，最小稳定性得分：{}".format(min_score))

        stable_models = []

        for exp in self.all_experiments:
            exp_name = exp['name']
            exp_path = exp['path']

            # 评估稳定性
            stability_result = self.evaluate_stability(exp_name, exp_path)

            if stability_result is None:
                continue

            # 检查稳定性得分
            if stability_result['stability_score'] >= min_score:
                stable_models.append(stability_result)
                print("模型通过稳定性筛选：{} (得分：{:.1f})".format(
                    exp_name, stability_result['stability_score']))
            else:
                print("模型未通过稳定性筛选：{} (得分：{:.1f})".format(
                    exp_name, stability_result['stability_score']))

        self.stable_models = stable_models
        print("稳定性筛选完成，{}/{} 个模型通过".format(
            len(stable_models), len(self.all_experiments)))

        # 保存稳定性分析汇总
        self._save_stability_summary(stable_models, min_score)

        return stable_models

    def collect_validation_metrics(self, stable_models):
        """
        从稳定模型中收集验证集指标

        参数:
            stable_models: 稳定模型列表

        返回:
            包含验证集指标的模型列表
        """
        print("开始收集验证集指标...")

        performance_data = []

        for model_info in stable_models:
            exp_name = model_info['experiment_name']
            exp_path = model_info['experiment_path']

            metrics_file = os.path.join(exp_path, "logs", "training_metrics.csv")
            df = pd.read_csv(metrics_file)

            # 提取最后一个epoch的验证集指标
            val_metrics = self._extract_validation_metrics(df)

            if val_metrics is None:
                print("  {} - 无验证集数据".format(exp_name))
                continue

            # 查找模型文件
            model_file = self._find_model_file(exp_path)

            # 读取配置
            config_file = os.path.join(exp_path, "config.json")
            config_data = {}
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

            performance_record = {
                'experiment_name': exp_name,
                'experiment_path': exp_path,
                'model_file': model_file,
                'stability_score': model_info['stability_score'],
                'training_health': model_info['training_health'],
                'config': config_data
            }

            # 添加验证集指标
            performance_record.update(val_metrics)

            performance_data.append(performance_record)

            print("  {} - val_ASR={:.4f}, val_CR={:.4f}, val_ARR={:.4f}".format(
                exp_name,
                val_metrics.get('val_ASR', 0),
                val_metrics.get('val_CR', 0),
                val_metrics.get('val_ARR', 0)
            ))

        self.performance_data = performance_data
        print("指标收集完成，共 {} 个模型".format(len(performance_data)))

        return performance_data

    def _extract_validation_metrics(self, df):
        """
        从训练指标CSV中提取最后一个epoch的验证集指标

        参数:
            df: 训练指标DataFrame

        返回:
            验证集指标字典
        """
        # 查找包含验证集指标的列
        val_columns = [col for col in df.columns if col.startswith('val_')]

        if not val_columns:
            return None

        # 找到最后一个有验证数据的行
        val_df = df[val_columns].dropna(how='all')

        if len(val_df) == 0:
            return None

        last_val = val_df.iloc[-1]

        # 提取主要指标
        metrics = {}

        # 尝试提取不同格式的指标
        for base_metric in ['ARR', 'AVol', 'MDD', 'ASR', 'CR', 'IR']:
            # 尝试多种可能的列名格式
            for col_pattern in ['val_{}'.format(base_metric),
                               'val_{}_env0'.format(base_metric),
                               'val_{}%_env0'.format(base_metric),
                               'val_SR_env0' if base_metric == 'ASR' else None]:
                if col_pattern and col_pattern in last_val.index:
                    value = last_val[col_pattern]
                    if not np.isnan(value):
                        metrics['val_{}'.format(base_metric)] = float(value)
                        break

        # 如果使用了validation_asr列名
        if 'validation_asr' in last_val.index and not np.isnan(last_val['validation_asr']):
            metrics['val_ASR'] = float(last_val['validation_asr'])

        # 如果使用了validation_reward列名（作为备选）
        if 'validation_reward' in last_val.index and not np.isnan(last_val['validation_reward']):
            if 'val_ARR' not in metrics:
                metrics['val_ARR'] = float(last_val['validation_reward'])

        return metrics if metrics else None

    def _find_model_file(self, exp_path):
        """
        查找实验目录中的模型文件

        参数:
            exp_path: 实验路径

        返回:
            模型文件路径或None
        """
        # 查找.pt文件
        for file in os.listdir(exp_path):
            if file.endswith('.pth'):
                return os.path.join(exp_path, file)

        return None

    def rank_by_metric(self, metric='val_ASR', top_n=10, ascending=False):
        """
        按指定指标排序

        参数:
            metric: 指标名称
            top_n: 返回前N个
            ascending: 是否升序（默认降序）

        返回:
            排序后的模型列表
        """
        print("按 {} 排序（取前{}个）...".format(metric, top_n))

        if not self.performance_data:
            print("错误：没有性能数据")
            return []

        # 过滤出有该指标的模型
        valid_models = [m for m in self.performance_data if metric in m and not np.isnan(m[metric])]

        if not valid_models:
            print("错误：没有模型包含指标 {}".format(metric))
            return []

        # 排序
        sorted_models = sorted(valid_models, key=lambda x: x[metric], reverse=not ascending)

        # 取前N个
        top_models = sorted_models[:top_n]

        print("排名结果：")
        for i, model in enumerate(top_models, 1):
            print("  {}. {} - {}={:.6f}, 稳定性={:.1f}".format(
                i, model['experiment_name'], metric, model[metric], model['stability_score']
            ))

        return top_models

    def find_best_stable_model(self, metric='val_ASR', min_stability=60):
        """
        找到综合最优模型（稳定性和性能）

        参数:
            metric: 优化的指标
            min_stability: 最小稳定性要求

        返回:
            最优模型信息
        """
        print("查找最优稳定模型：指标={}, 最小稳定性={}".format(metric, min_stability))

        # 筛选满足稳定性要求的模型
        candidates = [m for m in self.performance_data
                     if m['stability_score'] >= min_stability
                     and metric in m
                     and not np.isnan(m[metric])]

        if not candidates:
            print("错误：没有满足条件的模型")
            return None

        # 按指标排序（对于MDD，值越大越好即回撤越小）
        if metric == 'val_MDD':
            best_model = max(candidates, key=lambda x: x[metric])
        else:
            best_model = max(candidates, key=lambda x: x[metric])

        print("找到最优模型：{}".format(best_model['experiment_name']))
        print("  指标 {}={:.6f}".format(metric, best_model[metric]))
        print("  稳定性得分={:.1f}".format(best_model['stability_score']))
        print("  模型文件：{}".format(best_model.get('model_file', 'Not found')))

        return best_model

    def _save_stability_summary(self, stable_models, threshold):
        """保存稳定性分析汇总"""
        summary_file = os.path.join(self.experiments_dir, "stable_models_summary.json")

        summary = {
            "total_experiments": len(self.all_experiments),
            "stable_models_count": len(stable_models),
            "stability_threshold": threshold,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": [
                {
                    "experiment_name": m['experiment_name'],
                    "stability_score": float(m['stability_score']),
                    "training_health": float(m['training_health']),
                    "early_stop_confidence": int(m['early_stop_confidence']),
                    "loss_converged": m['loss_converged'],
                    "metric_stagnant_episodes": int(m['metric_stagnant_episodes'])
                } for m in stable_models
            ]
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("稳定性汇总已保存：{}".format(summary_file))

    def generate_best_models_report(self, min_stability=60,output_file=None):
        """
        生成最优模型推荐报告

        参数:
            output_file: 输出文件路径
        """
        print("生成最优模型推荐报告...")

        if output_file is None:
            output_file = os.path.join(self.experiments_dir, "best_stable_models.json")

        # 找到各指标的最优模型
        best_by_asr = self.find_best_stable_model('val_ASR', min_stability)
        best_by_cr = self.find_best_stable_model('val_CR', min_stability)
        best_by_arr = self.find_best_stable_model('val_ARR', min_stability)

        # Top 5排名
        top5_asr = self.rank_by_metric('val_ASR', top_n=5)
        top5_cr = self.rank_by_metric('val_CR', top_n=5)
        top5_arr = self.rank_by_metric('val_ARR', top_n=5)
        # 构建报告
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_stable_models": len(self.performance_data),
            "primary_recommendation": self._format_model_info(best_by_asr, "最高夏普比率且稳定性优秀") if best_by_asr else None,
            "best_by_calmar": self._format_model_info(best_by_cr, "最高卡尔玛比率且稳定性优秀") if best_by_cr else None,
            "best_by_return": self._format_model_info(best_by_arr, "最高年化收益率且稳定性优秀") if best_by_arr else None,
            "top_5_by_asr": [self._format_model_info(m) for m in top5_asr],
            "top_5_by_cr": [self._format_model_info(m) for m in top5_cr],
            "top_5_by_arr": [self._format_model_info(m) for m in top5_arr]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("最优模型推荐报告已保存：{}".format(output_file))

        # 生成Markdown报告
        md_file = output_file.replace('.json', '.md')
        lines = []
        lines.append("# 最优模型分析报告")
        lines.append("")
        lines.append("生成时间：{}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        lines.append("")

        # 稳定性筛选统计
        lines.append("## 一、稳定性筛选统计")
        lines.append("")
        lines.append("- 总实验数：{}".format(len(self.all_experiments)))
        lines.append("- 稳定模型数：{}".format(len(self.stable_models)))
        lines.append("- 通过率：{:.1f}%".format(
            len(self.stable_models) * 100.0 / len(self.all_experiments) if self.all_experiments else 0))
        lines.append("")

        # 最优模型推荐
        lines.append("## 二、最优模型推荐")
        lines.append("")

        if best_by_asr:
            lines.append("### 主要推荐（最高夏普比率）")
            lines.append("")
            lines.append("- 实验名称：{}".format(best_by_asr['experiment_name']))
            lines.append("- 模型文件：{}".format(best_by_asr.get('model_file', 'Not found')))
            lines.append("- 稳定性得分：{:.1f}/100".format(best_by_asr['stability_score']))
            lines.append("- 夏普比率：{:.4f}".format(best_by_asr.get('val_ASR', 0)))
            lines.append("- 卡尔玛比率：{:.4f}".format(best_by_asr.get('val_CR', 0)))
            lines.append("- 年化收益率：{:.4f}".format(best_by_asr.get('val_ARR', 0)))
            lines.append("- 最大回撤：{:.4f}".format(best_by_asr.get('val_MDD', 0)))
            lines.append("")

        if best_by_arr:
            lines.append("### 最高年化收益率")
            lines.append("")
            lines.append("- 实验名称：{}".format(best_by_arr['experiment_name']))
            lines.append("- 模型文件：{}".format(best_by_arr.get('model_file', 'Not found')))
            lines.append("- 稳定性得分：{:.1f}/100".format(best_by_arr['stability_score']))
            lines.append("- 年化收益率：{:.4f}".format(best_by_arr.get('val_ARR', 0)))
            lines.append("- 夏普比率：{:.4f}".format(best_by_arr.get('val_ASR', 0)))
            lines.append("- 卡尔玛比率：{:.4f}".format(best_by_arr.get('val_CR', 0)))
            lines.append("- 最大回撤：{:.4f}".format(best_by_arr.get('val_MDD', 0)))
            lines.append("")

        # Top 5排名
        lines.append("## 三、Top 5 模型排名")
        lines.append("")

        lines.append("### 按夏普比率（ASR）排序")
        lines.append("")
        lines.append("| 排名 | 实验名称 | 夏普比率 | 年化收益率 | 卡尔玛比率 | 稳定性得分 |")
        lines.append("|------|---------|---------|-----------|-----------|-----------|")

        for i, model in enumerate(top5_asr, 1):
            lines.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.1f} |".format(
                i,
                model['experiment_name'][:50],
                model.get('val_ASR', 0),
                model.get('val_ARR', 0),
                model.get('val_CR', 0),
                model['stability_score']
            ))
        lines.append("")

        lines.append("### 按年化收益率（ARR）排序")
        lines.append("")
        lines.append("| 排名 | 实验名称 | 年化收益率 | 夏普比率 | 卡尔玛比率 | 稳定性得分 |")
        lines.append("|------|---------|-----------|---------|-----------|-----------|")

        for i, model in enumerate(top5_arr, 1):
            lines.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.1f} |".format(
                i,
                model['experiment_name'][:50],
                model.get('val_ARR', 0),
                model.get('val_ASR', 0),
                model.get('val_CR', 0),
                model['stability_score']
            ))
        lines.append("")

        lines.append("### 按卡尔玛比率（CR）排序")
        lines.append("")
        lines.append("| 排名 | 实验名称 | 卡尔玛比率 | 夏普比率 | 年化收益率 | 稳定性得分 |")
        lines.append("|------|---------|-----------|---------|-----------|-----------|")

        for i, model in enumerate(top5_cr, 1):
            lines.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.1f} |".format(
                i,
                model['experiment_name'][:50],
                model.get('val_CR', 0),
                model.get('val_ASR', 0),
                model.get('val_ARR', 0),
                model['stability_score']
            ))
        lines.append("")

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print("Markdown报告已保存：{}".format(md_file))

        return report

    def _format_model_info(self, model, reason=""):
        """格式化模型信息用于报告"""
        if model is None:
            return None

        info = {
            "experiment_name": model['experiment_name'],
            "model_path": model.get('model_file', 'Not found'),
            "stability_score": float(model['stability_score']),
            "training_health": float(model['training_health'])
        }

        # 添加验证集指标
        for metric in ['val_ASR', 'val_CR', 'val_ARR', 'val_MDD', 'val_AVol', 'val_IR']:
            if metric in model and not np.isnan(model[metric]):
                info[metric] = float(model[metric])

        # 添加配置参数
        if 'config' in model and model['config']:
            config = model['config']
            info['config'] = {
                'lr': config.get('lr'),
                'batch_size': config.get('batch_size'),
                'seed': config.get('seed')
            }

        if reason:
            info['reason'] = reason

        return info




def main():
    parser = argparse.ArgumentParser(description='最优模型查找工具')
    parser.add_argument('--experiments-dir', default='experiments',
                       help='实验目录路径（默认：experiments）')
    parser.add_argument('--min-stability', type=float, default=60,
                       help='最小稳定性得分阈值（默认：60）')

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # evaluate-all命令
    eval_parser = subparsers.add_parser('evaluate-all', help='评估所有实验的稳定性')

    # list-stable命令
    list_parser = subparsers.add_parser('list-stable', help='显示稳定模型的性能排名')
    list_parser.add_argument('--metric', default='val_ASR',
                            help='排序指标（默认：val_ASR）')
    list_parser.add_argument('--top', type=int, default=10,
                            help='显示前N个（默认：10）')

    # find-best命令
    find_parser = subparsers.add_parser('find-best', help='找到最优模型')
    find_parser.add_argument('--metric', default='val_ASR',
                            help='优化指标（默认：val_ASR）')

    # report命令
    report_parser = subparsers.add_parser('report', help='生成完整分析报告')
    report_parser.add_argument('--output', help='输出文件路径')

    args = parser.parse_args()

    # 创建查找器
    finder = StableBestModelFinder(args.experiments_dir)

    # 扫描实验
    finder.scan_all_experiments()

    if args.command == 'evaluate-all':
        # 筛选稳定模型
        stable_models = finder.filter_stable_models(min_score=args.min_stability)
        print("评估完成，发现 {} 个稳定模型".format(len(stable_models)))

    elif args.command == 'list-stable':
        # 筛选稳定模型
        stable_models = finder.filter_stable_models(min_score=args.min_stability)

        # 收集验证集指标
        finder.collect_validation_metrics(stable_models)

        # 排序显示
        finder.rank_by_metric(args.metric, top_n=args.top)

    elif args.command == 'find-best':
        # 筛选稳定模型
        stable_models = finder.filter_stable_models(min_score=args.min_stability)

        # 收集验证集指标
        finder.collect_validation_metrics(stable_models)

        # 找到最优模型
        best_model = finder.find_best_stable_model(args.metric, min_stability=args.min_stability)

        if best_model:
            print("最优模型信息已保存")

    elif args.command == 'report':
        # 筛选稳定模型
        stable_models = finder.filter_stable_models(min_score=args.min_stability)

        # 收集验证集指标
        finder.collect_validation_metrics(stable_models)

        # 生成JSON报告,Markdown报告
        finder.generate_best_models_report(min_stability=args.min_stability)


        print("完整分析报告生成完成")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
