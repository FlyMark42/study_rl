# -*- coding: utf-8 -*-
"""
超参数优化功能模块
该文件应该放在项目根目录下
"""
import math
import os
import json
import numpy as np
import pandas as pd
import argparse
import re
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

def _get_exp_dir(experiment_name=""):
    return os.path.join("experiments", experiment_name)

# 设置英文字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

class ParameterSpaceManager:
    """参数空间管理器"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "param_spaces.json")
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """加载参数空间配置"""
        if not os.path.exists(self.config_path):
            print("Parameter space config not found: {}".format(self.config_path))
            return {}
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def detect_environment(self):
        """检测硬件环境"""
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            return {
                "type": "gpu",
                "gpu_count": gpu_count,
                "gpu_name": gpu_name,
                "recommended_params": "gpu_environment"
            }
        else:
            return {
                "type": "cpu", 
                "recommended_params": "cpu_environment"
            }
    
    def get_parameter_space(self, env_type=None):
        """获取参数空间定义"""
        if env_type is None:
            env_info = self.detect_environment()
            env_type = env_info["recommended_params"]
        
        return self.config.get(env_type, {})
    
    def show_parameter_space(self, env_type=None):
        """显示参数空间信息"""
        env_info = self.detect_environment()
        print("Environment Detection:")
        print("Type: {}".format(env_info["type"]))
        if env_info["type"] == "gpu":
            print("GPU Count: {}".format(env_info["gpu_count"]))
            print("GPU Name: {}".format(env_info["gpu_name"]))
        
        space = self.get_parameter_space(env_type)
        if not space:
            print("No parameter space found for environment: {}".format(env_type))
            return
        
        print("\nParameter Space: {}".format(space.get("description", "Unknown")))
        
        if "priority_parameters" in space:
            print("\nPriority Parameters:")
            for param, config in space["priority_parameters"].items():
                print("- {}: {} - {}".format(param, self._format_param_range(config), config.get("description", "")))
        
        if "optional_parameters" in space:
            print("\nOptional Parameters:")  
            for param, config in space["optional_parameters"].items():
                print("- {}: {} - {}".format(param, self._format_param_range(config), config.get("description", "")))
    
    def _format_param_range(self, param_config):
        """格式化参数范围显示"""
        if param_config["type"] == "choice":
            return "choices: {}".format(param_config["choices"])
        elif param_config["type"] == "log-uniform":
            return "log-uniform [{}, {}]".format(param_config["low"], param_config["high"])
        elif param_config["type"] == "uniform":
            return "uniform [{}, {}]".format(param_config["low"], param_config["high"])
        else:
            return "unknown type"

class RandomSearchOptimizer:
    """随机搜索优化器"""
    
    def __init__(self, param_space_manager):
        self.param_manager = param_space_manager
        self.results_dir = "hyperopt_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def random_search(self, template_name, num_samples=50, env_type=None,
                     train_start_date=None, train_window_months=None,
                     val_window_months=None, slide_step_months=None, val_start_date=None):
        """执行随机搜索"""
        param_space = self.param_manager.get_parameter_space(env_type)
        if not param_space:
            print("No parameter space found")
            return

        # 合并优先参数和可选参数
        all_params = {}
        if "priority_parameters" in param_space:
            all_params.update(param_space["priority_parameters"])
        if "optional_parameters" in param_space:
            all_params.update(param_space["optional_parameters"])

        # 生成数据集划分方案
        date_splits = self._generate_date_splits(
            num_samples, train_start_date, train_window_months,
            val_window_months, slide_step_months, val_start_date
        )

        search_id = "random_{}_{}_{}".format(template_name, datetime.now().strftime("%Y%m%d_%H%M%S"), num_samples)
        search_history = []

        print("Starting random search: {}".format(search_id))
        print("Template: {}, Samples: {}".format(template_name, num_samples))
        if date_splits:
            print("Using sliding window: train_window={} months, val_window={} months, slide_step={} months".format(
                train_window_months, val_window_months, slide_step_months))

        for i in range(num_samples):
            # 随机采样参数
            sampled_params = self._sample_parameters(all_params)

            # 添加日期划分参数
            if date_splits and i < len(date_splits):
                sampled_params.update(date_splits[i])

            # 记录采样结果
            sample_record = {
                "sample_id": i + 1,
                "search_id": search_id,
                "template": template_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "generated"
            }
            sample_record.update(sampled_params)
            search_history.append(sample_record)
            
            # 显示进度
            # if (i + 1) % 10 == 0 or i == 0:
            print("Generated sample {}/{}: {}".format(i + 1, num_samples,
                      {k: v for k, v in sampled_params.items()}))
        
        # 保存搜索历史
        history_path = os.path.join(self.results_dir, "{}_history.csv".format(search_id))
        history_df = pd.DataFrame(search_history)
        history_df.to_csv(history_path, index=False)
        
        # 生成实验配置文件
        self._generate_experiment_configs(search_history, template_name)
        
        print("Random search completed!")
        print("Search ID: {}".format(search_id))
        print("History saved to: {}".format(history_path))
        print("Generated {} experiment configurations".format(num_samples))
        
        return search_id, search_history
    
    def _generate_date_splits(self, num_samples, train_start_date, train_window_months,
                              val_window_months, slide_step_months, val_start_date):
        """生成数据集划分方案"""
        if not train_start_date or not train_window_months or not val_window_months:
            return None

        from dateutil.relativedelta import relativedelta
        from datetime import datetime as dt

        date_splits = []
        base_train_start = dt.strptime(train_start_date, "%Y-%m-%d")

        for i in range(num_samples):
            if val_start_date:
                # 如果指定了验证集开始日期
                # 训练集开始日期随滑动步长移动
                train_start = base_train_start + relativedelta(months=i * slide_step_months)
                # 训练集结束日期也随滑动步长移动，保持训练窗口大小不变
                train_end = train_start + relativedelta(months=train_window_months) - relativedelta(days=1)
                # 验证集开始日期和结束日期固定不变
                base_val_start = dt.strptime(val_start_date, "%Y-%m-%d")
                val_start = base_val_start
                val_end = val_start + relativedelta(months=val_window_months) - relativedelta(days=1)
                # 检查训练集结束日期不能晚于或等于验证集开始日期
                assert train_end < val_start, "训练集结束日期({})晚于或等于验证集开始日期({})，请调整train_window_months或slide_step_months参数".format(
                    train_end.strftime("%Y-%m-%d"), val_start.strftime("%Y-%m-%d"))
                # 实际的训练窗口月数就是指定的train_window_months
                actual_train_months = train_window_months
            else:
                # 使用滑动窗口
                train_start = base_train_start + relativedelta(months=i * slide_step_months)
                train_end = train_start + relativedelta(months=train_window_months) - relativedelta(days=1)
                val_start = train_end + relativedelta(days=1)
                val_end = val_start + relativedelta(months=val_window_months) - relativedelta(days=1)
                actual_train_months = train_window_months

            date_splits.append({
                "train_start_date": train_start.strftime("%Y-%m-%d"),
                "train_end_date": train_end.strftime("%Y-%m-%d"),
                "val_start_date": val_start.strftime("%Y-%m-%d"),
                "val_end_date": val_end.strftime("%Y-%m-%d"),
                "test_start_date": None,
                "test_end_date": None
            })
            print("Sample {}: train[{} to {}], val[{} to {}]".format(
                    i + 1,
                    train_start.strftime("%Y-%m-%d"),
                    train_end.strftime("%Y-%m-%d"),
                    val_start.strftime("%Y-%m-%d"),
                    val_end.strftime("%Y-%m-%d")
                ))

        return date_splits

    def _sample_parameters(self, param_space):
        """采样参数值"""
        sampled = {}

        for param_name, param_config in param_space.items():
            param_type = param_config["type"]

            if param_type == "choice":
                sampled[param_name] = np.random.choice(param_config["choices"])
            elif param_type == "log-uniform":
                log_low = np.log10(param_config["low"])
                log_high = np.log10(param_config["high"])
                log_value = np.random.uniform(log_low, log_high)
                sampled[param_name] = 10 ** log_value
            elif param_type == "uniform":
                sampled[param_name] = np.random.uniform(param_config["low"], param_config["high"])
            else:
                print("Unknown parameter type: {}".format(param_type))

        return sampled
    
    def _generate_experiment_configs(self, search_history, template_name):
        """为每个采样结果生成实验配置"""
        from experiment_manager import ExperimentManager
        manager = ExperimentManager()

        for record in search_history:
            # 构建参数字典
            params = {}
            for key, value in record.items():
                if key not in ["sample_id", "search_id", "template", "timestamp", "target_metric", "status", "experiment_name"]:
                    params[key] = value

            # 生成实验配置
            try:
                experiment_names = manager.generate_configs(template_name, **params)
                if experiment_names and len(experiment_names) > 0:
                    record["experiment_name"] = experiment_names[0]
            except Exception as e:
                print("Failed to generate config for sample {}: {}".format(record["sample_id"], e))

class BayesianOptimizer:
    """贝叶斯优化器"""
    
    def __init__(self, param_space_manager):
        self.param_manager = param_space_manager
        self.results_dir = "hyperopt_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def bayesian_search(self, template_name, max_iterations=50, env_type=None,
                       target_metric="asr", acquisition="EI", resume_from=None,
                       train_start_date=None, train_window_months=None,
                       val_window_months=None, slide_step_months=None, val_start_date=None):
        """执行贝叶斯优化搜索"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            print("scikit-optimize not installed. Please install with: pip install scikit-optimize")
            return None, None

        param_space = self.param_manager.get_parameter_space(env_type)
        if not param_space:
            print("No parameter space found")
            return None, None

        # 构建优化空间
        dimensions, param_names = self._build_skopt_space(param_space)

        search_id = "bayesian_{}_{}_{}".format(template_name, datetime.now().strftime("%Y%m%d_%H%M%S"), max_iterations)

        print("Starting Bayesian optimization: {}".format(search_id))
        print("Template: {}, Max iterations: {}".format(template_name, max_iterations))
        print("Acquisition function: {}".format(acquisition))
        if train_start_date and train_window_months and val_window_months:
            print("Using sliding window: train_window={} months, val_window={} months, slide_step={} months".format(
                train_window_months, val_window_months, slide_step_months))

        # 定义目标函数
        @use_named_args(dimensions)
        def objective(**params):
            # 这里应该返回验证集ASR的负值（因为skopt最小化）
            # 在实际应用中，需要运行实验并获取结果
            # 暂时返回随机值作为示例
            return np.random.random()

        # 执行贝叶斯优化
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=max_iterations,
            n_initial_points=max(1, min(10, max_iterations // 5)),
            acq_func=acquisition,
            random_state=42
        )

        # 保存优化结果
        self._save_bayesian_results(result, param_names, search_id, template_name)

        # 生成实验配置文件
        self._generate_experiment_configs_from_bayesian(
            result, param_names, search_id, template_name,
            train_start_date, train_window_months, val_window_months,
            slide_step_months, val_start_date
        )

        print("Bayesian optimization completed!")
        print("Best parameters: {}".format(dict(zip(param_names, result.x))))
        print("Best score: {:.6f}".format(result.fun))

        return search_id, result

    def resume_from_search(self, search_id, template_name, train_start_date=None,
                          train_window_months=None, val_window_months=None,
                          slide_step_months=None, val_start_date=None):
        """从已有搜索恢复并继续优化"""
        print("恢复搜索: {}".format(search_id))

        # 加载搜索历史
        history_path = os.path.join(self.results_dir, "{}_history.csv".format(search_id))
        if not os.path.exists(history_path):
            print("错误：搜索历史文件不存在: {}".format(history_path))
            return None

        # 查找并更新实验状态
        updated_experiments = self._update_experiment_status(search_id)

        # 检查是否有新完成的实验
        completed_experiments = [exp for exp in updated_experiments if exp['status'] == 'completed' and not exp.get('processed', False)]

        if not completed_experiments:
            print("没有新完成的实验需要处理")
            self._show_search_status(search_id, updated_experiments)
            return None

        print("发现 {} 个新完成的实验".format(len(completed_experiments)))

        # 提取ASR结果并更新到实验记录和CSV
        experiments_with_results = []
        for exp in completed_experiments:
            asr_result = self._extract_asr_from_experiment(exp['experiment_name'])
            if asr_result is not None:
                exp['asr_result'] = asr_result
                exp['processed'] = True
                experiments_with_results.append(exp)
                print("实验 {} ASR: {:.4f}".format(exp['experiment_name'], asr_result))

        if not experiments_with_results:
            print("无法从完成的实验中提取有效的ASR结果")
            return None

        # 更新CSV文件，保存asr_result
        self._update_search_history_with_results(search_id, updated_experiments)

        # 构建新结果用于贝叶斯优化
        new_results = []
        for exp in experiments_with_results:
            params = {}
            for key, value in exp.items():
                if key not in ['iteration', 'search_id', 'template', 'timestamp', 'status', 'processed', 'experiment_name', 'asr_result', 'is_best']:
                    params[key] = value
            new_results.append((params, -exp['asr_result']))  # 负值用于最小化

        # 生成新的实验配置
        new_search_id = self._generate_next_experiments(
            search_id, template_name, new_results,
            train_start_date, train_window_months, val_window_months,
            slide_step_months, val_start_date
        )

        print("成功生成新搜索: {}".format(new_search_id))
        return new_search_id

    def process_all_searches(self, search_pattern="bayesian_*", template_name="mask_sac"):
        """批量处理所有贝叶斯优化搜索"""
        import glob

        # 查找所有匹配的搜索历史文件
        pattern_path = os.path.join(self.results_dir, "{}_history.csv".format(search_pattern))
        history_files = glob.glob(pattern_path)

        if not history_files:
            print("没有找到匹配的搜索历史文件: {}".format(pattern_path))
            return

        print("找到 {} 个搜索历史文件".format(len(history_files)))

        processed_count = 0
        for history_file in history_files:
            # 从文件名提取search_id
            filename = os.path.basename(history_file)
            search_id = filename.replace("_history.csv", "")

            print("\n" + "="*60)
            print("处理搜索: {}".format(search_id))

            try:
                result = self.resume_from_search(search_id, template_name)
                if result:
                    processed_count += 1
            except Exception as e:
                print("处理搜索 {} 时出错: {}".format(search_id, e))

        print("\n" + "="*60)
        print("批量处理完成，成功处理 {} 个搜索".format(processed_count))

    def _update_experiment_status(self, search_id):
        """更新搜索中所有实验的状态"""
        history_path = os.path.join(self.results_dir, "{}_history.csv".format(search_id))

        try:
            df = pd.read_csv(history_path)

            # 确保必要的字段存在
            if 'experiment_name' not in df.columns:
                df['experiment_name'] = ''
            if 'processed' not in df.columns:
                df['processed'] = False
            if 'asr_result' not in df.columns:
                df['asr_result'] = None

            experiments = df.to_dict('records')

            for exp in experiments:
                if 'experiment_name' not in exp or pd.isna(exp['experiment_name']) or exp['experiment_name'] == '':
                    # 如果没有实验名称，根据参数生成
                    exp['experiment_name'] = self._generate_experiment_name_from_params(exp, search_id)

                # 检查实验状态
                current_status = self._check_experiment_status(exp['experiment_name'])
                exp['status'] = current_status

                # 确保processed字段存在
                if 'processed' not in exp:
                    exp['processed'] = False

                # 确保asr_result字段存在
                if 'asr_result' not in exp:
                    exp['asr_result'] = None

            # 更新CSV文件
            updated_df = pd.DataFrame(experiments)
            updated_df.to_csv(history_path, index=False)

            return experiments

        except Exception as e:
            print("更新实验状态时出错: {}".format(e))
            return []

    def _check_experiment_status(self, experiment_name):
        """检查单个实验的状态"""
        exp_dir = _get_exp_dir(experiment_name)

        if not os.path.exists(exp_dir):
            return "pending"

        # 检查是否有results.json文件（完成标志）
        results_file = os.path.join(exp_dir, "results.json")
        if os.path.exists(results_file):
            return "completed"

        # 检查是否有training_metrics.csv（运行中）
        metrics_file = os.path.join(exp_dir, "logs", "training_metrics.csv")
        if os.path.exists(metrics_file):
            return "running"

        return "pending"

    def _extract_asr_from_experiment(self, experiment_name):
        """从实验中提取ASR结果"""
        metrics_path = os.path.join(_get_exp_dir(experiment_name), "logs", "training_metrics.csv")

        if not os.path.exists(metrics_path):
            print("指标文件不存在: {}".format(metrics_path))
            return None

        try:
            df = pd.read_csv(metrics_path)
            if 'validation_asr' in df.columns:
                asr_values = df['validation_asr'].dropna()
                if len(asr_values) > 0:
                    return float(asr_values.iloc[-1])#取serials最后一个

            print("在指标文件中未找到validation_asr列")
            return None

        except Exception as e:
            print("提取ASR失败: {}".format(e))
            return None

    def _generate_experiment_name_from_params(self, exp_record, search_id):
        """根据参数生成实验名称"""
        # 提取关键参数
        key_params = []
        iteration = exp_record.get('iteration', 'unknown')

        # 添加搜索ID前缀
        base_name = "bayes_{}".format(search_id.split('_')[-2] if '_' in search_id else search_id[:8])
        key_params.append("iter_{}".format(iteration))

        # 添加主要参数
        for param_name in ['lr', 'learning_rate', 'batch_size', 'embed_dim']:
            if param_name in exp_record and not pd.isna(exp_record[param_name]):
                value = exp_record[param_name]
                if isinstance(value, float):
                    key_params.append("{}_{:.0e}".format(param_name, value))
                else:
                    key_params.append("{}_{}".format(param_name, value))

        # 添加种子
        seed = exp_record.get('seed', 3407)
        key_params.append("seed_{}".format(seed))

        return "_".join([base_name] + key_params)

    def _generate_next_experiments(self, old_search_id, template_name, new_results,
                                   train_start_date=None, train_window_months=None,
                                   val_window_months=None, slide_step_months=None, val_start_date=None):
        """基于新结果生成下一批实验"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args

            # 加载参数空间
            param_space = self.param_manager.get_parameter_space()
            dimensions, param_names = self._build_skopt_space(param_space)

            # 加载历史数据
            history_path = os.path.join(self.results_dir, "{}_history.csv".format(old_search_id))
            df = pd.read_csv(history_path)

            # 构建历史X, y数据用于贝叶斯优化
            X_history = []
            y_history = []

            for _, row in df.iterrows():
                # 检查是否有asr_result且不为空
                if 'asr_result' in row and not pd.isna(row['asr_result']) and row['asr_result'] != '':
                    row['asr_result'] = pd.to_numeric(row['asr_result'], errors='coerce')#有可能是字符串和无穷大无穷小，健壮点
                    if(pd.isna(row['asr_result'])):
                        print("从历史CSV文件中找到 {} 对应ASR结果是NaN,忽略这次实验".format(row['experiment_name']))
                        continue
                    if(math.isinf(row['asr_result'])):
                        print("从历史CSV文件中找到 {} 对应ASR结果是inf,忽略这次实验".format(row['experiment_name']))
                        continue
                    asr_result=row['asr_result']
                    x_point = []
                    for param_name in param_names:
                        if param_name in row and not pd.isna(row[param_name]):
                            x_point.append(row[param_name])
                        else:
                            # 使用默认值
                            x_point.append(0.001 if 'lr' in param_name else 64)

                    if len(x_point) == len(param_names):
                        X_history.append(x_point)
                        y_history.append(-asr_result)  # 负值用于最小化

            print("从历史CSV文件中找到 {} 个有效的ASR结果".format(len(X_history)))

            if len(X_history) < 1:
                print("历史数据点不足，需要至少1个完成的实验")
                return None

            # 创建新的搜索ID
            new_search_id = "bayesian_{}_{}_{}".format(
                template_name,
                datetime.now().strftime("%Y%m%d_%H%M%S"),
                "continue"
            )

            # 基于历史数据继续优化，生成3-5个新点
            n_new_points = min(5, max(3, len(new_results)))

            # 模拟贝叶斯优化过程
            result = gp_minimize(
                func=lambda x: 0,
                dimensions=dimensions,
                n_calls=len(X_history) + n_new_points,
                x0=X_history,
                y0=y_history,
                n_initial_points=0,
                acq_func="EI"
            )

            # 获取新建议的点
            new_points = result.x_iters[-n_new_points:]

            # 生成日期划分方案（复用RandomSearchOptimizer的方法）
            date_splits = self._generate_date_splits_for_bayesian(
                n_new_points, len(X_history), train_start_date, train_window_months,
                val_window_months, slide_step_months, val_start_date
            )

            # 生成新实验配置
            new_experiments = []
            for i, x_point in enumerate(new_points):
                params = dict(zip(param_names, x_point))

                # 添加日期划分参数
                if date_splits and i < len(date_splits):
                    params.update(date_splits[i])

                # 生成实验配置并获取实际的实验名称
                from experiment_manager import ExperimentManager
                manager = ExperimentManager()
                experiment_names = manager.generate_configs(template_name, **params)

                # 获取实际生成的实验名称
                if experiment_names and len(experiment_names) > 0:
                    actual_experiment_name = experiment_names[0]
                else:
                    print("警告：无法获取实验名称，使用默认命名")
                    actual_experiment_name = "exp_unknown_{}".format(i + 1)

                # 记录实验信息
                experiment_record = {
                    "iteration": len(X_history) + i + 1,
                    "search_id": new_search_id,
                    "template": template_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "pending",
                    "processed": False,
                    "parent_search": old_search_id,
                    "experiment_name": actual_experiment_name  # 使用实际的实验名称
                }
                experiment_record.update(params)
                new_experiments.append(experiment_record)

            # 保存新搜索历史
            new_history_path = os.path.join(self.results_dir, "{}_history.csv".format(new_search_id))
            new_df = pd.DataFrame(new_experiments)
            new_df.to_csv(new_history_path, index=False)

            print("生成了 {} 个新实验配置".format(len(new_experiments)))
            return new_search_id

        except Exception as e:
            print("生成新实验时出错: {}".format(e))
            return None

    def _update_search_history_with_results(self, search_id, experiments):
        """更新搜索历史文件，包含ASR结果"""
        history_path = os.path.join(self.results_dir, "{}_history.csv".format(search_id))

        try:
            # 创建DataFrame并确保包含所有必要字段
            df = pd.DataFrame(experiments)

            # 确保asr_result列存在
            if 'asr_result' not in df.columns:
                df['asr_result'] = None

            df.to_csv(history_path, index=False)
            print("已更新搜索历史，包含ASR结果: {}".format(history_path))
        except Exception as e:
            print("更新搜索历史时出错: {}".format(e))

    def _update_search_history(self, search_id, updated_experiments):
        """更新搜索历史文件"""
        history_path = os.path.join(self.results_dir, "{}_history.csv".format(search_id))

        try:
            df = pd.DataFrame(updated_experiments)
            df.to_csv(history_path, index=False)
            print("已更新搜索历史: {}".format(history_path))
        except Exception as e:
            print("更新搜索历史时出错: {}".format(e))

    def _show_search_status(self, search_id, experiments):
        """显示搜索状态信息"""
        print("\n搜索状态报告: {}".format(search_id))
        print("-" * 50)

        status_counts = {}
        for exp in experiments:
            status = exp.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        for status, count in status_counts.items():
            print("{}: {} 个实验".format(status, count))

        # 显示运行中的实验
        running_exps = [exp for exp in experiments if exp.get('status') == 'running']
        if running_exps:
            print("\n运行中的实验:")
            for exp in running_exps[:5]:  # 只显示前5个
                print("- {}".format(exp.get('experiment_name', 'unknown')))

        # 显示已完成但未处理的实验
        completed_unprocessed = [exp for exp in experiments
                               if exp.get('status') == 'completed' and not exp.get('processed', False)]
        if completed_unprocessed:
            print("\n已完成但未处理的实验:")
            for exp in completed_unprocessed:
                print("- {}".format(exp.get('experiment_name', 'unknown')))

    def _generate_date_splits_for_bayesian(self, num_samples, history_count, train_start_date,
                                           train_window_months, val_window_months, slide_step_months, val_start_date):
        """为贝叶斯优化生成日期划分方案"""
        if not train_start_date or not train_window_months or not val_window_months:
            return None

        from dateutil.relativedelta import relativedelta
        from datetime import datetime as dt

        date_splits = []
        base_train_start = dt.strptime(train_start_date, "%Y-%m-%d")

        # 从历史实验数量开始计数，确保日期连续
        for i in range(num_samples):
            actual_index = history_count + i

            if val_start_date:
                # 如果指定了验证集开始日期
                # 训练集开始日期随滑动步长移动
                train_start = base_train_start + relativedelta(months=actual_index * slide_step_months)
                # 训练集结束日期也随滑动步长移动，保持训练窗口大小不变
                train_end = train_start + relativedelta(months=train_window_months) - relativedelta(days=1)
                # 验证集开始日期和结束日期固定不变
                base_val_start = dt.strptime(val_start_date, "%Y-%m-%d")
                val_start = base_val_start
                val_end = val_start + relativedelta(months=val_window_months) - relativedelta(days=1)
                # 检查训练集结束日期不能晚于或等于验证集开始日期
                assert train_end < val_start, "训练集结束日期({})晚于或等于验证集开始日期({})，请调整train_window_months或slide_step_months参数".format(
                    train_end.strftime("%Y-%m-%d"), val_start.strftime("%Y-%m-%d"))
                # 实际的训练窗口月数就是指定的train_window_months
                actual_train_months = train_window_months
            else:
                # 使用滑动窗口
                train_start = base_train_start + relativedelta(months=actual_index * slide_step_months)
                train_end = train_start + relativedelta(months=train_window_months) - relativedelta(days=1)
                val_start = train_end + relativedelta(days=1)
                val_end = val_start + relativedelta(months=val_window_months) - relativedelta(days=1)
                actual_train_months = train_window_months

            date_splits.append({
                "train_start_date": train_start.strftime("%Y-%m-%d"),
                "train_end_date": train_end.strftime("%Y-%m-%d"),
                "val_start_date": val_start.strftime("%Y-%m-%d"),
                "val_end_date": val_end.strftime("%Y-%m-%d"),
                "test_start_date": None,
                "test_end_date": None
            })

            print("Bayesian Sample {}: train[{} to {}], val[{} to {}]".format(
                actual_index + 1,
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                val_start.strftime("%Y-%m-%d"),
                val_end.strftime("%Y-%m-%d")
            ))

        return date_splits

    def _build_skopt_space(self, param_space):
        """构建scikit-optimize的搜索空间"""
        from skopt.space import Real, Integer, Categorical
        
        dimensions = []
        param_names = []
        
        # 合并优先参数和可选参数
        all_params = {}
        if "priority_parameters" in param_space:
            all_params.update(param_space["priority_parameters"])
        if "optional_parameters" in param_space:
            all_params.update(param_space["optional_parameters"])
        
        for param_name, param_config in all_params.items():
            param_type = param_config["type"]
            
            if param_type == "choice":
                if all(isinstance(x, int) for x in param_config["choices"]):
                    dimensions.append(Categorical(param_config["choices"], name=param_name))
                else:
                    dimensions.append(Categorical(param_config["choices"], name=param_name))
            elif param_type == "log-uniform":
                dimensions.append(Real(param_config["low"], param_config["high"], 
                                     prior='log-uniform', name=param_name))
            elif param_type == "uniform":
                dimensions.append(Real(param_config["low"], param_config["high"], name=param_name))
            
            param_names.append(param_name)
        
        return dimensions, param_names
    
    def _save_bayesian_results(self, result, param_names, search_id, template_name):
        """保存贝叶斯优化结果"""
        # 找到最优结果的索引
        best_idx = np.argmin(result.func_vals)

        # 保存优化历史
        history_data = []
        for i, (x, y) in enumerate(zip(result.x_iters, result.func_vals)):
            record = {
                "iteration": i + 1,
                "search_id": search_id,
                "template": template_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "objective_value": y,
                "status": "evaluated",
                "is_best": (i == best_idx),
                "experiment_name": ""  # 初始为空，在生成实验后填充
            }
            for param_name, param_value in zip(param_names, x):
                record[param_name] = param_value
            history_data.append(record)

        history_path = os.path.join(self.results_dir, "{}_history.csv".format(search_id))
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(history_path, index=False)

        print("Optimization history saved to: {}".format(history_path))

    def _generate_experiment_configs_from_bayesian(self, result, param_names, search_id, template_name,
                                                   train_start_date=None, train_window_months=None,
                                                   val_window_months=None, slide_step_months=None, val_start_date=None):
        """从贝叶斯优化结果生成实验配置"""
        from experiment_manager import ExperimentManager
        manager = ExperimentManager()

        # 加载现有历史文件
        history_path = os.path.join(self.results_dir, "{}_history.csv".format(search_id))
        df = pd.read_csv(history_path)

        # 生成日期划分方案
        date_splits = self._generate_date_splits_for_bayesian(
            len(result.x_iters), 0, train_start_date, train_window_months,
            val_window_months, slide_step_months, val_start_date
        )

        # 为每次优化迭代生成配置
        generated_count = 0
        for i, x in enumerate(result.x_iters):
            # 构建参数字典
            params = dict(zip(param_names, x))

            # 添加日期划分参数
            if date_splits and i < len(date_splits):
                params.update(date_splits[i])

            # 生成实验配置
            try:
                # 调用generate_configs并获取实际生成的实验名称
                experiment_names = manager.generate_configs(template_name, **params)

                if experiment_names and len(experiment_names) > 0:
                    # 使用第一个生成的实验名称（通常只有一个）
                    experiment_name = experiment_names[0]
                    generated_count += 1

                    # 更新CSV中对应行的experiment_name
                    df.loc[i, 'experiment_name'] = experiment_name
                else:
                    print("No experiment name returned for iteration {}".format(i + 1))

            except Exception as e:
                print("Failed to generate config for iteration {}: {}".format(i + 1, e))

        # 保存更新后的历史文件
        df.to_csv(history_path, index=False)

        print("Generated {} experiment configurations from Bayesian optimization".format(generated_count))

class StabilityAnalyzer:
    """多种子稳定性评估器"""
    
    def __init__(self):
        self.experiments_dir = _get_exp_dir()
        self.results_dir = "hyperopt_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_stability(self, config_pattern=None, min_seeds=3):
        """分析多种子实验的稳定性"""
        if not os.path.exists(self.experiments_dir):
            print("Experiments directory not found")
            return
        
        # 收集实验结果
        experiment_groups = self._group_experiments_by_config(config_pattern)
        
        if not experiment_groups:
            print("No experiment groups found matching pattern: {}".format(config_pattern))
            return
        
        stability_results = []
        
        print("Analyzing stability for {} experiment groups...".format(len(experiment_groups)))
        
        for config_key, experiments in experiment_groups.items():
            if len(experiments) < min_seeds:
                print("Skipping {} (only {} seeds, need at least {})".format(
                    config_key, len(experiments), min_seeds))
                continue
            
            # 收集这组实验的ASR结果
            asr_values = []
            for exp_name, exp_data in experiments.items():
                asr = self._extract_final_asr(exp_data)
                if asr is not None:
                    asr_values.append(asr)
            
            if len(asr_values) < min_seeds:
                print("Skipping {} (insufficient ASR data)".format(config_key))
                continue
            
            # 计算稳定性统计量
            asr_array = np.array(asr_values)
            stability_stats = {
                "config": config_key,
                "num_seeds": len(asr_values),
                "mean_asr": np.mean(asr_array),
                "std_asr": np.std(asr_array),
                "var_asr": np.var(asr_array),
                "min_asr": np.min(asr_array),
                "max_asr": np.max(asr_array),
                "range_asr": np.max(asr_array) - np.min(asr_array),
                "cv_asr": np.std(asr_array) / np.mean(asr_array) if np.mean(asr_array) != 0 else np.inf
            }
            
            # 评估稳定性等级
            stability_stats["stability_level"] = self._assess_stability_level(stability_stats["var_asr"])
            stability_results.append(stability_stats)
            
            print("Config: {} | Seeds: {} | Mean ASR: {:.4f} ± {:.4f} | Level: {}".format(
                config_key, len(asr_values), stability_stats["mean_asr"], 
                stability_stats["std_asr"], stability_stats["stability_level"]))
        
        if not stability_results:
            print("No valid stability results found")
            return
        
        # 保存稳定性分析结果
        results_df = pd.DataFrame(stability_results)
        results_path = os.path.join(self.results_dir, "stability_analysis_{}.csv".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")))
        results_df.to_csv(results_path, index=False)
        
        # 生成稳定性报告
        self._generate_stability_report(results_df)
        
        print("Stability analysis completed!")
        print("Results saved to: {}".format(results_path))
        
        return results_df
    
    def _group_experiments_by_config(self, config_pattern=None):
        """按配置参数分组实验"""
        experiment_groups = defaultdict(dict)
        
        if not os.path.exists(self.experiments_dir):
            return {}
        
        for exp_dir in os.listdir(self.experiments_dir):
            if exp_dir in ['templates', 'configs']:
                continue
            
            exp_path = os.path.join(self.experiments_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue
            
            # 如果指定了模式，检查是否匹配
            if config_pattern and config_pattern not in exp_dir:
                continue
            
            # 读取实验配置
            results_path = os.path.join(exp_path, "results.json")
            if not os.path.exists(results_path):
                continue
            
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    exp_data = json.load(f)
                
                # 提取配置关键信息（排除seed）
                config_key = self._extract_config_key(exp_dir, exp_data)
                experiment_groups[config_key][exp_dir] = exp_data
                
            except Exception as e:
                print("Error reading {}: {}".format(results_path, e))
                continue
        
        return dict(experiment_groups)
    
    def _extract_config_key(self, exp_name, exp_data):
        """从实验名称和数据中提取配置关键信息"""
        # 从实验名称中提取参数（排除seed）
        parts = exp_name.split('_')
        config_parts = []
        
        skip_next = False
        for i, part in enumerate(parts):
            if skip_next:
                skip_next = False
                continue
            
            if part == "seed" and i + 1 < len(parts):
                skip_next = True  # 跳过seed值
                continue
            
            config_parts.append(part)
        
        return "_".join(config_parts)
    
    def _extract_final_asr(self, exp_data):
        """从实验数据中提取最终ASR值"""
        # 尝试从不同位置提取ASR值
        if "metrics" in exp_data and "final_asr" in exp_data["metrics"]:
            return exp_data["metrics"]["final_asr"]
        
        if "config" in exp_data and "final_performance" in exp_data["config"]:
            return exp_data["config"]["final_performance"].get("asr")
        
        # 可以添加更多提取逻辑
        return None
    
    def _assess_stability_level(self, variance):
        """评估稳定性等级"""
        if variance < 0.01:
            return "High"
        elif variance < 0.05:
            return "Medium"
        else:
            return "Low"
    
    def _generate_stability_report(self, results_df):
        """生成稳定性分析报告"""
        print("\n" + "="*80)
        print("STABILITY ANALYSIS REPORT")
        print("="*80)
        
        # 按稳定性排序
        sorted_df = results_df.sort_values('var_asr')
        
        print("TOP 5 Most Stable Configurations:")
        for i, row in sorted_df.head(5).iterrows():
            print("{}. {} | Mean ASR: {:.4f} ± {:.4f} | Level: {}".format(
                i+1, row['config'][:50], row['mean_asr'], row['std_asr'], row['stability_level']))
        
        print("\nStability Level Distribution:")
        level_counts = results_df['stability_level'].value_counts()
        for level, count in level_counts.items():
            print("{}: {} configurations".format(level, count))
        
        print("\nRecommendation:")
        best_config = sorted_df.iloc[0]
        print("Most reliable configuration: {}".format(best_config['config']))
        print("Expected ASR: {:.4f} ± {:.4f}".format(best_config['mean_asr'], best_config['std_asr']))
        
        # 生成稳定性图表
        self._plot_stability_analysis(results_df)
    
    def _plot_stability_analysis(self, results_df):
        """绘制稳定性分析图表"""
        if len(results_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Seed Stability Analysis', fontsize=16)
        
        # 1. ASR均值 vs 方差散点图
        ax1 = axes[0, 0]
        colors = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
        for level in results_df['stability_level'].unique():
            subset = results_df[results_df['stability_level'] == level]
            ax1.scatter(subset['mean_asr'], subset['var_asr'], 
                       c=colors[level], label=level, alpha=0.7, s=60)
        
        ax1.set_xlabel('Mean ASR')
        ax1.set_ylabel('ASR Variance')
        ax1.set_title('Stability vs Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 稳定性等级分布饼图
        ax2 = axes[0, 1]
        level_counts = results_df['stability_level'].value_counts()
        ax2.pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%',
                colors=[colors[level] for level in level_counts.index])
        ax2.set_title('Stability Level Distribution')
        
        # 3. ASR分布直方图
        ax3 = axes[1, 0]
        ax3.hist(results_df['mean_asr'], bins=min(20, len(results_df)), 
                alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Mean ASR')
        ax3.set_ylabel('Frequency')
        ax3.set_title('ASR Performance Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Top配置的ASR范围
        ax4 = axes[1, 1]
        top_configs = results_df.nsmallest(min(10, len(results_df)), 'var_asr')
        y_pos = range(len(top_configs))
        
        for i, (_, row) in enumerate(top_configs.iterrows()):
            ax4.barh(i, row['range_asr'], left=row['min_asr'], 
                    alpha=0.7, color=colors[row['stability_level']])
            ax4.plot(row['mean_asr'], i, 'ko', markersize=4)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([config[:20] + '...' if len(config) > 20 else config 
                            for config in top_configs['config']], fontsize=8)
        ax4.set_xlabel('ASR Range')
        ax4.set_title('Top Stable Configurations ASR Range')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.results_dir, "stability_analysis_{}.png".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")))
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Stability analysis plot saved to: {}".format(plot_path))

def main():
    parser = argparse.ArgumentParser(description='超参数优化工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # detect-environment命令
    subparsers.add_parser('detect-environment', help='检测硬件环境')
    
    # show-space命令
    space_parser = subparsers.add_parser('show-space', help='显示参数空间')
    space_parser.add_argument('--env', choices=['cpu', 'gpu'], help='环境类型')
    
    # random-search命令
    random_parser = subparsers.add_parser('random-search', help='随机搜索优化')
    random_parser.add_argument('--template', required=True, help='模板名称')
    random_parser.add_argument('--num-samples', type=int, default=50, help='采样数量')
    random_parser.add_argument('--env', choices=['cpu', 'gpu'], help='环境类型')
    random_parser.add_argument('--train-start-date', type=str, help='训练集开始日期，格式：YYYY-MM-DD')
    random_parser.add_argument('--train-window-months', type=int, help='训练集窗口大小（月）')
    random_parser.add_argument('--val-window-months', type=int, help='验证集窗口大小（月）')
    random_parser.add_argument('--slide-step-months', type=int, required=True, help='滑动步长（月），必须指定')
    random_parser.add_argument('--val-start-date', type=str, help='验证集开始日期（可选，如果指定则忽略滑动窗口）')
    
    # bayesian-search命令
    bayes_parser = subparsers.add_parser('bayesian-search', help='贝叶斯优化搜索')
    bayes_parser.add_argument('--template', required=True, help='模板名称')
    bayes_parser.add_argument('--max-iter', type=int, default=50, help='最大迭代次数')
    bayes_parser.add_argument('--env', choices=['cpu', 'gpu'], help='环境类型')
    bayes_parser.add_argument('--acquisition', default='EI', choices=['EI', 'PI', 'LCB'], help='获取函数')
    bayes_parser.add_argument('--resume-from', help='从已有搜索继续')
    bayes_parser.add_argument('--train-start-date', type=str, help='训练集开始日期，格式：YYYY-MM-DD')
    bayes_parser.add_argument('--train-window-months', type=int, help='训练集窗口大小（月）')
    bayes_parser.add_argument('--val-window-months', type=int, help='验证集窗口大小（月）')
    bayes_parser.add_argument('--slide-step-months', type=int, required=True, help='滑动步长（月），必须指定')
    bayes_parser.add_argument('--val-start-date', type=str, help='验证集开始日期（可选，如果指定则忽略滑动窗口）')

    # resume-from命令
    resume_parser = subparsers.add_parser('resume-from', help='从已有搜索恢复并继续优化')
    resume_parser.add_argument('--search-id', required=True, help='搜索ID')
    resume_parser.add_argument('--template', default='mask_sac', help='模板名称')
    resume_parser.add_argument('--train-start-date', type=str, help='训练集开始日期，格式：YYYY-MM-DD')
    resume_parser.add_argument('--train-window-months', type=int, help='训练集窗口大小（月）')
    resume_parser.add_argument('--val-window-months', type=int, help='验证集窗口大小（月）')
    resume_parser.add_argument('--slide-step-months', type=int, required=True, help='滑动步长（月），必须指定')
    resume_parser.add_argument('--val-start-date', type=str, help='验证集开始日期（可选，如果指定则忽略滑动窗口）')

    # process-all-searches命令
    process_parser = subparsers.add_parser('process-all-searches', help='批量处理所有贝叶斯优化搜索')
    process_parser.add_argument('--search-pattern', default='bayesian_*', help='搜索模式匹配')
    process_parser.add_argument('--template', default='mask_sac', help='模板名称')

    # analyze-stability命令
    stability_parser = subparsers.add_parser('analyze-stability', help='多种子稳定性分析')
    stability_parser.add_argument('--config-pattern', help='配置模式匹配')
    stability_parser.add_argument('--min-seeds', type=int, default=3, help='最少种子数量')
    
    args = parser.parse_args()
    
    # 初始化管理器
    param_manager = ParameterSpaceManager()
    
    if args.command == 'detect-environment':
        env_info = param_manager.detect_environment()
        print("Environment Type: {}".format(env_info["type"]))
        if env_info["type"] == "gpu":
            print("GPU Count: {}".format(env_info["gpu_count"]))
            print("GPU Name: {}".format(env_info["gpu_name"]))
        print("Recommended Parameter Space: {}".format(env_info["recommended_params"]))
        
    elif args.command == 'show-space':
        env_type = args.env + "_environment" if args.env else None
        param_manager.show_parameter_space(env_type)
        
    elif args.command == 'random-search':
        optimizer = RandomSearchOptimizer(param_manager)
        env_type = args.env + "_environment" if args.env else None
        search_id, history = optimizer.random_search(
            template_name=args.template,
            num_samples=args.num_samples,
            env_type=env_type,
            train_start_date=args.train_start_date,
            train_window_months=args.train_window_months,
            val_window_months=args.val_window_months,
            slide_step_months=args.slide_step_months,
            val_start_date=args.val_start_date
        )
        
    elif args.command == 'bayesian-search':
        optimizer = BayesianOptimizer(param_manager)
        env_type = args.env + "_environment" if args.env else None

        # 如果指定了resume-from，先处理恢复逻辑
        if args.resume_from:
            result = optimizer.resume_from_search(args.resume_from, args.template)
            if result:
                print("已从搜索 {} 生成新的优化搜索: {}".format(args.resume_from, result))
        else:
            search_id, result = optimizer.bayesian_search(
                template_name=args.template,
                max_iterations=args.max_iter,
                env_type=env_type,
                acquisition=args.acquisition,
                resume_from=args.resume_from,
                train_start_date=args.train_start_date,
                train_window_months=args.train_window_months,
                val_window_months=args.val_window_months,
                slide_step_months=args.slide_step_months,
                val_start_date=args.val_start_date
            )

    elif args.command == 'resume-from':
        optimizer = BayesianOptimizer(param_manager)
        result = optimizer.resume_from_search(
            args.search_id, args.template,
            train_start_date=args.train_start_date,
            train_window_months=args.train_window_months,
            val_window_months=args.val_window_months,
            slide_step_months=args.slide_step_months,
            val_start_date=args.val_start_date
        )
        if result:
            print("成功从搜索 {} 生成新的优化搜索: {}".format(args.search_id, result))
        else:
            print("搜索 {} 没有新的实验可以处理".format(args.search_id))

    elif args.command == 'process-all-searches':
        optimizer = BayesianOptimizer(param_manager)
        optimizer.process_all_searches(args.search_pattern, args.template)

    elif args.command == 'analyze-stability':
        analyzer = StabilityAnalyzer()
        results_df = analyzer.analyze_stability(
            config_pattern=args.config_pattern,
            min_seeds=args.min_seeds
        )
        
    else:
        parser.print_help()

if __name__ == '__main__':
    main()