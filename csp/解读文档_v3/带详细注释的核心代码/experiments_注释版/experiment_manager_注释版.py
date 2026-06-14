# -*- coding: utf-8 -*-
"""
==========================================================================================
实验生成器 ExperimentManager —— 逐行详细中文注释版（学习用）
对应原文件：EarnMore-main_v20260610/experiments/experiment_manager.py
==========================================================================================

【这份文件做什么】
它是大规模实验流水线的"第①步：生成实验"。把"一个模板 + 若干超参取值"展开成一批独立实验：
  每个实验 = experiments/exp_xxx/ 目录 + 一份可直接喂给 train.py 的 config.py。
后续由 scheduler.py 并发跑这些 config.py。

【核心思路】
  模板(json) 里有 fixed_parameters(固定) 和 variable_parameters(要搜索的)；
  对变量参数做"笛卡尔积"得到所有组合，每个组合 → 一个实验目录 + 一份 config.py。
  config.py 的生成靠"字符串精确替换"——把基础配置里的 lr=5e-5 等替换成本实验的取值。

【关键产出（每个 exp_xxx/）】
  config.py     训练用配置(scheduler 起 train.py 时 --config 指它)
  config.json   参数记录
  results.json  状态(created/running/completed/failed)
  logs/         指标 CSV 目录(监控器往里写)
==========================================================================================
"""
import os
import json
import itertools           # 用来对超参做笛卡尔积(所有组合)
import argparse
from datetime import datetime


def IsWin():
    """是否 Windows（影响实验命名：Win 上文件名不能太长/含特殊字符，故只取部分超参进名字）。"""
    if check_operating_system() == "Windows":
        return True
    else:
        return False


def check_operating_system():
    """判断当前是 Windows / Linux / 其它。"""
    os_name = os.name
    system_info = os.uname() if hasattr(os, 'uname') else None
    if os_name == 'nt':
        return "Windows"
    elif os_name == 'posix':
        if system_info and 'linux' in system_info.sysname.lower():
            return "Linux"
        else:
            return "Unix-like system ({})".format(system_info.sysname if system_info else 'unknown')
    else:
        return "Unknown operating system ({})".format(os_name)


class ExperimentManager:
    def __init__(self):
        self.templates_dir = "experiments/templates"     # 模板目录(json)
        self.experiments_dir = "experiments"             # 实验根目录(每个实验一个子目录)
        self.configs_dir = "experiments/configs"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保三个目录存在。"""
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)

    def list_templates(self):
        """列出 templates/ 下所有模板(json)及其 description。"""
        print("Available templates:")
        if not os.path.exists(self.templates_dir):
            print("No templates found.")
            return
        templates = [f for f in os.listdir(self.templates_dir) if f.endswith('.json')]
        if not templates:
            print("No templates found.")
            return
        for template in templates:
            template_name = template.replace('.json', '')
            template_path = os.path.join(self.templates_dir, template)
            with open(template_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            desc = config.get('description', 'No description')
            print("- {} ({})".format(template_name, desc))

    def show_template(self, template_name):
        """打印某模板的"固定参数"和"可变参数"。"""
        template_path = os.path.join(self.templates_dir, template_name + '.json')
        if not os.path.exists(template_path):
            print("Template {} not found.".format(template_name))
            return
        with open(template_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("Template: {}.json".format(template_name))
        print("Fixed parameters:")
        for key, value in config.get('fixed_parameters', {}).items():
            print("  {}: {}".format(key, value))
        print("Variable parameters:")
        for key, values in config.get('variable_parameters', {}).items():
            print("  {}: {}".format(key, values))

    def generate_configs(self, template_name, base_config=None, **kwargs):
        """★核心：基于模板 + 用户给的超参取值，生成一批实验（笛卡尔积展开）。

        例：generate_configs("mask_sac_base", lr="1e-4,5e-5", seed="123,456")
            → lr×seed = 2×2 = 4 个实验 exp_001..exp_004。
        """
        template_path = os.path.join(self.templates_dir, template_name + '.json')
        if not os.path.exists(template_path):
            print("Template {} not found.".format(template_name))
            return

        # 基础 Python 配置：所有实验都以它为蓝本，再替换各自的超参
        if base_config is None:
            base_config = "../configs/mask_sac_portfolio_management.py"
        if not os.path.exists(base_config):
            print("Base Python config not found: {}".format(base_config))
            return

        with open(template_path, 'r', encoding='utf-8') as f:
            template_config = json.load(f)
        fixed_params = template_config.get('fixed_parameters', {})
        var_params = template_config.get('variable_parameters', {})

        # 处理用户传入的超参：逗号分隔的字符串 → 列表（并做类型转换）
        param_combinations = {}
        for param, value_str in kwargs.items():
            if value_str is None:
                continue
            if param in var_params or param in fixed_params:
                if ',' in str(value_str):
                    param_combinations[param] = [self._convert_value(v.strip()) for v in str(value_str).split(',')]
                else:
                    param_combinations[param] = [self._convert_value(value_str)]

        param_names = list(param_combinations.keys())
        param_values = list(param_combinations.values())
        if not param_names:
            print("No variable parameters specified.")
            return

        # ★笛卡尔积：所有超参取值的组合
        combinations = list(itertools.product(*param_values))

        configs_generated = []
        for i, combination in enumerate(combinations, 1):
            exp_config = fixed_params.copy()         # 先放固定参数

            # 把本组合的各超参写进 exp_config，并拼实验名
            param_strs = []
            for param_name, param_value in zip(param_names, combination):
                exp_config[param_name] = param_value
                param_strs.append("{}_{}".format(param_name, param_value))

            # 实验命名：Windows 上只取 lr/batch_size/seed 进名字(避免名字过长/非法)
            if IsWin():
                win_param_strs = ["{}_{}".format(pn, pv) for pn, pv in zip(param_names, combination)
                                  if pn in ['lr', 'batch_size', 'seed']]
                exp_name = "exp_{:03d}_{}_{}".format(i, template_name, "_".join(win_param_strs))
            else:
                exp_name = "exp_{:03d}_{}_{}".format(i, template_name, "_".join(param_strs))

            # 建实验目录 + logs 子目录
            exp_dir = os.path.join(self.experiments_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

            # ★生成 config.py（字符串替换法，见 _generate_python_config）
            self._generate_python_config(base_config, exp_config, exp_name, template_name, exp_dir)

            # 写 config.json（参数记录）
            config_path = os.path.join(exp_dir, "config.json")
            exp_config['experiment_name'] = exp_name
            exp_config['created_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            exp_config['template'] = template_name
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(exp_config, f, indent=2, ensure_ascii=False)

            # 写 results.json（初始状态 = created）
            results_path = os.path.join(exp_dir, "results.json")
            initial_results = {
                "experiment_name": exp_name,
                "status": "created",
                "created_time": exp_config['created_time'],
                "template": template_name,
                "config": exp_config,
                "metrics": {}
            }
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(initial_results, f, indent=2, ensure_ascii=False)

            configs_generated.append(exp_name)

        print("Generated {} experiment configurations:".format(len(configs_generated)))
        for config_name in configs_generated:
            print("- {}".format(config_name))
        self._update_summary()        # 刷新 summary.json(花名册)
        return configs_generated

    def list_experiments(self):
        """列出所有实验及其状态（读各 exp_xxx/results.json）。"""
        if not os.path.exists(self.experiments_dir):
            print("No experiments found.")
            return
        experiments = []
        for exp_dir in os.listdir(self.experiments_dir):
            if exp_dir in ['templates', 'configs']:       # 跳过非实验目录
                continue
            exp_path = os.path.join(self.experiments_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue
            results_path = os.path.join(exp_path, "results.json")
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                experiments.append({
                    'id': exp_dir.split('_')[1] if '_' in exp_dir else '000',
                    'name': exp_dir,
                    'status': results.get('status', 'unknown'),
                    'template': results.get('template', 'unknown'),
                    'created': results.get('created_time', 'unknown')
                })
        if not experiments:
            print("No experiments found.")
            return
        experiments.sort(key=lambda x: x['id'])           # 按编号排序
        print("Total experiments: {}".format(len(experiments)))
        print("ID   Name                           Status    Template   Created")
        print("-" * 80)
        for exp in experiments:
            print("{:<4} {:<30} {:<9} {:<10} {}".format(
                exp['id'], exp['name'][:30], exp['status'],
                exp['template'], exp['created'].split()[0] if ' ' in exp['created'] else exp['created']))

    def show_experiments_tree(self):
        """以树状结构打印 experiments/ 目录（--tree 选项）。"""
        if not os.path.exists(self.experiments_dir):
            print("No experiments found.")
            return
        print("Experiments Directory Tree:")
        print("experiments/")
        exp_dirs = []
        for item in os.listdir(self.experiments_dir):
            if item in ['templates', 'configs']:
                continue
            if os.path.isdir(os.path.join(self.experiments_dir, item)):
                exp_dirs.append(item)
        exp_dirs.sort()
        for i, exp_dir in enumerate(exp_dirs):
            is_last = (i == len(exp_dirs) - 1)
            prefix = "└── " if is_last else "├── "
            print("{}{}".format(prefix, exp_dir))
            exp_path = os.path.join(self.experiments_dir, exp_dir)
            try:
                items = sorted(os.listdir(exp_path))
                for j, item in enumerate(items):
                    item_is_last = (j == len(items) - 1)
                    if is_last:
                        item_prefix = "    └── " if item_is_last else "    ├── "
                    else:
                        item_prefix = "│   └── " if item_is_last else "│   ├── "
                    item_path = os.path.join(exp_path, item)
                    print("{}{}{}".format(item_prefix, item, "/" if os.path.isdir(item_path) else ""))
            except PermissionError:
                pass

    def _convert_value(self, value_str):
        """把字符串值转成合适类型：true/false→bool，纯数字→int，否则尝试 float，再否则保持 str。"""
        value_str = str(value_str)
        if value_str.lower() == 'true':
            return True
        elif value_str.lower() == 'false':
            return False
        elif value_str.isdigit():
            return int(value_str)
        else:
            try:
                return float(value_str)
            except ValueError:
                return value_str

    def _update_summary(self):
        """汇总所有实验的 results.json 到 experiments/summary.json。"""
        summary_path = os.path.join(self.experiments_dir, "summary.json")
        experiments = []
        if os.path.exists(self.experiments_dir):
            for exp_dir in os.listdir(self.experiments_dir):
                if exp_dir in ['templates', 'configs']:
                    continue
                exp_path = os.path.join(self.experiments_dir, exp_dir)
                if not os.path.isdir(exp_path):
                    continue
                results_path = os.path.join(exp_path, "results.json")
                if os.path.exists(results_path):
                    with open(results_path, 'r', encoding='utf-8') as f:
                        experiments.append(json.load(f))
        summary = {
            "total_experiments": len(experiments),
            "updated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiments": experiments
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _generate_python_config(self, base_config_path, exp_config, exp_name, template_name, exp_dir):
        """★把基础配置 .py 复制一份，用"字符串精确替换"把超参改成本实验的取值，存为 exp_dir/config.py。

        ⚠️ 关键理解：这是靠"查找并替换固定写法的那一行"实现的！比如基础配置里必须**正好**有一行
           `lr = 5e-5`，才能被替换成 `lr = 1e-4`。所以改基础配置时，别破坏这些"锚点行"的写法，
           否则注参会悄悄失败（实验仍用默认值）。
        """
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_content = f.read()

        replacements = {}      # 旧字符串 → 新字符串 的替换表

        # 学习率 lr：基础配置里 lr/act_lr/cri_lr/rep_lr/beta_lr 都=5e-5，一起替换
        if 'lr' in exp_config:
            lr_val = exp_config['lr']
            replacements['lr = 5e-5'] = 'lr = {}'.format(lr_val)
            replacements['act_lr = 5e-5'] = 'act_lr = {}'.format(lr_val)
            replacements['cri_lr = 5e-5'] = 'cri_lr = {}'.format(lr_val)
            replacements['rep_lr = 5e-5'] = 'rep_lr = {}'.format(lr_val)
            replacements['beta_lr = 5e-5'] = 'beta_lr = {}'.format(lr_val)

        # batch_size：GPU 分支=32、CPU 分支=8，两种写法都替换
        if 'batch_size' in exp_config:
            bs_val = exp_config['batch_size']
            replacements['batch_size = 32'] = 'batch_size = {}'.format(bs_val)
            replacements['batch_size = 8'] = 'batch_size = {}'.format(bs_val)

        if 'seed' in exp_config:
            replacements['seed = 20258888'] = 'seed = {}'.format(exp_config['seed'])

        # embed_dim：同时改 embed_dim 和 decoder_embed_dim，GPU(128)/CPU(64) 两种写法
        if 'embed_dim' in exp_config:
            embed_val = exp_config['embed_dim']
            replacements['embed_dim = 128'] = 'embed_dim = {}'.format(embed_val)
            replacements['embed_dim = 64'] = 'embed_dim = {}'.format(embed_val)
            replacements['decoder_embed_dim = 128'] = 'decoder_embed_dim = {}'.format(embed_val)
            replacements['decoder_embed_dim = 64'] = 'decoder_embed_dim = {}'.format(embed_val)

        if 'num_stocks' in exp_config:
            stocks_val = exp_config['num_stocks']
            replacements['num_stocks = 1000'] = 'num_stocks = {}'.format(stocks_val)
            replacements['num_stocks = 5'] = 'num_stocks = {}'.format(stocks_val)

        if 'buffer_size' in exp_config:
            buffer_val = exp_config['buffer_size']
            replacements['buffer_size = 4096'] = 'buffer_size = {}'.format(buffer_val)
            replacements['buffer_size = 512'] = 'buffer_size = {}'.format(buffer_val)

        if 'num_episodes' in exp_config:
            episodes_val = exp_config['num_episodes']
            replacements['num_episodes = 200'] = 'num_episodes = {}'.format(episodes_val)
            replacements['num_episodes = 2'] = 'num_episodes = {}'.format(episodes_val)

        # 各日期窗口（滚动回测时由 hyperopt 算好后传进来）
        for key, anchor in [
            ('train_start_date', 'train_start_date = "2021-01-01"'),
            ('train_end_date',   'train_end_date = "2024-06-30"'),
            ('val_start_date',   'val_start_date = "2024-07-01"'),
            ('val_end_date',     'val_end_date = "2024-12-31"'),
            ('test_start_date',  'test_start_date = "2025-01-01"'),
            ('test_end_date',    'test_end_date = "2025-03-31"'),
        ]:
            if key in exp_config:
                left = anchor.split(' = ')[0]
                replacements[anchor] = '{} = "{}"'.format(left, exp_config[key])

        # workdir：指向本实验目录（注意实际路径是 experiments/experiments/exp_name）
        exp_relative_path = os.path.join("experiments", "experiments", exp_name)
        replacements['workdir = "workdir"'] = 'workdir = "{}"'.format(exp_relative_path)

        # tag：实验名。把基础配置里可能出现的各算法 tag 都替换成本实验名
        replacements['tag = "mask_sac"'] = 'tag = "{}"'.format(exp_name)
        for tag_val in ["sac", "dqn", "ddpg", "ppo", "td3", "mask_dqn", "mask_sync_dqn", "mask_ddqn", "mask_sync_sac"]:
            replacements['tag = "{}"'.format(tag_val)] = 'tag = "{}"'.format(exp_name)

        # 执行所有替换
        modified_content = base_content
        for old_value, new_value in replacements.items():
            modified_content = modified_content.replace(old_value, new_value)

        # 在文件头加一段"自动生成"注释，并去掉原文件重复的编码声明行
        header_comment = '''# -*- coding: utf-8 -*-
# 自动生成的实验配置文件
# 实验名称: {}
# 模板: {}
# 创建时间: {}
# 基于配置: {}

'''.format(exp_name, template_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), base_config_path)
        if modified_content.startswith('# -*- coding: utf-8 -*-'):
            modified_content = '\n'.join(modified_content.split('\n')[1:])
        final_content = header_comment + modified_content

        # 保存为 exp_dir/config.py（scheduler 起 train.py 时 --config 指它）
        py_config_path = os.path.join(exp_dir, "config.py")
        with open(py_config_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        print("Generated Python config: {}".format(py_config_path))


def main():
    # 命令行入口：python experiment_manager.py <子命令>
    parser = argparse.ArgumentParser(description='实验配置管理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    subparsers.add_parser('list-templates', help='查看可用模板')

    show_parser = subparsers.add_parser('show-template', help='查看模板详细内容')
    show_parser.add_argument('--name', required=True, help='模板名称')

    gen_parser = subparsers.add_parser('generate-configs', help='生成实验配置')
    gen_parser.add_argument('--template', required=True, help='模板名称')
    gen_parser.add_argument('--base-config', help='基础Python配置文件路径')
    gen_parser.add_argument('--lr', help='学习率，多个值用逗号分隔')
    gen_parser.add_argument('--batch_size', help='批大小，多个值用逗号分隔')
    gen_parser.add_argument('--embed_dim', help='嵌入维度，多个值用逗号分隔')
    gen_parser.add_argument('--seed', help='随机种子，多个值用逗号分隔')
    gen_parser.add_argument('--num_episodes', help='训练轮数，多个值用逗号分隔')

    list_parser = subparsers.add_parser('list-experiments', help='查看所有实验配置')
    list_parser.add_argument('--tree', action='store_true', help='以树状结构显示')

    args = parser.parse_args()
    manager = ExperimentManager()

    if args.command == 'list-templates':
        manager.list_templates()
    elif args.command == 'show-template':
        manager.show_template(args.name)
    elif args.command == 'generate-configs':
        kwargs = {}
        for k in ['lr', 'batch_size', 'embed_dim', 'seed', 'num_episodes']:
            v = getattr(args, k, None)
            if v:
                kwargs[k] = v
        manager.generate_configs(args.template, base_config=getattr(args, 'base_config', None), **kwargs)
    elif args.command == 'list-experiments':
        manager.show_experiments_tree() if args.tree else manager.list_experiments()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


# ==========================================================================================
# 【在流水线中的位置】
#   experiment_manager(本文件) 生成 exp_xxx/config.py
#        ↓
#   scheduler.py 扫描这些目录 → 为每个起一个 `python train.py --config exp_xxx/config.py` 子进程
#        ↓
#   train.py 训练 → monitor 写 logs/*.csv → best_model_finder 选最优
#
# 【最大的坑】_generate_python_config 用"字符串精确替换"注参：
#   基础配置里必须有"长得一模一样"的锚点行(如 `lr = 5e-5`、`workdir = "workdir"`)，
#   否则替换不上、实验会悄悄用默认值。改基础配置时务必保留这些锚点。
# ==========================================================================================
