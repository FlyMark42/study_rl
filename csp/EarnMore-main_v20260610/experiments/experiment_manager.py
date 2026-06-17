# -*- coding: utf-8 -*-
"""
实验配置管理功能模块
该文件应该放在项目根目录下
"""
import os
import json
import itertools
import argparse
from datetime import datetime

def IsWin():
    if check_operating_system()=="Windows":
        return True
    else:
        return False

def check_operating_system():
    """判断当前运行环境是Windows还是Linux"""
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
        self.templates_dir = "experiments/templates"
        self.experiments_dir = "experiments"
        self.configs_dir = "experiments/configs"
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """确保必要目录存在"""
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
    
    def list_templates(self):
        """查看可用的配置模板"""
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
        """查看模板详细内容"""
        template_path = os.path.join(self.templates_dir, template_name + '.json')
        if not os.path.exists(template_path):
            print("Template {} not found.".format(template_name))
            return
        
        with open(template_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("Template: {}.json".format(template_name))
        print("Fixed parameters:")
        fixed_params = config.get('fixed_parameters', {})
        for key, value in fixed_params.items():
            print("  {}: {}".format(key, value))
        
        print("Variable parameters:")
        var_params = config.get('variable_parameters', {})
        for key, values in var_params.items():
            print("  {}: {}".format(key, values))
    
    def generate_configs(self, template_name, base_config=None, **kwargs):
        """基于模板生成多个实验配置"""
        template_path = os.path.join(self.templates_dir, template_name + '.json')
        if not os.path.exists(template_path):
            print("Template {} not found.".format(template_name))
            return
        
        # 检查基础Python配置文件
        if base_config is None:
            base_config = "../configs/mask_sac_portfolio_management.py"
        
        if not os.path.exists(base_config):
            print("Base Python config not found: {}".format(base_config))
            return
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template_config = json.load(f)
        
        fixed_params = template_config.get('fixed_parameters', {})
        var_params = template_config.get('variable_parameters', {})
        
        # 处理用户指定的参数
        param_combinations = {}
        for param, value_str in kwargs.items():
            if value_str is None:
                continue
            if param in var_params or param in fixed_params:
                if ',' in str(value_str):
                    param_combinations[param] = [self._convert_value(v.strip()) for v in str(value_str).split(',')]
                else:
                    param_combinations[param] = [self._convert_value(value_str)]
        
        # 生成所有参数组合
        param_names = list(param_combinations.keys())
        param_values = list(param_combinations.values())
        
        if not param_names:
            print("No variable parameters specified.")
            return
        
        combinations = list(itertools.product(*param_values))
        
        configs_generated = []
        for i, combination in enumerate(combinations, 1):
            # 创建实验配置
            exp_config = fixed_params.copy()
            
            # 生成实验名称
            param_strs = []
            for param_name, param_value in zip(param_names, combination):
                exp_config[param_name] = param_value
                param_strs.append("{}_{}".format(param_name, param_value))
            
            if IsWin():
                win_param_strs = ["{}_{}".format(pn, pv) for pn, pv in zip(param_names, combination) if pn in ['lr', 'batch_size', 'seed']]
                exp_name = "exp_{:03d}_{}_{}".format(i, template_name, "_".join(win_param_strs))
            else:
                exp_name = "exp_{:03d}_{}_{}".format(i, template_name, "_".join(param_strs))

            # 创建实验目录
            exp_dir = os.path.join(self.experiments_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
            
            # 生成Python配置文件
            self._generate_python_config(base_config, exp_config, exp_name, template_name, exp_dir)
            config_path = os.path.join(exp_dir, "config.json")
            exp_config['experiment_name'] = exp_name
            exp_config['created_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            exp_config['template'] = template_name
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(exp_config, f, indent=2, ensure_ascii=False)
            
            # 初始化结果文件
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

        # 更新汇总信息
        self._update_summary()

        return configs_generated
    
    def list_experiments(self):
        """查看所有生成的实验配置"""
        if not os.path.exists(self.experiments_dir):
            print("No experiments found.")
            return
        
        experiments = []
        for exp_dir in os.listdir(self.experiments_dir):
            if exp_dir in ['templates', 'configs']:
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
        
        # 按ID排序
        experiments.sort(key=lambda x: x['id'])
        
        print("Total experiments: {}".format(len(experiments)))
        print("ID   Name                           Status    Template   Created")
        print("-" * 80)
        for exp in experiments:
            print("{:<4} {:<30} {:<9} {:<10} {}".format(
                exp['id'], exp['name'][:30], exp['status'], 
                exp['template'], exp['created'].split()[0] if ' ' in exp['created'] else exp['created']
            ))
    
    def show_experiments_tree(self):
        """以树状结构显示实验组织"""
        if not os.path.exists(self.experiments_dir):
            print("No experiments found.")
            return
        
        print("Experiments Directory Tree:")
        print("experiments/")
        
        # 获取所有实验目录
        exp_dirs = []
        for item in os.listdir(self.experiments_dir):
            if item in ['templates', 'configs']:
                continue
            exp_path = os.path.join(self.experiments_dir, item)
            if os.path.isdir(exp_path):
                exp_dirs.append(item)
        
        exp_dirs.sort()
        
        for i, exp_dir in enumerate(exp_dirs):
            is_last = (i == len(exp_dirs) - 1)
            prefix = "└── " if is_last else "├── "
            print("{}{}".format(prefix, exp_dir))
            
            # 显示子目录和文件
            exp_path = os.path.join(self.experiments_dir, exp_dir)
            try:
                items = os.listdir(exp_path)
                items.sort()
                for j, item in enumerate(items):
                    item_is_last = (j == len(items) - 1)
                    if is_last:
                        item_prefix = "    └── " if item_is_last else "    ├── "
                    else:
                        item_prefix = "│   └── " if item_is_last else "│   ├── "
                    
                    item_path = os.path.join(exp_path, item)
                    if os.path.isdir(item_path):
                        print("{}{}/ ".format(item_prefix, item))
                    else:
                        print("{}{}".format(item_prefix, item))
            except PermissionError:
                pass
    
    def _convert_value(self, value_str):
        """转换字符串值为适当的类型"""
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
        """更新实验汇总信息"""
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
                        results = json.load(f)
                    experiments.append(results)
        
        summary = {
            "total_experiments": len(experiments),
            "updated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiments": experiments
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _generate_python_config(self, base_config_path, exp_config, exp_name, template_name, exp_dir):
        """生成Python配置文件"""
        # 读取基础Python配置
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_content = f.read()
        
        # 参数替换映射
        replacements = {}
        
        # 根据实验配置生成替换字典
        if 'lr' in exp_config:
            lr_val = exp_config['lr']
            replacements['lr = 5e-5'] = 'lr = {}'.format(lr_val)
            replacements['act_lr = 5e-5'] = 'act_lr = {}'.format(lr_val)  
            replacements['cri_lr = 5e-5'] = 'cri_lr = {}'.format(lr_val)
            replacements['rep_lr = 5e-5'] = 'rep_lr = {}'.format(lr_val)
            replacements['beta_lr = 5e-5'] = 'beta_lr = {}'.format(lr_val)
        
        if 'batch_size' in exp_config:
            bs_val = exp_config['batch_size']
            replacements['batch_size = 32'] = 'batch_size = {}'.format(bs_val)
            replacements['batch_size = 8'] = 'batch_size = {}'.format(bs_val)
        
        if 'seed' in exp_config:
            seed_val = exp_config['seed']
            replacements['seed = 20258888'] = 'seed = {}'.format(seed_val)
        
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

        if 'train_start_date' in exp_config:
            date_val = exp_config['train_start_date']
            replacements['train_start_date = "2021-01-01"'] = 'train_start_date = "{}"'.format(date_val)

        if 'train_end_date' in exp_config:
            date_val = exp_config['train_end_date']
            replacements['train_end_date = "2024-06-30"'] = 'train_end_date = "{}"'.format(date_val)

        if 'val_start_date' in exp_config:
            date_val = exp_config['val_start_date']
            replacements['val_start_date = "2024-07-01"'] = 'val_start_date = "{}"'.format(date_val)

        if 'val_end_date' in exp_config:
            date_val = exp_config['val_end_date']
            replacements['val_end_date = "2024-12-31"'] = 'val_end_date = "{}"'.format(date_val)

        if 'test_start_date' in exp_config:
            date_val = exp_config['test_start_date']
            replacements['test_start_date = "2025-01-01"'] = 'test_start_date = "{}"'.format(date_val)

        if 'test_end_date' in exp_config:
            date_val = exp_config['test_end_date']
            replacements['test_end_date = "2025-03-31"'] = 'test_end_date = "{}"'.format(date_val)

        # 添加workdir和tag参数的替换
        # workdir是相对目录路径，实验实际创建于 experiments/experiments/exp_name
        exp_relative_path = os.path.join("experiments", "experiments", exp_name)
        replacements['workdir = "workdir"'] = 'workdir = "{}"'.format(exp_relative_path)
        
        # tag是实验名称
        replacements['tag = "mask_sac"'] = 'tag = "{}"'.format(exp_name)
        # 处理其他可能的tag值
        for tag_val in ["sac", "dqn", "ddpg", "ppo", "td3", "mask_dqn", "mask_sync_dqn", "mask_ddqn", "mask_sync_sac"]:
            replacements['tag = "{}"'.format(tag_val)] = 'tag = "{}"'.format(exp_name)
        
        # 执行替换
        modified_content = base_content
        for old_value, new_value in replacements.items():
            modified_content = modified_content.replace(old_value, new_value)
        
        # 在文件开头添加实验信息注释
        header_comment = '''# -*- coding: utf-8 -*-
# 自动生成的实验配置文件
# 实验名称: {}
# 模板: {}
# 创建时间: {}
# 基于配置: {}

'''.format(exp_name, template_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), base_config_path)
        
        # 移除原有的编码声明行（避免重复）
        if modified_content.startswith('# -*- coding: utf-8 -*-'):
            lines = modified_content.split('\n')
            modified_content = '\n'.join(lines[1:])
        
        final_content = header_comment + modified_content
        
        # 保存生成的Python配置文件
        py_config_path = os.path.join(exp_dir, "config.py")
        with open(py_config_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print("Generated Python config: {}".format(py_config_path))

def main():
    parser = argparse.ArgumentParser(description='实验配置管理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # list-templates命令
    subparsers.add_parser('list-templates', help='查看可用模板')
    
    # show-template命令
    show_parser = subparsers.add_parser('show-template', help='查看模板详细内容')
    show_parser.add_argument('--name', required=True, help='模板名称')
    
    # generate-configs命令
    gen_parser = subparsers.add_parser('generate-configs', help='生成实验配置')
    gen_parser.add_argument('--template', required=True, help='模板名称')
    gen_parser.add_argument('--base-config', help='基础Python配置文件路径，默认为configs/mask_sac_portfolio_management.py')
    gen_parser.add_argument('--lr', help='学习率参数，多个值用逗号分隔')
    gen_parser.add_argument('--batch_size', help='批大小参数，多个值用逗号分隔')
    gen_parser.add_argument('--embed_dim', help='嵌入维度参数，多个值用逗号分隔')
    gen_parser.add_argument('--seed', help='随机种子参数，多个值用逗号分隔')
    gen_parser.add_argument('--num_episodes', help='训练轮数参数，多个值用逗号分隔')
    
    # list-experiments命令
    list_parser = subparsers.add_parser('list-experiments', help='查看所有实验配置')
    list_parser.add_argument('--tree', action='store_true', help='以树状结构显示实验组织')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.command == 'list-templates':
        manager.list_templates()
    elif args.command == 'show-template':
        manager.show_template(args.name)
    elif args.command == 'generate-configs':
        kwargs = {}
        if args.lr:
            kwargs['lr'] = args.lr
        if args.batch_size:
            kwargs['batch_size'] = args.batch_size
        if args.embed_dim:
            kwargs['embed_dim'] = args.embed_dim
        if args.seed:
            kwargs['seed'] = args.seed
        if args.num_episodes:
            kwargs['num_episodes'] = args.num_episodes
        manager.generate_configs(args.template, base_config=getattr(args, 'base_config', None), **kwargs)
    elif args.command == 'list-experiments':
        if args.tree:
            manager.show_experiments_tree()
        else:
            manager.list_experiments()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()