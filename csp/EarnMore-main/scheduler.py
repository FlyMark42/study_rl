# -*- coding: utf-8 -*-
"""
实验调度功能模块
该文件应该放在项目根目录下
"""
import os
import sys
import json
import time
import argparse
import subprocess
import psutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import glob

g_strRoot="J:\\python_code\\stock_reinforcement\\EarnMore\\EarnMore-main"

class TaskQueue:
    """任务队列管理器"""
    
    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = experiments_dir
        self.queue = []
        print("初始化任务队列管理器")
    
    def scan_experiments(self):
        """扫描实验目录获取任务状态"""
        print("扫描实验目录获取任务状态...")
        
        if not os.path.exists(self.experiments_dir):
            print("实验目录不存在: {}".format(self.experiments_dir))
            return
        
        experiment_dirs = glob.glob(os.path.join(self.experiments_dir, "exp_*"))
        print("找到实验目录: {}".format([os.path.basename(d) for d in experiment_dirs]))
        
        # 清空队列重新扫描
        self.queue = []
        
        for exp_dir in experiment_dirs:
            if os.path.isdir(exp_dir):
                exp_name = os.path.basename(exp_dir)
                status_file = os.path.join(exp_dir, "status.json")
                
                if os.path.exists(status_file):
                    status = self._load_status(status_file)
                    print("加载状态文件 {}: status={}".format(exp_name, status["status"]))
                else:
                    status = self._init_status(exp_dir, exp_name)
                    print("初始化状态文件 {}: status={}".format(exp_name, status["status"]))
                
                self.queue.append({
                    "name": exp_name,
                    "path": exp_dir,
                    "status": status["status"],
                    "priority": status["priority"],
                    "retry_count": status["retry_count"],
                    "last_update": status["last_update"]
                })
        
        print("找到{}个实验".format(len(self.queue)))
    
    def _load_status(self, status_file):
        """加载实验状态文件"""
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print("加载状态文件失败 {}: {}".format(status_file, e))
            return self._create_default_status()
    
    def _init_status(self, exp_dir, exp_name):
        """初始化实验状态"""
        status = self._create_default_status()
        
        config_file = os.path.join(exp_dir, "config.py")
        
        # 只基于config.py文件判断初始状态，不依赖results.json
        if os.path.exists(config_file):
            status["status"] = "pending"
        else:
            status["status"] = "invalid"
        
        self._save_status(exp_dir, status)
        return status
    
    def _create_default_status(self):
        """创建默认状态"""
        return {
            "status": "pending",
            "priority": 0,
            "retry_count": 0,
            "max_retries": 3,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "failure_reason": "",
            "start_time": "",
            "end_time": ""
        }
    
    def _save_status(self, exp_dir, status):
        """保存实验状态到文件"""
        status_file = os.path.join(exp_dir, "status.json")
        try:
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print("保存状态文件失败 {}: {}".format(status_file, e))
    
    def get_pending_tasks(self):
        """获取待执行的任务列表"""
        print("队列中的所有任务:")
        for task in self.queue:
            print("  - {}: status={}".format(task["name"], task["status"]))
        
        pending_tasks = [task for task in self.queue if task["status"] == "pending"]
        pending_tasks.sort(key=lambda x: x["priority"], reverse=True)
        print("找到{}个待执行任务".format(len(pending_tasks)))
        return pending_tasks
    
    def get_running_tasks(self):
        """获取正在运行的任务列表"""
        return [task for task in self.queue if task["status"] == "running"]
    
    def update_task_status(self, exp_name, status, failure_reason=""):
        """更新任务状态"""
        for task in self.queue:
            if task["name"] == exp_name:
                task["status"] = status
                task["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                status_data = self._load_status(os.path.join(task["path"], "status.json"))
                status_data["status"] = status
                status_data["last_update"] = task["last_update"]
                
                if status == "running":
                    status_data["start_time"] = task["last_update"]
                elif status in ["completed", "failed"]:
                    status_data["end_time"] = task["last_update"]
                    if failure_reason:
                        status_data["failure_reason"] = failure_reason
                
                self._save_status(task["path"], status_data)
                print("更新任务状态: {} -> {}".format(exp_name, status))
                break
    
    def set_task_priority(self, exp_name, priority):
        """设置任务优先级"""
        for task in self.queue:
            if task["name"] == exp_name:
                task["priority"] = priority
                status_data = self._load_status(os.path.join(task["path"], "status.json"))
                status_data["priority"] = priority
                self._save_status(task["path"], status_data)
                print("设置任务优先级: {} = {}".format(exp_name, priority))
                return
        print("未找到任务: {}".format(exp_name))

class ResourceManager:
    """资源分配管理器"""
    
    def __init__(self, max_concurrent=2, cpu_threshold=80.0, memory_threshold=80.0):
        self.max_concurrent = max_concurrent
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.resource_log = []
        print("初始化资源分配管理器: max_concurrent={}, cpu_threshold={}%, memory_threshold={}%".format(
            max_concurrent, cpu_threshold, memory_threshold))
    
    def check_available_slots(self, current_running_count):
        """检查可用的执行槽位数量"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        self.resource_log.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "running_count": current_running_count
        })
        
        print("系统资源状态: CPU={:.1f}%, Memory={:.1f}%, Running={}".format(
            cpu_percent, memory_percent, current_running_count))
        
        # if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
        #     print("系统资源紧张，暂停启动新实验")
        #     return 0
        
        available_slots = self.max_concurrent - current_running_count
        return max(0, available_slots)
    
    def save_resource_log(self, log_file="resource_monitor.csv"):
        """保存资源使用历史到CSV文件"""
        try:
            import pandas as pd
            df = pd.DataFrame(self.resource_log)
            df.to_csv(log_file, index=False, encoding='utf-8')
            print("资源监控日志已保存: {}".format(log_file))
        except Exception as e:
            print("保存资源日志失败: {}".format(e))

class RetryManager:
    """失败重试机制管理器"""
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        print("初始化失败重试管理器: max_retries={}".format(max_retries))
    
    def should_retry(self, exp_path, current_retry_count):
        """判断是否应该重试"""
        if current_retry_count >= self.max_retries:
            print("已达最大重试次数，不再重试")
            return False
        
        status_file = os.path.join(exp_path, "status.json")
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                return status.get("retry_count", 0) < self.max_retries
            except Exception as e:
                print("读取状态文件失败: {}".format(e))
        
        return True
    
    def increment_retry_count(self, exp_path, failure_reason=""):
        """增加重试计数"""
        status_file = os.path.join(exp_path, "status.json")
        
        try:
            if os.path.exists(status_file):
                with open(status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
            else:
                status = {"retry_count": 0, "max_retries": self.max_retries}
            
            status["retry_count"] = status.get("retry_count", 0) + 1
            status["failure_reason"] = failure_reason
            status["last_retry"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
            
            print("重试计数增加: {} (第{}次)".format(os.path.basename(exp_path), status["retry_count"]))
            
        except Exception as e:
            print("更新重试计数失败: {}".format(e))

class RecoveryManager:
    """中断恢复机制管理器"""
    
    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = experiments_dir
        print("初始化中断恢复管理器")
    
    def recover_interrupted_experiments(self):
        """恢复中断的实验"""
        print("扫描中断的实验...")
        
        interrupted_count = 0
        experiment_dirs = glob.glob(os.path.join(self.experiments_dir, "exp_*"))
        
        for exp_dir in experiment_dirs:
            if self._is_interrupted_experiment(exp_dir):
                self._reset_experiment_status(exp_dir)
                interrupted_count += 1
        
        print("恢复了{}个中断的实验".format(interrupted_count))
        return interrupted_count
    
    def _is_interrupted_experiment(self, exp_dir):
        """检查实验是否被中断"""
        status_file = os.path.join(exp_dir, "status.json")
        
        if not os.path.exists(status_file):
            return False
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            
            # 只基于status.json判断：如果状态是running就认为被中断了
            if status.get("status") == "running":
                return True
                
        except Exception as e:
            print("检查中断状态失败 {}: {}".format(exp_dir, e))
        
        return False
    
    def _reset_experiment_status(self, exp_dir):
        """重置实验状态为pending"""
        status_file = os.path.join(exp_dir, "status.json")
        
        try:
            if os.path.exists(status_file):
                with open(status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
            else:
                status = {}
            
            status["status"] = "pending"
            status["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status["recovery_time"] = status["last_update"]
            
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
            
            print("重置实验状态: {}".format(os.path.basename(exp_dir)))
            
        except Exception as e:
            print("重置实验状态失败 {}: {}".format(exp_dir, e))

class ExperimentScheduler:
    """实验调度器主类"""
    
    def __init__(self, experiments_dir="experiments", max_concurrent=2):
        self.experiments_dir = experiments_dir
        self.max_concurrent = max_concurrent
        
        self.task_queue = TaskQueue(experiments_dir)
        self.resource_manager = ResourceManager(max_concurrent)
        self.retry_manager = RetryManager()
        self.recovery_manager = RecoveryManager(experiments_dir)
        
        self.running_processes = {}
        self.should_stop = False
        
        print("实验调度器初始化完成")
    
    def status(self):
        """显示调度器状态"""
        self.task_queue.scan_experiments()
        
        pending_tasks = self.task_queue.get_pending_tasks()
        running_tasks = self.task_queue.get_running_tasks()
        completed_tasks = [t for t in self.task_queue.queue if t["status"] == "completed"]
        failed_tasks = [t for t in self.task_queue.queue if t["status"] == "failed"]
        
        print("\n=== 实验调度器状态 ===")
        print("待执行实验: {}".format(len(pending_tasks)))
        print("正在运行: {}".format(len(running_tasks)))
        print("已完成: {}".format(len(completed_tasks)))
        print("失败: {}".format(len(failed_tasks)))
        print("总实验数: {}".format(len(self.task_queue.queue)))
        
        if running_tasks:
            print("\n正在运行的实验:")
            for task in running_tasks:
                print("  - {}".format(task["name"]))
        
        if pending_tasks:
            print("\n待执行实验队列 (按优先级排序):")
            for i, task in enumerate(pending_tasks[:10]):
                print("  {}. {} (优先级: {})".format(i+1, task["name"], task["priority"]))
            if len(pending_tasks) > 10:
                print("  ... 还有{}个待执行实验".format(len(pending_tasks) - 10))
    
    def set_priority(self, exp_name, priority):
        """设置实验优先级"""
        self.task_queue.scan_experiments()
        self.task_queue.set_task_priority(exp_name, priority)
    
    def rerunall_experiments(self):
        """重新运行所有实验"""
        self.task_queue.scan_experiments()
        count = 0
        
        for task in self.task_queue.queue:
            exp_name = task["name"]
            current_status = task["status"]
            
            # 将状态改为pending
            self.task_queue.update_task_status(exp_name, "pending")
            print("重置实验状态: {} {} -> pending".format(exp_name, current_status))
            count += 1
        
        print("已重置{}个实验状态为pending".format(count))
    
    def run(self):
        """运行调度器主循环"""
        print("启动实验调度器...")
        
        recovered_count = self.recovery_manager.recover_interrupted_experiments()
        if recovered_count > 0:
            print("恢复了{}个中断的实验".format(recovered_count))
        
        try:
            while not self.should_stop:
                self.task_queue.scan_experiments()
                self._process_running_experiments()
                self._check_pending_experiments_process_status()
                self._start_new_experiments()
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n收到中断信号，正在优雅关闭...")
            self._cleanup()
        
        print("调度器已停止")
    
    def _process_running_experiments(self):
        """处理正在运行的实验"""
        finished_experiments = []
        
        for exp_name, process in self.running_processes.items():
            if process.poll() is not None:
                return_code = process.returncode
                
                if return_code == 0:
                    self.task_queue.update_task_status(exp_name, "completed")
                    print("实验完成: {}".format(exp_name))
                else:
                    exp_path = None
                    for task in self.task_queue.queue:
                        if task["name"] == exp_name:
                            exp_path = task["path"]
                            break
                    
                    if exp_path and self.retry_manager.should_retry(exp_path, 0):
                        self.retry_manager.increment_retry_count(exp_path, "进程退出码: {}".format(return_code))
                        self.task_queue.update_task_status(exp_name, "pending")
                        print("实验失败，准备重试: {}".format(exp_name))
                    else:
                        self.task_queue.update_task_status(exp_name, "failed", "进程退出码: {}".format(return_code))
                        print("实验最终失败: {}".format(exp_name))
                
                finished_experiments.append(exp_name)
        
        for exp_name in finished_experiments:
            del self.running_processes[exp_name]
    
    def _check_pending_experiments_process_status(self):
        """检查pending状态实验的进程状态"""
        pending_tasks = [t for t in self.task_queue.queue if t["status"] == "pending"]

        for task in pending_tasks:
            exp_name = task["name"]

            # 检查该实验是否有正在运行的进程
            if exp_name in self.running_processes:
                process = self.running_processes[exp_name]

                # 如果进程还在运行，终止它
                if process.poll() is None:
                    try:
                        print("发现pending状态实验{}有运行中进程，正在终止...".format(exp_name))
                        process.terminate()
                        process.wait(timeout=5)
                        print("已终止实验{}的进程".format(exp_name))
                    except subprocess.TimeoutExpired:
                        process.kill()
                        print("强制终止实验{}的进程".format(exp_name))
                    except Exception as e:
                        print("终止实验{}进程失败: {}".format(exp_name, e))

                # 从运行进程列表中移除
                del self.running_processes[exp_name]

    def _start_new_experiments(self):
        """启动新实验"""
        available_slots = self.resource_manager.check_available_slots(len(self.running_processes))
        
        if available_slots <= 0:
            return
        
        pending_tasks = self.task_queue.get_pending_tasks()
        
        for i in range(min(available_slots, len(pending_tasks))):
            task = pending_tasks[i]
            self._start_single_experiment(task)
    
    def _start_single_experiment(self, task):
        """启动单个实验"""
        exp_name = task["name"]
        exp_path = task["path"]
        config_file = os.path.join(exp_path, "config.py")
        
        if not os.path.exists(config_file):
            print("配置文件不存在，跳过实验: {}".format(exp_name))
            self.task_queue.update_task_status(exp_name, "failed", "配置文件不存在")
            return
        
        try:
            cmd = [sys.executable, "tools/train.py", "--config", config_file, "--root", g_strRoot,"--monitor"]
            
            log_file = os.path.join(exp_path, "training.log")
            with open(log_file, 'w', encoding='utf-8') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd()
                )
            
            self.running_processes[exp_name] = process
            self.task_queue.update_task_status(exp_name, "running")
            print("启动实验: {} (PID: {})".format(exp_name, process.pid))
            
        except Exception as e:
            print("启动实验失败 {}: {}".format(exp_name, e))
            self.task_queue.update_task_status(exp_name, "failed", str(e))
    
    def _cleanup(self):
        """清理资源"""
        print("正在终止运行中的实验...")
        
        for exp_name, process in self.running_processes.items():
            try:
                process.terminate()
                process.wait(timeout=10)
                self.task_queue.update_task_status(exp_name, "pending")
                print("已终止实验: {}".format(exp_name))
            except subprocess.TimeoutExpired:
                process.kill()
                print("强制终止实验: {}".format(exp_name))
            except Exception as e:
                print("终止实验失败 {}: {}".format(exp_name, e))
        
        self.resource_manager.save_resource_log()
        print("资源清理完成")

def main():
    parser = argparse.ArgumentParser(description="实验调度器")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    status_parser = subparsers.add_parser("status", help="查看调度器状态")
    
    priority_parser = subparsers.add_parser("set-priority", help="设置实验优先级")
    priority_parser.add_argument("--experiment", required=True, help="实验名称")
    priority_parser.add_argument("--priority", type=int, required=True, help="优先级数值")
    
    rerunall_parser = subparsers.add_parser("rerunall", help="重新运行所有实验")
    
    run_parser = subparsers.add_parser("run", help="运行调度器")
    run_parser.add_argument("--max-concurrent", type=int, default=2, help="最大并发数")
    run_parser.add_argument("--max-retries", type=int, default=3, help="最大重试次数")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    scheduler = ExperimentScheduler(max_concurrent=getattr(args, 'max_concurrent', 2))
    
    if args.command == "status":
        scheduler.status()
    
    elif args.command == "set-priority":
        #scheduler.set_priority(args.experiment, args.priority)
        return
    
    elif args.command == "rerunall":
        scheduler.rerunall_experiments()
    
    elif args.command == "run":
        scheduler.run()

if __name__ == "__main__":
    main()