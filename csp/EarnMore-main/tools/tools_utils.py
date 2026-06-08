# -*- coding: utf-8 -*-
import os

def check_gpu_available():
    """检测是否有可用的GPU"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def check_operating_system():
    """判断当前运行环境是Windows还是Linux"""
    # 获取操作系统名称
    os_name = os.name
    # 获取更详细的系统信息
    system_info = os.uname() if hasattr(os, 'uname') else None

    # 判断逻辑
    if os_name == 'nt':
        return "Windows"
    elif os_name == 'posix':
        if system_info and 'linux' in system_info.sysname.lower():
            return "Linux"
        else:
            return f"Unix-like system ({system_info.sysname if system_info else 'unknown'})"
    else:
        return f"Unknown operating system ({os_name})"

# 使用示例
if __name__ == "__main__":
    gpu_available = check_gpu_available()
    print(f"GPU可用: {gpu_available}")
    os_type = check_operating_system()
    print(f"当前运行环境: {os_type}")
