#!/bin/bash

# 切换到脚本所在目录
cd /root/EarnMore-main_v20250707

# log file name with datetime
current_datetime=$(date +"%Y%m%d")
log_file="nohup_train_arr_${current_datetime}.out"

# rm -rf ./tests/results/progress/news_*
# 执行 Python 脚本，传入配置文件参数
nohup python ./tools/train_arr.py  >> "$log_file" 2>&1 &

