#!/bin/bash

# 获取当前时间，格式为 YYYY-MM-DD_HH-MM-SS
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# 日志文件路径
LOG_FILE="./result/log/train_log_${TIMESTAMP}.log"

# 运行 train.py，将标准输出和错误重定向到日志文件，不输出到控制台
nohup python train.py > "$LOG_FILE" 2>&1 &
PID=$!
echo "Training started... PID: $PID"
echo "Logs are being written to $LOG_FILE"