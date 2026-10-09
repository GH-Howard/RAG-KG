#!/bin/bash
# P7 抽取启动封装：固化 vLLM 0.31 在本机（8×3090，无系统 CUDA toolkit）所需环境
# 用法: bash p7_run.sh <a|b> <model_subdir> <gpu_ids> [extra args...]
# 例:  bash p7_run.sh a Qwen--Qwen3-32B-FP8 2,3
set -e
TRACK=$1; MODEL_DIR=$2; GPUS=$3; shift 3
VENV=/home/mountDisk2/user/guhao/.conda/envs/rag-kg-vllm
CU13=$VENV/lib/python3.10/site-packages/nvidia/cu13
MODEL=/home/mountDisk2/user/guhao/.cache/modelscope/models/$MODEL_DIR/snapshots/master
cd "$(dirname "$0")"
export CUDA_HOME=$CU13
export PATH=$VENV/bin:$CU13/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPUS
exec $VENV/bin/python -u extract_kg.py --track "$TRACK" --model "$MODEL" "$@"
