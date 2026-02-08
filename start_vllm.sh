#!/bin/bash

# vLLM startup script
# Configure environment variables for Chinese users
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
HOST="0.0.0.0"
PORT=8000
GPU_MEMORY_UTILIZATION=0.8
MAX_MODEL_LEN=32768

echo "Starting vLLM server..."
echo "Model: $MODEL_NAME"
echo "Host: $HOST"
echo "Port: $PORT"

# Start vLLM server with OpenAI compatible API
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size 1 \
    --dtype auto \
    --api-key sk-xxxxxxx \
    --served-model-name vllm-model