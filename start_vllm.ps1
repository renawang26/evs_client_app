# vLLM startup script for Windows PowerShell
# Configure environment variables for Chinese users
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:CUDA_VISIBLE_DEVICES = "0"

# Model configuration
$MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
$VLLM_HOST = "0.0.0.0"
$PORT = 8000
$GPU_MEMORY_UTILIZATION = 0.8
$MAX_MODEL_LEN = 32768

Write-Host "Starting vLLM server..." -ForegroundColor Green
Write-Host "Model: $MODEL_NAME" -ForegroundColor Yellow
Write-Host "Host: $VLLM_HOST" -ForegroundColor Yellow
Write-Host "Port: $PORT" -ForegroundColor Yellow

# Start vLLM server with OpenAI compatible API
python -m vllm.entrypoints.openai.api_server `
    --model $MODEL_NAME `
    --host $VLLM_HOST `
    --port $PORT `
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION `
    --max-model-len $MAX_MODEL_LEN `
    --tensor-parallel-size 1 `
    --dtype auto `
    --api-key sk-xxxxxxx `
    --served-model-name vllm-model