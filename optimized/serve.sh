#!/bin/sh
set -eu

artifact=${1:?usage: serve.sh OPTIMIZED_ARTIFACT}
served_model_name=${2:-gemma-4-12b-optimized}

exec vllm serve "$artifact" \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name "$served_model_name" \
  --enable-mm-embeds \
  --no-enable-prefix-caching \
  --skip-mm-profiling \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096
