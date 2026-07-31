#!/bin/sh
set -eu

artifact=${1:?usage: serve.sh OPTIMIZED_ARTIFACT}

exec vllm serve "$artifact" \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name gemma-4-12b-optimized \
  --enable-mm-embeds \
  --skip-mm-profiling \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096
