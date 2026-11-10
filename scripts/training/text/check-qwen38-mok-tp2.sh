#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail
RELAX_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${RAY_ADDRESS:?Set the Ray Jobs HTTP URL}"
: "${SUBMISSION_ID:?Set a unique Ray submission ID}"
: "${BENCHMARK_PYTHON:?Set the isolated Python executable}"
: "${BENCHMARK_OUTPUT:?Set the shared output JSON path}"
CUDNN_LIB_PATH=$("$BENCHMARK_PYTHON" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parents[1] / "nvidia/cudnn/lib")')
ray job submit --address="$RAY_ADDRESS" --submission-id="$SUBMISSION_ID" --no-wait \
    --runtime-env-json="$RUNTIME_ENV_JSON" --entrypoint-num-gpus=8 --entrypoint-num-cpus=16 \
    -- timeout 2400 env "CUDNN_PATH=${CUDNN_LIB_PATH%/lib}" "LD_LIBRARY_PATH=$CUDNN_LIB_PATH:${LD_LIBRARY_PATH:-}" \
    NCCL_NVLS_ENABLE=0 "$BENCHMARK_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "$RELAX_ROOT/scripts/benchmarks/check_qwen38_mok_tp2.py" --output "$BENCHMARK_OUTPUT" "$@"
