#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Opt into MoK for a compatible existing recipe; preserve its model/data/parallel arguments.
# The recipe must preserve MEGATRON/PYTHONPATH and disable shared-expert overlap.
set -euo pipefail
recipe="${1:?Usage: with-qwen38-mok.sh <compatible-training-recipe.sh> [recipe args]}"
shift
: "${MOK_MCORE:?Set MOK_MCORE to the patched MCore tree}"
: "${MOK_SITE:?Set MOK_SITE to the isolated MoK site-packages directory}"
export MEGATRON="$MOK_MCORE"
export QWEN38_SHARED_EXPERT_OVERLAP=0
export PYTHONPATH="$MOK_SITE:${PYTHONPATH:-}"
config="${MOK_BACKEND_CONFIG:-}"
if [[ -z "$config" ]]; then
    config='{"fwd_num_comm_sms":48,"bwd_num_comm_sms":56,"minibatch_size":4096,"macrobatch_size":65536,"schedule_capacity_multiplier":1.5,"token_capacity":65536}'
fi
exec bash "$recipe" --moe-megakernel-backend mok --moe-megakernel-backend-config "$config" "$@"
