# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh"


@pytest.mark.parametrize("mok", [False, True])
def test_public_recipe_preserves_backend_environment_and_overlap_policy(mok):
    env = dict(
        os.environ,
        MODEL_DIR="/model",
        DATA_DIR="/data",
        SAVE_DIR="/output",
        MEGATRON="/native/mcore",
        MOK_MCORE="/isolated/mcore",
        MOK_SITE="/isolated/mok site",
        PYTHONPATH="/extra/dependencies",
        RELAX_HYBRID_EP_COMMIT="test-image",
        DRY_RUN="1",
    )
    for key in ("RUNTIME_ENV_JSON", "QWEN38_SHARED_EXPERT_OVERLAP", "MOK_BACKEND_CONFIG"):
        env.pop(key, None)
    command = ["bash", str(RECIPE)]
    if mok:
        command = ["bash", str(ROOT / "scripts/training/text/with-qwen38-mok.sh"), str(RECIPE)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    argv = output["argv"]
    runtime = output["runtime_env"]["env_vars"]
    train = json.loads(argv[argv.index("--train-env-vars") + 1])
    for settings in (runtime, train):
        paths = settings["PYTHONPATH"].split(":")
        assert "/extra/dependencies" in paths
        assert settings["MEGATRON"] == ("/isolated/mcore" if mok else "/native/mcore")
        if mok:
            assert "/isolated/mok site" in paths
    assert ("--moe-shared-expert-overlap" in argv) is (not mok)
    if mok:
        assert argv[argv.index("--moe-megakernel-backend") + 1] == "mok"
        config = json.loads(argv[argv.index("--moe-megakernel-backend-config") + 1])
        assert config["token_capacity"] == 65536
    else:
        assert "--moe-megakernel-backend" not in argv
