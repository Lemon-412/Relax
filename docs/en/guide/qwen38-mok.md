# Optional MoK for Qwen3.8

MoK can run the routed experts with TP sequence parallelism while retaining the native TP shared expert. The default backend is unchanged. This integration passes the backend selection through Bridge and supplies isolated MCore/MoK patches, including the 128-wide FFN tail needed by FFN640.

## Validated scope

- Blackwell SM103, BF16, dropless MoE, TP2 with sequence parallelism, EP8 and ETP1.
- Qwen3.8-Flash-Next-F0: hidden size2560, FFN640, 512 experts, top10, native shared expert and output gate.
- Shared-expert overlap must be disabled in the training recipe. Keep the original TP/PP/CP/EP configuration; this wrapper does not change it.
- The module regression covers multiple outstanding microbatches, two Adam updates and full MoE recomputation. End-to-end short runs do not establish long-run numerical equivalence.
- FP8/MXFP8, other devices, other parallel layouts and cross-topology checkpoint resume are not established by this validation. TP1's fused shared path is distinct from the TP2 native shared path.

## Environment

Build from the compatible `qs-20261101-bfdd68d7-pr529-cu13-blackwell` base (Python3.12, Torch2.13.0+cu130):

```bash
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -f docker/Dockerfile.qwen38-mok -t relax-qwen38-mok .
```

The Dockerfile leaves the default Python and MCore untouched. It creates an isolated MoK environment and copies the base MCore before applying the existing checkpoint-padding and MTP CP fixes and the two MoK patches. The named base does not contain those two prerequisite fixes; a different base must be checked for patch compatibility before building. It does not install the experimental QSA dependencies or replace the QSA backend.

MoK is pinned to `ee7bdaba515fd3d557105d71809ec74cd523d5a1`, with its pinned ThunderKittens submodule and bundled build patch. The additional FFN128-tail patch changes the kernels as well as shape validation. A prebuilt wheel must match this source, CUDA/PyTorch ABI and architecture. Docker image construction is separate from patch-chain and runtime validation.

## Opt in to a compatible recipe

The public 32-GPU Qwen3.8 recipe preserves the isolated environment and uses ETP1. The wrapper disables shared-expert overlap through `QWEN38_SHARED_EXPERT_OVERLAP=0`. Model, data and output paths remain the recipe's responsibility and must be visible on every node.

```bash
export PYTHONPATH="/opt/relax/cublas13.3:${PYTHONPATH:-}"
export RELAX_SKIP_NONFINITE_GRAD=0
export MODEL_DIR=/shared/model DATA_DIR=/shared/data SAVE_DIR=/shared/output
bash scripts/training/text/with-qwen38-mok.sh \
  examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-32xgpu-blackwell.sh \
  --num-rollout 20
```

The image supplies `MOK_MCORE` and `MOK_SITE`. Outside that image, set them to the matching patched MCore and isolated MoK installation. The wrapper adds the existing `--moe-megakernel-backend mok` and `--moe-megakernel-backend-config` options; it adds no new training-parser flags. Running the original recipe directly retains its backend choice.

The default backend configuration uses token capacity65536 for the validated TP2/128K case. `MOK_BACKEND_CONFIG` can supply the existing backend JSON when the recipe needs another capacity. Capacity must cover the local routed input after TP/SP splitting; it is not the global batch token count.

Training logs include `MOK_BACKEND_PROVENANCE`. Check the expected MoE layer count on each pipeline stage, not merely the presence of that message. A layer constructed with another megakernel despite an explicit MoK request is rejected.

## Module regression

On an idle eight-GPU node with the compatible environment, run:

```bash
PYTHONPATH="$PWD:$MOK_MCORE:$MOK_SITE:/opt/relax/cublas13.3" \
  "$MOK_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
  scripts/benchmarks/check_qwen38_mok_tp2.py --output /shared/mok-module.json \
  --tokens 65536 --microbatches 4 --macrobatch-size 65536 --lr 2e-5 --uneven-tokens
```

This compares output, input/router/expert/shared gradients and optimizer updates against native MCore MoE. It is a numerical regression, not a HybridEP performance benchmark. `scripts/training/text/check-qwen38-mok-tp2.sh` provides the corresponding Ray Jobs wrapper when runtime environment and node allocation are managed by the caller.
