# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compare native MCore MoE and MoK with 512 experts/EP8 and two Adam updates.

The second update recomputes the whole MoE. Parameter overlap remains a
separate full-training gate. This is a numerical test, not a HybridEP speed
benchmark.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def relative_error(got: torch.Tensor, reference: torch.Tensor) -> float:
    assert torch.isfinite(got).all() and torch.isfinite(reference).all()
    return float((got.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recompute-first", action="store_true")
    parser.add_argument("--uneven-tokens", action="store_true")
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--macrobatch-size", type=int, choices=(4096, 65536), default=4096)
    args = parser.parse_args()
    if args.microbatches < 1:
        parser.error("microbatches must be positive")
    if args.tokens < 512:
        parser.error("MoK requires at least 512 tokens per rank")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 8
    from megatron.core import parallel_state as ps
    from megatron.core import tensor_parallel
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.layers import set_defaults_if_not_set_tensor_model_parallel_attributes
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig

    ps.initialize_model_parallel(
        tensor_model_parallel_size=2, expert_model_parallel_size=8, expert_tensor_parallel_size=1
    )
    torch.manual_seed(42)
    tensor_parallel.model_parallel_cuda_manual_seed(42)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    config = TransformerConfig(
        num_layers=1,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        hidden_size=2560,
        num_attention_heads=24,
        kv_channels=256,
        ffn_hidden_size=640,
        num_moe_experts=512,
        moe_ffn_hidden_size=640,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        moe_router_topk=10,
        moe_router_dtype="fp32",
        moe_router_score_function="softmax",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_token_dispatcher_type="alltoall",
        moe_shared_expert_intermediate_size=640,
        moe_shared_expert_gate=True,
        moe_shared_expert_overlap=False,
        gradient_accumulation_fusion=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    mok_config = dataclasses.replace(
        config,
        moe_megakernel_backend="mok",
        moe_megakernel_backend_config={
            "fwd_num_comm_sms": 48,
            "bwd_num_comm_sms": 56,
            "minibatch_size": 4096,
            "macrobatch_size": args.macrobatch_size,
            "schedule_capacity_multiplier": 1.5,
            **({"token_capacity": args.tokens} if args.uneven_tokens else {}),
        },
    )

    class Probe(MegatronModule):
        def __init__(self, cfg: TransformerConfig) -> None:
            super().__init__(cfg)
            self.moe = get_moe_module_spec(num_experts=512, moe_grouped_gemm=True)(
                config=cfg, pg_collection=pg, layer_number=1
            )
            self.recompute = False

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            def forward_moe(x: torch.Tensor) -> torch.Tensor:
                output, bias = self.moe(x)
                assert bias is None
                return output

            return tensor_parallel.checkpoint(forward_moe, False, value) if self.recompute else forward_moe(value)

    native, candidate = Probe(config).cuda(), Probe(mok_config).cuda()
    candidate.load_state_dict(native.state_dict(), strict=True)
    kernel = candidate.moe.megakernel_experts
    assert type(kernel).__name__ == "MoKMegakernel" and kernel.num_local_experts == 64
    assert kernel.shared_output_gate_weight is None
    assert not candidate.moe.megakernel_fuses_shared_expert
    assert native.moe.megakernel_experts is None
    modules = (native, candidate)
    models, optimizers = [], []
    for module, cfg in zip(modules, (config, mok_config)):
        for parameter in module.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(parameter)
        assert not module.moe.experts.linear_fc1.weight0.allreduce
        model = DistributedDataParallel(
            cfg,
            DistributedDataParallelConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                use_distributed_optimizer=True,
            ),
            module,
            disable_bucketing=True,
        )
        model.broadcast_params()
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=args.lr,
                min_lr=0.0,
                weight_decay=0.0,
                clip_grad=1.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=False,
            ),
            [model],
            use_gloo_process_groups=False,
        )
        models.append(model)
        optimizers.append(optimizer)
    rows = []
    for step in range(2):
        generator = torch.Generator(device="cuda").manual_seed(1000 + dist.get_rank() + step * 8)
        local_tokens = args.tokens - 37 * (dist.get_rank() // 2) - 13 * step if args.uneven_tokens else args.tokens
        value = torch.randn(local_tokens, 1, 2560, device="cuda", dtype=torch.bfloat16, generator=generator)
        with torch.no_grad():
            left_params, right_params = (dict(module.named_parameters()) for module in modules)
            parameter_error = max(relative_error(right_params[name], value) for name, value in left_params.items())
            if step == 0:
                assert parameter_error == 0.0, parameter_error
            native_probs, native_routes = native.moe.route(value, None, None)
            mok_probs, mok_routes = candidate.moe.route(value, None, None)
            expert_counts = native_routes.sum(dim=0).to(torch.int64)
            dist.all_reduce(expert_counts)
            local_experts = expert_counts.chunk(8)[ps.get_expert_model_parallel_rank()]
            assert int(local_experts.sum()) > args.macrobatch_size, "Probe must exercise multiple routed macrobatches"
            routing = {
                "local_routed_tokens": int(local_experts.sum()),
                "routes_equal": bool(torch.equal(native_routes, mok_routes)),
                "changed_token_fraction": float((native_routes != mok_routes).any(dim=-1).float().mean()),
                "router_parameter_relative_error": relative_error(
                    candidate.moe.router.weight, native.moe.router.weight
                ),
                "different_route_entries": int((native_routes != mok_routes).sum()),
                "probability_relative_error": relative_error(mok_probs, native_probs),
                "parameter_max_relative_error": parameter_error,
            }
        microbatch_values = [value.detach() + mb * 0.03125 for mb in range(args.microbatches)]
        upstreams = [
            torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
            / value.numel() ** 0.5
            for _ in microbatch_values
        ]
        routing_by_microbatch = []
        with torch.no_grad():
            for actual in microbatch_values:
                lp, lr = native.moe.route(actual, None, None)
                rp, rr = candidate.moe.route(actual, None, None)
                routing_by_microbatch.append(
                    {
                        "changed_token_fraction": float((lr != rr).any(dim=-1).float().mean()),
                        "probability_relative_error": relative_error(rp, lp),
                    }
                )
        inputs, outputs, gate_updates = [], [], []
        for model, module, optimizer in zip(models, modules, optimizers):
            module.recompute = args.recompute_first or step == 1
            module.set_is_first_microbatch()
            optimizer.zero_grad()
            model.zero_grad_buffer()
            # Retain multiple outstanding forwards before any backward, as in
            # pipeline warmup. Distinct inputs expose overwritten workspace state.
            batch_inputs = [actual.detach().clone().requires_grad_() for actual in microbatch_values]
            batch_outputs, snapshots = [], []
            for x in batch_inputs:
                out = model(x)
                batch_outputs.append(out)
                snapshots.append(out.detach().clone())
            for out, snapshot in zip(batch_outputs, snapshots):
                assert torch.equal(out.detach(), snapshot), "Outstanding output storage was overwritten"
            inputs.append(batch_inputs)
            outputs.append(snapshots)
            # TE first-microbatch wgrad overwrites, so backward follows FIFO
            # order just like non-interleaved pipeline microbatches.
            for out, grad in zip(batch_outputs, upstreams):
                (out.float() * grad.float()).sum().backward()
            finalize_model_grads([model], num_tokens=None)
        errors = {
            "output": max(relative_error(got, ref) for got, ref in zip(outputs[1], outputs[0])),
            "input_grad": max(relative_error(got.grad, ref.grad) for got, ref in zip(inputs[1], inputs[0])),
        }
        left, right = (dict(module.named_parameters()) for module in modules)
        assert left.keys() == right.keys()
        totals = {}
        for name in left:
            got, ref = right[name].main_grad, left[name].main_grad
            assert torch.isfinite(got).all() and torch.isfinite(ref).all(), name
            category = (
                "gate"
                if "gate_weight" in name
                else ("router" if ".router." in name else ("shared" if ".shared_experts." in name else "experts"))
            )
            diff, norm = totals.setdefault(category, [torch.zeros((), device="cuda"), torch.zeros((), device="cuda")])
            diff.add_((got.float() - ref.float()).square().sum())
            norm.add_(ref.float().square().sum())
            if category == "experts":
                if ".linear_fc1.weight" in name:
                    assert ref.shape == (1280, 2560), (name, ref.shape)
                    got_tail = torch.cat((got[512:640], got[1152:1280]))
                    ref_tail = torch.cat((ref[512:640], ref[1152:1280]))
                else:
                    assert ".linear_fc2.weight" in name and ref.shape == (2560, 640), (name, ref.shape)
                    got_tail, ref_tail = got[:, -128:], ref[:, -128:]
                tail_diff, tail_norm = totals.setdefault(
                    "expert_tail", [torch.zeros((), device="cuda"), torch.zeros((), device="cuda")]
                )
                tail_diff.add_((got_tail.float() - ref_tail.float()).square().sum())
                tail_norm.add_(ref_tail.float().square().sum())
        for category, (diff, norm) in totals.items():
            assert norm.item() > 0, category
            errors[category + "_grad"] = float((diff / norm).sqrt())
        diagnostic = {
            "step": step,
            "recompute": args.recompute_first or step == 1,
            "routing": routing,
            "routing_by_microbatch": routing_by_microbatch,
            "microbatch_errors": [
                {
                    "output": relative_error(outputs[1][mb], outputs[0][mb]),
                    "input_grad": relative_error(inputs[1][mb].grad, inputs[0][mb].grad),
                }
                for mb in range(args.microbatches)
            ],
            "errors": errors,
        }
        diagnostic_path = args.output.with_name(args.output.stem + f".rank{dist.get_rank()}.json")
        diagnostic_path.write_text(json.dumps({"completed": rows, "current": diagnostic}, indent=2) + "\n")
        assert all(error < 0.03 for error in errors.values()), diagnostic
        router_before = [module.moe.router.weight.detach().clone() for module in modules]
        for module, optimizer in zip(modules, optimizers):
            gate = module.moe.shared_experts.gate_weight
            before = gate.detach().clone()
            success, grad_norm, _ = optimizer.step()
            assert success and torch.isfinite(torch.as_tensor(grad_norm))
            delta = float((gate.detach().float() - before.float()).abs().max())
            assert delta > 0
            gate_updates.append(delta)
        router_updates = [
            module.moe.router.weight.detach().float() - before.float()
            for module, before in zip(modules, router_before)
        ]
        rows.append(
            {
                "step": step,
                "recompute": args.recompute_first or step == 1,
                "errors": errors,
                "routing": routing,
                "routing_by_microbatch": routing_by_microbatch,
                "microbatch_errors": diagnostic["microbatch_errors"],
                "gate_updates": gate_updates,
                "router_update_relative_error": relative_error(router_updates[1], router_updates[0]),
            }
        )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, rows)
    if dist.get_rank() == 0:
        args.output.write_text(
            json.dumps(
                {
                    "passed": True,
                    "experts": 512,
                    "tp": 2,
                    "etp": 1,
                    "macrobatch_size": args.macrobatch_size,
                    "ep": 8,
                    "tokens_per_rank": args.tokens,
                    "microbatches": args.microbatches,
                    "uneven_tokens": args.uneven_tokens,
                    "lr": args.lr,
                    "recompute_first": args.recompute_first,
                    "backend": type(kernel).__name__,
                    "ranks": gathered,
                    "scope": __doc__,
                },
                indent=2,
            )
            + "\n"
        )
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
