import argparse
import os
import pickle
from platform import system
from typing import List, Optional, Tuple

import numpy as np
import tvm
import tvm.testing
from tvm import relax
from tvm.runtime import NDArray

import web_llm
from web_llm import utils
from web_llm.relax_model import dolly, llama


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="dolly-v2-3b",
        choices=[
            "vicuna-7b-v1",
            "dolly-v2-3b",
            "dolly-v2-7b",
            "dolly-v2-12b",
        ],
    )
    args.add_argument("--target", type=str, default="auto")
    args.add_argument("--db-path", type=str, default="log_db/")
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument(
        "--use-cache",
        type=int,
        default=1,
        help="Whether to use previously pickled IRModule and skip trace.",
    )
    args.add_argument("--debug-dump", action="store_true", default=False)

    parsed = args.parse_args()
    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(parsed.artifact_path, parsed.model)
    if parsed.target == "auto":
        if system() == "Darwin":
            target = tvm.target.Target("apple/m1-gpu")
        else:
            has_gpu = tvm.cuda().exist
            target = tvm.target.Target("cuda" if has_gpu else "llvm")
        print(f"Automatically configuring target: {target}")
        parsed.target = tvm.target.Target(target, host="llvm")
    elif parsed.target == "webgpu":
        parsed.target = tvm.target.Target(
            "webgpu", host="llvm -mtriple=wasm32-unknown-unknown-wasm"
        )
    else:
        parsed.target = tvm.target.Target(parsed.target, host="llvm")
    return parsed


def debug_dump_script(mod, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    dump_path = os.path.join(args.artifact_path, "debug", name)
    with open(dump_path, "w") as outfile:
        outfile.write(mod.script(show_meta=True))
    print(f"Dump mod to {dump_path}")


def debug_dump_shader(ex, name, args):
    """Debug dump mode"""
    if not args.debug_dump:
        return
    target_kind = args.target.kind.default_keys[0]
    suffix_map = {
        "webgpu": ".wgsl",
        "cuda": ".cu",
        "metal": ".mtl",
    }
    suffix = suffix_map.get(target_kind, ".txt")
    dump_path = os.path.join(args.artifact_path, "debug", name + suffix)
    source = ex.mod.imported_modules[0].imported_modules[0].get_source()
    with open(dump_path, "w") as outfile:
        outfile.write(source)
    print(f"Dump shader to {dump_path}")


def get_models(
    config,
    model,
    max_sequence_length,
    params: Optional[List[str]] = None,
):
    if "vicuna" in model or "llama" in model:
        bb = relax.BlockBuilder()
        llama.create_encoding_func(bb, config)
        llama.create_encoding_func_without_cache(bb, config)
        llama.create_decoding_func(bb, config)
        mod = bb.get()
        for gv in mod.functions:
            func = mod[gv]
            if isinstance(func, relax.Function):
                mod[gv] = func.with_attr(
                    "tir_var_upper_bound", {"n": max_sequence_length}
                )
        return mod
    elif model in ["dolly-v2-3b", "dolly-v2-7b", "dolly-v2-12b"]:
        bb = relax.BlockBuilder()
        dolly.create_encoding_func(bb, config, params)
        dolly.create_decoding_func(bb, config, params)
        mod = bb.get()
        for gv in mod.functions:
            func = mod[gv]
            if isinstance(func, relax.Function):
                mod[gv] = func.with_attr(
                    "tir_var_upper_bound", {"n": max_sequence_length}
                )
        return mod
    else:
        raise ValueError(f"Model {model} not supported")


def get_named_params(
    config: dolly.GPTNeoXConfig,
    model,
    split_qkv: bool,
) -> List[Tuple[str, NDArray]]:
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    def _split_weight(qkv_weight):
        qkv_weight = qkv_weight.reshape(num_heads, 3, head_dim, hidden_size)
        return (
            qkv_weight[:, 0, :, :].reshape(hidden_size, hidden_size),
            qkv_weight[:, 1, :, :].reshape(hidden_size, hidden_size),
            qkv_weight[:, 2, :, :].reshape(hidden_size, hidden_size),
        )

    def _split_bias(qkv_bias):
        qkv_bias = qkv_bias.reshape(num_heads, 3, head_dim)
        return (
            qkv_bias[:, 0, :].reshape(hidden_size),
            qkv_bias[:, 1, :].reshape(hidden_size),
            qkv_bias[:, 2, :].reshape(hidden_size),
        )

    param_list: List[Tuple[str, NDArray]] = []
    for name, param in model.named_parameters():
        param = param.detach().cpu().numpy()
        if split_qkv:
            if name.endswith("query_key_value.weight"):
                name = name.replace("query_key_value.weight", "{}_proj.weight")
                assert param.ndim == 2
                q, k, v = _split_weight(param)
                param_list.append((name.format("q"), q))
                param_list.append((name.format("k"), k))
                param_list.append((name.format("v"), v))
                continue
            elif name.endswith("query_key_value.bias"):
                name = name.replace("query_key_value.bias", "{}_proj.bias")
                assert param.ndim == 1
                q, k, v = _split_bias(param)
                param_list.append((name.format("q"), q))
                param_list.append((name.format("k"), k))
                param_list.append((name.format("v"), v))
                continue
        param_list.append((name, param))
    tvm_param_list = []
    for name, param in param_list:
        tvm_param_list.append((name, tvm.nd.array(param, tvm.cpu())))
    print("Total parameters: ", len(param_list))
    return tvm_param_list


def get_params(config, model):
    import numpy as np

    param_list = []
    for _, param in model.named_parameters():
        param_list.append(tvm.nd.array(param.detach().cpu().numpy(), tvm.cpu()))

    ############ Rotary embedding constants ############
    head_dim = config.hidden_size / config.num_attention_heads
    inv_freq = 1.0 / (
        config.position_embedding_base
        ** (np.arange(0, head_dim, 2).astype("float32") / head_dim)
    )

    t = np.arange(config.max_sequence_length, dtype=inv_freq.dtype)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    param_list.append(tvm.nd.array(np.cos(emb), tvm.cpu()))
    param_list.append(tvm.nd.array(np.sin(emb), tvm.cpu()))
    ############ End ############

    return param_list


def mod_transform_before_build(
    mod: tvm.IRModule,
    model_params: List[tvm.nd.NDArray],
    args: argparse.Namespace,
) -> tvm.IRModule:
    """First-stage: Legalize ops and trace"""
    model_names = ["encoding", "decoding"]

    mod = web_llm.transform.GroupQuantize(group_size=32, sym=False)(mod)
    mod = web_llm.transform.FuseTransposeMatmul()(mod)

    # NOTE: enable pipeline after fusion getting fixed.
    # mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod["full"] = mod["full"].with_attr("op_pattern", 8)
    mod = relax.transform.FoldConstant()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)

    mod = web_llm.transform.FuseDecodeMatmulEwise()(mod)
    mod = relax.transform.DeadCodeElimination(model_names)(mod)
    mod = relax.transform.LiftTransformParams()(mod)
    mod_transform, mod_deploy = utils.split_transform_deploy_mod(mod, model_names)
    debug_dump_script(mod_transform, "mod_lift_params.py", args)

    new_params = utils.transform_params(mod_transform, model_params)
    utils.save_params(new_params, args.artifact_path)
    return mod_deploy


def build(
    mod_deploy: tvm.IRModule,
    args: argparse.Namespace,
) -> None:
    target_kind = args.target.kind.default_keys[0]

    debug_dump_script(mod_deploy, "mod_before_build.py", args)
    if target_kind != "cpu":
        from tvm import meta_schedule as ms

        db = ms.database.create(work_dir=args.db_path)
        with db, tvm.target.Target("apple/m1-gpu-restricted"):
            mod_deploy = relax.transform.MetaScheduleApplyDatabase()(mod_deploy)
            mod_deploy = web_llm.transform.DispatchTIROperator()(mod_deploy)
            mod_deploy = tvm.tir.transform.DefaultGPUSchedule()(mod_deploy)
            mod_deploy = tvm.tir.transform.ForceNarrowIndexToInt32()(mod_deploy)

    debug_dump_script(mod_deploy, "mod_build_stage.py", args)

    ex = relax.build(mod_deploy, args.target)

    if target_kind == "webgpu":
        output_filename = f"{args.model}_{target_kind}.wasm"
    else:
        output_filename = f"{args.model}_{target_kind}.so"

    debug_dump_shader(ex, f"{args.model}_{target_kind}", args)
    ex.export_library(os.path.join(args.artifact_path, output_filename))


if __name__ == "__main__":
    ARGS = _parse_args()
    os.makedirs(ARGS.artifact_path, exist_ok=True)
    os.makedirs(os.path.join(ARGS.artifact_path, "debug"), exist_ok=True)
    cache_path = os.path.join(ARGS.artifact_path, "mod_cache_before_build.pkl")
    use_cache = ARGS.use_cache and os.path.isfile(cache_path)
    if not use_cache:
        if ARGS.model in ["vicuna-7b"]:
            from transformers import AutoModelForCausalLM  # type: ignore

            hf_model = AutoModelForCausalLM.from_pretrained(ARGS.model_path)
            config = utils.get_config(hf_model.config, ARGS.model)
            mod = get_models(config, ARGS.model, config.max_sequence_length, None)
            params = get_params(config, hf_model)
            del hf_model
            mod = mod_transform_before_build(mod, params, ARGS)
            with open(cache_path, "wb") as outfile:
                pickle.dump(mod, outfile)
            print(f"Save a cached module to {cache_path}.")
        elif ARGS.model in ["dolly-v2-3b", "dolly-v2-7b", "dolly-v2-12b"]:
            config, hf_model = {
                "dolly-v2-3b": dolly.dolly_v2_3b,
                "dolly-v2-7b": dolly.dolly_v2_7b,
                "dolly-v2-12b": dolly.dolly_v2_12b,
            }[ARGS.model]()
            params = get_named_params(config, hf_model.model, split_qkv=True)
            mod = get_models(
                config,
                ARGS.model,
                max_sequence_length=2048,
                params=[k for k, _ in params],
            )
            del hf_model
            mod = mod_transform_before_build(mod, params, ARGS)
            with open(cache_path, "wb") as outfile:
                pickle.dump(mod, outfile)
            print(f"Save a cached module to {cache_path}.")
    else:
        print(
            f"Load cached module from {cache_path} and skip tracing. "
            "You can use --use-cache=0 to retrace"
        )
        mod = pickle.load(open(cache_path, "rb"))
    build(mod, ARGS)
