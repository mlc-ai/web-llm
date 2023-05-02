import tvm
from tvm import relax
from typing import List, Tuple
import tvm.testing
from tvm import meta_schedule as ms


def get_config(hf_config, model):
    if "vicuna" in model or "llama" in model or "wizardlm" in model:
        from .relax_model.llama import LlamaConfig as RelaxConfig

        return RelaxConfig(
            #max_sequence_length=hf_config.max_sequence_length,
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            hidden_act=hf_config.hidden_act,
            initializer_range=hf_config.initializer_range,
            rms_norm_eps=hf_config.rms_norm_eps,
            use_cache=hf_config.use_cache,
            pad_token_id=hf_config.pad_token_id,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            tie_word_embeddings=hf_config.tie_word_embeddings,
        )
    else:
        raise ValueError(f"Model {model} not supported")
    

def split_transform_deploy_mod(
    mod: tvm.IRModule, model_names: List[str]
) -> Tuple[tvm.IRModule, tvm.IRModule]:
    mod_transform = tvm.IRModule()
    mod_deploy = tvm.IRModule()

    transform_func_name = None
    gv_names = [gv.name_hint for gv in mod.get_global_vars()]
    for name in model_names:
        if (name + "_transform_params") in gv_names:
            transform_func_name = name + "_transform_params"
    assert transform_func_name is not None

    for gv in mod.functions:
        func = mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
            mod_deploy[gv] = func
        elif gv.name_hint == transform_func_name:
            mod_transform[gv] = func
        else:
            mod_deploy[gv] = func

    mod_transform = relax.transform.DeadCodeElimination([transform_func_name])(
        mod_transform
    )
    mod_deploy = relax.transform.DeadCodeElimination(model_names)(mod_deploy)

    return mod_transform, mod_deploy


def transform_params(
    mod_transform: tvm.IRModule, model_params: List[tvm.nd.NDArray]
) -> List[tvm.nd.NDArray]:
    transform_func_name = None
    for gv, func in mod_transform.functions.items():
        if isinstance(func, relax.Function):
            transform_func_name = gv.name_hint
    assert transform_func_name is not None

    ex = relax.build(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm.cpu())
    res = vm[transform_func_name](model_params)
    return res


def save_params(params: List[tvm.nd.NDArray], artifact_path: str) -> None:
    from tvm.contrib import tvmjs

    meta_data = {}
    param_dict = {}
    meta_data["ParamSize"] = len(params)
    for i, nd in enumerate(params):
        param_dict[f"param_{i}"] = nd
    tvmjs.dump_ndarray_cache(
        param_dict, f"{artifact_path}/params", meta_data=meta_data, encode_format="raw"
    )


def load_params(artifact_path: str, device) -> List[tvm.nd.NDArray]:
    from tvm.contrib import tvmjs

    params, meta = tvmjs.load_ndarray_cache(f"{artifact_path}/params", device)
    plist = []
    size = meta["ParamSize"]
    for i in range(size):
        plist.append(params[f"param_{i}"])
    return plist


def build_model_from_log(relax_mod, target, log_dir):
    db = ms.database.create(work_dir=log_dir)
    with target, db, tvm.transform.PassContext(opt_level=3):
        relax_mod = relax.transform.MetaScheduleApplyDatabase()(relax_mod)
    return relax_mod
