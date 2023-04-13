import argparse
import time

import torch

from web_llm import utils

import os
import tvm
from tvm import relax


from transformers import AutoTokenizer
from tvm.script import relax as R
from tvm.runtime import ShapeTuple
import numpy as np


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--prompt", type=str, default="The capital of Canada is")
    args.add_argument("--model", type=str, default="vicuna-7b")
    parsed = args.parse_args()
    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(parsed.artifact_path, parsed.model)
    if parsed.device_name == "auto":
        if tvm.cuda().exist:
            parsed.device_name = "cuda"
        elif tvm.metal().exist:
            parsed.device_name = "metal"
        else:
            raise ValueError("Cannot auto deduce device-name, please set it")
    return parsed


def deploy_to_pipeline(args) -> None:
    device = tvm.device(args.device_name)
    const_params = utils.load_params(args.artifact_path, device)
    ex = tvm.runtime.load_module(os.path.join(
        args.artifact_path, f"{args.model}_{args.device_name}.so"))
    vm = relax.VirtualMachine(ex, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print('tokenizing...')
    inputs = tvm.nd.array(
        tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy(),
        device,
    )
    first_sampled_token = tvm.nd.array(np.array([[6234]]).astype("int32"), device)
    seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1]])
    second_seq_len_shape = tvm.runtime.ShapeTuple([inputs.shape[1] + 1])
    fcreate_cache = tvm.get_global_func("vm.builtin.attention_kv_cache_create")
    kv_caches = []
    for i in range(64):
        kv_cache = fcreate_cache(
            tvm.nd.empty((1, 32, 128), device=device, dtype="float32"),
            tvm.runtime.ShapeTuple([6, 32, 128]),
            0
        )
        kv_caches.append(kv_cache)
    print("Running inference...")
    start = time.time()
    logits, kv_caches = vm["encoding"](inputs, seq_len_shape, kv_caches, const_params)
    device.sync()
    encoding_end = time.time()
    logits, kv_caches = vm["decoding"](
        first_sampled_token, second_seq_len_shape, kv_caches, const_params
    )
    device.sync()
    end = time.time()
    fcache_view = tvm.get_global_func("vm.builtin.attention_kv_cache_view")
    first_k_cache = fcache_view(kv_caches[0], ShapeTuple([7, 32, 128]))
    if args.debug_dump:
        print(f"output kv_cache[0]:\n{first_k_cache.numpy().transpose(1, 0, 2)}")
        print(f"output logits:\n{logits.numpy()}")
    print(f"Time elapsed: encoding {(encoding_end - start)} seconds, decodig {end - encoding_end} secs")


if __name__ == "__main__":
    ARGS = _parse_args()
    deploy_to_pipeline(ARGS)
