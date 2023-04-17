import argparse
import os
import time
from typing import List

import numpy as np
import torch
import tvm
from transformers import AutoTokenizer
from tvm import relax
from tvm.relax.testing.lib_comparator import LibCompareVMInstrument
from tvm.runtime import ShapeTuple

from web_llm import utils
from web_llm.relax_model import dolly

NUM_HIDDEN_LAYERS = 32
HIDDEN_DIM = 4096
NUM_ATTENTION_HEADS = 32


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--prompt", type=str, default="The capital of Canada is")
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
    args.add_argument("--profile", action="store_true", default=False)
    parsed = args.parse_args()
    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(parsed.artifact_path, parsed.model)
    if parsed.model in ["vicuna-7b"]:
        parsed.hf_model_path = parsed.model_path
    elif parsed.model in ["dolly-v2-3b", "dolly-v2-7b", "dolly-v2-12b"]:
        parsed.hf_model_path = "databricks/" + parsed.model
        cfg, _ = getattr(dolly, parsed.model.replace("-", "_"))(config_only=True)
        global NUM_HIDDEN_LAYERS
        global HIDDEN_DIM
        global NUM_ATTENTION_HEADS
        NUM_HIDDEN_LAYERS = cfg.num_hidden_layers
        HIDDEN_DIM = cfg.hidden_size
        NUM_ATTENTION_HEADS = cfg.num_attention_heads
    else:
        raise ValueError(f"Unknown model {parsed.model}")
    if parsed.device_name == "auto":
        if tvm.cuda().exist:
            parsed.device_name = "cuda"
        elif tvm.metal().exist:
            parsed.device_name = "metal"
        else:
            raise ValueError("Cannot auto deduce device-name, please set it")
    return parsed


class LibCompare(LibCompareVMInstrument):
    def __init__(self, mod, device):
        super().__init__(mod, device, True)
        self.time_eval_results = {}

    def compare(
        self,
        name: str,
        ref_args: List[tvm.nd.NDArray],
        new_args: List[tvm.nd.NDArray],
        ret_indices: List[int],
    ):
        super().compare(name, ref_args, new_args, ret_indices)
        if name not in self.time_eval_results:
            res = self.mod.time_evaluator(name, dev=self.device)(*new_args)
            self.time_eval_results[name] = res
            print(f"Time-eval result {name} on {self.device}: {res}")


def create_kv_caches(device, num_input_tokens):
    kv_caches = []
    fcreate_cache = tvm.get_global_func("vm.builtin.attention_kv_cache_create")
    for _ in range(NUM_HIDDEN_LAYERS * 2):
        kv_cache = fcreate_cache(
            tvm.nd.empty(
                (
                    1,
                    NUM_ATTENTION_HEADS,
                    HIDDEN_DIM // NUM_ATTENTION_HEADS,
                ),
                device=device,
                dtype="float32",
            ),
            tvm.runtime.ShapeTuple(
                [
                    num_input_tokens + 1,
                    NUM_ATTENTION_HEADS,
                    HIDDEN_DIM // NUM_ATTENTION_HEADS,
                ]
            ),
            0,
        )
        kv_caches.append(kv_cache)
    return kv_caches


def deploy_to_pipeline(args) -> None:
    device = tvm.device(args.device_name)
    const_params = utils.load_params(args.artifact_path, device)
    ex = tvm.runtime.load_module(
        os.path.join(args.artifact_path, f"{args.model}_{args.device_name}.so")
    )
    vm = relax.VirtualMachine(ex, device)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)

    print("Tokenizing...")
    inputs = tvm.nd.array(
        tokenizer(args.prompt, return_tensors="pt").input_ids.to(torch.int32).numpy(),
        device,
    )
    _, num_input_tokens = inputs.shape
    first_sampled_token = tvm.nd.array(np.array([[6234]]).astype("int32"), device)
    seq_len_shape = tvm.runtime.ShapeTuple([num_input_tokens])
    second_seq_len_shape = tvm.runtime.ShapeTuple([num_input_tokens + 1])
    kv_caches = create_kv_caches(device, num_input_tokens)
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
    first_k_cache = fcache_view(
        kv_caches[0],
        ShapeTuple(
            [
                num_input_tokens + 1,
                NUM_ATTENTION_HEADS,
                HIDDEN_DIM // NUM_ATTENTION_HEADS,
            ]
        ),
    )
    if args.debug_dump:
        print(f"output kv_cache[0]:\n{first_k_cache.numpy().transpose(1, 0, 2)}")
        print(f"output logits:\n{logits.numpy()}")
    print(
        f"Time elapsed: encoding {(encoding_end - start)} seconds, decoding {end - encoding_end} secs"
    )

    if args.profile:
        from contextlib import redirect_stdout

        cmp_instrument = LibCompare(ex, device)
        vm.set_instrument(cmp_instrument)

        print("Profiling...")
        profile_file_path = os.path.join(
            args.artifact_path, "debug", "evaluate_profile.log"
        )
        with open(profile_file_path, "w") as file:
            with redirect_stdout(file):
                kv_caches = create_kv_caches(device)
                print("======================= Starts Encoding =======================")
                logits, kv_caches = vm["encoding"](
                    inputs, seq_len_shape, kv_caches, const_params
                )
                print("======================= Starts Decoding =======================")
                logits, kv_caches = vm["decoding"](
                    first_sampled_token, second_seq_len_shape, kv_caches, const_params
                )
        print(f"Save the profiling results to {profile_file_path}")


if __name__ == "__main__":
    ARGS = _parse_args()
    deploy_to_pipeline(ARGS)
