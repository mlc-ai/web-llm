from typing import List, Optional
import argparse
import os

import torch
from web_llm import utils

from transformers import AutoTokenizer, AutoModelForCausalLM
import tvm
from tvm import relax
from web_llm.conversation import SeparatorStyle, conv_templates


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument("--model", type=str, default="vicuna-7b")
    args.add_argument("--max-gen-len", type=int, default=128)
    args.add_argument("--run-torch-model", action="store_true", default=False)
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


class ModelWrapper:
    def __init__(self, model: callable, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
    ):
        prompt_tokens = self.tokenizer.encode(prompt)

        total_len = max_gen_len + len(prompt_tokens)
        tokens = torch.full((1, total_len), self.tokenizer.pad_token_id).to(
            torch.int32
        )
        tokens[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens)
        start_pos = len(prompt_tokens)
        for cur_pos in range(start_pos, total_len):
            if cur_pos == start_pos:
                logits = self.model(tokens[:, :cur_pos], cur_pos, clear_cache=True)
            else:
                logits = self.model(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            tokens[:, cur_pos] = next_token
            # the following code assumes bsz == 1
            if next_token[0] == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            i = cur_pos - start_pos
            if i % stream_interval == 0 or i == max_gen_len - 1 or stopped:
                output = tokens[0, : cur_pos + 1]
                output = tokenizer.decode(output, skip_special_tokens=True)
                pos = output.rfind(stop_str, len(prompt))
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output
            if stopped:
                break


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def chat(model_wrapper, args):

    # Chat
    conv = conv_templates["v1"].copy()
    while True:
        try:
            inp = input(f"{conv.roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print(f"{conv.roles[1]}: ", end="", flush=True)
        pre = 0
        for outputs in model_wrapper.generate(
            prompt,
            args.max_gen_len,
            stop_str=conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        ):
            outputs = outputs[len(prompt) + 1 :].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                print(" ".join(outputs[pre : now - 1]), end=" ", flush=True)
                pre = now - 1
        print(" ".join(outputs[pre:]), flush=True)

        conv.messages[-1][-1] = " ".join(outputs)
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


def get_tvm_model(args):
    device = tvm.device(args.device_name)
    const_params = utils.load_params(args.artifact_path, device)
    ex = tvm.runtime.load_module(f"{args.artifact_path}/{args.model}_{args.device_name}.so")
    vm = relax.VirtualMachine(ex, device)

    class Model:
        def new_cache(self):
            fcreate_cache = tvm.get_global_func("vm.builtin.attention_kv_cache_create")
            self.kv_cache = []
            for i in range(64):  # num_layer
                kv_cache = fcreate_cache(
                    tvm.nd.empty((1, 32, 128), device=device, dtype="float32"),
                    tvm.runtime.ShapeTuple([32, 32, 128]),
                    0
                )
                self.kv_cache.append(kv_cache)

        def __init__(self) -> None:
            self.kv_cache = None
            self.new_cache()

        def forward(
            self, inputs: torch.Tensor, cur_pos: int, clear_cache: bool = False
        ) -> torch.Tensor:
            if clear_cache:
                self.new_cache()
            inputs = tvm.nd.array(inputs.numpy(), device=device)
            seq_len_shape = tvm.runtime.ShapeTuple([cur_pos])
            if inputs.shape[1] > 1:
                logits, kv_cache = vm["encoding"](
                    inputs, seq_len_shape, self.kv_cache, const_params
                )
            else:
                logits, kv_cache = vm["decoding"](
                    inputs, seq_len_shape, self.kv_cache, const_params
                )
            self.kv_cache = kv_cache

            return torch.from_numpy(logits.numpy())

    model = Model()
    return model.forward


def get_pytorch_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        logits = model(inputs, use_cache=False).logits
        return logits

    return forward


if __name__ == "__main__":
    ARGS = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if not ARGS.run_torch_model:
        model = ModelWrapper(get_tvm_model(ARGS), tokenizer)
    else:
        model = ModelWrapper(get_pytorch_model(ARGS), tokenizer)
    chat(model, ARGS)
