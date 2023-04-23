import argparse
import os
from typing import Callable

import torch
import tvm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tvm import relax

from web_llm import utils
from web_llm.conversation import SeparatorStyle, conv_templates, compute_skip_echo_len
from web_llm.relax_model import dolly

NUM_HIDDEN_LAYERS = 32
HIDDEN_DIM = 4096
NUM_ATTENTION_HEADS = 32
CONV = "vicuna_v1.1"


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--device-name", type=str, default="auto")
    args.add_argument("--debug-dump", action="store_true", default=False)
    args.add_argument("--artifact-path", type=str, default="dist")
    args.add_argument(
        "--model",
        type=str,
        default="dolly-v2-3b",
        choices=[
            "vicuna-7b-v1",
            "dolly-v2-3b",
            "dolly-v2-7b",
            "dolly-v2-12b",
            "stablelm-tuned-alpha-3b"
        ],
    )
    args.add_argument("--max-gen-len", type=int, default=1280)
    args.add_argument("--run-torch-model", action="store_true", default=False)
    parsed = args.parse_args()
    parsed.model_path = os.path.join(parsed.artifact_path, "models", parsed.model)
    parsed.artifact_path = os.path.join(parsed.artifact_path, parsed.model)

    if parsed.model in ["vicuna-7b"]:
        parsed.hf_model_path = parsed.model_path
    elif parsed.model in ["dolly-v2-3b", "dolly-v2-7b", "dolly-v2-12b"]:
        parsed.hf_model_path = "databricks/" + parsed.model
        cfg, _ = getattr(dolly, parsed.model.replace("-", "_"))(config_only=True)
        global NUM_HIDDEN_LAYERS, HIDDEN_DIM, NUM_ATTENTION_HEADS, CONV
        NUM_HIDDEN_LAYERS = cfg.num_hidden_layers
        HIDDEN_DIM = cfg.hidden_size
        NUM_ATTENTION_HEADS = cfg.num_attention_heads
        CONV = "dolly"
    elif parsed.model in ["stablelm-tuned-alpha-3b"]:
        parsed.hf_model_path = "stabilityai/" + parsed.model
        cfg, _ = getattr(dolly, parsed.model.replace("-", "_"))(config_only=True)
        # global NUM_HIDDEN_LAYERS, HIDDEN_DIM, NUM_ATTENTION_HEADS, CONV
        NUM_HIDDEN_LAYERS = cfg.num_hidden_layers
        HIDDEN_DIM = cfg.hidden_size
        NUM_ATTENTION_HEADS = cfg.num_attention_heads
        CONV = "stablelm"
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


class ModelWrapper:
    def __init__(self, model: Callable, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream_interval: int = 2,
        stop_str: str = None,
        stop_tokens = None,
    ):
        prompt_tokens = self.tokenizer.encode(prompt)
        stop_tokens = [tokenizer.eos_token_id] if stop_tokens is None else stop_tokens
        total_len = max_gen_len + len(prompt_tokens)
        tokens = torch.full((1, total_len), 0).to(torch.int32)
        tokens[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens)
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
            if next_token[0] in stop_tokens:
                stopped = True
            else:
                stopped = False

            i = cur_pos - start_pos
            if i % stream_interval == 0 or i == max_gen_len - 1 or stopped:
                output = tokens[0, : cur_pos + 1]
                output = tokenizer.decode(output, skip_special_tokens=True)
                if stop_str:
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
    conv = conv_templates[CONV].copy()
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
        skip_echo_len = compute_skip_echo_len(CONV, conv, prompt)
        for outputs in model_wrapper.generate(
            prompt,
            args.max_gen_len,
            stop_str=conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
            stop_tokens = [50278, 50279, 50277, 1, 0] if CONV == "stablelm" else None,
        ):
            outputs = outputs[skip_echo_len :].strip()
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
    ex = tvm.runtime.load_module(
        f"{args.artifact_path}/{args.model}_{args.device_name}.so"
    )
    vm = relax.VirtualMachine(ex, device)

    class Model:
        def new_cache(self):
            fcreate_cache = tvm.get_global_func("vm.builtin.attention_kv_cache_create")
            self.kv_cache = []
            for i in range(NUM_HIDDEN_LAYERS * 2):
                kv_cache = fcreate_cache(
                    tvm.nd.empty(
                        (1, NUM_ATTENTION_HEADS, HIDDEN_DIM // NUM_ATTENTION_HEADS),
                        device=device,
                        dtype="float32",
                    ),
                    tvm.runtime.ShapeTuple(
                        [32, NUM_ATTENTION_HEADS, HIDDEN_DIM // NUM_ATTENTION_HEADS]
                    ),
                    0,
                )
                self.kv_cache.append(kv_cache)

        def __init__(self) -> None:
            self.kv_cache = None  # type: ignore
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
    model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, device_map="auto")
    
    class Model:
        def __init__(self):
            self.past_key_values = None
        
        def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            output = model(inputs, use_cache=True, past_key_values=self.past_key_values)
            self.past_key_values = output.past_key_values
            return output.logits
    wrapped_model = Model()
    return wrapped_model.forward


if __name__ == "__main__":
    ARGS = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained(ARGS.hf_model_path)
    if "dolly" in ARGS.hf_model_path:
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
    if not ARGS.run_torch_model:
        model = ModelWrapper(get_tvm_model(ARGS), tokenizer)
    else:
        model = ModelWrapper(get_pytorch_model(ARGS), tokenizer)
    chat(model, ARGS)
