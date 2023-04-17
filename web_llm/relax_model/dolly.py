# pylint: disable=missing-docstring,too-few-public-methods,too-many-instance-attributes,invalid-name,too-many-locals,too-many-arguments
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tvm
from transformers import pipeline  # type: ignore
from tvm import relax, te
from tvm.relax.op import (
    astype,
    broadcast_to,
    full,
    matmul,
    maximum,
    permute_dims,
    reshape,
    squeeze,
    take,
    triu,
)
from tvm.relax.op.nn import gelu, layer_norm, softmax
from tvm.relax.testing import nn
from tvm.script import relax as R


@dataclass
class GPTNeoXConfig:  # pylint: disable=too-many-instance-attributes
    hidden_size: int = 5120
    intermediate_size: int = 20480
    num_attention_heads: int = 40
    num_hidden_layers: int = 36
    layer_norm_eps: float = 1e-05
    max_position_embeddings: int = 2048
    rotary_emb_base: int = 10000
    vocab_size: int = 50280

    rotary_pct: float = 0.25


def dolly_v2_3b(config_only: bool = False):
    config = GPTNeoXConfig(
        hidden_size=2560,
        intermediate_size=10240,
        num_attention_heads=32,
        num_hidden_layers=32,
    )
    if config_only:
        return config, None
    instruct_pipeline = pipeline(model="databricks/dolly-v2-3b", trust_remote_code=True)
    return config, instruct_pipeline


def dolly_v2_7b(config_only: bool = False):
    config = GPTNeoXConfig(
        hidden_size=4096,
        intermediate_size=16384,
        num_attention_heads=32,
        num_hidden_layers=32,
    )
    if config_only:
        return config, None
    instruct_pipeline = pipeline(model="databricks/dolly-v2-7b", trust_remote_code=True)
    return config, instruct_pipeline


def dolly_v2_12b(config_only: bool = False):
    config = GPTNeoXConfig(
        hidden_size=5120,
        intermediate_size=20480,
        num_attention_heads=40,
        num_hidden_layers=36,
    )
    if config_only:
        return config, None
    instruct_pipeline = pipeline(
        model="databricks/dolly-v2-12b", trust_remote_code=True
    )
    return config, instruct_pipeline


def _min_value(dtype) -> relax.Expr:
    return relax.const(tvm.tir.min_value(dtype).value, dtype)


def collect_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    params: Dict[str, nn.Parameter] = {}
    for name, module in model.__dict__.items():
        if isinstance(module, nn.Parameter):
            params[name] = module
        elif isinstance(module, ModuleList):
            for i, m in enumerate(module):
                for param_name, param in collect_parameters(m).items():
                    params[f"{name}.{i}.{param_name}"] = param
        elif isinstance(module, nn.Module):
            for param_name, param in collect_parameters(module).items():
                params[f"{name}.{param_name}"] = param
    return params


class ModuleList(nn.Module):
    def __init__(self, modules: List[nn.Module]):
        self.modules = modules

    def __iter__(self):
        return iter(self.modules)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def forward(self, x: relax.Expr) -> relax.Var:
        for module in self.modules:
            x = module(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((out_features, in_features), name="linear_weight")
        if bias:
            self.bias = nn.Parameter((out_features,), name="linear_bias")
        else:
            self.bias = None

    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(x, self.weight, self.bias))


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter((num_embeddings, embedding_dim), name="weight")

    def forward(self, x: relax.Expr) -> relax.Var:
        ndim = x.struct_info.ndim
        if ndim == 1:
            return nn.emit(take(self.weight, x, axis=0))
        x_shape = x.struct_info.shape.values
        emb_size = self.weight.struct_info.shape.values[-1]
        x = nn.emit(reshape(x, shape=[-1]))
        embedding = nn.emit(take(self.weight, x, axis=0))
        return nn.emit(reshape(embedding, [*x_shape, emb_size]))


class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter((hidden_size,), name="weight")
        self.bias = nn.Parameter((hidden_size,), name="bias")

    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(
            layer_norm(
                x,
                gamma=self.weight,
                beta=self.bias,
                axes=-1,
                epsilon=self.eps,
            )
        )


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        position_embedding_base: int,
        max_sequence_length: int,
        rotary_pct: float,
    ):
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        rotary_ndim = int(head_dim * rotary_pct)
        inv_freq = 1.0 / (
            position_embedding_base
            ** (np.arange(0, rotary_ndim, 2).astype("float32") / rotary_ndim)
        )
        t = np.arange(max_sequence_length, dtype=inv_freq.dtype)
        freq = np.einsum("i,j->ij", t, inv_freq)
        emb = np.concatenate((freq, freq), axis=-1)
        self.rotary_ndim = rotary_ndim
        self.cos_cached = relax.const(tvm.nd.array(np.cos(emb), device=tvm.cpu()))
        self.sin_cached = relax.const(tvm.nd.array(np.sin(emb), device=tvm.cpu()))

    def forward(
        self,
        q: relax.Expr,
        k: relax.Expr,
        offset: relax.Expr,
    ) -> Tuple[relax.Expr, relax.Expr]:
        def rotary_embedding(x, cos, sin, offset):
            def compute(
                i_batch_size,
                i_seq_len,
                i_num_heads,
                i_head_dim,
            ):
                n_feat_half = self.rotary_ndim // 2
                return tvm.tir.Select(
                    i_head_dim < self.rotary_ndim,
                    cos[
                        offset + i_seq_len,
                        i_head_dim,
                    ]
                    * x(i_batch_size, i_seq_len, i_num_heads, i_head_dim)
                    + sin[
                        offset + i_seq_len,
                        i_head_dim,
                    ]
                    * tvm.tir.Select(
                        i_head_dim < n_feat_half,
                        -x[
                            i_batch_size,
                            i_seq_len,
                            i_num_heads,
                            i_head_dim + n_feat_half,
                        ],
                        x[
                            i_batch_size,
                            i_seq_len,
                            i_num_heads,
                            i_head_dim - n_feat_half,
                        ],
                    ),
                    x(i_batch_size, i_seq_len, i_num_heads, i_head_dim),
                )

            return te.compute(x.shape, compute, name="rotary")

        cos, sin = self.cos_cached, self.sin_cached
        q_embed = nn.emit_te(
            rotary_embedding,
            q,
            cos,
            sin,
            offset,
            primfunc_name_hint="rotary_embedding",
        )
        k_embed = nn.emit_te(
            rotary_embedding,
            k,
            cos,
            sin,
            offset,
            primfunc_name_hint="rotary_embedding",
        )
        return q_embed, k_embed


class GPTNeoXAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_embedding: RotaryEmbedding,
    ):
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {hidden_size}"
                f" and `num_heads`: {num_heads})."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_embedding = rotary_embedding
        self.q_proj = Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = Linear(hidden_size, hidden_size, bias=True)
        self.dense = Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        hidden_states: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_value: Optional[Tuple[relax.Expr, relax.Expr]] = None,
        attention_mask: Optional[relax.Expr] = None,
    ) -> Tuple[relax.Expr, Union[Tuple[None, None], Tuple[relax.Expr, relax.Expr]]]:
        # hidden_states: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = hidden_states.struct_info.shape
        kv_seq_len = all_seq_len_shape.struct_info.values[0]

        def _project(proj):
            return nn.emit(
                reshape(
                    proj(hidden_states),
                    (batch_size, seq_len, self.num_heads, self.head_dim),
                )
            )

        # q/k/v states: [batch_size, seq_len, num_attention_heads, head_size]
        q, k, v = (
            _project(self.q_proj),
            _project(self.k_proj),
            _project(self.v_proj),
        )
        q, k = self.rotary_embedding(q, k, kv_seq_len - seq_len)

        if past_key_value is not None:
            f_kv_cache_append = relax.extern("vm.builtin.attention_kv_cache_append")
            f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
            k_cache, v_cache = past_key_value
            k_cache = nn.emit(
                relax.Call(
                    f_kv_cache_append,
                    args=[k_cache, squeeze(k, axis=0)],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
            v_cache = nn.emit(
                relax.Call(
                    f_kv_cache_append,
                    args=[v_cache, squeeze(v, axis=0)],
                    sinfo_args=[relax.ObjectStructInfo()],
                )
            )
            kv_cache_shape = R.shape(
                [
                    kv_seq_len,
                    k.struct_info.shape[2],
                    k.struct_info.shape[3],
                ]
            )
            kv_states_shape = R.shape(
                [
                    k.struct_info.shape[0],  # batch_size == 1
                    kv_seq_len,
                    k.struct_info.shape[2],
                    k.struct_info.shape[3],
                ]
            )
            k = nn.emit(
                relax.Call(
                    f_kv_cache_view,
                    args=[k_cache, kv_cache_shape],
                    sinfo_args=[R.Tensor(kv_cache_shape, k.struct_info.dtype)],
                )
            )
            v = nn.emit(
                relax.Call(
                    f_kv_cache_view,
                    args=[v_cache, kv_cache_shape],
                    sinfo_args=[R.Tensor(kv_cache_shape, v.struct_info.dtype)],
                )
            )
            k = nn.emit(reshape(k, kv_states_shape))
            v = nn.emit(reshape(v, kv_states_shape))
            past_key_value = (k_cache, v_cache)
        else:
            past_key_value = (None, None)

        q = nn.emit(permute_dims(q, [0, 2, 1, 3]))
        k = nn.emit(permute_dims(k, [0, 2, 1, 3]))
        v = nn.emit(permute_dims(v, [0, 2, 1, 3]))

        # Calculate QK
        attn_weights = nn.emit(
            matmul(q, permute_dims(k, [0, 1, 3, 2]))
            / relax.const(
                math.sqrt(self.head_dim),
                q.struct_info.dtype,
            )
        )
        # Apply attention mask
        attn_weights = nn.emit(attn_weights + attention_mask)
        attn_weights = nn.emit(
            maximum(
                attn_weights,
                _min_value(attn_weights.struct_info.dtype),
            )
        )
        # Calculate Softmax(QK)
        if attn_weights.struct_info.dtype != "float32":
            attn_weights = astype(attn_weights, "float32")
        attn_weights = nn.emit(softmax(attn_weights, axis=-1))
        if attn_weights.struct_info.dtype != q.struct_info.dtype:
            attn_weights = astype(attn_weights, q.struct_info.dtype)
        # Calculate Softmax(QK)V
        attn_output = nn.emit(matmul(attn_weights, v))
        # Apply output projection
        attn_output = self.dense(
            reshape(
                permute_dims(attn_output, [0, 2, 1, 3]),
                (batch_size, seq_len, self.hidden_size),
            )
        )
        return attn_output, past_key_value


class GPTNeoXMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.dense_h_to_4h = Linear(hidden_size, intermediate_size)
        self.dense_4h_to_h = Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return nn.emit(hidden_states)


class GPTNeoXLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        num_heads: int,
        rotary_embedding: RotaryEmbedding,
    ):
        self.input_layernorm = LayerNorm(
            hidden_size,
            eps=layer_norm_eps,
        )
        self.post_attention_layernorm = LayerNorm(
            hidden_size,
            eps=layer_norm_eps,
        )
        self.attention = GPTNeoXAttention(
            hidden_size,
            num_heads=num_heads,
            rotary_embedding=rotary_embedding,
        )
        self.mlp = GPTNeoXMLP(hidden_size, intermediate_size=intermediate_size)

    def forward(
        self,
        hidden_states,
        all_seq_len_shape: relax.Expr,
        past_key_value: Optional[Tuple[relax.Expr]] = None,
        attention_mask: Optional[relax.Expr] = None,
    ):
        attn_output, present_key_value = self.attention(
            self.input_layernorm(hidden_states),
            all_seq_len_shape,
            past_key_value,
            attention_mask,
        )
        mlp_output = self.mlp(
            self.post_attention_layernorm(hidden_states),
        )
        hidden_states = nn.emit(mlp_output + attn_output + hidden_states)
        return hidden_states, present_key_value


def _prepare_decoder_attention_mask(input_shape, src_len, dtype):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    if isinstance(input_shape[-1], tvm.tir.Var) or input_shape[-1] > 1:
        bsz, tgt_len = input_shape
        mask = full((tgt_len, tgt_len), _min_value(dtype))
        mask = triu(mask, k=1)
        mask = broadcast_to(mask, (bsz, 1, tgt_len, tgt_len))
    else:
        # Get src_len from input parameters
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        bsz, tgt_len = input_shape
        mask = relax.op.zeros((bsz, 1, tgt_len, src_len), dtype)
    return nn.emit(mask)


class GPTNeoXModel(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
    ):
        rotary_embedding = RotaryEmbedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            position_embedding_base=config.rotary_emb_base,
            max_sequence_length=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
        )
        self.embed_in = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = ModuleList(
            [
                GPTNeoXLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    layer_norm_eps=config.layer_norm_eps,
                    num_heads=config.num_attention_heads,
                    rotary_embedding=rotary_embedding,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = LayerNorm(
            hidden_size=config.hidden_size,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: Optional[Tuple[relax.Expr, relax.Expr]],
    ):
        batch_size, seq_length = input_ids.struct_info.shape
        seq_length_with_past = all_seq_len_shape.struct_info.values[0]
        # embed positions
        hidden_states = self.embed_in(input_ids)
        attention_mask = _prepare_decoder_attention_mask(
            (batch_size, seq_length),
            seq_length_with_past,
            dtype=hidden_states.struct_info.dtype,
        )
        present_kv_cache = []
        for i, layer in enumerate(self.layers):
            past_key_value = (
                (past_key_values[i * 2], past_key_values[i * 2 + 1])
                if past_key_values is not None
                else None
            )
            hidden_states, (present_k_cache, present_v_cache) = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                all_seq_len_shape=all_seq_len_shape,
            )
            present_kv_cache.append(present_k_cache)
            present_kv_cache.append(present_v_cache)
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, present_kv_cache


class GPTNeoXForCausalLM(nn.Module):
    def __init__(
        self,
        config: GPTNeoXConfig,
    ):
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: Optional[List[relax.Expr]],
    ):
        hidden_states, key_value_cache = self.gpt_neox(
            input_ids=input_ids,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )

        def _slice(x: te.Tensor):
            _, seq_len, hidden_dim = x.shape
            return te.compute(
                shape=(1, 1, hidden_dim),
                fcompute=lambda i, _, k: x[i, seq_len - 1, k],
                name="slice",
            )

        hidden_states = nn.emit_te(
            _slice,
            hidden_states,
            primfunc_name_hint="slice",
        )
        logits = self.embed_out(hidden_states)
        return logits, key_value_cache


def create_encoding_func(
    bb: relax.BlockBuilder,
    config: GPTNeoXConfig,
    ordered_params: List[str],
) -> None:
    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.Var("n", "int64")
    all_seq_len = seq_len
    with bb.function("encoding"):
        model = GPTNeoXForCausalLM(config)
        input_ids = nn.Placeholder(
            (batch_size, seq_len), dtype="int32", name="input_ids"
        )
        all_seq_len_shape = relax.Var(
            "all_seq_len",
            relax.ShapeStructInfo((all_seq_len,)),
        )
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids=input_ids,
                all_seq_len_shape=all_seq_len_shape,
                past_key_values=past_key_values,
            )
            params = [
                input_ids,
                all_seq_len_shape,
                past_key_values,
            ]
            named_params = collect_parameters(model)
            for name in ordered_params:
                params.append(named_params[name])
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)
    mod = bb.get()
    gv = mod.get_global_var("encoding")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_decoding_func(
    bb: relax.BlockBuilder,
    config: GPTNeoXConfig,
    ordered_params: List[str],
) -> None:
    batch_size = tvm.tir.IntImm("int64", 1)
    seq_len = tvm.tir.IntImm("int64", 1)
    all_seq_len = tvm.tir.Var("n", "int64")
    with bb.function("decoding"):
        model = GPTNeoXForCausalLM(config)
        input_ids = nn.Placeholder(
            (batch_size, seq_len), dtype="int32", name="input_ids"
        )
        all_seq_len_shape = relax.Var(
            "all_seq_len",
            relax.ShapeStructInfo((all_seq_len,)),
        )
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids=input_ids,
                all_seq_len_shape=all_seq_len_shape,
                past_key_values=past_key_values,
            )
            params = [
                input_ids,
                all_seq_len_shape,
                past_key_values,
            ]
            named_params = collect_parameters(model)
            for name in ordered_params:
                params.append(named_params[name])
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)
    mod = bb.get()
    gv = mod.get_global_var("decoding")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))
