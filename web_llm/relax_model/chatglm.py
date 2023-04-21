from typing import Optional, List, Tuple

import math

import tvm
from tvm import relax
from tvm import te
from tvm.relax.testing import nn
from tvm.script import relax as R


class ChatGLMConfig:
    def __init__(
        self,
        vocab_size=130528,
        hidden_size=4096,
        num_layers=28,
        num_attention_heads=32,
        layernorm_epsilon=1e-5,
        bos_token_id=130004,
        eos_token_id=130005,
        mask_token_id=130000,
        gmask_token_id=130001,
        pad_token_id=3,
        max_sequence_length=2048,
        inner_hidden_size=16384,
        position_encoding_2d=True,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.layernorm_epsilon = layernorm_epsilon
        self.inner_hidden_size = inner_hidden_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.gmask_token_id = gmask_token_id
        self.position_encoding_2d = position_encoding_2d
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.kwargs = kwargs


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((out_features, in_features), name="linear_weight", dtype="float32")
        if bias:
            self.bias = nn.Parameter((out_features,), name="linear_bias", dtype="float32")
        else:
            self.bias = None

    def forward(self, input: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(input, self.weight, self.bias))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        self.weight = nn.Parameter((hidden_size,), name="layer_norm_weight", dtype="float32")
        self.bias = nn.Parameter((hidden_size,), name="layer_norm_bias", dtype="float32")
        self.eps = eps

    def forward(self, hidden_states):
        return nn.emit(
            relax.op.nn.layer_norm(
                hidden_states,
                gamma=self.weight,
                beta=self.bias,
                axes=-1,
                epsilon=self.eps,
            )
        )


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            (num_embeddings, embedding_dim), dtype="float32", name="embedding_weight"
        )

    def forward(self, x: relax.Expr) -> relax.Var:
        from tvm.relax.op import reshape, take

        ndim = x.struct_info.ndim
        if ndim == 1:
            return nn.emit(take(self.weight, x, axis=0))
        else:
            x_shape = x.struct_info.shape.values
            emb_size = self.weight.struct_info.shape.values[-1]
            x = nn.emit(reshape(x, shape=[-1]))
            embedding = nn.emit(take(self.weight, x, axis=0))
            return nn.emit(reshape(embedding, [*x_shape, emb_size]))


def gelu(x: relax.Expr):
    return nn.emit(
        relax.const(0.5, x.struct_info.dtype)
        * x
        * (
            relax.const(1.0, x.struct_info.dtype)
            + relax.op.tanh(
                relax.const(0.7978845608028654, x.struct_info.dtype)
                * x
                * (
                    relax.const(1.0, x.struct_info.dtype)
                    + relax.const(0.044715, x.struct_info.dtype) * x * x
                )
            )
        )
    )


def apply_rotary_pos_emb_index(q, k, cos, sin, position_ids):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, hn] -> [sq, b, 1, hn]
    def f_rotary_embedding(tensor, cos, sin, position_ids):
        n_feat_half = tensor.shape[-1] // 2

        def rotary_compute(*idx):
            i, j = idx[0], idx[-1]
            offset = position_ids[i, 0]
            return cos[offset, j] * tensor(*idx) + sin[
                offset, j
            ] * tvm.tir.Select(
                j >= n_feat_half,
                tensor[i, idx[1], idx[2], j - n_feat_half], # true
                -tensor[i, idx[1], idx[2], j + n_feat_half], # false
            )

        return tvm.te.compute(tensor.shape, rotary_compute, name="rotary")

    q_embed = nn.emit_te(
        f_rotary_embedding,
        q,
        cos,
        sin,
        position_ids,
        primfunc_name_hint="rotary_embedding",
    )
    k_embed = nn.emit_te(
        f_rotary_embedding,
        k,
        cos,
        sin,
        position_ids,
        primfunc_name_hint="rotary_embedding",
    )
    return q_embed, k_embed


def attention_fn(
    query_layer: relax.Expr,  # [sq, b, np, hn]
    key_layer: relax.Expr,  # [sk, b, np, hn]
    value_layer: relax.Expr,
    all_seq_len_shape: relax.Expr,
    attention_mask: relax.Expr,
    hidden_size_per_partition: int,
    layer_id: int,
    past_key_value: Optional[Tuple[relax.Expr]] = None,
):
    from tvm.relax.op import reshape, matmul, permute_dims, full_like, where, squeeze
    from tvm.relax.op.nn import softmax

    kv_seq_len = all_seq_len_shape.struct_info.values[0]
    kv_states_shape = key_layer.struct_info.shape
    kv_states_dtype = key_layer.struct_info.dtype
    assert kv_states_shape[1] == 1  # bsz
    kv_states_shape = R.shape(
        [kv_seq_len, kv_states_shape[1], kv_states_shape[2], kv_states_shape[3]]
    )
    kv_cache_shape = R.shape([kv_seq_len, kv_states_shape[2], kv_states_shape[3]])
    if past_key_value is not None:
        squeezed_key = nn.emit(squeeze(key_layer, axis=1))
        squeezed_value = nn.emit(squeeze(value_layer, axis=1))
        k_cache, v_cache = past_key_value
        f_kv_cache_append = relax.extern("vm.builtin.attention_kv_cache_append")
        k_cache = nn.emit(
            relax.Call(
                f_kv_cache_append,
                args=[k_cache, squeezed_key],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )
        v_cache = nn.emit(
            relax.Call(
                f_kv_cache_append,
                args=[v_cache, squeezed_value],
                sinfo_args=[relax.ObjectStructInfo()],
            )
        )
        past_key_value = (k_cache, v_cache)
        f_kv_cache_view = relax.extern("vm.builtin.attention_kv_cache_view")
        k_cache = nn.emit(
            relax.Call(
                f_kv_cache_view,
                args=[k_cache, kv_cache_shape],
                sinfo_args=[R.Tensor(kv_cache_shape, kv_states_dtype)],
            )
        )
        v_cache = nn.emit(
            relax.Call(
                f_kv_cache_view,
                args=[v_cache, kv_cache_shape],
                sinfo_args=[R.Tensor(kv_cache_shape, kv_states_dtype)],
            )
        )
        key_layer = nn.emit(reshape(k_cache, kv_states_shape))
        value_layer = nn.emit(reshape(v_cache, kv_states_shape))

    # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
    seq_len, bsz, num_heads, head_dim = key_layer.struct_info.shape.values

    query_key_layer_scaling_coeff = float(layer_id + 1)

    query_layer = nn.emit(
        query_layer
        / (
            relax.const(math.sqrt(head_dim.value), query_layer.struct_info.dtype)
            * relax.const(query_key_layer_scaling_coeff, query_layer.struct_info.dtype)
        )
    )

    query_shape = query_layer.struct_info.shape
    key_shape = key_layer.struct_info.shape
    value_shape = value_layer.struct_info.shape
    # [b, np, sq, sk]
    output_size = (query_shape[1], query_shape[2], query_shape[0], key_shape[0])

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = nn.emit(
        reshape(
            query_layer,
            (output_size[2], output_size[0] * output_size[1], query_shape[-1]),
        )
    )

    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = nn.emit(
        reshape(
            key_layer, (output_size[3], output_size[0] * output_size[1], key_shape[-1])
        )
    )

    matmul_result = nn.emit(
        matmul(
            permute_dims(query_layer, [1, 0, 2]),  # b * np, sq, hn
            permute_dims(key_layer, [1, 2, 0]),  # b * np, hn, sk
        )
    )

    # change view to [b, np, sq, sk]
    attention_scores = nn.emit(reshape(matmul_result, output_size))

    attention_scores = nn.emit(
        where(
            attention_mask,
            full_like(
                attention_scores,
                relax.const(-10000.0, attention_scores.struct_info.dtype),
            ),
            attention_scores,
        )
    )

    attention_scores = nn.emit(
        attention_scores
        * relax.const(query_key_layer_scaling_coeff, attention_scores.struct_info.dtype)
    )

    attention_probs = nn.emit(softmax(attention_scores, axis=-1))

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_shape[1], value_shape[2], query_shape[0], value_shape[3])

    # change view [sk, b * np, hn]
    value_layer = nn.emit(
        reshape(
            value_layer,
            (value_shape[0], output_size[0] * output_size[1], value_shape[-1]),
        )
    )

    # change view [b * np, sq, sk]
    attention_probs = nn.emit(
        reshape(
            attention_probs,
            (output_size[0] * output_size[1], output_size[2], key_shape[0]),
        )
    )

    # matmul: [b * np, sq, hn]
    context_layer = nn.emit(
        matmul(attention_probs, permute_dims(value_layer, [1, 0, 2]))
    )

    # change view [b, np, sq, hn]
    context_layer = nn.emit(reshape(context_layer, output_size))

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = nn.emit(permute_dims(context_layer, [2, 0, 1, 3]))

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.struct_info.shape.values[:-2] + [
        hidden_size_per_partition,
    ]
    context_layer = nn.emit(reshape(context_layer, new_context_layer_shape))

    return context_layer, ((None, None) if past_key_value is None else past_key_value)


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        layer_id,
        hidden_size_per_attention_head=None,
        bias=True,
        position_encoding_2d=True,
    ):
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hidden_size_per_partition = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads
        self.position_encoding_2d = position_encoding_2d

        self.scale_mask_softmax = None

        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head

        self.inner_hidden_size = (
            num_attention_heads * self.hidden_size_per_attention_head
        )

        # Strided linear layer.
        self.query_key_value = Linear(
            hidden_size, 3 * self.inner_hidden_size, bias=bias
        )

        self.dense = Linear(self.inner_hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: relax.Expr,
        cos_cached: relax.Expr,
        sin_cache: relax.Expr,
        all_seq_len_shape: relax.Expr,
        position_ids: Tuple[relax.Expr],
        attention_mask: relax.Expr,
        layer_id: int,
        past_key_value: Optional[Tuple[relax.Expr]] = None,
    ):
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """
        from tvm.relax.op import reshape, permute_dims, split, concat, squeeze

        # [seq_len, batch, 3 * hidden_size]
        mixed_raw_layer = self.query_key_value(hidden_states)

        # [seq_len, batch, 3 * hidden_size] --> [seq_len, batch, num_attention_heads, 3 * hidden_size_per_attention_head]
        new_tensor_shape = mixed_raw_layer.struct_info.shape.values[:-1] + [
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        ]
        mixed_raw_layer = nn.emit(reshape(mixed_raw_layer, new_tensor_shape))

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        splited_layers = nn.emit(
            split(mixed_raw_layer, 3, axis=mixed_raw_layer.struct_info.ndim - 1)
        )
        query_layer = nn.emit(splited_layers[0])
        key_layer = nn.emit(splited_layers[1])
        value_layer = nn.emit(splited_layers[2])

        split_position_ids = nn.emit(
            split(position_ids, 2, axis=1)
        )
        position_ids = nn.emit(squeeze(split_position_ids[0], axis=1))
        block_position_ids = nn.emit(squeeze(split_position_ids[1], axis=1))
        
        # [b, sq] -> [sq, b]
        position_ids = nn.emit(
            permute_dims(
                position_ids,
                [1, 0]
            )
        )
        block_position_ids = nn.emit(
            permute_dims(
                block_position_ids,
                [1, 0]
            )
        )
        splited_query = nn.emit(
            split(
                query_layer,
                2,
                query_layer.struct_info.ndim - 1
            )
        )
        splited_key = nn.emit(
            split(
                key_layer,
                2,
                key_layer.struct_info.ndim - 1
            )
        )
        q1 = nn.emit(splited_query[0])
        q2 = nn.emit(splited_query[1])
        k1 = nn.emit(splited_key[0])
        k2 = nn.emit(splited_key[1])
        q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos_cached, sin_cache, position_ids)
        q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos_cached, sin_cache, block_position_ids)
        
        query_layer = nn.emit(
            concat(
                [q1, q2],
                axis=q1.struct_info.ndim-1
            )
        )
        key_layer = nn.emit(
            concat(
                [k1, k2],
                axis=k1.struct_info.ndim-1
            )
        )

        # [seq_len, batch, hidden_size]
        context_layer, present_key_value = attention_fn(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            all_seq_len_shape=all_seq_len_shape,
            attention_mask=attention_mask,
            hidden_size_per_partition=self.hidden_size_per_partition,
            layer_id=layer_id,
            past_key_value=past_key_value,
        )

        output = self.dense(context_layer)

        # if output_attentions:
        #     outputs += (attention_probs,)

        return output, present_key_value  # output, present, attention_probs


class GLU(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size=None, layer_id=None, bias=True):
        self.layer_id = layer_id

        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = Linear(self.hidden_size, self.inner_hidden_size, bias=bias)
        # Project back to h.
        self.dense_4h_to_h = Linear(self.inner_hidden_size, self.hidden_size, bias=bias)

    def forward(self, hidden_states: relax.Expr):
        """
        hidden_states: [seq_len, batch, hidden_size]
        """
        # [seq_len, batch, inner_hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = gelu(intermediate_parallel)

        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class GLMBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        layernorm_epsilon,
        layer_id,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        layernorm=LayerNorm,
        use_bias=True,
        num_layers=28,
    ):
        self.layer_id = layer_id

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            bias=use_bias,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        self.num_layers = num_layers

        # GLU
        self.mlp = GLU(
            hidden_size,
            inner_hidden_size=inner_hidden_size,
            bias=use_bias,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        cos_cached: relax.Expr,
        sin_cached: relax.Expr,
        all_seq_len_shape: relax.Expr,
        position_ids: Tuple[relax.Expr],
        attention_mask: relax.Expr,
        layer_id: int,
        past_key_value: Optional[Tuple[relax.Expr]] = None,
    ):
        """
        hidden_states: [seq_len, batch, hidden_size]
        attention_mask: [(1, 1), seq_len, seq_len]
        """

        # Layer norm at the begining of the transformer layer.
        # [seq_len, batch, hidden_size]
        attention_input = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, present_key_value = self.attention(
            hidden_states=attention_input,
            cos_cached=cos_cached,
            sin_cache=sin_cached,
            all_seq_len_shape=all_seq_len_shape,
            position_ids=position_ids,
            attention_mask=attention_mask,
            layer_id=layer_id,
            past_key_value=past_key_value,
        )

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = nn.emit(
            attention_input * relax.const(alpha, attention_input.struct_info.dtype)
            + attention_output
        )

        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input)

        # Second residual connection.
        output = nn.emit(
            mlp_input * relax.const(alpha, attention_input.struct_info.dtype)
            + mlp_output
        )

        return output, present_key_value  # hidden_states, present, attentions


class ChatGLMModel(nn.Module):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config: ChatGLMConfig):
        # recording parameters
        self.max_sequence_length = config.max_sequence_length
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.layernorm_epsilon = config.layernorm_epsilon
        self.inner_hidden_size = config.inner_hidden_size
        self.hidden_size_per_attention_head = (
            self.hidden_size // self.num_attention_heads
        )

        self.word_embeddings = Embedding(self.vocab_size, self.hidden_size)

        def get_layer(layer_id):
            return GLMBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.layernorm_epsilon,
                layer_id,
                inner_hidden_size=self.inner_hidden_size,
                hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                layernorm=LayerNorm,
                use_bias=True,
            )

        self.layers = [get_layer(layer_id) for layer_id in range(self.num_layers)]

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)

    def forward(
        self,
        input_ids: relax.Expr,
        cos_cached: relax.Expr,
        sin_cached: relax.Expr,
        position_ids: Tuple[relax.Expr],
        attention_mask: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: Optional[List[relax.Expr]] = None,
    ):
        from tvm.relax.op import permute_dims

        inputs_embeds = self.word_embeddings(input_ids)

        # [seq_len, batch, hidden_size]
        hidden_states = nn.emit(permute_dims(inputs_embeds, [1, 0, 2]))

        next_decoder_cache = ()

        for i, layer in enumerate(self.layers):
            past_key_value = (
                (past_key_values[i * 2], past_key_values[i * 2 + 1])
                if past_key_values is not None
                else None
            )

            hidden_states, key_values_cache = layer(
                hidden_states,
                cos_cached=cos_cached,
                sin_cached=sin_cached,
                all_seq_len_shape=all_seq_len_shape,
                position_ids=position_ids,
                attention_mask=attention_mask,
                layer_id=i,
                past_key_value=past_key_value,
            )

            next_decoder_cache += key_values_cache

        # Final layer norm.
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, next_decoder_cache


class ChatGLMForConditionalGeneration(nn.Module):
    def __init__(self, config: ChatGLMConfig):
        self.max_sequence_length = config.max_sequence_length

        self.transformer = ChatGLMModel(config)

        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

        self.config = config

        ############ Rotary embedding constants ############
        assert config.hidden_size % config.num_attention_heads == 0
        self.cos_cached = nn.Parameter(
            (config.max_sequence_length, config.hidden_size // (config.num_attention_heads * 2)), name="cos_cached", dtype="float32"
        )
        self.sin_cached = nn.Parameter(
            (config.max_sequence_length, config.hidden_size // (config.num_attention_heads * 2)), name="sin_cached", dtype="float32"
        )
        ############ End ############

    def forward(
        self,
        input_ids: relax.Expr,
        position_ids: Tuple[relax.Expr],
        attention_mask: relax.Expr,
        all_seq_len_shape: relax.Expr,
        past_key_values: Optional[List[relax.Expr]] = None,
    ):
        from tvm.relax.op import permute_dims

        hidden_states, key_value_cache = self.transformer(
            input_ids=input_ids,
            cos_cached=self.cos_cached,
            sin_cached=self.sin_cached,
            position_ids=position_ids,
            attention_mask=attention_mask,
            all_seq_len_shape=all_seq_len_shape,
            past_key_values=past_key_values,
        )
        
        def te_slicing(x: te.Tensor):
            return te.compute(
                shape=(1, 1, x.shape[-1]),
                fcompute=lambda i, j, k: x[i, x.shape[1] - 1, k],
                name="slice",
            )
        
        lm_logits = nn.emit(permute_dims(self.lm_head(hidden_states), [1, 0, 2]))
        
        logits = nn.emit_te(te_slicing, lm_logits, primfunc_name_hint="slice")

        return logits, key_value_cache


def create_encoding_func(bb: relax.BlockBuilder, config: ChatGLMConfig) -> None:
    bsz = 1
    seq_len = tvm.tir.Var("n", "int64")
    
    with bb.function("encoding"):
        model = ChatGLMForConditionalGeneration(config)
        input_ids = nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        position_ids = nn.Placeholder((bsz, 2, seq_len), dtype="int64", name="position_ids")
        attention_mask = nn.Placeholder((1, 1, seq_len, seq_len), dtype="bool", name="input_ids")
        all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((seq_len,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_layers * 2)]
            )
        )
        
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids, position_ids, attention_mask, all_seq_len_shape, past_key_values
            )
            params = [
                input_ids,
                position_ids,
                attention_mask,
                all_seq_len_shape,
                past_key_values
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var("encoding")
    bb.update_func(gv, mod[gv].with_attr("num_input", 5))
        
        
def create_decoding_func(bb: relax.BlockBuilder, config: ChatGLMConfig) -> None:
    bsz = 1
    seq_len = tvm.tir.Var("n", "int64")
    
    with bb.function("decoding"):
        model = ChatGLMForConditionalGeneration(config)
        input_ids = nn.Placeholder((bsz, 1), dtype="int32", name="input_ids")
        position_ids = nn.Placeholder((bsz, 2, 1), dtype="int64", name="position_ids") # 1,2,seq_len
        attention_mask = nn.Placeholder((1, 1), dtype="bool", name="input_ids") # 1,1,seq_len,seq_len
        all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((seq_len,)))
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_layers * 2)]
            )
        )
        
        with bb.dataflow():
            logits, key_value_cache = model(
                input_ids, position_ids, attention_mask, all_seq_len_shape, past_key_values
            )
            params = [
                input_ids,
                position_ids,
                attention_mask,
                all_seq_len_shape,
                past_key_values
            ] + model.parameters()
            gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var("decoding")
    bb.update_func(gv, mod[gv].with_attr("num_input", 5))
        