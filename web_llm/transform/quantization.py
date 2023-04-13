"""Relax quantization passes."""

import tvm
from tvm import relax
from tvm import te, tir
from tvm.ir.module import IRModule
from tvm.relax.expr_functor import mutator, PyExprMutator
from tvm.relax.analysis import remove_all_unused
from tvm.ir import Array
from tvm.relax import Tuple

from tvm.script import tir as T


# fmt: off
def _tir_f32x2_to_bf16x2_to_u32(v0: tir.PrimExpr, v1: tir.PrimExpr, round_to_even: bool=True):
    mask = tir.const((1 << 16) - 1, "uint32")
    res = []
    for data in [v0, v1]:
        u32_val = tir.reinterpret("uint32", data)
        if round_to_even:
            rounding_bias = ((u32_val >> tir.const(16, "uint32")) & tir.const(1, "uint32")) + tir.const(0x7FFF, "uint32")
            u32_val += rounding_bias
        res.append((u32_val >> tir.const(16, "uint32")) & mask)
    return res[0] | (res[1] << tir.const(16, "uint32"))


def _tir_u32_to_bf16x2_to_f32x2(x: tir.PrimExpr):
    mask = tir.const((1 << 16) - 1, "uint32")
    x0 = x & mask
    x1 = (x >> 16) & mask
    return (tir.reinterpret("float32", x << tir.const(16, "uint32")) for x in [x0, x1])


def _tir_u32_to_i4_to_f32(val: tir.PrimExpr, pos: tir.PrimExpr):
    assert val.dtype == "uint32"
    mask = tvm.tir.const((1 << 4) - 1, "uint32")
    return tir.Cast("float32", (val >> (pos * 4).astype("uint32")) & mask)


def encoding_func_asym(group_size: int):
    def te_encode_asym(weight: te.Tensor):
        assert weight.shape[1] % group_size == 0
        n_group = weight.shape[1] // group_size

        scale_min_shape = (weight.shape[0], n_group)
        k = te.reduce_axis((0, group_size), name="k")
        min_value = te.compute(shape=scale_min_shape, fcompute=lambda i, j: te.min(weight[i, j * group_size + k], axis=k), name="min")
        max_value = te.compute(shape=scale_min_shape, fcompute=lambda i, j: te.max(weight[i, j * group_size + k], axis=k), name="max")
        scale = te.compute(shape=scale_min_shape, fcompute=lambda i, j: (max_value[i, j] - min_value[i, j]) / tir.const(15, "float32"), name="scale")

        def f_scale_weight(i, j):
            group_idx = j // group_size
            w_scaled = tir.round((weight[i, j] - min_value[i, group_idx]) / scale[i, group_idx]).astype("int32")
            w_scaled = T.min(T.max(w_scaled, tir.const(0, "int32")), tir.const(15, "int32"))
            w_scaled = w_scaled.astype("uint32")
            return w_scaled

        k = te.reduce_axis((0, 8), name="k")
        reducer = te.comm_reducer(fcombine=lambda x, y: tir.bitwise_or(x, y), fidentity=lambda dtype: tir.const(0, dtype), name="bitwise_or")
        w_gathered = te.compute(shape=(weight.shape[0], weight.shape[1] // 8), fcompute=lambda i, j: reducer(f_scale_weight(i, j * 8 + k) << (k * 4).astype("uint32"), axis=k), name="w_gathered")
        scale_bias = te.compute(shape=(weight.shape[0], n_group), fcompute=lambda i, j: _tir_f32x2_to_bf16x2_to_u32(scale[i, j], min_value[i, j], round_to_even=True), name="scale_min")

        return w_gathered, scale_bias

    return te_encode_asym


def decoding_func_asym(group_size: int):
    def te_decode_asym(data: te.Tensor, scale_bias_bf16x2: te.Tensor):
        def f_decode_asym(i, j):
            data_f32 = _tir_u32_to_i4_to_f32(data[i, j // 8], j % 8)
            scale_f32, bias_f32 = _tir_u32_to_bf16x2_to_f32x2(scale_bias_bf16x2[i, j // group_size])
            return data_f32 * scale_f32 + bias_f32

        return te.compute(shape=(data.shape[0], data.shape[1] * 8), fcompute=f_decode_asym, name="decode")

    return te_decode_asym
# fmt: on


@tvm.transform.module_pass(opt_level=0, name="GroupQuantize")
class GroupQuantize:
    def __init__(self, group_size: int = 64, sym: bool = False) -> None:
        # NOTE: symmetric quantization is not supported at this moment.
        assert sym == False
        self.group_size = group_size
        self.sym = sym

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        @mutator
        class QuantizeMutator(PyExprMutator):
            def __init__(self, mod: IRModule, group_size: int, sym: bool):
                super().__init__(mod)
                self.mod = mod
                self._params = set()
                self.group_size = group_size
                self.sym = sym
                self.decode_cache = dict()

            def emit_te_encode_decode(self, x):
                encoded_data = self.builder_.emit_te(
                    encoding_func_asym(self.group_size), x, primfunc_name_hint="encode"
                )

                decode_args = []
                decode_args.append(self.builder_.emit(relax.TupleGetItem(encoded_data, 0)))
                decode_args.append(self.builder_.emit(relax.TupleGetItem(encoded_data, 1)))
                decode_args[0] = self.builder_.emit(
                    relax.op.builtin.stop_lift_params(decode_args[0])
                )

                decoded_data = self.builder_.emit_te(
                    decoding_func_asym(self.group_size), *decode_args, primfunc_name_hint="decode"
                )
                return decoded_data

            def transform(self) -> IRModule:
                for global_var, func in self.mod.functions.items():
                    if not isinstance(func, relax.Function):
                        continue
                    if not "num_input" in func.attrs:
                        continue
                    num_inputs = func.attrs["num_input"]
                    for i in range(int(num_inputs), len(func.params)):
                        self._params.add(func.params[i])
                    updated_func = self.visit_expr(func)
                    updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(global_var, updated_func)
                return self.builder_.get()

            def process_args(self, args):
                if isinstance(args, (Array, Tuple)):
                    updated = False
                    new_args = []
                    for arg in args:
                        new_arg, arg_updated = self.process_args(arg)
                        new_args.append(new_arg)
                        updated = updated or arg_updated
                    return (Tuple(new_args) if isinstance(args, Tuple) else new_args), updated
                elif isinstance(args, relax.Var):
                    if args in self._params:
                        if args.struct_info.ndim != 2:
                            return args, False

                        if args not in self.decode_cache:
                            updated_arg = self.emit_te_encode_decode(args)
                            self.decode_cache[args] = updated_arg
                            return updated_arg, True
                        else:
                            return self.decode_cache[args], True
                    else:
                        return args, False
                else:
                    return args, False

            def visit_call_(self, call):
                call = self.visit_expr_post_order(call)
                new_args, updated = self.process_args(call.args)
                if not updated:
                    return call
                new_call = relax.Call(call.op, new_args, call.attrs, call.sinfo_args, call.span)
                return new_call

        return QuantizeMutator(mod, self.group_size, self.sym).transform()
