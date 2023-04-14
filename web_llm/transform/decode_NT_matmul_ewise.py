import tvm
from tvm import IRModule
from tvm import relax, tir
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.dpl.pattern import GlobalVarPattern, TuplePattern


def check_x_1dim(ctx: relax.transform.PatternCheckContext) -> bool:
    x = ctx.annotated_expr["x"]
    n = x.struct_info.shape[-2]
    return isinstance(n, tir.IntImm) and n.value == 1


def check_decoding(ctx: relax.transform.PatternCheckContext) -> bool:
    call = ctx.annotated_expr["w"]
    gv = call.args[0]
    return gv.name_hint.startswith("decode")


def check_NT_matmul(ctx: relax.transform.PatternCheckContext) -> bool:
    call = ctx.annotated_expr["NT_matmul"]
    gv = call.args[0]
    return gv.name_hint.startswith("NT_matmul") or gv.name_hint.startswith("fused_NT_matmul")


def pattern_check(ctx: relax.transform.PatternCheckContext) -> bool:
    return check_x_1dim(ctx) and check_decoding(ctx) and check_NT_matmul(ctx)


def decode_NT_matmul_pattern():
    w_scaled = wildcard()
    scale_min = wildcard()
    x = wildcard()
    w = is_op("relax.call_tir")(
        GlobalVarPattern(), TuplePattern([w_scaled, scale_min]), add_constraint=False
    )
    NT_matmul = is_op("relax.call_tir")(
        GlobalVarPattern(), TuplePattern([x, w]), add_constraint=False
    )

    annotations = {
        "NT_matmul": NT_matmul,
        "w": w,
        "x": x,
        "w_scaled": w_scaled,
        "scale_min": scale_min,
    }

    return NT_matmul, annotations, pattern_check


def decode_NT_matmul_ewise_pattern():
    w_scaled = wildcard()
    scale_min = wildcard()
    x = wildcard()
    y = wildcard()
    w = is_op("relax.call_tir")(
        GlobalVarPattern(), TuplePattern([w_scaled, scale_min]), add_constraint=False
    )
    NT_matmul_ewise = is_op("relax.call_tir")(
        GlobalVarPattern(), TuplePattern([x, w, y]), add_constraint=False
    )

    annotations = {
        "NT_matmul": NT_matmul_ewise,
        "w": w,
        "x": x,
        "w_scaled": w_scaled,
        "scale_min": scale_min,
    }

    return NT_matmul_ewise, annotations, pattern_check


@tvm.transform.module_pass(opt_level=0, name="FuseDecodeNTMatmulEwise")
class FuseDecodeNTMatmulEwise:
    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        mod = relax.transform.FuseOpsByPattern([("decode_NT_matmul", *decode_NT_matmul_pattern())])(
            mod
        )
        mod = relax.transform.FuseOpsByPattern(
            [("decode_NT_matmul_ewise", *decode_NT_matmul_ewise_pattern())]
        )(mod)
        mod = relax.transform.FuseTIR()(mod)

        return mod
