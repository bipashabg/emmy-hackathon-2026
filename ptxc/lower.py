from ptxc.ptx import PTXBuilder, f32_hex


def extent_to_int(axis):
    expr = axis.extent.expr

    if type(expr).__name__ != "Literal":
        raise NotImplementedError(
            f"Non-literal tile extent: axis {getattr(axis, 'name', '?')!r} "
            f"has extent expr {type(expr).__name__}; only Literal extents "
            f"are supported so far"
        )

    value = expr.value
    if not isinstance(value, int) or isinstance(value, bool):
        raise NotImplementedError(
            f"Non-integer literal extent: {value!r} "
            f"(dtype={getattr(expr, 'dtype', '?')!r})"
        )

    return value


class LowerContext:

    def __init__(self, builder):
        self.b = builder
        self.axis_env = {}
        self.var_env = {}
        self.ptr_env = {}
        self.acc_env = {}
        self.smem_env = {}
        self.tid_reg = None
        self._lane_reg = None
        self._warp_reg = None
        self._label_counts = {}

    def fresh_label(self, stem):
        n = self._label_counts.get(stem, 0)
        self._label_counts[stem] = n + 1
        return f"{stem}_{n}"


def lower_kernel(kernel, kspec):
    b = PTXBuilder()
    ctx = LowerContext(b)

    entry = kspec["entry"]
    arg_order = kspec["arg_order"]

    for name in arg_order:
        raw = b.reg("u64")
        b.emit(f"ld.param.u64 {raw}, [param_{name}];")
        ptr = b.reg("u64")
        b.emit(f"cvta.to.global.u64 {ptr}, {raw};")
        ctx.ptr_env[name] = ptr

    for tile in kernel.body:
        lower_tile(tile, ctx)

    b.emit("ret;")

    return b.render_kernel(entry, arg_order)


def lower_tile(tile, ctx):
    b = ctx.b

    if len(tile.axes) != 1:
        raise NotImplementedError(
            f"lower_tile handles one axis for now; got {len(tile.axes)}"
        )

    ctx._lane_reg = None
    ctx._warp_reg = None

    axis = tile.axes[0]
    extent = extent_to_int(axis)

    tid = b.reg("s32")
    b.emit(f"mov.u32 {tid}, {b.sreg('tid.x')};")
    ctx.tid_reg = tid
    ntid = b.reg("s32")
    b.emit(f"mov.u32 {ntid}, {b.sreg('ntid.x')};")
    ctaid = b.reg("s32")
    b.emit(f"mov.u32 {ctaid}, {b.sreg('ctaid.x')};")

    idx = b.reg("s32")
    b.emit(f"mad.lo.s32 {idx}, {ctaid}, {ntid}, {tid};")

    pred = b.reg("pred")
    b.emit(f"setp.ge.s32 {pred}, {idx}, {extent};")

    done = ctx.fresh_label("DONE")
    b.emit(f"bra {done};", pred=pred)

    ctx.axis_env[axis.name] = idx

    for stmt in tile.body:
        lower_stmt(stmt, ctx)

    b.put_label(done)


def lower_stmt(stmt, ctx):
    kind = type(stmt).__name__

    handler = {
        "Load": lower_load,
        "Assign": lower_assign,
        "Write": lower_write,
        "StridedLoop": lower_strided_loop,
        "Accum": lower_accum,
        "WarpShuffle": lower_warp_shuffle,
        "Smem": lower_smem,
        "Cond": lower_cond,
        "Sync": lower_sync,
        "TreeHalve": lower_tree_halve,
    }.get(kind)

    if handler is None:
        raise NotImplementedError(
            f"lower_stmt does not handle {kind} yet: {stmt!r}"
        )
    return handler(stmt, ctx)


def _get_lane(ctx):
    if ctx._lane_reg is None:
        r = ctx.b.reg("s32")
        ctx.b.emit(f"and.b32 {r}, {ctx.tid_reg}, 31;")
        ctx._lane_reg = r
    return ctx._lane_reg


def _get_warp(ctx):
    if ctx._warp_reg is None:
        r = ctx.b.reg("s32")
        ctx.b.emit(f"shr.u32 {r}, {ctx.tid_reg}, 5;")
        ctx._warp_reg = r
    return ctx._warp_reg


def _resolve_var(name, ctx):
    if name in ctx.axis_env:
        return ctx.axis_env[name]
    if name == "lane":
        return _get_lane(ctx)
    if name == "warp":
        return _get_warp(ctx)
    raise NotImplementedError(
        f"Unbound variable {name!r}; axes={sorted(ctx.axis_env)}"
    )


def _value_reg(name, ctx):
    if name in ctx.var_env:
        return ctx.var_env[name]
    if name in ctx.acc_env:
        return ctx.acc_env[name]
    raise NotImplementedError(
        f"unbound value {name!r}; "
        f"var_env={sorted(ctx.var_env)} acc_env={sorted(ctx.acc_env)}"
    )



_INT_BINOP = {
    "*": "mul.lo.s32",  "mul": "mul.lo.s32",
    "+": "add.s32",     "add": "add.s32",
    "-": "sub.s32",     "sub": "sub.s32",
}


def _binop_key(op):
    if isinstance(op, str):
        return op.lower()
    return getattr(op, "name", str(op)).lower()


def lower_expr(expr, ctx):
    kind = type(expr).__name__

    if kind == "Var":
        return _resolve_var(expr.name, ctx)

    if kind == "Literal":
        return expr.value

    if kind == "BinaryExpr":
        opcode = _INT_BINOP.get(_binop_key(expr.op))
        if opcode is None:
            raise NotImplementedError(
                f"Unsupported integer operator {expr.op!r} in BinaryExpr"
            )
        left = lower_expr(expr.left, ctx)
        right = lower_expr(expr.right, ctx)
        dst = ctx.b.reg("s32")
        ctx.b.emit(f"{opcode} {dst}, {left}, {right};")
        return dst

    raise NotImplementedError(
        f"lower_expr does not handle {kind} yet: {expr!r}"
    )



_DTYPE = {
    "f32": ("f32", 4),
    "f64": ("f64", 8),
}

_SMEM_DTYPE = {
    "float": ("f32", 4),
    "double": ("f64", 8),
}


def _dtype_key(dt):
    if isinstance(dt, str):
        return dt.lower()
    return getattr(dt, "name", str(dt)).lower()


def _resolve_index(index_tuple, ctx, what):
    if len(index_tuple) != 1:
        raise NotImplementedError(
            f"multi-dimensional {what} index not supported yet: "
            f"{len(index_tuple)} dims"
        )
    index = lower_expr(index_tuple[0], ctx)
    if isinstance(index, int):
        idx_reg = ctx.b.reg("s32")
        ctx.b.emit(f"mov.s32 {idx_reg}, {index};")
        index = idx_reg
    return index


def _byte_address(base, index, elem_size, ctx):
    b = ctx.b
    offset = b.reg("s64")
    b.emit(f"mul.wide.s32 {offset}, {index}, {elem_size};")
    addr = b.reg("s64")
    b.emit(f"add.s64 {addr}, {base}, {offset};")
    return addr


def lower_load(stmt, ctx):
    b = ctx.b

    if len(stmt.names) != 1:
        raise NotImplementedError(
            f"load binding {len(stmt.names)} names not supported yet: "
            f"{tuple(stmt.names)!r}"
        )

    key = _dtype_key(stmt.dtype)
    if key not in _DTYPE:
        raise NotImplementedError(
            f"unsupported load dtype {stmt.dtype!r} (normalized {key!r})"
        )
    ptx_ty, elem_size = _DTYPE[key]

    try:
        base = ctx.ptr_env[stmt.input]
    except KeyError:
        raise NotImplementedError(
            f"no base pointer for tensor {stmt.input!r}; "
            f"bound tensors are {sorted(ctx.ptr_env)}"
        )

    index = _resolve_index(stmt.index, ctx, "load")
    addr = _byte_address(base, index, elem_size, ctx)

    dst = b.reg(ptx_ty)
    b.emit(f"ld.global.{ptx_ty} {dst}, [{addr}];")

    ctx.var_env[stmt.names[0]] = dst
    return dst


def lower_write(stmt, ctx):
    b = ctx.b

    if getattr(stmt, "atomic", False):
        raise NotImplementedError("atomic Write not supported yet")
    if getattr(stmt, "swizzle", "NONE") not in (None, "NONE"):
        raise NotImplementedError(
            f"Write swizzle {stmt.swizzle!r} not supported yet"
        )

    key = _dtype_key(stmt.value_dtype)
    if key not in _DTYPE:
        raise NotImplementedError(
            f"unsupported write dtype {stmt.value_dtype!r} (normalized {key!r})"
        )
    ptx_ty, elem_size = _DTYPE[key]

    out = stmt.output

    if out in ctx.smem_env:
        space = "shared"
        elem = ctx.smem_env[out]
        base = b.reg("b32")
        b.emit(f"mov.u32 {base}, {out};")
        if len(stmt.index) == 0:
            addr = base
        else:
            if len(stmt.index) != 1:
                raise NotImplementedError("multi-dim shared write")
            index = lower_expr(stmt.index[0], ctx)
            off = b.reg("s32")
            if isinstance(index, int):
                b.emit(f"mov.s32 {off}, {index * elem};")
            else:
                b.emit(f"mul.lo.s32 {off}, {index}, {elem};")
            addr = b.reg("s32")
            b.emit(f"add.s32 {addr}, {base}, {off};")

    elif out in ctx.ptr_env:
        space = "global"
        base = ctx.ptr_env[out]
        if len(stmt.index) == 0:
            addr = base
        else:
            index = _resolve_index(stmt.index, ctx, "write")
            addr = _byte_address(base, index, elem_size, ctx)

    else:
        raise NotImplementedError(
            f"no destination for Write output {out!r}; "
            f"ptr_env={sorted(ctx.ptr_env)} smem_env={sorted(ctx.smem_env)}"
        )

    for i, name in enumerate(stmt.values):
        val = _value_reg(name, ctx)
        disp = i * elem_size
        slot = f"[{addr}]" if disp == 0 else f"[{addr}+{disp}]"
        b.emit(f"st.{space}.{ptx_ty} {slot}, {val};")


def lower_smem(stmt, ctx):
    dt = stmt.dtype
    if dt not in _SMEM_DTYPE:
        raise NotImplementedError(f"unsupported smem dtype {dt!r}")
    _, elem = _SMEM_DTYPE[dt]

    if len(stmt.extents) != 1:
        raise NotImplementedError("multi-dim smem not supported yet")

    n = stmt.extents[0]
    size = n * elem
    align = stmt.align or elem
    ctx.b.shared(stmt.name, size, align)
    ctx.smem_env[stmt.name] = elem


def lower_sync(stmt, ctx):
    ctx.b.emit(f"bar.sync {stmt.barrier_id};")


_FLOAT_BINOP = {
    "add": "add",
    "subtract": "sub",
    "sub": "sub",
    "multiply": "mul",
    "mul": "mul",
    "divide": "div.rn",
    "true_divide": "div.rn",
    "div": "div.rn",
    "maximum": "max",
    "max": "max",
    "minimum": "min",
    "min": "min",
}

_IDENTITY_VAL = {
    "add": 0.0,
    "subtract": 0.0,
    "sub": 0.0,
    "multiply": 1.0,
    "mul": 1.0,
    "maximum": float("-inf"),
    "max": float("-inf"),
    "minimum": float("inf"),
    "min": float("inf"),
}

_CMP_NEG = {
    "==": "setp.ne.s32",
    "!=": "setp.eq.s32",
    "<":  "setp.ge.s32",
    "<=": "setp.gt.s32",
    ">":  "setp.le.s32",
    ">=": "setp.lt.s32",
}


def lower_assign(stmt, ctx):
    b = ctx.b

    key = _dtype_key(stmt.dtype)
    if key not in _DTYPE:
        raise NotImplementedError(
            f"unsupported assign dtype {stmt.dtype!r} (normalized {key!r})"
        )
    ptx_ty, _ = _DTYPE[key]

    op_name = str(getattr(stmt.op, "name", stmt.op)).lower()

    srcs = [_value_reg(a, ctx) for a in stmt.args]

    dst = b.reg(ptx_ty)

    binop = _FLOAT_BINOP.get(op_name)
    if binop is not None:
        if len(srcs) != 2:
            raise NotImplementedError(
                f"{op_name!r} expects 2 operands, got {len(srcs)}: {tuple(stmt.args)!r}"
            )
        b.emit(f"{binop}.{ptx_ty} {dst}, {srcs[0]}, {srcs[1]};")
    elif op_name == "relu":
        if len(srcs) != 1:
            raise NotImplementedError(
                f"relu expects 1 operand, got {len(srcs)}: {tuple(stmt.args)!r}"
            )
        if ptx_ty != "f32":
            raise NotImplementedError(f"relu only lowers f32 for now, got {ptx_ty!r}")
        b.emit(f"max.f32 {dst}, {srcs[0]}, {f32_hex(0.0)};")
    else:
        raise NotImplementedError(
            f"unsupported elementwise op {op_name!r} in Assign"
        )

    ctx.var_env[stmt.name] = dst
    return dst


def _seed_accum(accum, ctx):
    b = ctx.b
    key = _dtype_key(accum.dtype)
    if key not in _DTYPE:
        raise NotImplementedError(f"unsupported accum dtype {accum.dtype!r}")
    ptx_ty, _ = _DTYPE[key]
    if ptx_ty != "f32":
        raise NotImplementedError(f"accum seed only f32 for now, got {ptx_ty!r}")

    op_name = str(getattr(accum.op, "name", accum.op)).lower()
    if op_name not in _IDENTITY_VAL:
        raise NotImplementedError(f"no identity for accum op {op_name!r}")

    reg = b.reg(ptx_ty)
    b.emit(f"mov.{ptx_ty} {reg}, {f32_hex(_IDENTITY_VAL[op_name])};")
    ctx.acc_env[accum.name] = reg


def lower_accum(stmt, ctx):
    b = ctx.b

    if stmt.base is not None:
        raise NotImplementedError(f"Accum with base {stmt.base!r} not supported yet")

    acc = ctx.acc_env.get(stmt.name)
    if acc is None:
        raise NotImplementedError(
            f"accumulator {stmt.name!r} not seeded; seeded={sorted(ctx.acc_env)}"
        )

    key = _dtype_key(stmt.dtype)
    ptx_ty, _ = _DTYPE[key]
    op_name = str(getattr(stmt.op, "name", stmt.op)).lower()
    base = _FLOAT_BINOP.get(op_name)
    if base is None:
        raise NotImplementedError(f"unsupported accum op {op_name!r}")

    val = _value_reg(stmt.value, ctx)
    b.emit(f"{base}.{ptx_ty} {acc}, {acc}, {val};")
    return acc


def lower_strided_loop(stmt, ctx):
    b = ctx.b
    axis = stmt.axis
    extent = extent_to_int(axis)
    step = lower_expr(stmt.step, ctx)

    if stmt.seed:
        for s in stmt.body:
            if type(s).__name__ == "Accum":
                _seed_accum(s, ctx)

    start = lower_expr(stmt.start, ctx)
    iv = b.reg("s32")
    b.emit(f"mov.u32 {iv}, {start};")
    ctx.axis_env[axis.name] = iv

    loop = ctx.fresh_label("LOOP")
    loop_end = ctx.fresh_label("LOOP_END")

    b.put_label(loop)
    pred = b.reg("pred")
    b.emit(f"setp.ge.s32 {pred}, {iv}, {extent};")
    b.emit(f"bra {loop_end};", pred=pred)

    for s in stmt.body:
        lower_stmt(s, ctx)

    b.emit(f"add.s32 {iv}, {iv}, {step};")
    b.emit(f"bra {loop};")
    b.put_label(loop_end)


def lower_warp_shuffle(stmt, ctx):
    b = ctx.b
    key = _dtype_key(stmt.dtype)
    ptx_ty, _ = _DTYPE[key]

    length = stmt.length
    mask = length - 1

    offsets = []
    o = length // 2
    while o >= 1:
        offsets.append(o)
        o //= 2

    for off in offsets:
        for st_name, combine in zip(stmt.state, stmt.combine_states):
            acc = ctx.acc_env[st_name]
            op_name = str(getattr(combine.op, "name", combine.op)).lower()
            opc = _FLOAT_BINOP.get(op_name)
            if opc is None:
                raise NotImplementedError(f"unsupported warp combine op {op_name!r}")

            tmp = b.reg("b32")
            b.emit(f"mov.b32 {tmp}, {acc};")
            shf = b.reg("b32")
            b.emit(f"shfl.sync.bfly.b32 {shf}, {tmp}, {off}, {hex(mask)}, 0xffffffff;")
            partner = b.reg(ptx_ty)
            b.emit(f"mov.b32 {partner}, {shf};")
            b.emit(f"{opc}.{ptx_ty} {acc}, {acc}, {partner};")


def lower_cond(stmt, ctx):
    b = ctx.b
    cond = stmt.cond

    if type(cond).__name__ != "BinaryExpr" or cond.op not in _CMP_NEG:
        raise NotImplementedError(f"unsupported Cond condition {cond!r}")

    if stmt.else_body:
        raise NotImplementedError("Cond else_body not supported yet")

    left = lower_expr(cond.left, ctx)
    right = lower_expr(cond.right, ctx)
    pred = b.reg("pred")
    b.emit(f"{_CMP_NEG[cond.op]} {pred}, {left}, {right};")

    end = ctx.fresh_label("ENDIF")
    b.emit(f"bra {end};", pred=pred)

    for s in stmt.body:
        lower_stmt(s, ctx)

    b.put_label(end)


def lower_tree_halve(stmt, ctx):
    b = ctx.b
    key = _dtype_key(stmt.dtype)
    ptx_ty, _ = _DTYPE[key]

    buf = stmt.bufs[0]
    if buf not in ctx.smem_env:
        raise NotImplementedError(f"TreeHalve buffer {buf!r} not declared")
    elem = ctx.smem_env[buf]

    acc = ctx.acc_env[stmt.state[0]]
    combine = stmt.combine_states[0]
    op_name = str(getattr(combine.op, "name", combine.op)).lower()
    opc = _FLOAT_BINOP.get(op_name)
    if opc is None:
        raise NotImplementedError(f"unsupported tree combine op {op_name!r}")

    length = stmt.length
    tid_reg = _resolve_var(stmt.tid_var, ctx)

    pred = b.reg("pred")
    b.emit(f"setp.ne.s32 {pred}, {tid_reg}, 0;")
    skip = ctx.fresh_label("TREE_SKIP")
    b.emit(f"bra {skip};", pred=pred)

    base = b.reg("b32")
    b.emit(f"mov.u32 {base}, {buf};")
    b.emit(f"ld.shared.{ptx_ty} {acc}, [{base}];")
    for i in range(1, length):
        tmp = b.reg(ptx_ty)
        b.emit(f"ld.shared.{ptx_ty} {tmp}, [{base}+{i * elem}];")
        b.emit(f"{opc}.{ptx_ty} {acc}, {acc}, {tmp};")

    b.put_label(skip)
