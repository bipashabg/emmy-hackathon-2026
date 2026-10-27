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

    axis = tile.axes[0]
    extent = extent_to_int(axis)

    tid = b.reg("s32")
    b.emit(f"mov.u32 {tid}, {b.sreg('tid.x')};")
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

    if kind == "Load":
        return lower_load(stmt, ctx)
    if kind == "Assign":
        return lower_assign(stmt, ctx)
    if kind == "Write":
        return lower_write(stmt, ctx)

    raise NotImplementedError(
        f"lower_stmt does not handle {kind} yet: {stmt!r}"
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
        name = expr.name
        try:
            return ctx.axis_env[name]
        except KeyError:
            raise NotImplementedError(
                f"Unbound variable {name!r} in index expr; "
                f"bound axes are {sorted(ctx.axis_env)}"
            )

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
            f"load binding {len(stmt.names)} names not supported yet "
            f"(expected a single scalar result): {tuple(stmt.names)!r}"
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


def lower_assign(stmt, ctx):
    b = ctx.b

    key = _dtype_key(stmt.dtype)
    if key not in _DTYPE:
        raise NotImplementedError(
            f"unsupported assign dtype {stmt.dtype!r} (normalized {key!r})"
        )
    ptx_ty, _ = _DTYPE[key]

    op_name = str(getattr(stmt.op, "name", stmt.op)).lower()

    srcs = []
    for a in stmt.args:
        try:
            srcs.append(ctx.var_env[a])
        except KeyError:
            raise NotImplementedError(
                f"unbound operand {a!r} in Assign; "
                f"bound values are {sorted(ctx.var_env)}"
            )

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

    try:
        base = ctx.ptr_env[stmt.output]
    except KeyError:
        raise NotImplementedError(
            f"no base pointer for output tensor {stmt.output!r}; "
            f"bound tensors are {sorted(ctx.ptr_env)}"
        )

    index = _resolve_index(stmt.index, ctx, "write")
    addr = _byte_address(base, index, elem_size, ctx)

    for i, name in enumerate(stmt.values):
        try:
            val = ctx.var_env[name]
        except KeyError:
            raise NotImplementedError(
                f"unbound value {name!r} in Write; "
                f"bound values are {sorted(ctx.var_env)}"
            )
        disp = i * elem_size
        slot = f"[{addr}]" if disp == 0 else f"[{addr}+{disp}]"
        b.emit(f"st.global.{ptx_ty} {slot}, {val};")
