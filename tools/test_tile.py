
from ptxc.loader import load_problem, find_kernel
from ptxc.ptx import PTXBuilder
from ptxc.lower import LowerContext, lower_tile

ir_path = "problems/p00_scalar_add/k_add_pointwise.ir.json"

graph, meta = load_problem(ir_path)
kernel = find_kernel(graph, meta["kernels"][0])
kspec = meta["kernels"][0]

builder = PTXBuilder()
ctx = LowerContext(builder)

for arg in kspec["args"]:
    ctx.shape_env[arg["name"]] = tuple(arg["shape"])

for name in kspec["arg_order"]:
    raw = builder.reg("u64")
    builder.emit(f"ld.param.u64 {raw}, [param_{name}];")

    ptr = builder.reg("u64")
    builder.emit(f"cvta.to.global.u64 {ptr}, {raw};")

    ctx.ptr_env[name] = ptr

lower_tile(kernel.body[0], ctx)

builder.emit("ret;")
print(builder.render())

