from ptxc.loader import load_problem, find_kernel
from ptxc.ptx import PTXBuilder
from ptxc.lower import LowerContext, lower_expr

graph, meta = load_problem("problems/p00_scalar_add/k_add_pointwise.ir.json")
kernel = find_kernel(graph, meta["kernels"][0])
tile = kernel.body[0]

builder = PTXBuilder()
ctx = LowerContext(builder)

axis = tile.axes[0]
ctx.axis_env[axis.name] = builder.reg("s32")
builder.emit(f"mov.u32 {ctx.axis_env[axis.name]}, %tid.x;")

for stmt in tile.body:
    for expr in getattr(stmt, "index", ()):
        result = lower_expr(expr, ctx)
        print(f"{type(expr).__name__} -> {result}")

print("\nGenerated instructions:")
print(builder.render())
