from ptxc.loader import load_problem, find_kernel
from ptxc.ptx import PTXBuilder
from ptxc.lower import LowerContext, lower_tile

ir_path = "problems/p00_scalar_add/k_add_pointwise.ir.json"

graph, meta = load_problem(ir_path)
kernel = find_kernel(graph, meta["kernels"][0])

builder = PTXBuilder()
ctx = LowerContext(builder)

lower_tile(kernel.body[0], ctx)

print(builder.render())
