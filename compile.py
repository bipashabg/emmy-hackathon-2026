#!/usr/bin/env python3
import sys

from ptxc.loader import load_problem, find_kernel
from ptxc.lower import lower_kernel


def compile_ir(ir_path):
    graph, meta = load_problem(ir_path)

    kspecs = meta["kernels"]
    if len(kspecs) != 1:
        raise NotImplementedError(
            f"single-kernel problems only for now; got {len(kspecs)}"
        )

    kspec = kspecs[0]
    kernel = find_kernel(graph, kspec)
    return lower_kernel(kernel, kspec)


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <path-to-ir.json>", file=sys.stderr)
        return 2
    sys.stdout.write(compile_ir(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
