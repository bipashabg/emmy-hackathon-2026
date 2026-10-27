
import json
from pathlib import Path

from emmy.compiler.graph import Graph


def load_problem(ir_path):
    ir_path = Path(ir_path)

    if not ir_path.is_file():
        raise FileNotFoundError(f"IR file not found: {ir_path}")

    meta_path = ir_path.parent / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    graph_data = json.loads(ir_path.read_text())
    meta = json.loads(meta_path.read_text())
    graph = Graph.from_dict(graph_data)

    return graph, meta


def find_kernel(graph, kernel_spec):
    node_id = kernel_spec["node"]

    if node_id not in graph.nodes:
        raise KeyError(
            f"Kernel node {node_id!r} from meta.json "
            "was not found in the IR graph"
        )

    node = graph.nodes[node_id]

    if type(node.op).__name__ != "KernelOp":
        raise TypeError(
            f"Node {node_id!r} is {type(node.op).__name__}, "
            "not KernelOp"
        )

    return node.op

