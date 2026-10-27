import struct


class PTXBuilder:
    _TYPES = {
        "pred": ("%p", ".pred"),
        "f32": ("%f", ".f32"),
        "f64": ("%fd", ".f64"),
        "u32": ("%r", ".b32"),
        "s32": ("%r", ".b32"),
        "b32": ("%r", ".b32"),
        "u64": ("%rd", ".b64"),
        "s64": ("%rd", ".b64"),
        "b64": ("%rd", ".b64"),
    }

    def __init__(self):
        self.lines = []
        self._register_counts = {}
        self._labels = set()
        self._shared_declarations = []

    def reg(self, ty):
        if ty not in self._TYPES:
            raise ValueError(f"Unsupported PTX register type: {ty}")

        prefix, _ = self._TYPES[ty]
        index = self._register_counts.get(prefix, 0)
        self._register_counts[prefix] = index + 1
        return f"{prefix}{index}"

    def regs(self, ty, count):
        if count < 0:
            raise ValueError("Register count cannot be negative")
        return [self.reg(ty) for _ in range(count)]

    def register_declarations(self):
        declarations = []
        for prefix, ptx_type in (
            ("%p", ".pred"),
            ("%f", ".f32"),
            ("%fd", ".f64"),
            ("%r", ".b32"),
            ("%rd", ".b64"),
        ):
            count = self._register_counts.get(prefix, 0)
            if count:
                declarations.append(
                    f".reg {ptx_type} {prefix}<{count}>;"
                )
        return declarations

    def label(self, name):
        return name

    def put_label(self, name):
        if name in self._labels:
            raise ValueError(f"Duplicate PTX label: {name}")
        self._labels.add(name)
        self.lines.append(f"{name}:")

    def emit(self, instruction, pred=None):
        if pred is None:
            self.lines.append(f"    {instruction}")
        else:
            self.lines.append(f"    @{pred} {instruction}")

    def comment(self, text):
        self.lines.append(f"    // {text}")

    def shared(self, name, size, align=4):
        if size < 0:
            raise ValueError("Shared-memory size cannot be negative")
        self._shared_declarations.append(
            f".shared .align {align} .b8 {name}[{size}];"
        )

    def sreg(self, name):
        allowed = {"tid.x", "ntid.x", "ctaid.x"}
        if name not in allowed:
            raise ValueError(f"Unsupported special register: {name}")
        return f"%{name}"


    def render_kernel(self, entry, arg_order):
        if not entry.isidentifier():
            raise ValueError(f"Invalid PTX entry name: {entry}")

        if len(set(arg_order)) != len(arg_order):
            raise ValueError("Duplicate kernel argument names")

        lines = [
            ".version 8.0",
            ".target sm_89",
            ".address_size 64",
            "",
        ]

        params = ",\n".join(
            f"    .param .u64 param_{name}"
            for name in arg_order
        )
        lines.append(f".visible .entry {entry}(")
        lines.append(params)
        lines.append(") {")

        lines.extend(f"    {decl}" for decl in self._shared_declarations)
        lines.extend(
            f"    {decl}" for decl in self.register_declarations()
        )
        lines.extend(self.lines)
        lines.append("}")

        return "\n".join(lines) + "\n"


    def render(self):
        return "\n".join(
            self._shared_declarations + self.lines
        ) + "\n"


def f32_hex(value):
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f"0f{bits:08X}"

