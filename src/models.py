from __future__ import annotations
import ast
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

NORM_TYPES = ["None", "BatchNorm2d", "InstanceNorm2d", "GroupNorm", "LayerNorm"]
POOL_TYPES = ["None", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "AdaptiveMaxPool2d"]
ACT_TYPES = ["None", "ReLU", "LeakyReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "ELU", "PReLU", "Softmax"]
CONV_DROPOUT_TYPES = ["None", "Dropout", "Dropout2d"]

HEAD_NORM_TYPES = ["None", "BatchNorm1d", "LayerNorm"]
HEAD_DROPOUT_TYPES = ["None", "Dropout"]

FLATTEN_MODES = ["Flatten (use input H/W)", "Global Average Pool -> Flatten"]

OPTIMIZER_TYPES = ["Adam", "SGD", "AdamW", "RMSprop", "Adagrad"]
LOSS_TYPES = ["CrossEntropyLoss", "NLLLoss", "BCEWithLogitsLoss"]
VAL_METRIC_CHOICES = [
    "Best Validation Accuracy",
    "Best Validation F1",
    "Best Validation Loss",
    "Best Train Accuracy",
    "Best Train F1",
    "Best Train Loss",
    "Every Epoch",
]
WEIGHT_MODES = ["Auto-balance (Inverse Frequency)", "Manual (comma-separated)"]

IMAGEFOLDER_WARNING = (
    "FORMAT REQUIREMENT FOR ImageFolder:\n"
    "The dataset directory must contain a subfolder for each class:\n\n"
    "  root_dir/\n"
    "  ├── class_a/\n"
    "  │   ├── img01.jpg\n"
    "  │   └── img02.png\n"
    "  └── class_b/\n"
    "      ├── img03.jpg\n"
    "      └── img04.png\n\n"
    "Subfolder names will automatically be used as class labels."
)

CSV_FOLDER_WARNING = (
    "FORMAT REQUIREMENT FOR Folder + CSV:\n"
    "1. An images directory containing all raw image files.\n"
    "2. A CSV file containing AT LEAST these EXACT column names:\n"
    "   id, img_name, label\n\n"
    "   • 'id': sample identifier (e.g. 1, 2, 3...)\n"
    "   • 'img_name': image filename (e.g. 'cat_01.jpg') relative to images folder\n"
    "   • 'label': class name string or class integer\n\n"
    "Example CSV header and rows:\n"
    "   id,img_name,label\n"
    "   1,img001.jpg,cat\n"
    "   2,img002.jpg,dog"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConvLayerBlock:
    # Convolution
    out_channels: int = 32
    kernel_size: str = "3"      # "3" or "3,5" (H,W)
    stride: str = "1"           # "1" or "1,2"
    padding: str = "1"          # "1" or "1,2"
    dilation: str = "1"
    groups: int = 1
    bias: bool = True
    # Normalization
    norm_type: str = "BatchNorm2d"
    norm_eps: str = "1e-5"
    norm_momentum: str = "0.1"
    norm_affine: bool = True
    norm_num_groups: str = "8"          # used only for GroupNorm
    # Pooling
    pool_type: str = "None"
    pool_kernel_size: str = "2"
    pool_stride: str = "2"
    pool_padding: str = "0"
    pool_output_size: str = "1"         # used only for Adaptive*Pool2d
    # Dropout
    dropout_type: str = "None"
    dropout_p: str = "0.0"
    # Activation
    activation_type: str = "ReLU"
    act_negative_slope: str = "0.01"    # LeakyReLU
    act_alpha: str = "1.0"              # ELU
    act_dim: str = "1"                  # Softmax

    def summary(self) -> str:
        parts = [f"Conv2d(out={self.out_channels}, k={self.kernel_size}, "
                 f"s={self.stride}, p={self.padding})"]
        if self.norm_type != "None":
            parts.append(self.norm_type)
        if self.pool_type != "None":
            if "Adaptive" in self.pool_type:
                parts.append(f"{self.pool_type}(out={self.pool_output_size})")
            else:
                parts.append(f"{self.pool_type}(k={self.pool_kernel_size})")
        if self.activation_type != "None":
            parts.append(self.activation_type)
        if self.dropout_type != "None":
            parts.append(f"{self.dropout_type}(p={self.dropout_p})")
        return "  ->  ".join(parts)


@dataclass
class HeadLayerBlock:
    out_features: int = 128
    bias: bool = True
    norm_type: str = "None"
    dropout_type: str = "None"
    dropout_p: str = "0.5"
    activation_type: str = "ReLU"
    act_negative_slope: str = "0.01"
    act_alpha: str = "1.0"

    def summary(self) -> str:
        parts = [f"Linear(out={self.out_features})"]
        if self.norm_type != "None":
            parts.append(self.norm_type)
        if self.activation_type != "None":
            parts.append(self.activation_type)
        if self.dropout_type != "None":
            parts.append(f"{self.dropout_type}(p={self.dropout_p})")
        return "  ->  ".join(parts)


@dataclass
class ModelDesign:
    in_channels: int = 3
    input_height: int = 224
    input_width: int = 224
    num_classes: int = 10
    flatten_mode: str = FLATTEN_MODES[0]
    backbone: List[ConvLayerBlock] = field(default_factory=list)
    head: List[HeadLayerBlock] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(text: str) -> "ModelDesign":
        d = json.loads(text)
        backbone = [ConvLayerBlock(**b) for b in d.get("backbone", [])]
        head = [HeadLayerBlock(**h) for h in d.get("head", [])]
        design = ModelDesign(
            in_channels=d.get("in_channels", 3),
            input_height=d.get("input_height", 224),
            input_width=d.get("input_width", 224),
            num_classes=d.get("num_classes", 10),
            flatten_mode=d.get("flatten_mode", FLATTEN_MODES[0]),
            backbone=backbone,
            head=head,
        )
        return design

    def get_nn_module_class(self, class_name: str = "GeneratedCNN"):
        try:
            import torch   #(imported here so the GUI itself never needs torch)
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to build the nn.Module class. "
                "Install it with `pip install torch` and try again."
            ) from exc

        code = generate_model_code(self, class_name=class_name)
        namespace: Dict[str, Any] = {}
        exec(compile(code, f"<generated:{class_name}>", "exec"), namespace)
        return namespace[class_name]


def _parse_num(s: str) -> float:
    s = s.strip()
    try:
        f = float(s)
    except ValueError as exc:
        raise ValueError(f"'{s}' is not a valid number") from exc
    return int(f) if f.is_integer() and ("." not in s and "e" not in s.lower()) else f


def _parse_int_or_tuple(s: str) -> Tuple[int, ...]:
    s = s.strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return tuple(int(float(p)) for p in parts)
    return (int(float(s)),)


def _parse_int_or_pair(s: str) -> Tuple[int, int]:
    """Parse 'x' -> (x, x) or 'x,y' -> (x, y). Always returns a 2-tuple (a, b)."""
    vals = _parse_int_or_tuple(s)
    if len(vals) == 1:
        return (vals[0], vals[0])
    if len(vals) == 2:
        return (vals[0], vals[1])
    raise ValueError(f"Expected 1 or 2 values, got {len(vals)} in '{s}'")


def _literal_src(s: str) -> str:
    vals = _parse_int_or_tuple(s)
    if len(vals) == 1:
        return str(vals[0])
    return "(" + ", ".join(str(v) for v in vals) + ")"


def _float_src(s: str) -> str:
    s = s.strip()
    float(s)
    return s


def _square(s: str) -> int:
    return _parse_int_or_tuple(s)[0]


def validate_conv_block(b: ConvLayerBlock) -> Optional[str]:
    try:
        ks = _parse_int_or_tuple(b.kernel_size)
        if any(v < 1 for v in ks):
            return f"Kernel size values must be >= 1, got {b.kernel_size}"

        strides = _parse_int_or_tuple(b.stride)
        if any(v < 1 for v in strides):
            return f"Stride values must be >= 1 (stride=0 is invalid), got {b.stride}"

        paddings = _parse_int_or_tuple(b.padding)
        if any(v < 0 for v in paddings):
            return f"Padding values must be >= 0, got {b.padding}"

        dilations = _parse_int_or_tuple(b.dilation)
        if any(v < 1 for v in dilations):
            return f"Dilation values must be >= 1, got {b.dilation}"

        oc = int(b.out_channels)
        if oc < 1:
            return f"Out channels must be >= 1, got {oc}"

        g = int(b.groups)
        if g < 1:
            return f"Groups must be >= 1, got {g}"
        if oc % g != 0:
            return f"out_channels ({oc}) must be divisible by groups ({g})"

        float(b.dropout_p)
        if b.pool_type != "None":
            if "Adaptive" in b.pool_type:
                pout = _parse_int_or_tuple(b.pool_output_size)
                if any(v < 1 for v in pout):
                    return f"Adaptive pool output size must be >= 1"
            else:
                pk = _parse_int_or_tuple(b.pool_kernel_size)
                ps = _parse_int_or_tuple(b.pool_stride)
                pp = _parse_int_or_tuple(b.pool_padding)
                if any(v < 1 for v in pk):
                    return f"Pool kernel size must be >= 1"
                if any(v < 1 for v in ps):
                    return f"Pool stride must be >= 1"
                if any(v < 0 for v in pp):
                    return f"Pool padding must be >= 0"
        if b.norm_type == "GroupNorm":
            ng = int(b.norm_num_groups)
            if ng < 1:
                return f"num_groups must be >= 1"
        if b.norm_type != "None":
            float(b.norm_eps)
            float(b.norm_momentum)
        if b.activation_type == "LeakyReLU":
            float(b.act_negative_slope)
        if b.activation_type == "ELU":
            float(b.act_alpha)
    except ValueError as exc:
        return str(exc)
    return None


def validate_head_block(h: HeadLayerBlock) -> Optional[str]:
    try:
        of = int(h.out_features)
        if of < 1:
            return f"Out features must be >= 1, got {of}"
        float(h.dropout_p)
        if h.activation_type == "LeakyReLU":
            float(h.act_negative_slope)
        if h.activation_type == "ELU":
            float(h.act_alpha)
    except ValueError as exc:
        return str(exc)
    return None


def _conv_out_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _pool_out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - kernel) // stride + 1


def trace_spatial_size(design: ModelDesign) -> Tuple[int, int, int]:
    c = design.in_channels
    h = design.input_height
    w = design.input_width

    for b in design.backbone:
        # Parse all parameters as (H, W) pairs
        kH, kW = _parse_int_or_pair(b.kernel_size)
        sH, sW = _parse_int_or_pair(b.stride)
        pH, pW = _parse_int_or_pair(b.padding)
        dH, dW = _parse_int_or_pair(b.dilation)

        h = _conv_out_size(h, kH, sH, pH, dH)
        w = _conv_out_size(w, kW, sW, pW, dW)
        c = int(b.out_channels)

        if b.pool_type in ("MaxPool2d", "AvgPool2d"):
            pkH, pkW = _parse_int_or_pair(b.pool_kernel_size)
            psH, psW = _parse_int_or_pair(b.pool_stride)
            ppH, ppW = _parse_int_or_pair(b.pool_padding)
            h = _pool_out_size(h, pkH, psH, ppH)
            w = _pool_out_size(w, pkW, psW, ppW)
        elif b.pool_type in ("AdaptiveAvgPool2d", "AdaptiveMaxPool2d"):
            out_pair = _parse_int_or_pair(b.pool_output_size)
            h = out_pair[0]
            w = out_pair[1]

    return c, max(h, 1), max(w, 1)


def compute_conv_block_output_shape(block: ConvLayerBlock, in_c: int, in_h: int, in_w: int) -> Tuple[int, int, int]:
    #Calculate the output tensor shape (out_channels, out_h, out_w) for a single conv block.
    kH, kW = _parse_int_or_pair(block.kernel_size)
    sH, sW = _parse_int_or_pair(block.stride)
    pH, pW = _parse_int_or_pair(block.padding)
    dH, dW = _parse_int_or_pair(block.dilation)

    h = _conv_out_size(in_h, kH, sH, pH, dH)
    w = _conv_out_size(in_w, kW, sW, pW, dW)
    out_c = int(block.out_channels)

    if block.pool_type in ("MaxPool2d", "AvgPool2d"):
        pkH, pkW = _parse_int_or_pair(block.pool_kernel_size)
        psH, psW = _parse_int_or_pair(block.pool_stride)
        ppH, ppW = _parse_int_or_pair(block.pool_padding)
        h = _pool_out_size(h, pkH, psH, ppH)
        w = _pool_out_size(w, pkW, psW, ppW)
    elif block.pool_type in ("AdaptiveAvgPool2d", "AdaptiveMaxPool2d"):
        out_pair = _parse_int_or_pair(block.pool_output_size)
        h = out_pair[0]
        w = out_pair[1]

    return out_c, h, w


def compute_layer_shapes(design: ModelDesign) -> List[Dict[str, Any]]:
    layers: List[Dict[str, Any]] = []
    curr_c = design.in_channels
    curr_h = design.input_height
    curr_w = design.input_width

    # 1. Backbone conv layers
    for idx, b in enumerate(design.backbone):
        in_shape = (curr_c, curr_h, curr_w)
        try:
            out_c, out_h, out_w = compute_conv_block_output_shape(b, curr_c, curr_h, curr_w)
            out_shape = (out_c, out_h, out_w)
            curr_c, curr_h, curr_w = out_c, out_h, out_w
        except Exception:
            out_shape = ("?", "?", "?")

        layers.append({
            "idx": idx + 1,
            "section": "Backbone",
            "name": f"Backbone Layer #{idx + 1}",
            "type": "ConvLayerBlock",
            "input_shape": in_shape,
            "input_str": f"({in_shape[0]}, {in_shape[1]}, {in_shape[2]})",
            "output_shape": out_shape,
            "output_str": f"({out_shape[0]}, {out_shape[1]}, {out_shape[2]})" if isinstance(out_shape, tuple) else str(out_shape),
            "summary": b.summary(),
        })

    # 2. Transition (Flatten or GAP)
    use_gap = design.flatten_mode == FLATTEN_MODES[1]
    in_trans = (curr_c, curr_h, curr_w)
    if use_gap:
        flat_features = curr_c
        trans_summary = "AdaptiveAvgPool2d(1) -> Flatten"
    else:
        flat_features = curr_c * max(curr_h, 1) * max(curr_w, 1)
        trans_summary = f"Flatten({curr_c} × {curr_h} × {curr_w})"

    layers.append({
        "idx": len(layers) + 1,
        "section": "Transition",
        "name": "Transition (Before Head)",
        "type": "GAP -> Flatten" if use_gap else "Flatten",
        "input_shape": in_trans,
        "input_str": f"({curr_c}, {curr_h}, {curr_w})",
        "output_shape": flat_features,
        "output_str": f"{flat_features}",
        "summary": trans_summary,
    })

    # 3. Head blocks
    curr_feat = flat_features
    for idx, h in enumerate(design.head):
        in_feat = curr_feat
        try:
            out_feat = int(h.out_features)
            curr_feat = out_feat
        except Exception:
            out_feat = "?"

        layers.append({
            "idx": len(layers) + 1,
            "section": "Head",
            "name": f"Head Layer #{idx + 1}",
            "type": "HeadLayerBlock",
            "input_shape": in_feat,
            "input_str": f"{in_feat}",
            "output_shape": out_feat,
            "output_str": f"{out_feat}",
            "summary": h.summary(),
        })

    # 4. Final Classification Layer 
    layers.append({
        "idx": len(layers) + 1,
        "section": "Output",
        "name": "Final Classifier Layer",
        "type": "Linear (Auto)",
        "input_shape": curr_feat,
        "input_str": f"{curr_feat}",
        "output_shape": design.num_classes,
        "output_str": f"{design.num_classes}",
        "summary": f"Linear(in={curr_feat}, out={design.num_classes}) -> Logits",
    })

    return layers



# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def generate_conv_layer_lines(b: ConvLayerBlock, in_channels: int, indent: str = " " * 12) -> List[str]:
    lines = []
    lines.append(
        f"{indent}nn.Conv2d(in_channels={in_channels}, out_channels={b.out_channels}, "
        f"kernel_size={_literal_src(b.kernel_size)}, stride={_literal_src(b.stride)}, "
        f"padding={_literal_src(b.padding)}, dilation={_literal_src(b.dilation)}, "
        f"groups={b.groups}, bias={b.bias}),"
    )
    if b.norm_type == "BatchNorm2d":
        lines.append(f"{indent}nn.BatchNorm2d({b.out_channels}, eps={_float_src(b.norm_eps)}, "
                      f"momentum={_float_src(b.norm_momentum)}, affine={b.norm_affine}),")
    elif b.norm_type == "InstanceNorm2d":
        lines.append(f"{indent}nn.InstanceNorm2d({b.out_channels}, eps={_float_src(b.norm_eps)}, "
                      f"affine={b.norm_affine}),")
    elif b.norm_type == "GroupNorm":
        lines.append(f"{indent}nn.GroupNorm(num_groups={b.norm_num_groups}, "
                      f"num_channels={b.out_channels}, eps={_float_src(b.norm_eps)}, "
                      f"affine={b.norm_affine}),")
    elif b.norm_type == "LayerNorm":
        lines.append(f"{indent}nn.GroupNorm(num_groups=1, num_channels={b.out_channels}, "
                      f"eps={_float_src(b.norm_eps)}, affine={b.norm_affine}),  "
                      f"# LayerNorm-over-channels emulated via GroupNorm(1, C)")

    if b.pool_type == "MaxPool2d":
        lines.append(f"{indent}nn.MaxPool2d(kernel_size={_literal_src(b.pool_kernel_size)}, "
                      f"stride={_literal_src(b.pool_stride)}, padding={_literal_src(b.pool_padding)}),")
    elif b.pool_type == "AvgPool2d":
        lines.append(f"{indent}nn.AvgPool2d(kernel_size={_literal_src(b.pool_kernel_size)}, "
                      f"stride={_literal_src(b.pool_stride)}, padding={_literal_src(b.pool_padding)}),")
    elif b.pool_type == "AdaptiveAvgPool2d":
        lines.append(f"{indent}nn.AdaptiveAvgPool2d(output_size={_literal_src(b.pool_output_size)}),")
    elif b.pool_type == "AdaptiveMaxPool2d":
        lines.append(f"{indent}nn.AdaptiveMaxPool2d(output_size={_literal_src(b.pool_output_size)}),")

    if b.activation_type == "ReLU":
        lines.append(f"{indent}nn.ReLU(inplace=True),")
    elif b.activation_type == "LeakyReLU":
        lines.append(f"{indent}nn.LeakyReLU(negative_slope={_float_src(b.act_negative_slope)}, inplace=True),")
    elif b.activation_type == "GELU":
        lines.append(f"{indent}nn.GELU(),")
    elif b.activation_type == "SiLU":
        lines.append(f"{indent}nn.SiLU(inplace=True),")
    elif b.activation_type == "Sigmoid":
        lines.append(f"{indent}nn.Sigmoid(),")
    elif b.activation_type == "Tanh":
        lines.append(f"{indent}nn.Tanh(),")
    elif b.activation_type == "ELU":
        lines.append(f"{indent}nn.ELU(alpha={_float_src(b.act_alpha)}, inplace=True),")
    elif b.activation_type == "PReLU":
        lines.append(f"{indent}nn.PReLU(),")
    elif b.activation_type == "Softmax":
        lines.append(f"{indent}nn.Softmax(dim={b.act_dim}),")

    if b.dropout_type == "Dropout":
        lines.append(f"{indent}nn.Dropout(p={_float_src(b.dropout_p)}),")
    elif b.dropout_type == "Dropout2d":
        lines.append(f"{indent}nn.Dropout2d(p={_float_src(b.dropout_p)}),")

    return lines


def generate_head_layer_lines(h: HeadLayerBlock, in_features: int, indent: str = " " * 12) -> List[str]:
    lines = [f"{indent}nn.Linear(in_features={in_features}, out_features={h.out_features}, bias={h.bias}),"]
    if h.norm_type == "BatchNorm1d":
        lines.append(f"{indent}nn.BatchNorm1d({h.out_features}),")
    elif h.norm_type == "LayerNorm":
        lines.append(f"{indent}nn.LayerNorm({h.out_features}),")

    if h.activation_type == "ReLU":
        lines.append(f"{indent}nn.ReLU(inplace=True),")
    elif h.activation_type == "LeakyReLU":
        lines.append(f"{indent}nn.LeakyReLU(negative_slope={_float_src(h.act_negative_slope)}, inplace=True),")
    elif h.activation_type == "GELU":
        lines.append(f"{indent}nn.GELU(),")
    elif h.activation_type == "SiLU":
        lines.append(f"{indent}nn.SiLU(inplace=True),")
    elif h.activation_type == "Sigmoid":
        lines.append(f"{indent}nn.Sigmoid(),")
    elif h.activation_type == "Tanh":
        lines.append(f"{indent}nn.Tanh(),")
    elif h.activation_type == "ELU":
        lines.append(f"{indent}nn.ELU(alpha={_float_src(h.act_alpha)}, inplace=True),")

    if h.dropout_type == "Dropout":
        lines.append(f"{indent}nn.Dropout(p={_float_src(h.dropout_p)}),")

    return lines


def generate_model_code(design: ModelDesign, class_name: str = "GeneratedCNN") -> str:
    if not design.backbone:
        raise ValueError("Add at least one backbone (conv) layer before generating code.")

    backbone_lines: List[str] = []
    in_ch = design.in_channels
    for b in design.backbone:
        backbone_lines.extend(generate_conv_layer_lines(b, in_ch))
        in_ch = int(b.out_channels)

    final_c, final_h, final_w = trace_spatial_size(design)
    use_gap = design.flatten_mode == FLATTEN_MODES[1]
    flat_features = final_c if use_gap else final_c * final_h * final_w

    head_lines: List[str] = []
    in_feat = flat_features
    for h in design.head:
        head_lines.extend(generate_head_layer_lines(h, in_feat))
        in_feat = int(h.out_features)
    # Final classifier layer -> num_classes (no activation; use with CrossEntropyLoss)
    head_lines.append(f"{' ' * 12}nn.Linear(in_features={in_feat}, out_features={design.num_classes}),")

    gap_line = "nn.AdaptiveAvgPool2d(1)," if use_gap else "# (using nn.Flatten on raw spatial size below)"

    code = f'''"""
Auto-generated by VisualCNN Architecture Designer.
Input assumed: (batch, {design.in_channels}, {design.input_height}, {design.input_width})
Traced output of backbone: (batch, {final_c}, {final_h}, {final_w})
{"Global-average-pooled before the head." if use_gap else "Flattened directly (H * W * C) before the head."}
NOTE: Spatial-size tracing correctly handles asymmetric (H, W) kernel/stride/padding values.
"""

import torch
import torch.nn as nn


class {class_name}(nn.Module):
    def __init__(self, in_channels: int = {design.in_channels}, num_classes: int = {design.num_classes}):
        super().__init__()

        self.backbone = nn.Sequential(
{chr(10).join(backbone_lines)}
        )

        {gap_line}
        self.gap = nn.AdaptiveAvgPool2d(1) if {use_gap} else None
        self.flatten = nn.Flatten()

        self.head = nn.Sequential(
{chr(10).join(head_lines)}
        )

    def forward(self, x):
        x = self.backbone(x)
        if self.gap is not None:
            x = self.gap(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = {class_name}()
    dummy = torch.randn(2, {design.in_channels}, {design.input_height}, {design.input_width})
    out = model(dummy)
    print(model)
    print("Output shape:", out.shape)
'''
    return code
