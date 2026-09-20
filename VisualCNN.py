from __future__ import annotations
import copy
import json
import os
import sys
import time
import threading
import subprocess
import tkinter as tk
from dataclasses import asdict
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from src.models import (
    ConvLayerBlock,
    HeadLayerBlock,
    ModelDesign,
    NORM_TYPES,
    POOL_TYPES,
    ACT_TYPES,
    CONV_DROPOUT_TYPES,
    HEAD_NORM_TYPES,
    HEAD_DROPOUT_TYPES,
    FLATTEN_MODES,
    validate_conv_block,
    validate_head_block,
    trace_spatial_size,
    generate_model_code,
    compute_conv_block_output_shape,
    compute_layer_shapes,
)
from src.dataset import (
    CSVImageDataset,
    create_image_folder_dataset,
    get_dataset_transform,
    OPTIMIZER_TYPES,
    LOSS_TYPES,
    VAL_METRIC_CHOICES,
    WEIGHT_MODES,
    IMAGEFOLDER_WARNING,
    CSV_FOLDER_WARNING,
)
from src.deployment import (
    export_model_package,
    ModelDeployer,
    LocalAPIServer,
)


def labeled_entry(parent, row, label, default, width=12, on_change=None):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
    var = tk.StringVar(value=str(default))
    ent = ttk.Entry(parent, textvariable=var, width=width)
    ent.grid(row=row, column=1, sticky="w", padx=4, pady=3)
    var.widget = ent
    if on_change:
        ent.bind("<KeyRelease>", lambda e: on_change())
        ent.bind("<FocusOut>", lambda e: on_change())
    return var


def set_enabled(var: "tk.StringVar", enabled: bool):
    w = getattr(var, "widget", None)
    if w is not None:
        w.configure(state="normal" if enabled else "disabled")


def labeled_combo(parent, row, label, values, default, width=18, on_change=None):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
    var = tk.StringVar(value=default)
    combo = ttk.Combobox(parent, textvariable=var, values=values, width=width, state="readonly")
    combo.grid(row=row, column=1, sticky="w", padx=4, pady=3)
    if on_change:
        combo.bind("<<ComboboxSelected>>", lambda e: on_change())
    var.widget = combo
    return var


def labeled_check(parent, row, label, default, on_change=None):
    var = tk.BooleanVar(value=default)
    chk = ttk.Checkbutton(parent, text=label, variable=var, command=on_change if on_change else None)
    chk.grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=3)
    var.widget = chk
    return var


# ---------------------------------------------------------------------------
# Dialog: View PyTorch Code
# ---------------------------------------------------------------------------

class CodeViewerDialog(tk.Toplevel):
    def __init__(self, master, design: ModelDesign):
        super().__init__(master)
        self.title("PyTorch Model Code Preview")
        self.geometry("820x620")
        self.minsize(650, 450)
        self.design = design

        top_fr = ttk.Frame(self, padding=8)
        top_fr.pack(fill="x")
        ttk.Label(
            top_fr,
            text="Generated Standalone PyTorch nn.Module Implementation",
            font=("TkDefaultFont", 11, "bold"),
            foreground="#1565C0",
        ).pack(side="left")

        btn_copy = ttk.Button(top_fr, text="📋 Copy Code", command=self._copy_to_clipboard)
        btn_copy.pack(side="right", padx=4)
        btn_save = ttk.Button(top_fr, text="💾 Save to .py", command=self._save_to_file)
        btn_save.pack(side="right", padx=4)

        try:
            self.code_text = generate_model_code(design, class_name="AutomateCNNModel")
        except Exception as exc:
            self.code_text = f"# Error generating PyTorch code:\n# {exc}"

        txt_fr = ttk.Frame(self, padding=8)
        txt_fr.pack(fill="both", expand=True)

        self.txt = scrolledtext.ScrolledText(txt_fr, wrap="none", font=("Courier", 10))
        self.txt.pack(fill="both", expand=True)
        self.txt.insert(tk.END, self.code_text)
        self.txt.configure(state="disabled")

        btm_fr = ttk.Frame(self, padding=8)
        btm_fr.pack(fill="x")
        ttk.Button(btm_fr, text="Close", command=self.destroy).pack(side="right", padx=4)

        self.transient(master)
        self.grab_set()

    def _copy_to_clipboard(self):
        self.clipboard_clear()
        self.clipboard_append(self.code_text)
        messagebox.showinfo("Copied", "PyTorch model code copied to clipboard!", parent=self)

    def _save_to_file(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")],
            initialfile="cnn_model.py",
            parent=self,
        )
        if p:
            with open(p, "w") as f:
                f.write(self.code_text)
            messagebox.showinfo("Saved", f"Model code saved to:\n{p}", parent=self)


# ---------------------------------------------------------------------------
# Dialog: Network Recapitulation / Tensor Flow Summary
# ---------------------------------------------------------------------------

class NetworkRecapitulationDialog(tk.Toplevel):
    def __init__(self, master, design: ModelDesign, on_confirm=None):
        super().__init__(master)
        self.title("Architecture Recapitulation & Layer Tensor Flow")
        self.geometry("900x560")
        self.minsize(780, 480)
        self.design = design
        self.on_confirm = on_confirm

        top_fr = ttk.Frame(self, padding=10)
        top_fr.pack(fill="x")
        ttk.Label(
            top_fr,
            text="Complete Layer-by-Layer Network Tensor Flow Recapitulation",
            font=("TkDefaultFont", 12, "bold"),
            foreground="#1B5E20",
        ).pack(anchor="w")
        info_txt = (
            f"Input Image Tensor: ({design.in_channels}, {design.input_height}, {design.input_width})  ➔  "
            f"Target Classes: {design.num_classes}  |  "
            f"Backbone Blocks: {len(design.backbone)}  |  Head Blocks: {len(design.head)}"
        )
        ttk.Label(top_fr, text=info_txt, foreground="#555555").pack(anchor="w", pady=(2, 0))

        tree_fr = ttk.Frame(self, padding=8)
        tree_fr.pack(fill="both", expand=True)

        columns = ("step", "section", "layer_name", "input_shape", "output_shape", "details")
        self.tree = ttk.Treeview(tree_fr, columns=columns, show="headings", height=14)
        self.tree.heading("step", text="Step")
        self.tree.heading("section", text="Section")
        self.tree.heading("layer_name", text="Layer / Stage")
        self.tree.heading("input_shape", text="Input Tensor")
        self.tree.heading("output_shape", text="Output Tensor")
        self.tree.heading("details", text="Layer Details & Operations")

        self.tree.column("step", width=45, anchor="center")
        self.tree.column("section", width=95, anchor="center")
        self.tree.column("layer_name", width=160, anchor="w")
        self.tree.column("input_shape", width=120, anchor="center")
        self.tree.column("output_shape", width=120, anchor="center")
        self.tree.column("details", width=340, anchor="w")

        scroll = ttk.Scrollbar(tree_fr, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # Tags styling
        self.tree.tag_configure("backbone", background="#F1F8E9")
        self.tree.tag_configure("trans", background="#FFF8E1")
        self.tree.tag_configure("head", background="#E3F2FD")
        self.tree.tag_configure("output", background="#E8F5E9", font=("TkDefaultFont", 9, "bold"))

        self._populate_layers()

        btm_fr = ttk.Frame(self, padding=10)
        btm_fr.pack(fill="x")
        ttk.Button(btm_fr, text="⮌ Back to Designer", command=self.destroy).pack(side="left", padx=4)
        btn_proceed = ttk.Button(
            btm_fr, text="Proceed to Dataset & Training ➔", command=self._on_proceed
        )
        btn_proceed.pack(side="right", padx=4)

        self.transient(master)
        self.grab_set()

    def _populate_layers(self):
        layers = compute_layer_shapes(self.design)
        for layer in layers:
            tag = layer["section"].lower()
            if tag.startswith("backbone"):
                tag = "backbone"
            elif tag.startswith("head"):
                tag = "head"
            elif tag.startswith("output"):
                tag = "output"
            else:
                tag = "trans"

            self.tree.insert(
                "",
                "end",
                values=(
                    layer["idx"],
                    layer["section"],
                    layer["name"],
                    layer["input_str"],
                    layer["output_str"],
                    layer["summary"],
                ),
                tags=(tag,),
            )

    def _on_proceed(self):
        self.destroy()
        if self.on_confirm:
            self.on_confirm()


# ---------------------------------------------------------------------------
# Dialog: Configure Backbone Layer Block with Live Tensor Shape Tracing
# ---------------------------------------------------------------------------

class ConvBlockDialog(tk.Toplevel):
    def __init__(
        self,
        master,
        block: Optional[ConvLayerBlock] = None,
        in_shape: Tuple[int, int, int] = (3, 224, 224),
    ):
        super().__init__(master)
        self.title("Configure Backbone Layer Block")
        self.resizable(False, False)
        self.result: Optional[ConvLayerBlock] = None
        self.in_shape = in_shape
        b = block if block is not None else ConvLayerBlock()

        outer = ttk.Frame(self, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")

        # Shape Live Tracing Banner
        self.shape_fr = ttk.Labelframe(outer, text="Live Tensor Shape Tracing", padding=8)
        self.shape_fr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 8))

        self.lbl_in_shape = ttk.Label(
            self.shape_fr,
            text=f"Expected Input Tensor:  (Channels: {in_shape[0]}, Height: {in_shape[1]}, Width: {in_shape[2]})",
            font=("TkDefaultFont", 9, "bold"),
            foreground="#37474F",
        )
        self.lbl_in_shape.pack(anchor="w")

        self.lbl_out_shape = ttk.Label(
            self.shape_fr,
            text="Calculated Output Tensor: Calculating...",
            font=("TkDefaultFont", 10, "bold"),
            foreground="#1565C0",
        )
        self.lbl_out_shape.pack(anchor="w", pady=(3, 0))

        # 1. Convolution
        conv_fr = ttk.Labelframe(outer, text="1. Convolution (Conv2d)", padding=8)
        conv_fr.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.out_channels = labeled_entry(conv_fr, 0, "Out channels:", b.out_channels, on_change=self._update_shape_preview)
        self.kernel_size = labeled_entry(conv_fr, 1, "Kernel size (e.g. 3 or 3,5):", b.kernel_size, on_change=self._update_shape_preview)
        self.stride = labeled_entry(conv_fr, 2, "Stride (>= 1):", b.stride, on_change=self._update_shape_preview)
        self.padding = labeled_entry(conv_fr, 3, "Padding:", b.padding, on_change=self._update_shape_preview)
        self.dilation = labeled_entry(conv_fr, 4, "Dilation (>= 1):", b.dilation, on_change=self._update_shape_preview)
        self.groups = labeled_entry(conv_fr, 5, "Groups:", b.groups, on_change=self._update_shape_preview)
        self.bias = labeled_check(conv_fr, 6, "Use bias", b.bias)

        # 2. Normalization
        norm_fr = ttk.Labelframe(outer, text="2. Normalization", padding=8)
        norm_fr.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self.norm_type = labeled_combo(norm_fr, 0, "Type:", NORM_TYPES, b.norm_type, on_change=lambda: self._sync_norm_fields())
        self.norm_eps = labeled_entry(norm_fr, 1, "eps:", b.norm_eps)
        self.norm_momentum = labeled_entry(norm_fr, 2, "momentum (BatchNorm):", b.norm_momentum)
        self.norm_affine = labeled_check(norm_fr, 3, "Affine (learnable scale/shift)", b.norm_affine)
        self.norm_num_groups = labeled_entry(norm_fr, 4, "num_groups (GroupNorm only):", b.norm_num_groups)

        # 3. Pooling
        pool_fr = ttk.Labelframe(outer, text="3. Pooling", padding=8)
        pool_fr.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        self.pool_type = labeled_combo(pool_fr, 0, "Type:", POOL_TYPES, b.pool_type, on_change=lambda: self._sync_pool_fields())
        self.pool_kernel_size = labeled_entry(pool_fr, 1, "Kernel size:", b.pool_kernel_size, on_change=self._update_shape_preview)
        self.pool_stride = labeled_entry(pool_fr, 2, "Stride:", b.pool_stride, on_change=self._update_shape_preview)
        self.pool_padding = labeled_entry(pool_fr, 3, "Padding:", b.pool_padding, on_change=self._update_shape_preview)
        self.pool_output_size = labeled_entry(pool_fr, 4, "Output size (Adaptive* only, e.g. 1,1):", b.pool_output_size, on_change=self._update_shape_preview)

        # 4. Activation
        act_fr = ttk.Labelframe(outer, text="4. Activation", padding=8)
        act_fr.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)
        self.activation_type = labeled_combo(act_fr, 0, "Type:", ACT_TYPES, b.activation_type, on_change=lambda: self._sync_act_fields())
        self.act_negative_slope = labeled_entry(act_fr, 1, "negative_slope (LeakyReLU):", b.act_negative_slope)
        self.act_alpha = labeled_entry(act_fr, 2, "alpha (ELU):", b.act_alpha)
        self.act_dim = labeled_entry(act_fr, 3, "dim (Softmax):", b.act_dim)

        # 5. Dropout
        drop_fr = ttk.Labelframe(outer, text="5. Dropout", padding=8)
        drop_fr.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)
        self.dropout_type = labeled_combo(drop_fr, 0, "Type:", CONV_DROPOUT_TYPES, b.dropout_type, on_change=lambda: self._sync_dropout_fields())
        self.dropout_p = labeled_entry(drop_fr, 1, "p:", b.dropout_p)

        self._sync_norm_fields()
        self._sync_pool_fields()
        self._sync_dropout_fields()
        self._sync_act_fields()
        self._update_shape_preview()

        btn_fr = ttk.Frame(outer)
        btn_fr.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky="e")
        ttk.Button(btn_fr, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(btn_fr, text="OK", command=self._on_ok).pack(side="right", padx=4)

        self.transient(master)
        self.grab_set()
        self.wait_window(self)

    def _sync_norm_fields(self):
        t = self.norm_type.get()
        set_enabled(self.norm_eps, t != "None")
        set_enabled(self.norm_momentum, t == "BatchNorm2d")
        set_enabled(self.norm_num_groups, t == "GroupNorm")

    def _sync_pool_fields(self):
        t = self.pool_type.get()
        adaptive = t in ("AdaptiveAvgPool2d", "AdaptiveMaxPool2d")
        regular = t in ("MaxPool2d", "AvgPool2d")
        set_enabled(self.pool_kernel_size, regular)
        set_enabled(self.pool_stride, regular)
        set_enabled(self.pool_padding, regular)
        set_enabled(self.pool_output_size, adaptive)
        self._update_shape_preview()

    def _sync_dropout_fields(self):
        set_enabled(self.dropout_p, self.dropout_type.get() != "None")

    def _sync_act_fields(self):
        t = self.activation_type.get()
        set_enabled(self.act_negative_slope, t == "LeakyReLU")
        set_enabled(self.act_alpha, t == "ELU")
        set_enabled(self.act_dim, t == "Softmax")

    def _update_shape_preview(self):
        temp_block = ConvLayerBlock(
            out_channels=self.out_channels.get(),
            kernel_size=self.kernel_size.get(),
            stride=self.stride.get(),
            padding=self.padding.get(),
            dilation=self.dilation.get(),
            groups=self.groups.get(),
            bias=self.bias.get(),
            norm_type=self.norm_type.get(),
            pool_type=self.pool_type.get(),
            pool_kernel_size=self.pool_kernel_size.get(),
            pool_stride=self.pool_stride.get(),
            pool_padding=self.pool_padding.get(),
            pool_output_size=self.pool_output_size.get(),
        )
        try:
            val_err = validate_conv_block(temp_block)
            if val_err:
                self.lbl_out_shape.configure(
                    text=f"⚠️ Invalid Parameter: {val_err}", foreground="#C62828"
                )
                return

            c, h, w = compute_conv_block_output_shape(
                temp_block, self.in_shape[0], self.in_shape[1], self.in_shape[2]
            )
            if h <= 0 or w <= 0:
                self.lbl_out_shape.configure(
                    text=f"⚠️ Spatial Dimension Collapsed to {h}×{w}! Check kernel size/stride/padding.",
                    foreground="#C62828",
                )
            else:
                self.lbl_out_shape.configure(
                    text=f"Calculated Output Tensor:  (Channels: {c}, Height: {h}, Width: {w})  ✓",
                    foreground="#2E7D32",
                )
        except Exception as exc:
            self.lbl_out_shape.configure(text=f"⚠️ Calculation error: {exc}", foreground="#C62828")

    def _on_ok(self):
        block = ConvLayerBlock(
            out_channels=self.out_channels.get(),
            kernel_size=self.kernel_size.get(),
            stride=self.stride.get(),
            padding=self.padding.get(),
            dilation=self.dilation.get(),
            groups=self.groups.get(),
            bias=self.bias.get(),
            norm_type=self.norm_type.get(),
            norm_eps=self.norm_eps.get(),
            norm_momentum=self.norm_momentum.get(),
            norm_affine=self.norm_affine.get(),
            norm_num_groups=self.norm_num_groups.get(),
            pool_type=self.pool_type.get(),
            pool_kernel_size=self.pool_kernel_size.get(),
            pool_stride=self.pool_stride.get(),
            pool_padding=self.pool_padding.get(),
            pool_output_size=self.pool_output_size.get(),
            dropout_type=self.dropout_type.get(),
            dropout_p=self.dropout_p.get(),
            activation_type=self.activation_type.get(),
            act_negative_slope=self.act_negative_slope.get(),
            act_alpha=self.act_alpha.get(),
            act_dim=self.act_dim.get(),
        )
        try:
            block.out_channels = int(block.out_channels)
            block.groups = int(block.groups)
        except ValueError:
            messagebox.showerror("Invalid input", "Out channels and groups must be integers.", parent=self)
            return
        err = validate_conv_block(block)
        if err:
            messagebox.showerror("Invalid input", err, parent=self)
            return

        try:
            _, h, w = compute_conv_block_output_shape(
                block, self.in_shape[0], self.in_shape[1], self.in_shape[2]
            )
            if h <= 0 or w <= 0:
                messagebox.showerror(
                    "Invalid Dimensions",
                    f"Spatial dimensions collapsed to {h}×{w}. Please adjust kernel size, stride, or padding.",
                    parent=self,
                )
                return
        except Exception as exc:
            messagebox.showerror("Shape Calculation Error", str(exc), parent=self)
            return

        self.result = block
        self.destroy()


# ---------------------------------------------------------------------------
# Dialog: Configure Head Layer Block with Live Feature Tracing
# ---------------------------------------------------------------------------

class HeadBlockDialog(tk.Toplevel):
    def __init__(self, master, block: Optional[HeadLayerBlock] = None, in_features: int = 128):
        super().__init__(master)
        self.title("Configure Head Layer Block")
        self.resizable(False, False)
        self.result: Optional[HeadLayerBlock] = None
        self.in_features = in_features
        h = block if block is not None else HeadLayerBlock()

        outer = ttk.Frame(self, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")

        # Feature shape preview
        shape_fr = ttk.Labelframe(outer, text="Live Feature Tracing", padding=8)
        shape_fr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 8))

        self.lbl_in_feat = ttk.Label(
            shape_fr,
            text=f"Expected Input Features:  {in_features}",
            font=("TkDefaultFont", 9, "bold"),
            foreground="#37474F",
        )
        self.lbl_in_feat.pack(anchor="w")

        self.lbl_out_feat = ttk.Label(
            shape_fr,
            text=f"Output Features:  {h.out_features}  ✓",
            font=("TkDefaultFont", 10, "bold"),
            foreground="#2E7D32",
        )
        self.lbl_out_feat.pack(anchor="w", pady=(3, 0))

        lin_fr = ttk.Labelframe(outer, text="1. Linear (Fully Connected)", padding=8)
        lin_fr.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.out_features = labeled_entry(lin_fr, 0, "Out features:", h.out_features, on_change=self._update_feat_preview)
        self.bias = labeled_check(lin_fr, 1, "Use bias", h.bias)

        norm_fr = ttk.Labelframe(outer, text="2. Normalization", padding=8)
        norm_fr.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        self.norm_type = labeled_combo(norm_fr, 0, "Type:", HEAD_NORM_TYPES, h.norm_type)

        act_fr = ttk.Labelframe(outer, text="3. Activation", padding=8)
        act_fr.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        self.activation_type = labeled_combo(act_fr, 0, "Type:", ACT_TYPES, h.activation_type, on_change=lambda: self._sync_act_fields())
        self.act_negative_slope = labeled_entry(act_fr, 1, "negative_slope (LeakyReLU):", h.act_negative_slope)
        self.act_alpha = labeled_entry(act_fr, 2, "alpha (ELU):", h.act_alpha)

        drop_fr = ttk.Labelframe(outer, text="4. Dropout", padding=8)
        drop_fr.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)
        self.dropout_type = labeled_combo(drop_fr, 0, "Type:", HEAD_DROPOUT_TYPES, h.dropout_type, on_change=lambda: self._sync_dropout_fields())
        self.dropout_p = labeled_entry(drop_fr, 1, "p:", h.dropout_p)

        self._sync_dropout_fields()
        self._sync_act_fields()

        btn_fr = ttk.Frame(outer)
        btn_fr.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="e")
        ttk.Button(btn_fr, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(btn_fr, text="OK", command=self._on_ok).pack(side="right", padx=4)

        self.transient(master)
        self.grab_set()
        self.wait_window(self)

    def _update_feat_preview(self):
        try:
            of = int(self.out_features.get())
            if of > 0:
                self.lbl_out_feat.configure(text=f"Output Features:  {of}  ✓", foreground="#2E7D32")
            else:
                self.lbl_out_feat.configure(text="⚠️ Out features must be >= 1", foreground="#C62828")
        except ValueError:
            self.lbl_out_feat.configure(text="⚠️ Enter a valid integer", foreground="#C62828")

    def _sync_dropout_fields(self):
        set_enabled(self.dropout_p, self.dropout_type.get() != "None")

    def _sync_act_fields(self):
        t = self.activation_type.get()
        set_enabled(self.act_negative_slope, t == "LeakyReLU")
        set_enabled(self.act_alpha, t == "ELU")

    def _on_ok(self):
        block = HeadLayerBlock(
            out_features=self.out_features.get(),
            bias=self.bias.get(),
            norm_type=self.norm_type.get(),
            dropout_type=self.dropout_type.get(),
            dropout_p=self.dropout_p.get(),
            activation_type=self.activation_type.get(),
            act_negative_slope=self.act_negative_slope.get(),
            act_alpha=self.act_alpha.get(),
        )
        try:
            block.out_features = int(block.out_features)
        except ValueError:
            messagebox.showerror("Invalid input", "Out features must be an integer.", parent=self)
            return
        err = validate_head_block(block)
        if err:
            messagebox.showerror("Invalid input", err, parent=self)
            return
        self.result = block
        self.destroy()


# ---------------------------------------------------------------------------
# Dialog: Training Graphs Display (Loss & Evaluation Metrics)
# ---------------------------------------------------------------------------

class TrainingGraphsDialog(tk.Toplevel):
    def __init__(
        self,
        master,
        history: List[Dict[str, Any]],
        metric_name: str = "Accuracy",
        best_epoch: Optional[int] = None,
    ):
        super().__init__(master)
        self.title("Training & Validation Progress Graphs")
        self.geometry("860x640")
        self.minsize(720, 500)

        if not history:
            ttk.Label(self, text="No training history recorded yet.").pack(padx=20, pady=20)
            return

        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history if h.get("val_loss") is not None]
        train_metric = [h["train_metric"] for h in history]
        val_metric = [h["val_metric"] for h in history if h.get("val_metric") is not None]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.patch.set_facecolor("#FAFAFA")

        # Subplot 1: Loss
        ax1.set_facecolor("#FFFFFF")
        ax1.plot(epochs, train_loss, label="Train Loss", color="#D32F2F", linewidth=2, marker="o", markersize=4)
        if len(val_loss) == len(epochs):
            ax1.plot(epochs, val_loss, label="Validation Loss", color="#1976D2", linewidth=2, linestyle="--", marker="s", markersize=4)
        ax1.set_ylabel("Loss", fontsize=10, fontweight="bold")
        ax1.set_title("Cross-Entropy Loss Progression", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")

        if best_epoch is not None and best_epoch in epochs:
            ax1.axvline(x=best_epoch, color="#388E3C", linestyle=":", alpha=0.7, label=f"Best Model (Ep {best_epoch})")

        # Subplot 2: Metric Score
        ax2.set_facecolor("#FFFFFF")
        ax2.plot(epochs, train_metric, label=f"Train {metric_name}", color="#E65100", linewidth=2, marker="o", markersize=4)
        if len(val_metric) == len(epochs):
            ax2.plot(epochs, val_metric, label=f"Val {metric_name}", color="#388E3C", linewidth=2, linestyle="--", marker="^", markersize=4)
        ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
        ax2.set_ylabel(f"{metric_name} (%)", fontsize=10, fontweight="bold")
        ax2.set_title(f"{metric_name} Progression", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right")

        if best_epoch is not None and best_epoch in epochs:
            ax2.axvline(x=best_epoch, color="#388E3C", linestyle=":", alpha=0.7)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

        toolbar_fr = ttk.Frame(self)
        toolbar_fr.pack(fill="x", padx=8, pady=(0, 4))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_fr)
        toolbar.update()

        btm_fr = ttk.Frame(self, padding=8)
        btm_fr.pack(fill="x")
        ttk.Button(btm_fr, text="Close", command=self.destroy).pack(side="right", padx=4)


# ---------------------------------------------------------------------------
# Dialog: Model Deployment & Live Testing
# ---------------------------------------------------------------------------

class DeploymentDialog(tk.Toplevel):
    def __init__(self, master, export_dir: str):
        super().__init__(master)
        self.title("Deploy & Test Model - " + os.path.basename(export_dir))
        self.geometry("740x640")
        self.minsize(680, 540)
        self.export_dir = export_dir

        try:
            self.deployer = ModelDeployer(export_dir)
        except Exception as exc:
            messagebox.showerror("Load Failed", f"Failed to initialize deployer from {export_dir}:\n{exc}", parent=master)
            self.destroy()
            return

        self.api_server = None
        self.selected_img_path = tk.StringVar(value="")
        self.pred_result_var = tk.StringVar(value="Select an image and click 'Run Prediction'.")
        self.server_status_var = tk.StringVar(value="Status: Stopped")
        self.server_port_var = tk.StringVar(value="8000")
        self.thumbnail_label = None
        self._thumb_img = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        top_fr = ttk.Labelframe(self, text="Deployed Model Specification", padding=10)
        top_fr.pack(fill="x", padx=10, pady=6)

        cfg = self.deployer.config
        info_text = (
            f"Model: {cfg.get('model_name', 'AutomateCNNModel')} | "
            f"Input: ({cfg['in_channels']}, {cfg['input_height']}, {cfg['input_width']}) | "
            f"Classes ({cfg['num_classes']}): {', '.join(cfg['classes'][:6])}"
            + ("..." if len(cfg['classes']) > 6 else "")
        )
        ttk.Label(top_fr, text=info_text, font=("TkDefaultFont", 9, "bold"), foreground="#1565C0").pack(anchor="w")
        ttk.Label(top_fr, text=f"Package Folder: {self.export_dir}", foreground="#555555").pack(anchor="w", pady=(2, 0))

        pred_fr = ttk.Labelframe(self, text="1. Live Single-Image Classification", padding=10)
        pred_fr.pack(fill="both", expand=True, padx=10, pady=6)

        pick_row = ttk.Frame(pred_fr)
        pick_row.pack(fill="x", pady=4)
        ttk.Button(pick_row, text="Select Test Image...", command=self._browse_image).pack(side="left", padx=4)
        ttk.Label(pick_row, textvariable=self.selected_img_path, foreground="#333333").pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(pick_row, text="▶ Run Prediction", command=self._predict_image).pack(side="right", padx=4)

        content_fr = ttk.Frame(pred_fr)
        content_fr.pack(fill="both", expand=True, pady=6)

        self.thumbnail_label = ttk.Label(content_fr, text="[No Image Selected]", relief="solid", borderwidth=1, anchor="center")
        self.thumbnail_label.pack(side="left", padx=8, pady=4)
        self.thumbnail_label.configure(width=20)

        res_fr = ttk.Frame(content_fr)
        res_fr.pack(side="left", fill="both", expand=True, padx=8)

        self.lbl_big_pred = ttk.Label(res_fr, textvariable=self.pred_result_var, font=("TkDefaultFont", 11, "bold"), foreground="#2E7D32")
        self.lbl_big_pred.pack(anchor="w", pady=(0, 6))

        ttk.Label(res_fr, text="Class Probability Distribution:").pack(anchor="w")
        self.prob_text = scrolledtext.ScrolledText(res_fr, height=6, font=("Courier", 9), wrap="word")
        self.prob_text.pack(fill="both", expand=True, pady=2)

        api_fr = ttk.Labelframe(self, text="2. Local REST API Inference Server", padding=10)
        api_fr.pack(fill="x", padx=10, pady=6)

        api_top = ttk.Frame(api_fr)
        api_top.pack(fill="x", pady=2)
        ttk.Label(api_top, text="Port:").pack(side="left", padx=4)
        ttk.Entry(api_top, textvariable=self.server_port_var, width=6).pack(side="left", padx=4)

        self.btn_start_server = ttk.Button(api_top, text="▶ Start Server", command=self._start_server)
        self.btn_start_server.pack(side="left", padx=6)
        self.btn_stop_server = ttk.Button(api_top, text="⏹ Stop Server", command=self._stop_server, state="disabled")
        self.btn_stop_server.pack(side="left", padx=4)

        self.lbl_server_status = ttk.Label(api_top, textvariable=self.server_status_var, font=("TkDefaultFont", 9, "bold"), foreground="#C62828")
        self.lbl_server_status.pack(side="left", padx=12)

        curl_row = ttk.Frame(api_fr)
        curl_row.pack(fill="x", pady=(4, 0))
        curl_cmd = "curl -X POST -F \"image=@sample.jpg\" http://localhost:8000/predict"
        ttk.Label(curl_row, text=f"Sample Request: {curl_cmd}", font=("Courier", 8), foreground="#555555").pack(anchor="w")

        btm_fr = ttk.Frame(self)
        btm_fr.pack(fill="x", padx=10, pady=(6, 10))
        ttk.Button(btm_fr, text="📁 Open Package Folder", command=self._open_folder).pack(side="left", padx=4)
        ttk.Button(btm_fr, text="Close", command=self._on_close).pack(side="right", padx=4)

    def _browse_image(self):
        f = filedialog.askopenfilename(
            title="Select Test Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")],
        )
        if f:
            self.selected_img_path.set(f)
            try:
                img = Image.open(f)
                img.thumbnail((140, 140))
                self._thumb_img = ImageTk.PhotoImage(img)
                self.thumbnail_label.configure(image=self._thumb_img, text="")
            except Exception:
                self.thumbnail_label.configure(image="", text="[Preview Failed]")

    def _predict_image(self):
        p = self.selected_img_path.get().strip()
        if not p or not os.path.isfile(p):
            messagebox.showwarning("No Image", "Please select a valid test image file first.", parent=self)
            return

        try:
            res = self.deployer.predict(p)
            self.pred_result_var.set(f"Prediction: {res['predicted_class']} ({res['confidence_pct']} confidence)")
            self.lbl_big_pred.configure(foreground="#1565C0")

            self.prob_text.delete("1.0", tk.END)
            for cls_name, prob in res["ranked_probabilities"]:
                bar = "█" * int(prob * 25)
                self.prob_text.insert(tk.END, f"{cls_name:18s}: {prob*100:5.1f}% {bar}\n")
        except Exception as exc:
            messagebox.showerror("Inference Error", str(exc), parent=self)

    def _start_server(self):
        try:
            port = int(self.server_port_var.get())
        except ValueError:
            messagebox.showerror("Invalid Port", "Port must be an integer.", parent=self)
            return

        try:
            self.api_server = LocalAPIServer(self.export_dir, port=port)
            self.api_server.start()
            self.server_status_var.set(f"● Running on http://127.0.0.1:{port} (POST /predict)")
            self.lbl_server_status.configure(foreground="#2E7D32")
            self.btn_start_server.configure(state="disabled")
            self.btn_stop_server.configure(state="normal")
        except Exception as exc:
            messagebox.showerror("Server Error", f"Failed to start API server:\n{exc}", parent=self)

    def _stop_server(self):
        if self.api_server:
            self.api_server.stop()
            self.api_server = None
        self.server_status_var.set("Status: Stopped")
        self.lbl_server_status.configure(foreground="#C62828")
        self.btn_start_server.configure(state="normal")
        self.btn_stop_server.configure(state="disabled")

    def _open_folder(self):
        if sys.platform == "win32":
            os.startfile(self.export_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self.export_dir])
        else:
            subprocess.Popen(["xdg-open", self.export_dir])

    def _on_close(self):
        self._stop_server()
        self.destroy()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class DesignerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VisualCNN — CNN Architecture Designer, Trainer & Deployment Studio")
        self.geometry("1120x820")
        self.minsize(1020, 720)
        self.design = ModelDesign()

        self._init_training_state()
        self._build_global_settings()
        self._build_notebook()
        self._build_bottom_bar()
        self._build_fullscreen_training_view()

        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _init_training_state(self):
        # Dataset variables
        self.train_data_mode = tk.StringVar(value="ImageFolder")
        self.train_img_folder = tk.StringVar(value="")
        self.train_csv_folder = tk.StringVar(value="")
        self.train_csv_file = tk.StringVar(value="")

        # Validation variables
        self.val_strategy = tk.StringVar(value="split")
        self.val_split_size = tk.StringVar(value="0.2")
        self.val_split_seed = tk.StringVar(value="42")
        self.val_split_stratify = tk.BooleanVar(value=True)

        self.val_data_mode = tk.StringVar(value="ImageFolder")
        self.val_img_folder = tk.StringVar(value="")
        self.val_csv_folder = tk.StringVar(value="")
        self.val_csv_file = tk.StringVar(value="")

        # Hyperparameters
        self.train_optimizer_name = tk.StringVar(value="Adam")
        self.train_lr = tk.StringVar(value="0.001")
        self.train_momentum = tk.StringVar(value="0.9")
        self.train_weight_decay = tk.StringVar(value="0.0001")
        self.train_lr_scheduler = tk.StringVar(value="None")

        self.train_loss_name = tk.StringVar(value="CrossEntropyLoss")
        self.train_use_weights = tk.BooleanVar(value=False)
        self.train_weight_mode = tk.StringVar(value=WEIGHT_MODES[0])
        self.train_manual_weights = tk.StringVar(value="")

        self.train_epochs = tk.StringVar(value="10")
        self.train_batch_size = tk.StringVar(value="32")
        self.train_device = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        self.train_workers = tk.StringVar(value="0")

        # Advanced Settings
        self.train_grad_clip = tk.StringVar(value="0.0")
        self.train_early_stop = tk.StringVar(value="0")

        # Evaluation Metric: Accuracy vs F1
        self.eval_metric_choice = tk.StringVar(value="Accuracy")
        self.save_metric_choice = tk.StringVar(value=VAL_METRIC_CHOICES[0])

        # Data Augmentation
        self.aug_hflip = tk.BooleanVar(value=True)
        self.aug_crop = tk.BooleanVar(value=False)
        self.aug_color_jitter = tk.BooleanVar(value=False)

        # Checkpointing & Export
        self.save_checkpoint_dir = tk.StringVar(value=os.path.abspath("./checkpoints"))
        self.export_dir_var = tk.StringVar(value=os.path.abspath("./exported_model"))

        # Progress bars variables for full training view
        self.prog_batch_var = tk.DoubleVar(value=0.0)
        self.prog_eval_var = tk.DoubleVar(value=0.0)
        self.prog_epoch_var = tk.DoubleVar(value=0.0)

        self.lbl_prog_batch_text = tk.StringVar(value="Waiting to start...")
        self.lbl_prog_eval_text = tk.StringVar(value="Evaluation idle.")
        self.lbl_prog_epoch_text = tk.StringVar(value="Epoch 0 / 10 (0.0%)")

        # Big KPI cards variables
        self.kpi_train_loss = tk.StringVar(value="--")
        self.kpi_val_loss = tk.StringVar(value="--")
        self.kpi_train_metric = tk.StringVar(value="--")
        self.kpi_val_metric = tk.StringVar(value="--")
        self.kpi_best_metric = tk.StringVar(value="--")
        self.kpi_status = tk.StringVar(value="Status: Ready to Train")
        self.kpi_best_banner = tk.StringVar(value="")

        # Best Model Tracking
        self.best_model_state_dict: Optional[Dict[str, Any]] = None
        self.best_classes: List[str] = []
        self.best_metric_info: Dict[str, Any] = {}
        self.last_export_path: Optional[str] = None
        self.history_records: List[Dict[str, Any]] = []
        self.best_epoch_record: Optional[int] = None

        # Worker control
        self._train_thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._is_training = False

    # -- Global Settings Panel with High Visibility --------------------
    def _build_global_settings(self):
        self.global_settings_fr = ttk.Labelframe(
            self,
            text="⚙️ Global Architecture Specifications & Input Geometry",
            padding=10,
        )
        self.global_settings_fr.pack(fill="x", padx=8, pady=6)

        self.var_in_channels = tk.StringVar(value=str(self.design.in_channels))
        self.var_in_h = tk.StringVar(value=str(self.design.input_height))
        self.var_in_w = tk.StringVar(value=str(self.design.input_width))
        self.var_num_classes = tk.StringVar(value=str(self.design.num_classes))
        self.var_flatten_mode = tk.StringVar(value=self.design.flatten_mode)

        # Top row: prominently styled inputs
        r1 = ttk.Frame(self.global_settings_fr)
        r1.pack(fill="x", pady=2)

        ttk.Label(r1, text="Input Channels (C):", font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=(4, 2))
        ent_c = ttk.Entry(r1, textvariable=self.var_in_channels, width=5, font=("TkDefaultFont", 10, "bold"))
        ent_c.pack(side="left", padx=(0, 10))
        ent_c.bind("<KeyRelease>", lambda e: self._on_global_dim_change())

        ttk.Label(r1, text="Image Height (H):", font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=(4, 2))
        ent_h = ttk.Entry(r1, textvariable=self.var_in_h, width=6, font=("TkDefaultFont", 10, "bold"))
        ent_h.pack(side="left", padx=(0, 10))
        ent_h.bind("<KeyRelease>", lambda e: self._on_global_dim_change())

        ttk.Label(r1, text="Image Width (W):", font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=(4, 2))
        ent_w = ttk.Entry(r1, textvariable=self.var_in_w, width=6, font=("TkDefaultFont", 10, "bold"))
        ent_w.pack(side="left", padx=(0, 14))
        ent_w.bind("<KeyRelease>", lambda e: self._on_global_dim_change())

        # Highly visible Classes box
        cls_badge_fr = tk.Frame(r1, bg="#E8EAF6", bd=1, relief="solid", padx=6, pady=2)
        cls_badge_fr.pack(side="left", padx=(0, 14))
        tk.Label(
            cls_badge_fr,
            text="🏷️ Number of Classes:",
            font=("TkDefaultFont", 10, "bold"),
            bg="#E8EAF6",
            fg="#1A237E",
        ).pack(side="left", padx=2)
        self.ent_classes = tk.Entry(
            cls_badge_fr,
            textvariable=self.var_num_classes,
            width=6,
            font=("TkDefaultFont", 11, "bold"),
            justify="center",
            bg="#FFFFFF",
            fg="#0D47A1",
        )
        self.ent_classes.pack(side="left", padx=4)
        self.ent_classes.bind("<KeyRelease>", lambda e: self._on_global_dim_change())

        ttk.Label(r1, text="Before Head:").pack(side="left", padx=(4, 2))
        cbo_flatten = ttk.Combobox(
            r1,
            textvariable=self.var_flatten_mode,
            values=FLATTEN_MODES,
            state="readonly",
            width=25,
        )
        cbo_flatten.pack(side="left", padx=(0, 6))
        cbo_flatten.bind("<<ComboboxSelected>>", lambda e: self._on_global_dim_change())

        # Row 2: Status badge and Python code button right under global settings
        r2 = ttk.Frame(self.global_settings_fr)
        r2.pack(fill="x", pady=(6, 0))

        self.lbl_global_badge = ttk.Label(
            r2,
            text=f"Tensor Input Shape: ({self.design.in_channels}, {self.design.input_height}, {self.design.input_width})  ➔  Classes: {self.design.num_classes}",
            font=("TkDefaultFont", 9, "bold"),
            foreground="#1565C0",
        )
        self.lbl_global_badge.pack(side="left", padx=4)

        # "See python code for current configuration" button
        btn_code = ttk.Button(
            r2,
            text="📜 See Python Code for Current Configuration",
            command=self.show_python_code_dialog,
        )
        btn_code.pack(side="right", padx=4)

    def _on_global_dim_change(self):
        err = self._pull_global_settings()
        if not err:
            self.lbl_global_badge.configure(
                text=f"Tensor Input Shape: ({self.design.in_channels}, {self.design.input_height}, {self.design.input_width})  ➔  Classes: {self.design.num_classes}"
            )
            self._refresh_backbone_tree()
            self._refresh_head_tree()

    # -- Notebook tabs -------------------------------------------------
    def _build_notebook(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=6)

        self.backbone_tab = ttk.Frame(self.notebook)
        self.head_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)

        # Tab 1 and 2 added by default. Tab 3 is hidden and only unlocked via 'Verify Architecture & Proceed'
        self.notebook.add(self.backbone_tab, text="1. Backbone (Convolutional Layers)")
        self.notebook.add(self.head_tab, text="2. Head (Classifier Layers)")

        self.backbone_tree = self._build_backbone_tab(self.backbone_tab)
        self.head_tree = self._build_head_tab(self.head_tab)
        self._build_training_settings_tab(self.training_tab)

        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

    def _on_notebook_tab_changed(self, event=None):
        try:
            selected = self.notebook.select()
            if selected == str(self.training_tab):
                self.btn_next.pack_forget()
            else:
                self.btn_next.pack(side="right", padx=4)
        except Exception:
            pass

    def _proceed_to_training_tab(self):
        tabs = [str(t) for t in self.notebook.tabs()]
        if str(self.training_tab) not in tabs:
            self.notebook.add(self.training_tab, text="3. Dataset & Training Settings")
        self.notebook.select(self.training_tab)
        self.btn_next.pack_forget()

    def _back_to_designer_tabs(self):
        self.notebook.select(self.backbone_tab)
        try:
            self.notebook.forget(self.training_tab)
        except Exception:
            pass
        self.btn_next.pack(side="right", padx=4)

    def _build_backbone_tab(self, parent):
        tree_fr = ttk.Frame(parent)
        tree_fr.pack(fill="both", expand=True, side="left", padx=(0, 6), pady=6)

        # Columns show input tensor and output tensor for each conv block!
        columns = ("idx", "in_shape", "out_shape", "summary")
        tree = ttk.Treeview(tree_fr, columns=columns, show="headings", height=16)
        tree.heading("idx", text="#")
        tree.heading("in_shape", text="Input Tensor Shape")
        tree.heading("out_shape", text="Output Tensor Shape")
        tree.heading("summary", text="Convolutional Block Configuration")

        tree.column("idx", width=40, anchor="center")
        tree.column("in_shape", width=140, anchor="center")
        tree.column("out_shape", width=140, anchor="center")
        tree.column("summary", width=520, anchor="w")

        tree.pack(fill="both", expand=True, side="left")
        scroll = ttk.Scrollbar(tree_fr, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        # Bottom info label for backbone reduction
        self.lbl_backbone_summary = ttk.Label(
            parent, text="Backbone: No layers added yet.", foreground="#555555"
        )
        self.lbl_backbone_summary.pack(side="bottom", anchor="w", padx=6, pady=2)

        btn_fr = ttk.Frame(parent)
        btn_fr.pack(side="left", fill="y", pady=6)

        ttk.Button(btn_fr, text="Add Conv Layer...", command=self.add_backbone_layer).pack(fill="x", pady=3)
        ttk.Button(btn_fr, text="Edit Layer...", command=self.edit_backbone_layer).pack(fill="x", pady=3)
        ttk.Button(btn_fr, text="Duplicate Layer", command=self.duplicate_backbone_layer).pack(fill="x", pady=3)
        ttk.Button(btn_fr, text="Delete Layer", command=self.delete_backbone_layer).pack(fill="x", pady=3)
        ttk.Separator(btn_fr).pack(fill="x", pady=6)
        ttk.Button(btn_fr, text="Move Up", command=lambda: self.move_layer(True, -1)).pack(fill="x", pady=3)
        ttk.Button(btn_fr, text="Move Down", command=lambda: self.move_layer(True, 1)).pack(fill="x", pady=3)

        return tree

    def _build_head_tab(self, parent):
        tree_fr = ttk.Frame(parent)
        tree_fr.pack(fill="both", expand=True, side="left", padx=(0, 6), pady=6)

        # Columns for head block and last output layer
        columns = ("idx", "in_feat", "out_feat", "summary")
        tree = ttk.Treeview(tree_fr, columns=columns, show="headings", height=16)
        tree.heading("idx", text="#")
        tree.heading("in_feat", text="Input Features")
        tree.heading("out_feat", text="Output Features")
        tree.heading("summary", text="Linear Layer Block Configuration")

        tree.column("idx", width=55, anchor="center")
        tree.column("in_feat", width=130, anchor="center")
        tree.column("out_feat", width=130, anchor="center")
        tree.column("summary", width=500, anchor="w")

        tree.pack(fill="both", expand=True, side="left")
        scroll = ttk.Scrollbar(tree_fr, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        # Tags for final output layer
        tree.tag_configure("final_layer", background="#E8F5E9", foreground="#1B5E20")

        self.lbl_head_summary = ttk.Label(
            parent, text="Head: 1 final classifier layer (auto-managed).", foreground="#555555"
        )
        self.lbl_head_summary.pack(side="bottom", anchor="w", padx=6, pady=2)

        btn_fr = ttk.Frame(parent)
        btn_fr.pack(side="left", fill="y", pady=6)

        ttk.Button(btn_fr, text="Add Head Layer...", command=self.add_head_layer).pack(fill="x", pady=3)
        ttk.Button(btn_fr, text="Edit Layer...", command=self.edit_head_layer).pack(fill="x", pady=3)
        ttk.Button(btn_fr, text="Duplicate Layer", command=self.duplicate_head_layer).pack(fill="x", pady=3)
        ttk.Button(btn_fr, text="Delete Layer", command=self.delete_head_layer).pack(fill="x", pady=3)
        ttk.Separator(btn_fr).pack(fill="x", pady=6)
        ttk.Button(btn_fr, text="Move Up", command=lambda: self.move_layer(False, -1)).pack(fill="x", pady=3)
        ttk.Button(btn_fr, text="Move Down", command=lambda: self.move_layer(False, 1)).pack(fill="x", pady=3)

        return tree

    # -- Bottom Bar with Next and Code buttons -------------------------
    def _build_bottom_bar(self):
        self.bottom_fr = ttk.Frame(self)
        self.bottom_fr.pack(fill="x", padx=8, pady=(0, 8))

        ttk.Button(self.bottom_fr, text="💾 Save Design (JSON)", command=self.save_design).pack(side="left", padx=4)
        ttk.Button(self.bottom_fr, text="📂 Load Design (JSON)", command=self.load_design).pack(side="left", padx=4)
        ttk.Button(self.bottom_fr, text="📜 See PyTorch Code", command=self.show_python_code_dialog).pack(side="left", padx=4)

        # Next button -> brings recapitulation popup then goes to training tab
        self.btn_next = ttk.Button(
            self.bottom_fr, text="Next: Verify Architecture & Proceed  ➜", command=self.on_next_clicked
        )
        self.btn_next.pack(side="right", padx=4)

    def show_python_code_dialog(self):
        err = self._pull_global_settings()
        if err:
            messagebox.showerror("Invalid Settings", err, parent=self)
            return
        if len(self.design.backbone) == 0:
            messagebox.showinfo("No Backbone", "Add at least one convolutional layer to view model code.", parent=self)
            return
        CodeViewerDialog(self, self.design)

    def on_next_clicked(self):
        """Verify architecture and show recapitulation modal dialog."""
        err = self._pull_global_settings()
        if err:
            messagebox.showerror("Invalid Settings", err, parent=self)
            return

        if len(self.design.backbone) == 0:
            messagebox.showerror(
                "Architecture Incomplete",
                "You must add at least ONE convolutional layer block in the Backbone before proceeding.",
                parent=self,
            )
            self.notebook.select(self.backbone_tab)
            return

        if len(self.design.head) == 0:
            messagebox.showerror(
                "Architecture Incomplete",
                "You must add at least ONE layer block in the Classifier Head before proceeding.",
                parent=self,
            )
            self.notebook.select(self.head_tab)
            return

        try:
            c, h, w = trace_spatial_size(self.design)
            if h <= 0 or w <= 0:
                messagebox.showerror(
                    "Spatial Dimension Error",
                    f"Backbone spatial dimensions collapsed to {h}×{w}.\n"
                    "Please adjust your kernel sizes, strides, or pooling layers so dimensions remain positive.",
                    parent=self,
                )
                return
        except Exception as exc:
            messagebox.showerror("Verification Error", f"Error tracing spatial dimensions:\n{exc}", parent=self)
            return

        # Pop up the network recapitulation dialog with tensor flow per layer
        NetworkRecapitulationDialog(
            self,
            self.design,
            on_confirm=self._proceed_to_training_tab,
        )

    # -- Tab 3: Dedicated Dataset & Training Configuration Settings Only -
    def _build_training_settings_tab(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        vscroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_content = ttk.Frame(canvas, padding=8)

        scroll_content.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas_win = canvas.create_window((0, 0), window=scroll_content, anchor="nw")

        def _on_canvas_resize(event):
            canvas.itemconfig(canvas_win, width=event.width)

        canvas.bind("<Configure>", _on_canvas_resize)
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        # Top navigation banner to return to backbone/head designer
        nav_top_fr = ttk.Frame(scroll_content)
        nav_top_fr.pack(fill="x", padx=6, pady=(0, 8))
        ttk.Button(
            nav_top_fr,
            text="⮌ Back to Architecture Designer (Backbone & Head)",
            command=self._back_to_designer_tabs,
        ).pack(side="left")
        ttk.Label(
            nav_top_fr,
            text="✓ Architecture Verified & Locked for Training",
            font=("TkDefaultFont", 9, "bold"),
            foreground="#2E7D32",
        ).pack(side="right", padx=4)

        # Section 1: Training Dataset
        ds_fr = ttk.Labelframe(scroll_content, text="1. Training Dataset Selection", padding=10)
        ds_fr.pack(fill="x", padx=6, pady=4)

        ttk.Label(ds_fr, text="Dataset Loader Format:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        r_fr = ttk.Frame(ds_fr)
        r_fr.grid(row=0, column=1, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Radiobutton(
            r_fr, text="ImageFolder (Subfolder per class)", variable=self.train_data_mode,
            value="ImageFolder", command=self._sync_dataset_view
        ).pack(side="left", padx=4)
        ttk.Radiobutton(
            r_fr, text="Folder of images + CSV (id, img_name, label)", variable=self.train_data_mode,
            value="Folder + CSV", command=self._sync_dataset_view
        ).pack(side="left", padx=4)

        self.lbl_train_format_warning = tk.Label(
            ds_fr, text="", justify="left", relief="solid", bd=1,
            bg="#FFF9C4", fg="#3E2723", font=("Courier", 8), padx=6, pady=4
        )
        self.lbl_train_format_warning.grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=4)

        self.lbl_imgfolder = ttk.Label(ds_fr, text="Dataset Root Folder:")
        self.ent_imgfolder = ttk.Entry(ds_fr, textvariable=self.train_img_folder)
        self.btn_imgfolder = ttk.Button(ds_fr, text="Browse...", command=lambda: self._browse_dir(self.train_img_folder))

        self.lbl_csv_folder = ttk.Label(ds_fr, text="Images Directory:")
        self.ent_csv_folder = ttk.Entry(ds_fr, textvariable=self.train_csv_folder)
        self.btn_csv_folder = ttk.Button(ds_fr, text="Browse...", command=lambda: self._browse_dir(self.train_csv_folder))

        self.lbl_csv_file = ttk.Label(ds_fr, text="CSV File (id,img_name,label):")
        self.ent_csv_file = ttk.Entry(ds_fr, textvariable=self.train_csv_file)
        self.btn_csv_file = ttk.Button(ds_fr, text="Browse...", command=lambda: self._browse_csv(self.train_csv_file))

        v_fr = ttk.Frame(ds_fr)
        v_fr.grid(row=5, column=0, columnspan=3, sticky="ew", padx=4, pady=6)
        ttk.Button(v_fr, text="🔍 Inspect / Verify Dataset", command=self.verify_dataset_action).pack(side="left", padx=2)
        self.lbl_dataset_status = ttk.Label(v_fr, text="No dataset scanned yet.", foreground="#555555")
        self.lbl_dataset_status.pack(side="left", padx=8)

        # Section 2: Validation Strategy
        val_fr = ttk.Labelframe(scroll_content, text="2. Validation Data Strategy", padding=10)
        val_fr.pack(fill="x", padx=6, pady=4)

        ttk.Radiobutton(
            val_fr, text="Split training dataset (train_test_split)",
            variable=self.val_strategy, value="split", command=self._sync_val_view
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=2)

        self.val_split_subfr = ttk.Frame(val_fr)
        self.val_split_subfr.grid(row=1, column=0, columnspan=3, sticky="w", padx=20, pady=2)
        ttk.Label(self.val_split_subfr, text="Val split ratio:").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(self.val_split_subfr, textvariable=self.val_split_size, width=6).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(self.val_split_subfr, text="(e.g. 0.2 for 20%)").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(self.val_split_subfr, text="Random seed:").grid(row=0, column=3, sticky="w", padx=6)
        ttk.Entry(self.val_split_subfr, textvariable=self.val_split_seed, width=6).grid(row=0, column=4, sticky="w", padx=4)
        ttk.Checkbutton(self.val_split_subfr, text="Stratify by class", variable=self.val_split_stratify).grid(row=0, column=5, sticky="w", padx=6)

        ttk.Radiobutton(
            val_fr, text="Use separate validation dataset",
            variable=self.val_strategy, value="separate", command=self._sync_val_view
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 2))

        self.val_sep_subfr = ttk.Frame(val_fr)
        self.val_sep_subfr.grid(row=3, column=0, columnspan=3, sticky="ew", padx=20, pady=2)

        ttk.Radiobutton(
            self.val_sep_subfr, text="ImageFolder", variable=self.val_data_mode,
            value="ImageFolder", command=self._sync_val_view
        ).grid(row=0, column=0, sticky="w", padx=4)
        ttk.Radiobutton(
            self.val_sep_subfr, text="Folder + CSV", variable=self.val_data_mode,
            value="Folder + CSV", command=self._sync_val_view
        ).grid(row=0, column=1, sticky="w", padx=4)

        self.lbl_val_format_warning = tk.Label(
            self.val_sep_subfr, text="", justify="left", relief="solid", bd=1,
            bg="#FFF9C4", fg="#3E2723", font=("Courier", 8), padx=6, pady=4
        )
        self.lbl_val_format_warning.grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=4)

        self.lbl_val_imgfolder = ttk.Label(self.val_sep_subfr, text="Val Root Folder:")
        self.ent_val_imgfolder = ttk.Entry(self.val_sep_subfr, textvariable=self.val_img_folder)
        self.btn_val_imgfolder = ttk.Button(self.val_sep_subfr, text="Browse...", command=lambda: self._browse_dir(self.val_img_folder))

        self.lbl_val_csv_folder = ttk.Label(self.val_sep_subfr, text="Val Images Folder:")
        self.ent_val_csv_folder = ttk.Entry(self.val_sep_subfr, textvariable=self.val_csv_folder)
        self.btn_val_csv_folder = ttk.Button(self.val_sep_subfr, text="Browse...", command=lambda: self._browse_dir(self.val_csv_folder))

        self.lbl_val_csv_file = ttk.Label(self.val_sep_subfr, text="Val CSV (id,img_name,label):")
        self.ent_val_csv_file = ttk.Entry(self.val_sep_subfr, textvariable=self.val_csv_file)
        self.btn_val_csv_file = ttk.Button(self.val_sep_subfr, text="Browse...", command=lambda: self._browse_csv(self.val_csv_file))

        # Section 3: Hyperparameters, Scheduler & Optimization
        opt_fr = ttk.Labelframe(scroll_content, text="3. Training Hyperparameters & Optimization", padding=10)
        opt_fr.pack(fill="x", padx=6, pady=4)

        ttk.Label(opt_fr, text="Optimizer:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.cbo_opt = ttk.Combobox(opt_fr, textvariable=self.train_optimizer_name, values=OPTIMIZER_TYPES, state="readonly", width=12)
        self.cbo_opt.grid(row=0, column=1, sticky="w", padx=4, pady=3)
        self.cbo_opt.bind("<<ComboboxSelected>>", lambda e: self._sync_optimizer_fields())

        ttk.Label(opt_fr, text="Learning rate:").grid(row=0, column=2, sticky="w", padx=6, pady=3)
        ttk.Entry(opt_fr, textvariable=self.train_lr, width=10).grid(row=0, column=3, sticky="w", padx=4, pady=3)

        ttk.Label(opt_fr, text="Momentum:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.ent_momentum = ttk.Entry(opt_fr, textvariable=self.train_momentum, width=12)
        self.ent_momentum.grid(row=1, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(opt_fr, text="Weight decay:").grid(row=1, column=2, sticky="w", padx=6, pady=3)
        ttk.Entry(opt_fr, textvariable=self.train_weight_decay, width=10).grid(row=1, column=3, sticky="w", padx=4, pady=3)

        ttk.Label(opt_fr, text="LR Scheduler:").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        schedulers = ["None", "StepLR (step=epoch//3, gamma=0.5)", "CosineAnnealingLR", "ReduceLROnPlateau"]
        ttk.Combobox(opt_fr, textvariable=self.train_lr_scheduler, values=schedulers, state="readonly", width=26).grid(
            row=2, column=1, columnspan=2, sticky="w", padx=4, pady=3
        )

        ttk.Label(opt_fr, text="Epochs:").grid(row=3, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(opt_fr, textvariable=self.train_epochs, width=12).grid(row=3, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(opt_fr, text="Batch size:").grid(row=3, column=2, sticky="w", padx=6, pady=3)
        ttk.Entry(opt_fr, textvariable=self.train_batch_size, width=10).grid(row=3, column=3, sticky="w", padx=4, pady=3)

        ttk.Label(opt_fr, text="Device:").grid(row=4, column=0, sticky="w", padx=4, pady=3)
        dev_choices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        ttk.Combobox(opt_fr, textvariable=self.train_device, values=dev_choices, state="readonly", width=12).grid(
            row=4, column=1, sticky="w", padx=4, pady=3
        )

        ttk.Label(opt_fr, text="Num workers:").grid(row=4, column=2, sticky="w", padx=6, pady=3)
        ttk.Entry(opt_fr, textvariable=self.train_workers, width=10).grid(row=4, column=3, sticky="w", padx=4, pady=3)

        ttk.Label(opt_fr, text="Grad clip norm:").grid(row=5, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(opt_fr, textvariable=self.train_grad_clip, width=12).grid(row=5, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(opt_fr, text="Early stopping:").grid(row=5, column=2, sticky="w", padx=6, pady=3)
        ttk.Entry(opt_fr, textvariable=self.train_early_stop, width=10).grid(row=5, column=3, sticky="w", padx=4, pady=3)
        ttk.Label(opt_fr, text="(0 = disabled, or e.g. 5 epochs)").grid(row=5, column=4, sticky="w", padx=4, pady=3)

        # Section 4: Evaluation Metric & Loss Function
        loss_fr = ttk.Labelframe(scroll_content, text="4. Evaluation Metric & Loss Function", padding=10)
        loss_fr.pack(fill="x", padx=6, pady=4)

        # Ask whether to use F1 or accuracy for evaluation metric
        metric_pick_fr = tk.Frame(loss_fr, bg="#E8F5E9", bd=1, relief="solid", padx=8, pady=6)
        metric_pick_fr.grid(row=0, column=0, columnspan=4, sticky="ew", padx=4, pady=4)

        tk.Label(
            metric_pick_fr,
            text="🎯 Primary Evaluation Metric:",
            font=("TkDefaultFont", 10, "bold"),
            bg="#E8F5E9",
            fg="#1B5E20",
        ).pack(side="left", padx=(4, 12))

        ttk.Radiobutton(
            metric_pick_fr,
            text="Accuracy (%) [Recommended for balanced datasets]",
            variable=self.eval_metric_choice,
            value="Accuracy",
            command=self._sync_metric_choice,
        ).pack(side="left", padx=8)

        ttk.Radiobutton(
            metric_pick_fr,
            text="Macro F1-Score (%) [Recommended for imbalanced classes]",
            variable=self.eval_metric_choice,
            value="F1-Score",
            command=self._sync_metric_choice,
        ).pack(side="left", padx=8)

        ttk.Label(loss_fr, text="Save model by metric:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.cbo_save_metric = ttk.Combobox(loss_fr, textvariable=self.save_metric_choice, values=VAL_METRIC_CHOICES, state="readonly", width=28)
        self.cbo_save_metric.grid(row=1, column=1, sticky="w", padx=4, pady=3)

        ttk.Label(loss_fr, text="Loss function:").grid(row=2, column=0, sticky="w", padx=4, pady=3)
        ttk.Combobox(loss_fr, textvariable=self.train_loss_name, values=LOSS_TYPES, state="readonly", width=20).grid(
            row=2, column=1, sticky="w", padx=4, pady=3
        )

        ttk.Checkbutton(
            loss_fr, text="Use class weights (imbalance handling)",
            variable=self.train_use_weights, command=self._sync_weights_fields
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=3)

        self.w_subfr = ttk.Frame(loss_fr)
        self.w_subfr.grid(row=4, column=0, columnspan=4, sticky="ew", padx=16, pady=2)

        ttk.Label(self.w_subfr, text="Weight calculation:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.cbo_weight_mode = ttk.Combobox(
            self.w_subfr, textvariable=self.train_weight_mode, values=WEIGHT_MODES, state="readonly", width=28
        )
        self.cbo_weight_mode.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        self.cbo_weight_mode.bind("<<ComboboxSelected>>", lambda e: self._sync_weights_fields())

        self.lbl_manual_w = ttk.Label(self.w_subfr, text="Manual weights (comma-separated):")
        self.lbl_manual_w.grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.ent_manual_w = ttk.Entry(self.w_subfr, textvariable=self.train_manual_weights, width=28)
        self.ent_manual_w.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        # Section 5: Data Augmentation
        aug_fr = ttk.Labelframe(scroll_content, text="5. Training Data Augmentation", padding=10)
        aug_fr.pack(fill="x", padx=6, pady=4)

        ttk.Checkbutton(aug_fr, text="Random Horizontal Flip (50% probability)", variable=self.aug_hflip).pack(anchor="w", padx=4, pady=2)
        ttk.Checkbutton(aug_fr, text="Random Crop with padding (shifts & small crops)", variable=self.aug_crop).pack(anchor="w", padx=4, pady=2)
        ttk.Checkbutton(aug_fr, text="Color Jitter (brightness/contrast/saturation perturbation)", variable=self.aug_color_jitter).pack(anchor="w", padx=4, pady=2)

        # Section 6: Checkpoint & Saving
        ckpt_fr = ttk.Labelframe(scroll_content, text="6. Model Checkpoint & Saving", padding=10)
        ckpt_fr.pack(fill="x", padx=6, pady=4)

        ttk.Label(ckpt_fr, text="Save directory:").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(ckpt_fr, textvariable=self.save_checkpoint_dir, width=36).grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(ckpt_fr, text="Browse...", command=lambda: self._browse_dir(self.save_checkpoint_dir)).grid(row=0, column=2, padx=4, pady=3)

        ttk.Label(ckpt_fr, text="Best model filename:").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        ttk.Label(ckpt_fr, text="best_model.pth (auto-saved)", font=("TkDefaultFont", 9, "bold"), foreground="#2E7D32").grid(row=1, column=1, sticky="w", padx=4, pady=3)

        # Big Start Training Button
        btn_start_fr = tk.Frame(scroll_content, bg="#F5F5F5", pady=12)
        btn_start_fr.pack(fill="x", padx=6, pady=8)

        self.btn_big_train = tk.Button(
            btn_start_fr,
            text="START TRAINING",
            font=("TkDefaultFont", 13, "bold"),
            bg="#2E7D32",
            fg="white",
            activebackground="#1B5E20",
            activeforeground="white",
            relief="raised",
            bd=3,
            padx=16,
            pady=10,
            cursor="hand2",
            command=self.start_training,
        )
        self.btn_big_train.pack(fill="x", padx=12)

        # Initial view sync
        self._sync_dataset_view()
        self._sync_val_view()
        self._sync_optimizer_fields()
        self._sync_weights_fields()

    def _sync_metric_choice(self):
        m = self.eval_metric_choice.get()
        if m == "F1-Score":
            self.save_metric_choice.set("Best Validation F1")
        else:
            self.save_metric_choice.set("Best Validation Accuracy")

    # -- Full-Screen Training Page -------------------------------------
    def _build_fullscreen_training_view(self):
        self.training_fullscreen_frame = ttk.Frame(self, padding=8)

        # Top Header Banner
        head_fr = tk.Frame(self.training_fullscreen_frame, bg="#1A237E", padx=12, pady=10)
        head_fr.pack(fill="x", pady=(0, 8))

        lbl_title = tk.Label(
            head_fr,
            text="VisualCNN — Training",
            font=("TkDefaultFont", 13, "bold"),
            fg="white",
            bg="#1A237E",
        )
        lbl_title.pack(side="left")

        self.lbl_train_status_badge = tk.Label(
            head_fr,
            textvariable=self.kpi_status,
            font=("TkDefaultFont", 11, "bold"),
            fg="#FFD54F",
            bg="#1A237E",
        )
        self.lbl_train_status_badge.pack(side="right")

        # Main Paned View: Left (Progress & Live Dashboard) vs Right (History Table)
        paned = ttk.PanedWindow(self.training_fullscreen_frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left_container = ttk.Frame(paned, padding=6)
        right_container = ttk.Frame(paned, padding=6)
        paned.add(left_container, weight=5)
        paned.add(right_container, weight=4)

        # === LEFT: 3 Progress Bars ===
        prog_box = ttk.Labelframe(left_container, text="Training Progress Bars", padding=10)
        prog_box.pack(fill="x", pady=(0, 6))

        # Bar 1: Training in current epoch
        ttk.Label(prog_box, text="1. Training in Current Epoch:", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        ttk.Label(prog_box, textvariable=self.lbl_prog_batch_text, foreground="#1565C0").pack(anchor="w", pady=(1, 2))
        self.bar_batch = ttk.Progressbar(prog_box, orient="horizontal", mode="determinate", variable=self.prog_batch_var, maximum=100.0)
        self.bar_batch.pack(fill="x", pady=(0, 6))

        # Bar 2: Evaluation within current epoch
        ttk.Label(prog_box, text="2. Evaluation within Current Epoch:", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        ttk.Label(prog_box, textvariable=self.lbl_prog_eval_text, foreground="#7B1FA2").pack(anchor="w", pady=(1, 2))
        self.bar_eval = ttk.Progressbar(prog_box, orient="horizontal", mode="determinate", variable=self.prog_eval_var, maximum=100.0)
        self.bar_eval.pack(fill="x", pady=(0, 6))

        # Bar 3: Overall epoch progress
        ttk.Label(prog_box, text="3. Overall Training Progress (Epochs):", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        ttk.Label(prog_box, textvariable=self.lbl_prog_epoch_text, foreground="#2E7D32").pack(anchor="w", pady=(1, 2))
        self.bar_total = ttk.Progressbar(prog_box, orient="horizontal", mode="determinate", variable=self.prog_epoch_var, maximum=100.0)
        self.bar_total.pack(fill="x", pady=(0, 4))

        # === LEFT: High-Visibility Live Output KPI Cards ===
        kpi_fr = ttk.Labelframe(left_container, text="Live Performance Numbers", padding=8)
        kpi_fr.pack(fill="x", pady=4)

        kpi_grid = tk.Frame(kpi_fr, bg="#F5F5F5", padx=6, pady=6)
        kpi_grid.pack(fill="x")

        # Card 1: Loss
        c1 = tk.Frame(kpi_grid, bg="#FFFFFF", bd=1, relief="solid", padx=10, pady=8)
        c1.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(c1, text="TRAIN LOSS", font=("TkDefaultFont", 8, "bold"), fg="#757575", bg="#FFFFFF").pack()
        tk.Label(c1, textvariable=self.kpi_train_loss, font=("TkDefaultFont", 16, "bold"), fg="#C62828", bg="#FFFFFF").pack()
        tk.Label(c1, text="Val Loss:", font=("TkDefaultFont", 8), fg="#616161", bg="#FFFFFF").pack()
        tk.Label(c1, textvariable=self.kpi_val_loss, font=("TkDefaultFont", 11, "bold"), fg="#D32F2F", bg="#FFFFFF").pack()

        # Card 2: Metric Score
        c2 = tk.Frame(kpi_grid, bg="#FFFFFF", bd=1, relief="solid", padx=10, pady=8)
        c2.pack(side="left", fill="both", expand=True, padx=4)
        self.lbl_card_metric_title = tk.Label(c2, text="TRAIN METRIC", font=("TkDefaultFont", 8, "bold"), fg="#757575", bg="#FFFFFF")
        self.lbl_card_metric_title.pack()
        tk.Label(c2, textvariable=self.kpi_train_metric, font=("TkDefaultFont", 16, "bold"), fg="#1565C0", bg="#FFFFFF").pack()
        self.lbl_card_val_title = tk.Label(c2, text="Val Metric:", font=("TkDefaultFont", 8), fg="#616161", bg="#FFFFFF")
        self.lbl_card_val_title.pack()
        tk.Label(c2, textvariable=self.kpi_val_metric, font=("TkDefaultFont", 11, "bold"), fg="#0D47A1", bg="#FFFFFF").pack()

        # Card 3: Best Model Recorded
        c3 = tk.Frame(kpi_grid, bg="#E8F5E9", bd=1, relief="solid", padx=10, pady=8)
        c3.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(c3, text="★ BEST MODEL RECORD", font=("TkDefaultFont", 8, "bold"), fg="#2E7D32", bg="#E8F5E9").pack()
        tk.Label(c3, textvariable=self.kpi_best_metric, font=("TkDefaultFont", 16, "bold"), fg="#1B5E20", bg="#E8F5E9").pack()
        tk.Label(c3, text="Saved to 'best_model.pth'", font=("TkDefaultFont", 8), fg="#388E3C", bg="#E8F5E9").pack()

        # Best Model Result Banner
        self.banner_best_fr = tk.Frame(left_container, bg="#E8F5E9", bd=1, relief="solid", padx=8, pady=6)
        self.lbl_banner_best = tk.Label(
            self.banner_best_fr,
            textvariable=self.kpi_best_banner,
            font=("TkDefaultFont", 10, "bold"),
            fg="#1B5E20",
            bg="#E8F5E9",
            wraplength=550,
            justify="left",
        )
        self.lbl_banner_best.pack(fill="x")

        # Action Buttons in Training View
        ctl_fr = ttk.Frame(left_container)
        ctl_fr.pack(fill="x", pady=6)

        self.btn_train_stop = ttk.Button(ctl_fr, text="⏹ Stop Training", command=self.stop_training)
        self.btn_train_stop.pack(side="left", padx=3)

        self.btn_view_graphs = ttk.Button(
            ctl_fr, text="📈 View Graphs", command=self.show_training_graphs, state="disabled"
        )
        self.btn_view_graphs.pack(side="left", padx=3)

        self.btn_train_export = ttk.Button(
            ctl_fr, text="📦 Export Best Model", command=self.export_best_model_action, state="disabled"
        )
        self.btn_train_export.pack(side="left", padx=3)

        self.btn_train_deploy = ttk.Button(
            ctl_fr, text="🚀 Deploy & Test", command=self.start_deploying_action, state="disabled"
        )
        self.btn_train_deploy.pack(side="left", padx=3)

        self.btn_return_designer = ttk.Button(
            ctl_fr, text="⮌ Return to Designer", command=self._return_to_designer_view, state="disabled"
        )
        self.btn_return_designer.pack(side="right", padx=3)

        # Scrolled Text Activity Log Console
        log_box = ttk.Labelframe(left_container, text="Training Activity Log", padding=6)
        log_box.pack(fill="both", expand=True, pady=4)

        self.txt_log = scrolledtext.ScrolledText(log_box, wrap="word", font=("Courier", 9), height=7)
        self.txt_log.pack(fill="both", expand=True)

        # === RIGHT: History Table ===
        hist_box = ttk.Labelframe(right_container, text="Epoch History & Best Model Marker", padding=8)
        hist_box.pack(fill="both", expand=True)

        cols = ("epoch", "train_loss", "train_metric", "val_loss", "val_metric", "best")
        self.history_tree = ttk.Treeview(hist_box, columns=cols, show="headings", height=20)
        self.history_tree.heading("epoch", text="Epoch")
        self.history_tree.heading("train_loss", text="Train Loss")
        self.history_tree.heading("train_metric", text="Train Score")
        self.history_tree.heading("val_loss", text="Val Loss")
        self.history_tree.heading("val_metric", text="Val Score")
        self.history_tree.heading("best", text="Best Model?")

        self.history_tree.column("epoch", width=55, anchor="center")
        self.history_tree.column("train_loss", width=85, anchor="center")
        self.history_tree.column("train_metric", width=95, anchor="center")
        self.history_tree.column("val_loss", width=85, anchor="center")
        self.history_tree.column("val_metric", width=95, anchor="center")
        self.history_tree.column("best", width=80, anchor="center")

        self.history_tree.tag_configure("best_row", background="#E8F5E9", foreground="#1B5E20", font=("TkDefaultFont", 9, "bold"))
        self.history_tree.tag_configure("regular_row", background="#FFFFFF")

        h_scroll = ttk.Scrollbar(hist_box, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=h_scroll.set)
        self.history_tree.pack(side="left", fill="both", expand=True)
        h_scroll.pack(side="right", fill="y")

    def _switch_to_training_view(self):
        """Hides previous designer tabs and displays dedicated full training page."""
        self.global_settings_fr.pack_forget()
        self.notebook.pack_forget()
        self.bottom_fr.pack_forget()
        self.training_fullscreen_frame.pack(fill="both", expand=True, padx=8, pady=8)

    def _return_to_designer_view(self):
        """Restores the designer notebook tabs and global settings."""
        if self._is_training:
            messagebox.showwarning("Training Running", "Please stop training before returning to designer.", parent=self)
            return
        self.training_fullscreen_frame.pack_forget()
        self.global_settings_fr.pack(fill="x", padx=8, pady=6)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=6)
        self.bottom_fr.pack(fill="x", padx=8, pady=(0, 8))

    # -- Dynamic UI synchronization handlers ---------------------------
    def _sync_dataset_view(self):
        mode = self.train_data_mode.get()
        if mode == "ImageFolder":
            self.lbl_train_format_warning.config(text=IMAGEFOLDER_WARNING)
            self.lbl_csv_folder.grid_remove()
            self.ent_csv_folder.grid_remove()
            self.btn_csv_folder.grid_remove()
            self.lbl_csv_file.grid_remove()
            self.ent_csv_file.grid_remove()
            self.btn_csv_file.grid_remove()

            self.lbl_imgfolder.grid(row=2, column=0, sticky="w", padx=4, pady=3)
            self.ent_imgfolder.grid(row=2, column=1, sticky="ew", padx=4, pady=3)
            self.btn_imgfolder.grid(row=2, column=2, padx=4, pady=3)
        else:
            self.lbl_train_format_warning.config(text=CSV_FOLDER_WARNING)
            self.lbl_imgfolder.grid_remove()
            self.ent_imgfolder.grid_remove()
            self.btn_imgfolder.grid_remove()

            self.lbl_csv_folder.grid(row=2, column=0, sticky="w", padx=4, pady=3)
            self.ent_csv_folder.grid(row=2, column=1, sticky="ew", padx=4, pady=3)
            self.btn_csv_folder.grid(row=2, column=2, padx=4, pady=3)

            self.lbl_csv_file.grid(row=3, column=0, sticky="w", padx=4, pady=3)
            self.ent_csv_file.grid(row=3, column=1, sticky="ew", padx=4, pady=3)
            self.btn_csv_file.grid(row=3, column=2, padx=4, pady=3)

    def _sync_val_view(self):
        strat = self.val_strategy.get()
        if strat == "split":
            self.val_split_subfr.grid()
            self.val_sep_subfr.grid_remove()
        else:
            self.val_split_subfr.grid_remove()
            self.val_sep_subfr.grid()

            vmode = self.val_data_mode.get()
            if vmode == "ImageFolder":
                self.lbl_val_format_warning.config(text=IMAGEFOLDER_WARNING)
                self.lbl_val_csv_folder.grid_remove()
                self.ent_val_csv_folder.grid_remove()
                self.btn_val_csv_folder.grid_remove()
                self.lbl_val_csv_file.grid_remove()
                self.ent_val_csv_file.grid_remove()
                self.btn_val_csv_file.grid_remove()

                self.lbl_val_imgfolder.grid(row=2, column=0, sticky="w", padx=4, pady=2)
                self.ent_val_imgfolder.grid(row=2, column=1, sticky="ew", padx=4, pady=2)
                self.btn_val_imgfolder.grid(row=2, column=2, padx=4, pady=2)
            else:
                self.lbl_val_format_warning.config(text=CSV_FOLDER_WARNING)
                self.lbl_val_imgfolder.grid_remove()
                self.ent_val_imgfolder.grid_remove()
                self.btn_val_imgfolder.grid_remove()

                self.lbl_val_csv_folder.grid(row=2, column=0, sticky="w", padx=4, pady=2)
                self.ent_val_csv_folder.grid(row=2, column=1, sticky="ew", padx=4, pady=2)
                self.btn_val_csv_folder.grid(row=2, column=2, padx=4, pady=2)

                self.lbl_val_csv_file.grid(row=3, column=0, sticky="w", padx=4, pady=2)
                self.ent_val_csv_file.grid(row=3, column=1, sticky="ew", padx=4, pady=2)
                self.btn_val_csv_file.grid(row=3, column=2, padx=4, pady=2)

    def _sync_optimizer_fields(self):
        opt = self.train_optimizer_name.get()
        has_momentum = opt in ("SGD", "RMSprop")
        self.ent_momentum.configure(state="normal" if has_momentum else "disabled")

    def _sync_weights_fields(self):
        use_w = self.train_use_weights.get()
        if not use_w:
            self.cbo_weight_mode.configure(state="disabled")
            self.ent_manual_w.configure(state="disabled")
        else:
            self.cbo_weight_mode.configure(state="readonly")
            is_manual = "Manual" in self.train_weight_mode.get()
            self.ent_manual_w.configure(state="normal" if is_manual else "disabled")

    def _browse_dir(self, target_var: tk.StringVar):
        d = filedialog.askdirectory()
        if d:
            target_var.set(d)

    def _browse_csv(self, target_var: tk.StringVar):
        f = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if f:
            target_var.set(f)

    def _log(self, text: str):
        t_str = time.strftime("[%H:%M:%S] ")
        self.txt_log.insert(tk.END, t_str + text + "\n")
        self.txt_log.see(tk.END)

    def _clear_log(self):
        self.txt_log.delete("1.0", tk.END)

    # -- Dataset verification action -----------------------------------
    def verify_dataset_action(self):
        mode = self.train_data_mode.get()
        try:
            if mode == "ImageFolder":
                p = self.train_img_folder.get().strip()
                if not p:
                    messagebox.showwarning("Dataset Path Missing", "Please select or enter the ImageFolder directory path.", parent=self)
                    return
                ds = create_image_folder_dataset(p, in_channels=int(self.var_in_channels.get()))
                total = len(ds)
                classes = ds.classes
            else:
                img_p = self.train_csv_folder.get().strip()
                csv_p = self.train_csv_file.get().strip()
                if not img_p or not csv_p:
                    messagebox.showwarning("Paths Missing", "Please select both the images folder and the CSV file.", parent=self)
                    return
                ds = CSVImageDataset(img_p, csv_p, in_channels=int(self.var_in_channels.get()))
                total = len(ds)
                classes = ds.classes

            msg = f"✓ Found {total} images across {len(classes)} classes: {', '.join(classes[:8])}"
            if len(classes) > 8:
                msg += f"... (+{len(classes)-8} more)"
            self.lbl_dataset_status.config(text=msg, foreground="#2E7D32")
            self._log(f"Dataset Verified: {total} total images, {len(classes)} classes.")

            # Automatically synchronize global num_classes if different
            curr_classes = int(self.var_num_classes.get())
            if curr_classes != len(classes):
                self.var_num_classes.set(str(len(classes)))
                self.design.num_classes = len(classes)
                self.lbl_global_badge.configure(
                    text=f"Tensor Input Shape: ({self.design.in_channels}, {self.design.input_height}, {self.design.input_width})  ➔  Classes: {self.design.num_classes}"
                )
                self._refresh_head_tree()
                self._log(f"Global 'Num classes' automatically updated from {curr_classes} to {len(classes)}.")

            messagebox.showinfo("Dataset Verification", f"Dataset verified successfully!\n\n{msg}", parent=self)
        except Exception as exc:
            self.lbl_dataset_status.config(text="Verification failed.", foreground="#C62828")
            self._log(f"Verification Error: {exc}")
            messagebox.showerror("Verification Failed", str(exc), parent=self)

    # -- Training execution pipeline -----------------------------------
    def start_training(self):
        if self._is_training:
            return

        err = self._pull_global_settings()
        if err:
            messagebox.showerror("Invalid Settings", err, parent=self)
            return

        if len(self.design.backbone) == 0:
            messagebox.showerror("No Backbone", "Add at least one convolutional layer to the backbone.", parent=self)
            return
        if len(self.design.head) == 0:
            messagebox.showerror("No Head", "Add at least one layer to the classifier head.", parent=self)
            return

        try:
            epochs = int(self.train_epochs.get())
            batch_size = int(self.train_batch_size.get())
            lr = float(self.train_lr.get())
            momentum = float(self.train_momentum.get())
            weight_decay = float(self.train_weight_decay.get())
            num_workers = int(self.train_workers.get())
            grad_clip = float(self.train_grad_clip.get())
            early_stop = int(self.train_early_stop.get())
            if epochs <= 0 or batch_size <= 0 or lr <= 0:
                raise ValueError("Epochs, batch size, and learning rate must be positive numbers.")
        except ValueError as exc:
            messagebox.showerror("Invalid Hyperparameters", str(exc), parent=self)
            return

        save_dir = self.save_checkpoint_dir.get().strip()
        if not save_dir:
            messagebox.showerror("Save Directory Missing", "Please select a directory to save model checkpoints.", parent=self)
            return

        # Switch to full training view and reset monitoring states
        self._switch_to_training_view()

        self._is_training = True
        self._stop_requested = False
        self.btn_train_stop.configure(state="normal")
        self.btn_view_graphs.configure(state="disabled")
        self.btn_train_export.configure(state="disabled")
        self.btn_train_deploy.configure(state="disabled")
        self.btn_return_designer.configure(state="disabled")

        self.kpi_status.set("Status: Initializing...")
        self.kpi_best_banner.set("")
        self.kpi_train_loss.set("--")
        self.kpi_val_loss.set("--")
        self.kpi_train_metric.set("--")
        self.kpi_val_metric.set("--")
        self.kpi_best_metric.set("--")

        # Set KPI card titles based on chosen evaluation metric
        metric_label = self.eval_metric_choice.get()
        self.lbl_card_metric_title.configure(text=f"TRAIN {metric_label.upper()}")
        self.lbl_card_val_title.configure(text=f"Val {metric_label}:")

        self.prog_batch_var.set(0.0)
        self.prog_eval_var.set(0.0)
        self.prog_epoch_var.set(0.0)
        self.lbl_prog_batch_text.set("Initializing dataset...")
        self.lbl_prog_eval_text.set("Evaluation will start after epoch training.")
        self.lbl_prog_epoch_text.set(f"Epoch 0 / {epochs} (0.0%)")

        self.history_records.clear()
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)

        self._log(f"=== Starting Training Session ({epochs} Epochs) ===")
        self._log(f"Evaluation Metric: {metric_label} | Optimizer: {self.train_optimizer_name.get()} (lr={lr})")

        self._train_thread = threading.Thread(
            target=self._run_training_worker,
            args=(epochs, batch_size, lr, momentum, weight_decay, num_workers, grad_clip, early_stop, save_dir),
            daemon=True,
        )
        self._train_thread.start()

    def stop_training(self):
        if not self._is_training:
            return
        self._stop_requested = True
        self.btn_train_stop.configure(state="disabled")
        self.kpi_status.set("Status: Stopping after current batch...")
        self._log("Stop requested by user. Terminating training safely...")

    def _run_training_worker(
        self,
        epochs: int,
        batch_size: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        num_workers: int,
        grad_clip: float,
        early_stop: int,
        save_dir: str,
    ):
        device_str = self.train_device.get()
        device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
        eval_metric = self.eval_metric_choice.get()

        try:
            self.after(0, self._log, f"Hardware device: {device}")
            in_ch = self.design.in_channels
            in_h = self.design.input_height
            in_w = self.design.input_width

            # Transforms: Training with data augmentation, Validation clean
            train_transform = get_dataset_transform(
                in_ch, in_h, in_w,
                aug_hflip=self.aug_hflip.get(),
                aug_crop=self.aug_crop.get(),
                aug_color_jitter=self.aug_color_jitter.get(),
            )
            clean_transform = get_dataset_transform(in_ch, in_h, in_w)

            mode = self.train_data_mode.get()
            if mode == "ImageFolder":
                raw_train_ds = create_image_folder_dataset(self.train_img_folder.get(), in_channels=in_ch, transform=train_transform)
                raw_eval_ds = create_image_folder_dataset(self.train_img_folder.get(), in_channels=in_ch, transform=clean_transform)
            else:
                raw_train_ds = CSVImageDataset(self.train_csv_folder.get(), self.train_csv_file.get(), transform=train_transform, in_channels=in_ch)
                raw_eval_ds = CSVImageDataset(self.train_csv_folder.get(), self.train_csv_file.get(), transform=clean_transform, in_channels=in_ch)

            classes = raw_train_ds.classes
            num_classes = len(classes)
            self.best_classes = classes

            if num_classes != self.design.num_classes:
                self.design.num_classes = num_classes
                self.after(0, lambda n=num_classes: self.var_num_classes.set(str(n)))
                self.after(0, self._log, f"Model num_classes synchronized to {num_classes} classes.")

            # Validation split / dataset
            val_strat = self.val_strategy.get()
            if val_strat == "split":
                val_ratio = float(self.val_split_size.get())
                seed = int(self.val_split_seed.get())
                stratify_flag = self.val_split_stratify.get()

                indices = list(range(len(raw_train_ds)))
                targets = raw_train_ds.targets
                strat = None
                if stratify_flag:
                    counts = pd.Series(targets).value_counts()
                    if counts.min() >= 2 and len(counts) > 1:
                        strat = targets

                train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=seed, stratify=strat)
                train_ds = Subset(raw_train_ds, train_idx)
                val_ds = Subset(raw_eval_ds, val_idx)
                self.after(0, self._log, f"Dataset split: {len(train_ds)} train samples, {len(val_ds)} val samples.")
            else:
                train_ds = raw_train_ds
                val_mode = self.val_data_mode.get()
                if val_mode == "ImageFolder":
                    val_ds = create_image_folder_dataset(self.val_img_folder.get(), in_channels=in_ch, transform=clean_transform)
                else:
                    val_ds = CSVImageDataset(
                        self.val_csv_folder.get(), self.val_csv_file.get(), transform=clean_transform,
                        class_to_idx=raw_train_ds.class_to_idx, in_channels=in_ch
                    )
                self.after(0, self._log, f"Separate datasets: {len(train_ds)} train samples, {len(val_ds)} val samples.")

            # DataLoaders
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if val_ds is not None else None

            # Build Model
            ModelClass = self.design.get_nn_module_class("AutomateCNNModel")
            model = ModelClass(in_channels=in_ch, num_classes=num_classes).to(device)

            # Setup Class Weights & Loss
            weight_tensor = None
            if self.train_use_weights.get():
                w_mode = self.train_weight_mode.get()
                if "Auto" in w_mode:
                    if isinstance(train_ds, Subset):
                        train_targets = [train_ds.dataset.targets[i] for i in train_ds.indices]
                    else:
                        train_targets = train_ds.targets
                    c_counts = np.bincount(train_targets, minlength=num_classes)
                    total_s = len(train_targets)
                    weights = total_s / (num_classes * np.maximum(c_counts, 1).astype(float))
                    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
                    self.after(0, self._log, f"Auto class weights applied: {np.round(weights, 3).tolist()}")
                else:
                    raw_w = [float(p.strip()) for p in self.train_manual_weights.get().split(",") if p.strip()]
                    if len(raw_w) != num_classes:
                        raise ValueError(f"Manual weights count ({len(raw_w)}) must match num_classes ({num_classes}).")
                    weight_tensor = torch.tensor(raw_w, dtype=torch.float, device=device)
                    self.after(0, self._log, f"Manual class weights applied: {raw_w}")

            loss_name = self.train_loss_name.get()
            if loss_name == "CrossEntropyLoss":
                criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            elif loss_name == "NLLLoss":
                criterion = nn.NLLLoss(weight=weight_tensor)
            elif loss_name == "BCEWithLogitsLoss":
                criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
            else:
                criterion = nn.CrossEntropyLoss(weight=weight_tensor)

            # Setup Optimizer
            opt_name = self.train_optimizer_name.get()
            if opt_name == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif opt_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            elif opt_name == "AdamW":
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif opt_name == "RMSprop":
                optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            elif opt_name == "Adagrad":
                optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr)

            # Setup LR Scheduler
            sched_choice = self.train_lr_scheduler.get()
            scheduler = None
            if "StepLR" in sched_choice:
                step_sz = max(1, epochs // 3)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_sz, gamma=0.5)
            elif "CosineAnnealingLR" in sched_choice:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            elif "ReduceLROnPlateau" in sched_choice:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min" if "Loss" in self.save_metric_choice.get() else "max", patience=2, factor=0.5
                )

            # Training loop variables
            best_metric_val: Optional[float] = None
            self.best_epoch_record = None
            metric_choice = self.save_metric_choice.get()
            num_batches = len(train_loader)
            num_val_batches = len(val_loader) if val_loader else 0
            epochs_no_improve = 0

            for epoch in range(epochs):
                if self._stop_requested:
                    break

                self.after(0, lambda ep=epoch + 1: self.kpi_status.set(f"Status: Training Epoch {ep}/{epochs}..."))

                model.train()
                running_loss = 0.0
                all_train_preds: List[int] = []
                all_train_targets: List[int] = []
                total_samples = 0

                # Reset Bar 1 for current epoch
                self.after(0, lambda: self.prog_batch_var.set(0.0))

                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    if self._stop_requested:
                        break

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)

                    if loss_name == "BCEWithLogitsLoss":
                        targets_one_hot = torch.zeros(outputs.shape, device=device).scatter_(
                            1, targets.unsqueeze(1), 1.0
                        )
                        loss = criterion(outputs, targets_one_hot)
                    else:
                        loss = criterion(outputs, targets)

                    loss.backward()

                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)

                    all_train_preds.extend(preds.cpu().numpy().tolist())
                    all_train_targets.extend(targets.cpu().numpy().tolist())
                    total_samples += inputs.size(0)

                    # Calculate live metric
                    if eval_metric == "F1-Score":
                        batch_metric = f1_score(targets.cpu().numpy(), preds.cpu().numpy(), average="macro", zero_division=0) * 100.0
                    else:
                        batch_metric = (torch.sum(preds == targets.data).item() / inputs.size(0)) * 100.0

                    batch_pct = ((batch_idx + 1) / num_batches) * 100.0
                    cur_loss = running_loss / total_samples

                    # UI update for Bar 1 (batch progress through epoch X)
                    self.after(
                        0,
                        self._update_training_batch_progress,
                        epoch + 1,
                        epochs,
                        batch_idx + 1,
                        num_batches,
                        batch_pct,
                        loss.item(),
                        batch_metric,
                        cur_loss,
                    )

                if self._stop_requested:
                    break

                # Epoch training scores
                epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0.0
                if eval_metric == "F1-Score":
                    epoch_train_metric = f1_score(all_train_targets, all_train_preds, average="macro", zero_division=0) * 100.0
                else:
                    epoch_train_metric = (np.sum(np.array(all_train_preds) == np.array(all_train_targets)) / total_samples) * 100.0

                # Validation pass: Bar 2 (evaluation progress through epoch X)
                epoch_val_loss = None
                epoch_val_metric = None

                if val_loader is not None and not self._stop_requested:
                    self.after(0, lambda ep=epoch + 1: self.kpi_status.set(f"Status: Evaluating Epoch {ep}/{epochs}..."))
                    self.after(0, lambda: self.prog_eval_var.set(0.0))

                    model.eval()
                    v_loss_sum = 0.0
                    all_val_preds: List[int] = []
                    all_val_targets: List[int] = []
                    v_total = 0

                    with torch.no_grad():
                        for v_idx, (v_inputs, v_targets) in enumerate(val_loader):
                            if self._stop_requested:
                                break
                            v_inputs = v_inputs.to(device)
                            v_targets = v_targets.to(device)
                            v_out = model(v_inputs)

                            if loss_name == "BCEWithLogitsLoss":
                                v_targets_one_hot = torch.zeros(v_out.shape, device=device).scatter_(
                                    1, v_targets.unsqueeze(1), 1.0
                                )
                                v_loss = criterion(v_out, v_targets_one_hot)
                            else:
                                v_loss = criterion(v_out, v_targets)

                            v_loss_sum += v_loss.item() * v_inputs.size(0)
                            _, v_preds = torch.max(v_out, 1)

                            all_val_preds.extend(v_preds.cpu().numpy().tolist())
                            all_val_targets.extend(v_targets.cpu().numpy().tolist())
                            v_total += v_inputs.size(0)

                            v_pct = ((v_idx + 1) / num_val_batches) * 100.0
                            self.after(
                                0,
                                self._update_val_batch_progress,
                                v_idx + 1,
                                num_val_batches,
                                v_pct,
                                v_loss.item(),
                            )

                    if v_total > 0:
                        epoch_val_loss = v_loss_sum / v_total
                        if eval_metric == "F1-Score":
                            epoch_val_metric = f1_score(all_val_targets, all_val_preds, average="macro", zero_division=0) * 100.0
                        else:
                            epoch_val_metric = (np.sum(np.array(all_val_preds) == np.array(all_val_targets)) / v_total) * 100.0

                # LR Scheduler step
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        score_to_step = epoch_val_loss if "Loss" in metric_choice else (epoch_val_metric or epoch_train_metric)
                        scheduler.step(score_to_step)
                    else:
                        scheduler.step()

                # Determine if this epoch is the BEST model
                is_best = False
                metric_val_for_log = 0.0

                if metric_choice == "Best Validation Accuracy" or metric_choice == "Best Validation F1":
                    val_to_check = epoch_val_metric if epoch_val_metric is not None else epoch_train_metric
                    metric_val_for_log = val_to_check
                    if best_metric_val is None or val_to_check > best_metric_val:
                        is_best = True
                        best_metric_val = val_to_check
                elif metric_choice == "Best Validation Loss":
                    val_to_check = epoch_val_loss if epoch_val_loss is not None else epoch_train_loss
                    metric_val_for_log = val_to_check
                    if best_metric_val is None or val_to_check < best_metric_val:
                        is_best = True
                        best_metric_val = val_to_check
                elif metric_choice == "Best Train Accuracy" or metric_choice == "Best Train F1":
                    metric_val_for_log = epoch_train_metric
                    if best_metric_val is None or epoch_train_metric > best_metric_val:
                        is_best = True
                        best_metric_val = epoch_train_metric
                elif metric_choice == "Best Train Loss":
                    metric_val_for_log = epoch_train_loss
                    if best_metric_val is None or epoch_train_loss < best_metric_val:
                        is_best = True
                        best_metric_val = epoch_train_loss
                elif metric_choice == "Every Epoch":
                    is_best = True
                    metric_val_for_log = epoch_val_metric if epoch_val_metric is not None else epoch_train_metric

                # Early stopping check
                if is_best:
                    epochs_no_improve = 0
                    self.best_epoch_record = epoch + 1
                else:
                    epochs_no_improve += 1

                # Save best model checkpoint
                saved_path = None
                if is_best:
                    os.makedirs(save_dir, exist_ok=True)
                    ckpt_name = "best_model.pth"
                    saved_path = os.path.join(save_dir, ckpt_name)

                    self.best_model_state_dict = copy.deepcopy(model.state_dict())
                    self.best_metric_info = {
                        "metric_monitored": metric_choice,
                        "metric_value": metric_val_for_log,
                        "epoch": epoch + 1,
                        "train_loss": epoch_train_loss,
                        "train_metric": epoch_train_metric,
                        "val_loss": epoch_val_loss,
                        "val_metric": epoch_val_metric,
                    }

                    # Transformation dictionary (classes conversion to 0..n)
                    class_map = {cls: idx for idx, cls in enumerate(classes)}
                    class_map_file = os.path.join(save_dir, "class_map.json")
                    with open(class_map_file, "w") as f:
                        json.dump(
                            {
                                "description": "Class name to integer index conversion mapping",
                                "classes": classes,
                                "class_to_idx": class_map,
                                "idx_to_class": {str(idx): cls for cls, idx in class_map.items()},
                            },
                            f,
                            indent=2,
                        )

                    checkpoint = {
                        "epoch": epoch + 1,
                        "model_state_dict": self.best_model_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": epoch_train_loss,
                        "train_metric": epoch_train_metric,
                        "val_loss": epoch_val_loss,
                        "val_metric": epoch_val_metric,
                        "metric_monitored": metric_choice,
                        "metric_value": metric_val_for_log,
                        "classes": classes,
                        "class_to_idx": class_map,
                        "design": asdict(self.design),
                    }
                    torch.save(checkpoint, saved_path)

                # Record into history
                record = {
                    "epoch": epoch + 1,
                    "train_loss": epoch_train_loss,
                    "train_metric": epoch_train_metric,
                    "val_loss": epoch_val_loss,
                    "val_metric": epoch_val_metric,
                    "is_best": is_best,
                }
                self.history_records.append(record)

                # UI update for Bar 3 (epoch overall) & history table
                overall_pct = ((epoch + 1) / epochs) * 100.0
                self.after(
                    0,
                    self._update_epoch_completed,
                    epoch + 1,
                    epochs,
                    overall_pct,
                    epoch_train_loss,
                    epoch_train_metric,
                    epoch_val_loss,
                    epoch_val_metric,
                    best_metric_val,
                    is_best,
                )

                # Early stopping trigger
                if early_stop > 0 and epochs_no_improve >= early_stop:
                    self.after(0, self._log, f"Early stopping triggered: no improvement for {early_stop} epochs.")
                    break

        except Exception as exc:
            self.after(0, self._log, f"❌ Training Error: {exc}")
            self.after(0, lambda e=str(exc): messagebox.showerror("Training Error", e, parent=self))
        finally:
            self.after(0, self._on_training_finished)

    def _update_training_batch_progress(
        self, epoch: int, total_epochs: int, batch: int, total_batches: int,
        batch_pct: float, batch_loss: float, batch_metric: float, cur_loss: float
    ):
        self.prog_batch_var.set(batch_pct)
        metric_name = self.eval_metric_choice.get()
        self.lbl_prog_batch_text.set(
            f"Batch {batch} / {total_batches} ({batch_pct:.1f}%) | "
            f"Batch Loss: {batch_loss:.4f} | Batch {metric_name}: {batch_metric:.1f}%"
        )
        self.kpi_train_loss.set(f"{cur_loss:.4f}")

    def _update_val_batch_progress(self, batch: int, total_batches: int, val_pct: float, batch_loss: float):
        self.prog_eval_var.set(val_pct)
        self.lbl_prog_eval_text.set(f"Eval Batch {batch} / {total_batches} ({val_pct:.1f}%) | Loss: {batch_loss:.4f}")

    def _update_epoch_completed(
        self,
        epoch: int,
        total_epochs: int,
        overall_pct: float,
        train_loss: float,
        train_metric: float,
        val_loss: Optional[float],
        val_metric: Optional[float],
        best_metric_val: Optional[float],
        is_best: bool,
    ):
        self.prog_epoch_var.set(overall_pct)
        self.lbl_prog_epoch_text.set(f"Epoch {epoch} / {total_epochs} ({overall_pct:.1f}%)")

        self.kpi_train_loss.set(f"{train_loss:.4f}")
        self.kpi_train_metric.set(f"{train_metric:.2f}%")

        v_loss_str = f"{val_loss:.4f}" if val_loss is not None else "--"
        v_metric_str = f"{val_metric:.2f}%" if val_metric is not None else "--"
        self.kpi_val_loss.set(v_loss_str)
        self.kpi_val_metric.set(v_metric_str)

        if best_metric_val is not None:
            self.kpi_best_metric.set(f"★ {best_metric_val:.2f}% (Ep {self.best_epoch_record})")

        # Insert row in history table
        best_tag = "★ BEST" if is_best else "-"
        row_tag = "best_row" if is_best else "regular_row"
        self.history_tree.insert(
            "",
            "end",
            values=(
                f"Ep {epoch}",
                f"{train_loss:.4f}",
                f"{train_metric:.2f}%",
                v_loss_str,
                v_metric_str,
                best_tag,
            ),
            tags=(row_tag,),
        )
        # Auto scroll to bottom
        self.history_tree.yview_moveto(1.0)

        metric_name = self.eval_metric_choice.get()
        log_line = (
            f"Epoch {epoch}/{total_epochs} ➔ Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.2f}% | "
            f"Val Loss: {v_loss_str}, Val {metric_name}: {v_metric_str}"
        )
        if is_best:
            log_line += " | 💾 Best Model Saved (best_model.pth)"
        self._log(log_line)

    def _on_training_finished(self):
        self._is_training = False
        self._stop_requested = False
        self.btn_train_stop.configure(state="disabled")
        self.btn_return_designer.configure(state="normal")
        self.kpi_status.set("Status: Completed / Stopped")

        metric_name = self.eval_metric_choice.get()

        if self.best_model_state_dict is not None and self.best_epoch_record is not None:
            self.btn_train_export.configure(state="normal")
            self.btn_view_graphs.configure(state="normal")

            best_val = self.best_metric_info.get("metric_value", 0.0)
            banner_txt = (
                f"🏆 Training Finished! Best Model Achieved at Epoch {self.best_epoch_record} "
                f"with {metric_name} = {best_val:.2f}%.\n"
                f"Model weights automatically saved as 'best_model.pth' and class conversion table saved as 'class_map.json'."
            )
            self.kpi_best_banner.set(banner_txt)
            self._log(f"\n{banner_txt}")

            # Prompt to view graphs
            self.after(500, self.show_training_graphs)
        else:
            self.kpi_best_banner.set("Training ended without saving a checkpoint.")

    def show_training_graphs(self):
        if not self.history_records:
            messagebox.showinfo("No History", "No training history records available yet.", parent=self)
            return
        TrainingGraphsDialog(
            self,
            self.history_records,
            metric_name=self.eval_metric_choice.get(),
            best_epoch=self.best_epoch_record,
        )

    # -- Export & Deploy actions ---------------------------------------
    def export_best_model_action(self):
        """Export best model package with config.json, class_map.json, best_model.pth, model.py, predict.py, api_server.py."""
        if self.best_model_state_dict is None:
            messagebox.showinfo("No Model Trained", "Please run a training session first before exporting.", parent=self)
            return

        export_dir = filedialog.askdirectory(
            title="Select Destination Folder for Exported Model Package",
            initialdir=os.path.abspath("./"),
            parent=self,
        )
        if not export_dir:
            return

        target_dir = os.path.join(export_dir, "deployed_model")
        os.makedirs(target_dir, exist_ok=True)

        try:
            res = export_model_package(
                design=self.design,
                model_state_dict=self.best_model_state_dict,
                classes=self.best_classes,
                export_dir=target_dir,
                metric_info=self.best_metric_info,
            )
            self.last_export_path = target_dir
            self.btn_train_deploy.configure(state="normal")

            self._log(f"📦 Model package successfully exported to: {target_dir}")
            self._log(f"   Files: config.json, class_map.json, best_model.pth, model.py, predict.py, api_server.py")

            msg = (
                f"Model exported successfully!\n\n"
                f"Export Location:\n{target_dir}\n\n"
                f"Package contents:\n"
                f"• config.json (Metadata, input specs, and class names)\n"
                f"• class_map.json (Class name to 0, 1, ..., n integer conversion dictionary)\n"
                f"• best_model.pth (Best trained PyTorch model weights)\n"
                f"• model.py (Standalone PyTorch nn.Module class)\n"
                f"• predict.py (Plug-and-play CLI prediction script)\n"
                f"• api_server.py (Zero-dependency REST API server)\n\n"
                f"You can now click '🚀 Deploy & Test' to test single images or launch the inference server!"
            )
            messagebox.showinfo("Export Successful", msg, parent=self)
        except Exception as exc:
            messagebox.showerror("Export Failed", str(exc), parent=self)

    def start_deploying_action(self):
        if not self.last_export_path or not os.path.isdir(self.last_export_path):
            messagebox.showinfo("Export Required", "Please click 'Export Best Model' first before deploying.", parent=self)
            return
        DeploymentDialog(self, self.last_export_path)

    def _on_window_close(self):
        if self._is_training:
            if messagebox.askyesno("Exit Confirmation", "A training run is currently active. Stop training and exit?", parent=self):
                self._stop_requested = True
                self.destroy()
        else:
            self.destroy()

    # -- Global Settings Synchronizer ----------------------------------
    def _pull_global_settings(self) -> Optional[str]:
        try:
            self.design.in_channels = int(self.var_in_channels.get())
            self.design.input_height = int(self.var_in_h.get())
            self.design.input_width = int(self.var_in_w.get())
            self.design.num_classes = int(self.var_num_classes.get())
            if self.design.in_channels < 1 or self.design.input_height < 1 or self.design.input_width < 1 or self.design.num_classes < 1:
                return "Channels, H, W, and num_classes must be positive integers."
        except ValueError:
            return "Input channels / H / W / num classes must be integers."
        self.design.flatten_mode = self.var_flatten_mode.get()
        return None

    # -- Refresh Treeviews with Tensor Flow & Final Layer --------------
    def _refresh_backbone_tree(self):
        self.backbone_tree.delete(*self.backbone_tree.get_children())
        layers = compute_layer_shapes(self.design)
        backbone_layers = [l for l in layers if l["section"] == "Backbone"]

        for i, b_layer in enumerate(backbone_layers):
            self.backbone_tree.insert(
                "",
                "end",
                iid=str(i),
                values=(
                    i + 1,
                    b_layer["input_str"],
                    b_layer["output_str"],
                    b_layer["summary"],
                ),
            )

        if backbone_layers:
            first_in = backbone_layers[0]["input_str"]
            last_out = backbone_layers[-1]["output_str"]
            self.lbl_backbone_summary.configure(
                text=f"Backbone Flow: Input {first_in}  ➔  Output {last_out} across {len(backbone_layers)} layers."
            )
        else:
            self.lbl_backbone_summary.configure(text="Backbone: No convolutional layers added yet.")

    def _refresh_head_tree(self):
        self.head_tree.delete(*self.head_tree.get_children())
        layers = compute_layer_shapes(self.design)
        head_layers = [l for l in layers if l["section"] == "Head"]
        output_layer = [l for l in layers if l["section"] == "Output"]

        for i, h_layer in enumerate(head_layers):
            self.head_tree.insert(
                "",
                "end",
                iid=str(i),
                values=(
                    i + 1,
                    h_layer["input_str"],
                    h_layer["output_str"],
                    h_layer["summary"],
                ),
            )

        # Insert final classification layer (maps to num_classes)
        if output_layer:
            out_info = output_layer[0]
            self.head_tree.insert(
                "",
                "end",
                iid="final_output_layer",
                values=(
                    "★ Out",
                    out_info["input_str"],
                    out_info["output_str"],
                    f"nn.Linear(in={out_info['input_str']}, out={out_info['output_str']}) ➔ Final Classification Logits (Auto-managed)",
                ),
                tags=("final_layer",),
            )

        self.lbl_head_summary.configure(
            text=f"Head: {len(head_layers)} custom hidden layer(s) + 1 final output layer (mapping to {self.design.num_classes} classes)."
        )

    def _selected_index(self, tree) -> Optional[int]:
        sel = tree.selection()
        if not sel:
            return None
        if sel[0] == "final_output_layer":
            messagebox.showinfo(
                "Final Classifier Layer",
                f"The final output layer is automatically configured to output {self.design.num_classes} classes.\n\n"
                f"To change its output size, simply update 'Number of Classes' in Global Settings.",
                parent=self,
            )
            return None
        return int(sel[0])

    def _get_input_shape_for_backbone_index(self, idx: int) -> Tuple[int, int, int]:
        c, h, w = self.design.in_channels, self.design.input_height, self.design.input_width
        for i in range(idx):
            b = self.design.backbone[i]
            try:
                c, h, w = compute_conv_block_output_shape(b, c, h, w)
            except Exception:
                pass
        return c, h, w

    def _get_input_features_for_head_index(self, idx: int) -> int:
        layers = compute_layer_shapes(self.design)
        trans = [l for l in layers if l["section"] == "Transition"]
        in_feat = trans[0]["output_shape"] if trans else 128
        for i in range(idx):
            try:
                in_feat = int(self.design.head[i].out_features)
            except Exception:
                pass
        return in_feat

    # -- Backbone CRUD -------------------------------------------------
    def add_backbone_layer(self):
        in_shape = self._get_input_shape_for_backbone_index(len(self.design.backbone))
        dlg = ConvBlockDialog(self, in_shape=in_shape)
        if dlg.result is not None:
            self.design.backbone.append(dlg.result)
            self._refresh_backbone_tree()
            self._refresh_head_tree()

    def edit_backbone_layer(self):
        idx = self._selected_index(self.backbone_tree)
        if idx is None:
            return
        in_shape = self._get_input_shape_for_backbone_index(idx)
        dlg = ConvBlockDialog(self, block=copy.deepcopy(self.design.backbone[idx]), in_shape=in_shape)
        if dlg.result is not None:
            self.design.backbone[idx] = dlg.result
            self._refresh_backbone_tree()
            self._refresh_head_tree()

    def duplicate_backbone_layer(self):
        idx = self._selected_index(self.backbone_tree)
        if idx is None:
            return
        self.design.backbone.insert(idx + 1, copy.deepcopy(self.design.backbone[idx]))
        self._refresh_backbone_tree()
        self._refresh_head_tree()

    def delete_backbone_layer(self):
        idx = self._selected_index(self.backbone_tree)
        if idx is None:
            return
        del self.design.backbone[idx]
        self._refresh_backbone_tree()
        self._refresh_head_tree()

    # -- Head CRUD -----------------------------------------------------
    def add_head_layer(self):
        in_feat = self._get_input_features_for_head_index(len(self.design.head))
        dlg = HeadBlockDialog(self, in_features=in_feat)
        if dlg.result is not None:
            self.design.head.append(dlg.result)
            self._refresh_head_tree()

    def edit_head_layer(self):
        idx = self._selected_index(self.head_tree)
        if idx is None:
            return
        in_feat = self._get_input_features_for_head_index(idx)
        dlg = HeadBlockDialog(self, block=copy.deepcopy(self.design.head[idx]), in_features=in_feat)
        if dlg.result is not None:
            self.design.head[idx] = dlg.result
            self._refresh_head_tree()

    def duplicate_head_layer(self):
        idx = self._selected_index(self.head_tree)
        if idx is None:
            return
        self.design.head.insert(idx + 1, copy.deepcopy(self.design.head[idx]))
        self._refresh_head_tree()

    def delete_head_layer(self):
        idx = self._selected_index(self.head_tree)
        if idx is None:
            return
        del self.design.head[idx]
        self._refresh_head_tree()

    # -- Move layer up/down --------------------------------------------
    def move_layer(self, is_backbone: bool, delta: int):
        tree = self.backbone_tree if is_backbone else self.head_tree
        lst = self.design.backbone if is_backbone else self.design.head
        idx = self._selected_index(tree)
        if idx is None:
            return
        new_idx = idx + delta
        if 0 <= new_idx < len(lst):
            lst[idx], lst[new_idx] = lst[new_idx], lst[idx]
            if is_backbone:
                self._refresh_backbone_tree()
                self._refresh_head_tree()
            else:
                self._refresh_head_tree()
            tree.selection_set(str(new_idx))

    # -- Save / Load Design --------------------------------------------
    def save_design(self):
        err = self._pull_global_settings()
        if err:
            messagebox.showerror("Invalid settings", err, parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON file", "*.json")],
            initialfile="cnn_design.json",
            parent=self,
        )
        if not path:
            return
        with open(path, "w") as f:
            f.write(self.design.to_json())
        messagebox.showinfo("Saved", f"Design saved to:\n{path}", parent=self)

    def load_design(self):
        path = filedialog.askopenfilename(filetypes=[("JSON file", "*.json")], parent=self)
        if not path:
            return
        try:
            with open(path) as f:
                self.design = ModelDesign.from_json(f.read())
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc), parent=self)
            return
        self.var_in_channels.set(str(self.design.in_channels))
        self.var_in_h.set(str(self.design.input_height))
        self.var_in_w.set(str(self.design.input_width))
        self.var_num_classes.set(str(self.design.num_classes))
        self.var_flatten_mode.set(self.design.flatten_mode)
        self.lbl_global_badge.configure(
            text=f"Tensor Input Shape: ({self.design.in_channels}, {self.design.input_height}, {self.design.input_width})  ➔  Classes: {self.design.num_classes}"
        )
        self._refresh_backbone_tree()
        self._refresh_head_tree()


def main():
    app = DesignerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
