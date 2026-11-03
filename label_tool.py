from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from label_obb import Box, atomic_write, decode, encode, normalize_angle
from synth_engine import (  # noqa: F401  (re-exported for scripts and tests)
    AUTO_ERROR_MODES, CONFIRM_ABOVE, CUT_SIDES, EFFECT_MODES, IMAGE_EXTENSIONS,
    JITTER_ANGLE, JITTER_ANGLE_DEG, JITTER_SHIFT, MAX_ATTEMPTS, MAX_JITTER_ANGLE_DEG, STATE_VERSION, BatchSettings, Code,
    Composition, Generator, Job, Target, all_mode_names, auto_job_name, box_inside_box,
    character_asset, clip_box_to_side, code_mode_problem, contact_sheet, default_state,
    encode_out_lines, erase_boxes, first_free_number, image_files, image_to_local,
    layout_conflict, layout_boxes, line_crosses_box, line_side_mask, load_code, load_target,
    local_to_image, normalize_state, oriented_line, out_line_hits, overlapping_boxes,
    plan_batch, previous_split, propose_character_boxes, read_manifest, read_out_lines,
    read_region, run_batch, scale_region,
    split_rows, state_to_json, state_uses_indices, task_rng, text_angle,
)


BASE = Path(__file__).resolve().parent
SAMPLE_ROOT = BASE / "group_samples" / "images"
BACKGROUND_ROOT = BASE / "group_background"
OUTPUT_ROOT = BASE / "synthetic_output"
CONFIG_PATH = BASE / "label_tool_config.json"
MODEL_PATH = BASE / "best_char_obb.pt"
CHECK_ON, CHECK_OFF = "☑", "☐"


def sample_label_path(path):
    return SAMPLE_ROOT / "labels" / f"{path.stem}.txt"


def region_path(path):
    return BACKGROUND_ROOT / "labels" / f"{path.stem}.txt"


def out_line_path(path):
    """Out-lines of a background (see synth_engine.read_out_lines)."""
    return BACKGROUND_ROOT / "lines" / f"{path.stem}.json"


def read_int(variable, default, low, high):
    """Read a Tk number variable whose widget may hold half-typed text."""
    try:
        value = float(variable.get())
    except (tk.TclError, ValueError, TypeError):
        return default
    if not math.isfinite(value):
        return default
    return max(low, min(high, int(round(value))))


def read_float(variable, default, low, high):
    try:
        value = float(variable.get())
    except (tk.TclError, ValueError, TypeError):
        return default
    if not math.isfinite(value):
        return default
    return max(low, min(high, value))

class ProgressDialog:
    """Modal progress window that keeps Tk responsive during a long export."""

    def __init__(self, parent, title, total):
        self.cancelled = False
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.transient(parent)
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)
        self.text = tk.StringVar(value="Đang chuẩn bị…")
        ttk.Label(frame, textvariable=self.text, width=60).pack(anchor="w")
        self.bar = ttk.Progressbar(frame, maximum=max(1, total), length=420)
        self.bar.pack(fill="x", pady=8)
        ttk.Button(frame, text="Hủy", command=self.cancel).pack(anchor="e")
        self.window.update()
        try:
            # Modal: keeps clicks/keys away from the main window meanwhile.
            # A grab needs a mapped window, hence the update() first.
            self.window.grab_set()
        except tk.TclError:
            pass
        self.window.focus_set()

    def cancel(self):
        self.cancelled = True
        self.text.set("Đang dừng…")

    def step(self, done, text):
        self.bar["value"] = done
        if not self.cancelled:
            self.text.set(text)
        self.window.update()

    def close(self):
        try:
            self.window.grab_release()
            self.window.destroy()
        except tk.TclError:
            pass


class SyntheticLabelTool:
    COLORS = ("#00e5ff", "#ffca28", "#ff66aa", "#76ff03")

    def __init__(self, root):
        self.root = root
        root.title("Synthetic Date OBB Tool")
        root.geometry("1680x960")
        root.minsize(1100, 700)
        SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
        BACKGROUND_ROOT.mkdir(parents=True, exist_ok=True)
        self.generator = Generator()
        self.size_by_region = tk.BooleanVar(value=False)
        self.batch = BatchSettings()
        # Checkbox choices the user made; names not listed use the default
        # (code: checked when it has a box, background: always checked).
        self.code_checks, self.background_checks = {}, {}
        self._status_cache = {}
        self.region_queue = None
        self.batch_dialog = None
        self.samples, self.backgrounds = [], []
        self.sample_index = self.background_index = -1
        self.sample_path = self.background_path = None
        self.sample_image = self.background_image = None
        self.sample_boxes = []
        self.region = None
        self.out_lines = []  # background pixels; touching one = "out" error
        self.selected_box = None
        self.history = []
        self.cut_lines = []  # date-local coordinates, see local_to_image
        self.cut_side = tk.StringVar(value="none")
        self.mode = tk.StringVar(value="normal")
        self.last_effect_mode = "normal"
        self.saved_modes = []
        self.saved_mode_serial = 0
        self.synthetic_output_dir = tk.StringVar(value=str(OUTPUT_ROOT))
        self.export_count = tk.IntVar(value=10)
        self.seed_text = tk.StringVar(value="")
        self.stack_size = tk.IntVar(value=3)
        self.arc_height = tk.DoubleVar(value=35.0)
        self.blur_radius = tk.DoubleVar(value=2.0)
        self.spacing_groups = []
        self.group_indices_text = tk.StringVar(value="0-2")
        self.group_x_factor = tk.DoubleVar(value=1.0)
        self.delete_scope = tk.StringVar(value="box")
        self.deleted_indices = set()
        self.preview_cache = None
        self.preview_boxes = []
        self.preview_source_indices = []
        self.preview_blank = []
        self.view = tk.StringVar(value="sample")
        self.draw_mode = tk.StringVar(value="edit")
        self._rotate_key, self._rotate_time = None, 0.0
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self.photo = None
        self.gesture = None
        self.busy = False
        self.close_requested = False
        self.progress = None
        self._asset_cache = {}
        self._size_cache = {}
        self._warned_duplicates = set()
        self.status = tk.StringVar(value="Sẵn sàng")
        self.info = tk.StringVar(value="")
        self._build_ui()
        config_warning = self.load_config()
        self.refresh_files()
        if config_warning:
            self.status.set(config_warning)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------ config

    def config_data(self):
        """Return JSON-safe settings that should survive an application restart."""
        return {
            "synthetic_output_dir": self.synthetic_output_dir.get(),
            "export_count": read_int(self.export_count, 10, 1, 10000),
            "seed": self.seed_text.get().strip(),
            "stack_size": read_int(self.stack_size, 3, 1, 5),
            "arc_height": read_float(self.arc_height, 35.0, -100.0, 100.0),
            "blur_radius": read_float(self.blur_radius, 2.0, .2, 8.0),
            "group_indices_text": self.group_indices_text.get(),
            "delete_scope": self.delete_scope.get(),
            "saved_mode_serial": self.saved_mode_serial,
            "saved_modes": [
                {"name": item["name"], "state": state_to_json(item["state"])}
                for item in self.saved_modes
            ],
            "batch": self.batch_config(),
        }

    def batch_config(self):
        settings = self.batch
        return {
            "codes": sorted(name for name, on in self.code_checks.items() if on),
            "codes_unchecked": sorted(name for name, on in self.code_checks.items() if not on),
            "backgrounds": sorted(name for name, on in self.background_checks.items() if on),
            "backgrounds_unchecked": sorted(
                name for name, on in self.background_checks.items() if not on),
            "pairing": settings.pairing,
            "max_backgrounds_per_code": settings.max_backgrounds_per_code,
            "count": read_int(self.export_count, 10, 1, 100000),
            "count_mode": settings.count_mode,
            "modes": settings.selected_modes(all_mode_names(self.saved_modes)),
            "modes_unchecked": settings.modes_off or [],
            "val_percent": settings.val_percent,
            "size_mode": "region" if self.size_by_region.get() else "image",
            "angle_jitter_deg": settings.angle_jitter_deg,
        }

    def save_config(self):
        try:
            atomic_write(CONFIG_PATH, json.dumps(
                self.config_data(), ensure_ascii=False, indent=2) + "\n")
        except (OSError, TypeError, ValueError, tk.TclError):
            # Closing the UI must not fail just because settings cannot be saved.
            pass

    def load_config(self):
        """Load settings; return a warning message or None."""
        if not CONFIG_PATH.exists():
            return None
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("cấu hình phải là object")
        except (OSError, ValueError) as exc:
            return f"Không đọc được cấu hình cũ ({exc}); đang dùng cấu hình mặc định"
        problems = []

        def setting(name, apply):
            if name not in data:
                return
            try:
                apply(data[name])
            except (TypeError, ValueError, tk.TclError):
                problems.append(name)

        setting("synthetic_output_dir", lambda v: self.synthetic_output_dir.set(str(v)))
        setting("export_count", lambda v: self.export_count.set(max(1, min(100000, int(v)))))
        setting("batch", self.apply_batch_config)
        setting("seed", lambda v: self.seed_text.set(str(v)))
        setting("stack_size", lambda v: self.stack_size.set(max(1, min(5, int(v)))))
        setting("arc_height", lambda v: self.arc_height.set(max(-100.0, min(100.0, float(v)))))
        setting("blur_radius", lambda v: self.blur_radius.set(max(.2, min(8.0, float(v)))))
        setting("group_indices_text", lambda v: self.group_indices_text.set(str(v)))
        setting("delete_scope", lambda v: self.delete_scope.set(
            v if v in ("box", "group") else "box"))
        setting("saved_mode_serial", lambda v: setattr(
            self, "saved_mode_serial", max(0, int(v))))
        skipped = []
        items = data.get("saved_modes", [])
        for item in items if isinstance(items, list) else []:
            try:
                saved = {"name": str(item["name"]), "state": normalize_state(item["state"])}
            except (KeyError, TypeError, ValueError) as exc:
                name = item.get("name", "?") if isinstance(item, dict) else "?"
                skipped.append(f"{name} ({exc})")
                continue
            self.saved_modes.append(saved)
            self.saved_mode_list.insert("end", self.saved_mode_label(saved))
        messages = []
        if problems:
            messages.append("bỏ qua giá trị lỗi: " + ", ".join(problems))
        if skipped:
            messages.append("bỏ qua mode đã lưu không dùng được: " + "; ".join(skipped))
        return ("Cấu hình cũ: " + " | ".join(messages)) if messages else None

    def apply_batch_config(self, data):
        if not isinstance(data, dict):
            raise ValueError("batch phải là object")
        self.batch = BatchSettings.from_config(data)
        self.size_by_region.set(self.batch.size_mode == "region")
        # The generator drives the editor's test images and every plan, so it
        # must start from the saved settings too.
        self.generator.size_mode = self.batch.size_mode
        self.generator.angle_jitter_deg = self.batch.angle_jitter_deg
        if "count" in data:
            self.export_count.set(self.batch.count)

        def names(key):
            value = data.get(key, [])
            return [str(item) for item in value] if isinstance(value, list) else []

        self.code_checks = {name: True for name in names("codes")}
        self.code_checks.update({name: False for name in names("codes_unchecked")})
        self.background_checks = {name: True for name in names("backgrounds")}
        self.background_checks.update({name: False for name in names("backgrounds_unchecked")})

    def close(self):
        if self.busy:
            # Finish (or cancel) the running export first; close afterwards.
            self.close_requested = True
            if self.progress is not None:
                self.progress.cancel()
            return
        try:
            self.save_config()
        finally:
            self.root.destroy()

    # ---------------------------------------------------------------------- UI

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")
        for text, command in (("Thêm ảnh mẫu", self.add_samples),
                              ("Thêm background", self.add_backgrounds),
                              ("Vẽ/sửa box", self.enable_box_edit),
                              ("Undo Ctrl+Z", self.undo),
                              ("Xóa Delete", self.delete_selected),
                              ("Lưu box mẫu", self.save_sample_boxes),
                              ("Tạo background từ box", self.create_background_from_boxes),
                              ("Lưu vùng đặt", self.save_region),
                              ("TỰ ĐỘNG TẠO TẤT CẢ LỖI", self.open_batch_dialog)):
            ttk.Button(top, text=text, command=command).pack(side="left", padx=3)
        ttk.Button(top, text="Vừa ảnh", command=self.fit).pack(side="left", padx=3)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)
        side_host = ttk.Frame(body, width=450)
        body.add(side_host, weight=0)
        self.side_canvas = tk.Canvas(side_host, width=430, highlightthickness=0)
        side_scrollbar = ttk.Scrollbar(side_host, orient="vertical",
                                       command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=side_scrollbar.set)
        side_scrollbar.pack(side="right", fill="y")
        self.side_canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(self.side_canvas, padding=8)
        self.side_panel = side
        self.side_window = self.side_canvas.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda _event: self.side_canvas.configure(
            scrollregion=self.side_canvas.bbox("all")))
        self.side_canvas.bind("<Configure>", lambda event: self.side_canvas.itemconfigure(
            self.side_window, width=event.width))
        ttk.Label(side, text="1. Ảnh lấy date (code) — group_samples · ☑ = dùng khi tạo hàng loạt"
                  ).pack(anchor="w")
        self.sample_list = self._file_tree(side, self.choose_sample, self.toggle_code)
        sample_buttons = ttk.Frame(side)
        sample_buttons.pack(fill="x", pady=(2, 0))
        ttk.Button(sample_buttons, text="Tự tìm ký tự",
                   command=self.auto_propose_boxes).pack(side="left", fill="x", expand=True)
        ttk.Button(sample_buttons, text="Chọn hết",
                   command=lambda: self.check_all("code", True)).pack(side="left", padx=(4, 0))
        ttk.Button(sample_buttons, text="Bỏ hết",
                   command=lambda: self.check_all("code", False)).pack(side="left", padx=(4, 0))
        ttk.Label(side, text="2. Ảnh chứa date (background) — group_background").pack(
            anchor="w", pady=(10, 0))
        self.background_list = self._file_tree(side, self.choose_background,
                                               self.toggle_background)
        background_buttons = ttk.Frame(side)
        background_buttons.pack(fill="x", pady=(2, 0))
        ttk.Button(background_buttons, text="Vẽ vùng lần lượt",
                   command=self.start_region_queue).pack(side="left", fill="x", expand=True)
        ttk.Button(background_buttons, text="Áp vùng này cho…",
                   command=self.copy_region_to_checked).pack(side="left", fill="x", expand=True,
                                                             padx=(4, 0))
        ttk.Button(background_buttons, text="Tổng quan",
                   command=self.show_region_overview).pack(side="left", padx=(4, 0))
        background_buttons2 = ttk.Frame(side)
        background_buttons2.pack(fill="x", pady=(2, 0))
        ttk.Button(background_buttons2, text="Chọn hết",
                   command=lambda: self.check_all("background", True)).pack(
                       side="left", fill="x", expand=True)
        ttk.Button(background_buttons2, text="Bỏ hết",
                   command=lambda: self.check_all("background", False)).pack(
                       side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(background_buttons2, text="Xóa đường out-line",
                   command=self.clear_out_lines).pack(side="left", fill="x", expand=True,
                                                      padx=(4, 0))

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="Vẽ / thao tác chuột").pack(anchor="w")
        for text, value in (("Vẽ/sửa box (ảnh mẫu) hoặc vùng (background)", "edit"),
                            ("Kéo ngang chỉnh spacing/overlap", "adjust_horizontal"),
                            ("Kéo dọc chỉnh cong", "adjust_arc"),
                            ("Kẻ đường cắt", "cut_line"),
                            ("Kẻ đường out-line trên background (date chạm = lỗi out)",
                             "out_line")):
            ttk.Radiobutton(side, text=text, variable=self.draw_mode, value=value,
                            command=self._sync_view).pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="Dạng lỗi — mỗi lần chỉ chọn một").pack(anchor="w")
        for text, value in (("Không lỗi", "normal"), ("Dãn cách", "spacing"),
                            ("Dồn/chồng box", "overlap"), ("Cong lên/xuống", "arc"),
                            ("Làm mờ date", "blur"),
                            ("Cắt ngang", "cut"),
                            ("Xóa object đã đặt", "delete_objects")):
            ttk.Radiobutton(side, text=text, variable=self.mode, value=value,
                            command=self.on_mode_change).pack(anchor="w")

        export_panel = ttk.LabelFrame(side, text="Các mode lỗi tự tạo đã lưu", padding=6)
        export_panel.pack(fill="x", pady=(8, 0))
        self.saved_mode_list = tk.Listbox(export_panel, height=5, exportselection=False)
        self.saved_mode_list.pack(fill="x")
        mode_buttons = ttk.Frame(export_panel)
        mode_buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(mode_buttons, text="Lưu mode hiện tại",
                   command=self.save_current_mode).pack(side="left", fill="x", expand=True)
        ttk.Button(mode_buttons, text="Xóa mode",
                   command=self.delete_saved_mode).pack(side="left", padx=(4, 0))
        ttk.Button(export_panel, text="Làm mới để tạo mode khác",
                   command=self.reset_effect_editor).pack(fill="x", pady=(4, 0))

        self.spacing_panel = ttk.LabelFrame(side, text="Cụm box dãn/co theo trục X", padding=6)
        ttk.Label(self.spacing_panel, text="Thứ tự box, ví dụ: 0-3,5,7").pack(anchor="w")
        group_row = ttk.Frame(self.spacing_panel)
        group_row.pack(fill="x", pady=3)
        ttk.Entry(group_row, textvariable=self.group_indices_text).pack(
            side="left", fill="x", expand=True)
        ttk.Button(group_row, text="Add cụm", command=self.add_spacing_group).pack(
            side="left", padx=(4, 0))
        self.spacing_group_list = tk.Listbox(
            self.spacing_panel, height=5, exportselection=False)
        self.spacing_group_list.pack(fill="x")
        self.spacing_group_list.bind("<<ListboxSelect>>", self.choose_spacing_group)
        ttk.Button(self.spacing_panel, text="Xóa cụm đang chọn",
                   command=self.delete_spacing_group).pack(fill="x", pady=3)
        ttk.Label(self.spacing_panel,
                  text="Dãn/co đồng thời khoảng cách giữa tất cả cụm").pack(anchor="w")
        ttk.Scale(self.spacing_panel, from_=0.1, to=4.0,
                  variable=self.group_x_factor,
                  command=self.change_group_factor).pack(fill="x")
        self.group_factor_label = ttk.Label(self.spacing_panel, text="1.00×")
        self.group_factor_label.pack(anchor="e")

        self.delete_panel = ttk.LabelFrame(side, text="Xóa object trên background", padding=6)
        ttk.Label(self.delete_panel, text="Chọn phạm vi rồi click vào object").pack(anchor="w")
        ttk.Radiobutton(self.delete_panel, text="Xóa một box", variable=self.delete_scope,
                        value="box").pack(anchor="w")
        ttk.Radiobutton(self.delete_panel, text="Xóa cả cụm", variable=self.delete_scope,
                        value="group").pack(anchor="w")
        ttk.Button(self.delete_panel, text="Khôi phục tất cả object",
                   command=self.restore_deleted_objects).pack(fill="x", pady=(4, 0))
        ttk.Label(side, text="Số box/chồng (1–5)").pack(anchor="w")
        stack_spin = ttk.Spinbox(side, from_=1, to=5, textvariable=self.stack_size, width=8,
                                 command=self.show_preview)
        stack_spin.pack(anchor="w")
        # command= only fires on the arrow buttons; typed values refresh too.
        stack_spin.bind("<KeyRelease>", lambda _event: self.show_preview())
        stack_spin.bind("<FocusOut>", lambda _event: self.stack_size.set(
            read_int(self.stack_size, 3, 1, 5)))
        ttk.Label(side, text="Độ cong: âm=lên, dương=xuống").pack(anchor="w")
        ttk.Scale(side, from_=-100, to=100, variable=self.arc_height,
                  command=lambda _=None: self.show_preview()).pack(fill="x")
        ttk.Label(side, text="Độ mờ date").pack(anchor="w")
        ttk.Scale(side, from_=0.2, to=8.0, variable=self.blur_radius,
                  command=lambda _=None: self.show_preview()).pack(fill="x")
        ttk.Label(side, text="Xóa phía nào của đường cắt").pack(anchor="w", pady=(6, 0))
        for text, value in (("Không cắt", "none"), ("Phía trên", "above"),
                            ("Phía dưới", "below")):
            ttk.Radiobutton(side, text=text, variable=self.cut_side, value=value,
                            command=self.show_preview).pack(anchor="w")
        ttk.Button(side, text="Xóa các đường cắt",
                   command=self.clear_cut_lines).pack(fill="x", pady=(2, 0))
        ttk.Button(side, text="Xem ảnh mẫu", command=lambda: self.set_view("sample")).pack(fill="x", pady=(8, 2))
        ttk.Button(side, text="Xem background/preview", command=lambda: self.set_view("preview")).pack(fill="x")
        output_panel = ttk.LabelFrame(side, text="Thư mục data giả / xem trước", padding=6)
        output_panel.pack(fill="x", pady=(8, 0))
        ttk.Entry(output_panel, textvariable=self.synthetic_output_dir).pack(fill="x")
        count_row = ttk.Frame(output_panel)
        count_row.pack(fill="x", pady=(4, 0))
        ttk.Label(count_row, text="Số ảnh cho mỗi dạng lỗi").pack(side="left")
        count_spin = ttk.Spinbox(count_row, from_=1, to=100000, textvariable=self.export_count,
                                 width=9)
        count_spin.pack(side="right")
        count_spin.bind("<FocusOut>", lambda _event: self.export_count.set(
            read_int(self.export_count, 10, 1, 100000)))
        ttk.Checkbutton(output_panel, text="Cỡ ký tự theo vùng (date cao 85 % vùng)",
                        variable=self.size_by_region,
                        command=self.on_size_mode_change).pack(anchor="w", pady=(4, 0))
        seed_row = ttk.Frame(output_panel)
        seed_row.pack(fill="x", pady=(4, 0))
        ttk.Label(seed_row, text="Seed (trống = ngẫu nhiên)").pack(side="left")
        ttk.Entry(seed_row, textvariable=self.seed_text, width=12).pack(side="right")
        output_buttons = ttk.Frame(output_panel)
        output_buttons.pack(fill="x", pady=(4, 0))
        ttk.Button(output_buttons, text="Chọn thư mục",
                   command=self.choose_synthetic_output).pack(side="left", fill="x", expand=True)
        ttk.Button(output_buttons, text="Tạo ảnh kiểm tra",
                   command=self.save_synthetic_previews).pack(side="left", fill="x", expand=True,
                                                              padx=(4, 0))
        self.on_mode_change()

        self.canvas = tk.Canvas(body, background="#20242b", highlightthickness=0)
        body.add(self.canvas, weight=1)
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill="x")
        # Messages and the always-current summary live in separate labels so a
        # redraw never wipes a message (e.g. a config or auto-select notice).
        ttk.Label(status_bar, textvariable=self.status, padding=6).pack(side="left")
        ttk.Label(status_bar, textvariable=self.info, padding=6).pack(side="right")
        self.canvas.bind("<Configure>", lambda event: self.render())
        self.canvas.bind("<Button-1>", self.press)
        self.canvas.bind("<Button-3>", self.delete_sample_box)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.root.bind_all("<MouseWheel>", self.side_mousewheel, add="+")
        self.root.bind("<Control-s>", self.on_save_key)
        self.root.bind("<Control-z>", self.on_undo_key)
        self.root.bind("<Delete>", self.on_delete_key)
        for key, direction in (("q", -1), ("Q", -1), ("e", 1), ("E", 1)):
            self.root.bind(f"<KeyPress-{key}>",
                           lambda event, direction=direction: self.on_rotate_key(event, direction))
        for key, step in (("<Return>", 1), ("<Right>", 1), ("<Left>", -1)):
            self.root.bind(key, lambda event, step=step: self.on_queue_key(event, step))
        self.root.bind("<Escape>", lambda event: self.on_queue_key(event, 0))

    def _file_tree(self, parent, on_select, on_toggle):
        """A list with a checkbox column: click the box to toggle, the row to open."""
        frame = ttk.Frame(parent)
        frame.pack(fill="x")
        tree = ttk.Treeview(frame, columns=("check", "name", "status"), show="headings",
                            height=8, selectmode="browse")
        for column, text, width, stretch in (("check", "✓", 28, False),
                                             ("name", "Ảnh", 215, True),
                                             ("status", "Trạng thái", 170, True)):
            tree.heading(column, text=text)
            tree.column(column, width=width, minwidth=width if not stretch else 60,
                        stretch=stretch, anchor="center" if column == "check" else "w")
        tree.tag_configure("warn", foreground="#b26a00")
        tree.tag_configure("bad", foreground="#c62828")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        tree.pack(side="left", fill="x", expand=True)
        tree.bind("<<TreeviewSelect>>", on_select)

        def click(event):
            if tree.identify_region(event.x, event.y) == "cell" and \
                    tree.identify_column(event.x) == "#1":
                row = tree.identify_row(event.y)
                if row:
                    on_toggle(int(row))
                    return "break"
            return None

        tree.bind("<Button-1>", click)
        tree.bind("<space>", lambda _event: (on_toggle(int(tree.focus())), "break")[1]
                  if tree.focus() else None)
        return tree

    @staticmethod
    def _typing(event):
        """True when a key event belongs to a text field (Entry, Spinbox, …)."""
        return isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, tk.Text))

    def on_save_key(self, _event=None):
        if self.view.get() == "sample":
            self.save_sample_boxes()
        else:
            self.save_region()

    def on_undo_key(self, event):
        if not self._typing(event):
            self.undo()

    def on_delete_key(self, event):
        if not self._typing(event):
            self.delete_selected()

    def on_rotate_key(self, event, direction):
        if self._typing(event):
            return None
        shift = event.state & 0x0001
        self.rotate_selected(direction * math.radians(5 if shift else 1))
        return "break"

    # ------------------------------------------------------------------- files

    @staticmethod
    def _select(tree, index):
        tree.selection_set(())
        if index >= 0 and tree.exists(str(index)):
            tree.selection_set(str(index))
            tree.focus(str(index))
            tree.see(str(index))

    @staticmethod
    def _chosen(tree):
        selection = tree.selection()
        return int(selection[0]) if selection else None

    # ---------------------------------------------------- list status, checks

    def _stamp(self, *paths):
        stamp = []
        for path in paths:
            try:
                stamp.append(path.stat().st_mtime_ns)
            except OSError:
                stamp.append(None)
        return tuple(stamp)

    def code_info(self, path):
        """Status of one code: ``{"boxes", "rows", "text", "level"}``, cached."""
        label = sample_label_path(path)
        live = path == self.sample_path and self.sample_image is not None
        stamp = ("live", len(self.sample_boxes)) if live else self._stamp(path, label)
        cached = self._status_cache.get(("code", path))
        if cached and cached[0] == stamp and not live:
            return cached[1]
        try:
            code = self.current_code() if live else load_code(path, label)
        except (OSError, ValueError) as exc:
            info = {"boxes": 0, "rows": 0, "text": f"✗ lỗi đọc: {exc}", "level": "bad"}
            self._status_cache[("code", path)] = (stamp, info)
            return info
        boxes = code.boxes
        if not boxes:
            info = {"boxes": 0, "rows": 0, "text": "chưa có box", "level": "warn"}
        else:
            rows = len(split_rows(boxes))
            parts, level = [f"{len(boxes)} box · {rows} hàng"], ""
            blank = code.blank_boxes()
            if blank:
                parts.append(f"⚠ box {blank} không có nét in")
                level = "bad"
            overlaps = overlapping_boxes(boxes)
            if overlaps:
                parts.append("⚠ box chồng: " + ", ".join(f"{a}/{b}" for a, b in overlaps[:3])
                             + ("…" if len(overlaps) > 3 else ""))
                level = level or "warn"
            runnable = [mode for mode in AUTO_ERROR_MODES if code_mode_problem(code, mode) is None]
            if len(runnable) < len(AUTO_ERROR_MODES):
                parts.append("⚠ chỉ chạy được " + " / ".join(runnable))
                level = level or "warn"
            info = {"boxes": len(boxes), "rows": rows, "text": "  ".join(parts), "level": level}
        self._status_cache[("code", path)] = (stamp, info)
        return info

    def background_region(self, path):
        """The region saved for a background (None if none), cached by mtime."""
        if path == self.background_path and self.background_image is not None:
            return self.region
        label = region_path(path)
        stamp = self._stamp(path, label)
        cached = self._status_cache.get(("region", path))
        if cached and cached[0] == stamp:
            return cached[1]
        size = self.image_size(path)
        try:
            region = read_region(label, size) if size else None
        except (OSError, ValueError, UnicodeError):
            region = None
        self._status_cache[("region", path)] = (stamp, region)
        return region

    def background_out_lines(self, path):
        """The out-lines saved for a background ([] if none), cached by mtime."""
        if path == self.background_path and self.background_image is not None:
            return self.out_lines
        lines_file = out_line_path(path)
        stamp = self._stamp(path, lines_file)
        cached = self._status_cache.get(("lines", path))
        if cached and cached[0] == stamp:
            return cached[1]
        size = self.image_size(path)
        try:
            lines = read_out_lines(lines_file, size) if size else []
        except (OSError, ValueError, TypeError, UnicodeError):
            lines = []
        self._status_cache[("lines", path)] = (stamp, lines)
        return lines

    def background_info(self, path):
        region = self.background_region(path)
        lines = self.background_out_lines(path)
        suffix = f" · {len(lines)} out-line" if lines else ""
        if region is None:
            return {"text": "✗ chưa có vùng" + suffix, "level": "warn"}
        return {"text": f"✓ vùng {region.w:.0f}×{region.h:.0f}" + suffix, "level": ""}

    def code_checked(self, path):
        if path.name in self.code_checks:
            return self.code_checks[path.name]
        return self.code_info(path)["boxes"] > 0

    def background_checked(self, path):
        return self.background_checks.get(path.name, True)

    def update_code_row(self, index):
        if not 0 <= index < len(self.samples) or not self.sample_list.exists(str(index)):
            return
        path = self.samples[index]
        info = self.code_info(path)
        self.sample_list.item(str(index), values=(
            CHECK_ON if self.code_checked(path) else CHECK_OFF, path.name, info["text"]),
            tags=(info["level"],) if info["level"] else ())

    def update_background_row(self, index):
        if not 0 <= index < len(self.backgrounds) or \
                not self.background_list.exists(str(index)):
            return
        path = self.backgrounds[index]
        info = self.background_info(path)
        self.background_list.item(str(index), values=(
            CHECK_ON if self.background_checked(path) else CHECK_OFF, path.name, info["text"]),
            tags=(info["level"],) if info["level"] else ())

    def update_rows(self):
        for index in range(len(self.samples)):
            self.update_code_row(index)
        for index in range(len(self.backgrounds)):
            self.update_background_row(index)
        if self.batch_dialog is not None:
            self.batch_dialog.schedule_refresh()

    def toggle_code(self, index):
        path = self.samples[index]
        self.code_checks[path.name] = not self.code_checked(path)
        self.update_code_row(index)
        self.after_check_change()

    def toggle_background(self, index):
        path = self.backgrounds[index]
        self.background_checks[path.name] = not self.background_checked(path)
        self.update_background_row(index)
        self.after_check_change()

    def check_all(self, kind, value):
        paths, checks = ((self.samples, self.code_checks) if kind == "code"
                         else (self.backgrounds, self.background_checks))
        for path in paths:
            checks[path.name] = value
        self.update_rows()
        self.after_check_change()

    def after_check_change(self):
        self.save_config()
        if self.batch_dialog is not None:
            self.batch_dialog.schedule_refresh()

    def checked_codes(self):
        return [path for path in self.samples if self.code_checked(path)]

    def checked_backgrounds(self):
        return [path for path in self.backgrounds if self.background_checked(path)]

    def refresh_files(self):
        self.samples, skipped_samples = image_files(SAMPLE_ROOT)
        self.backgrounds, skipped_backgrounds = image_files(BACKGROUND_ROOT)
        names = {path.name for path in self.samples}
        self.code_checks = {name: on for name, on in self.code_checks.items() if name in names}
        names = {path.name for path in self.backgrounds}
        self.background_checks = {name: on for name, on in self.background_checks.items()
                                  if name in names}
        for tree, paths in ((self.sample_list, self.samples),
                            (self.background_list, self.backgrounds)):
            tree.delete(*tree.get_children())
            for index, path in enumerate(paths):
                tree.insert("", "end", iid=str(index), values=("", path.name, ""))
        # Re-locate the loaded files by path: the sorted lists may have changed,
        # and a stale index would make autosave write another file's label.
        self.sample_index = (self.samples.index(self.sample_path)
                             if self.sample_path in self.samples else -1)
        self.background_index = (self.backgrounds.index(self.background_path)
                                 if self.background_path in self.backgrounds else -1)
        cleared = False
        if self.sample_index < 0 and self.sample_path is not None:
            self.clear_sample()
            cleared = True
        if self.background_index < 0 and self.background_path is not None:
            self.clear_background()
            cleared = True
        self._select(self.sample_list, self.sample_index)
        self._select(self.background_list, self.background_index)
        if self.samples and self.sample_index < 0:
            self.load_sample(0)
        if self.backgrounds and self.background_index < 0:
            self.load_background(0)
        if cleared:
            self.show_preview()
        self.update_rows()
        duplicates = [name for name in skipped_samples + skipped_backgrounds
                      if name not in self._warned_duplicates]
        if duplicates:
            self._warned_duplicates.update(duplicates)
            messagebox.showwarning(
                "Trùng tên ảnh",
                "Nhãn được lưu theo tên file không có đuôi, nên các ảnh sau bị bỏ qua "
                "vì trùng tên với một ảnh khác cùng thư mục:\n" + "\n".join(duplicates))

    def clear_sample(self):
        self.sample_index, self.sample_path, self.sample_image = -1, None, None
        self.sample_boxes = []
        self._asset_cache.clear()
        self.selected_box = None
        self.history.clear()
        self.spacing_groups.clear()
        self.deleted_indices.clear()
        self.refresh_spacing_groups()

    def clear_background(self):
        self.background_index, self.background_path = -1, None
        self.background_image = self.region = None
        self.out_lines = []
        self.selected_box = None
        self.history.clear()

    def _copy_images(self, target):
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        if not paths:
            return
        target.mkdir(parents=True, exist_ok=True)
        existing = {}
        for path in sorted(target.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                existing.setdefault(path.stem.casefold(), path)
        added_names, skipped = [], []
        for value in paths:
            source = Path(value)
            destination = target / source.name
            try:
                if destination.exists() and source.resolve() == destination.resolve():
                    added_names.append(destination.name)
                    continue
                clash = existing.get(source.stem.casefold())
                if clash is not None:
                    skipped.append(f"{source.name}: trùng tên với {clash.name} "
                                   f"(nhãn dùng chung {source.stem}.txt)")
                    continue
                shutil.copy2(source, destination)
            except OSError as exc:
                skipped.append(f"{source.name}: {exc}")
                continue
            existing[source.stem.casefold()] = destination
            added_names.append(destination.name)
        if skipped:
            messagebox.showwarning("Đã bỏ qua", "Không thêm được:\n" + "\n".join(skipped))
        self.refresh_files()
        if added_names and target == BACKGROUND_ROOT:
            lookup = {path.name: index for index, path in enumerate(self.backgrounds)}
            if added_names[-1] in lookup:
                self.load_background(lookup[added_names[-1]])
        elif added_names and target == SAMPLE_ROOT:
            lookup = {path.name: index for index, path in enumerate(self.samples)}
            if added_names[-1] in lookup:
                self.load_sample(lookup[added_names[-1]])

    def add_samples(self):
        self._copy_images(SAMPLE_ROOT)

    def add_backgrounds(self):
        self._copy_images(BACKGROUND_ROOT)

    def choose_sample(self, _event=None):
        chosen = self._chosen(self.sample_list)
        if chosen is not None and chosen != self.sample_index:
            self.load_sample(chosen)
        if chosen is not None and chosen == self.sample_index:
            # The list column is narrow: show the whole status here.
            self.status.set(f"{self.sample_path.name}: {self.code_info(self.sample_path)['text']}")

    def choose_background(self, _event=None):
        chosen = self._chosen(self.background_list)
        if chosen is not None and chosen != self.background_index:
            self.load_background(chosen)

    def load_sample(self, index):
        path = self.samples[index]
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            label = sample_label_path(path)
            boxes = decode(label.read_text(encoding="utf-8-sig"), *image.size, 1) if label.exists() else []
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không mở được ảnh mẫu", f"{path.name}: {exc}")
            self._select(self.sample_list, self.sample_index)
            return False
        self.sample_index, self.sample_path = index, path
        self.sample_image, self.sample_boxes = image, boxes
        self._asset_cache.clear()
        self.selected_box = None
        self.history.clear()
        self.spacing_groups.clear()
        self.deleted_indices.clear()
        self.refresh_spacing_groups()
        self._select(self.sample_list, index)
        self.show_preview()
        return True

    def load_background(self, index):
        path = self.backgrounds[index]
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            saved = region_path(path)
            saved_boxes = decode(saved.read_text(encoding="utf-8-sig"), *image.size, 1) if saved.exists() else []
            if len(saved_boxes) > 1:
                raise ValueError("Nhãn background chỉ được chứa một OBB vùng đặt date.")
            region = saved_boxes[0] if saved_boxes else None
            out_lines = read_out_lines(out_line_path(path), image.size)
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("Không mở được background", f"{path.name}: {exc}")
            self._select(self.background_list, self.background_index)
            return False
        self.background_index, self.background_path = index, path
        self.background_image, self.region = image, region
        self.out_lines = out_lines
        self.selected_box = None
        self.history.clear()
        self._select(self.background_list, index)
        message = None
        nearest = self.nearest_sample_index(image.size)
        if nearest is not None and nearest != self.sample_index:
            name = self.samples[nearest].name
            if self.sample_image is None or not self.sample_boxes:
                # Only replace a sample that cannot be used anyway; a sample
                # the user picked (with its groups and deletions) is kept.
                if self.load_sample(nearest):
                    message = f"Đã tự chọn mẫu gần kích thước background nhất: {name}"
            else:
                message = (f"Gợi ý: mẫu có kích thước gần background nhất là {name} "
                           "(vẫn giữ mẫu đang chọn)")
        self.show_preview()
        if message:
            self.status.set(message)
        return True

    def image_size(self, path):
        """Return an image's size, cached by modification time."""
        try:
            stamp = path.stat().st_mtime_ns
        except OSError:
            return None
        cached = self._size_cache.get(path)
        if cached and cached[0] == stamp:
            return cached[1]
        try:
            with Image.open(path) as image:
                size = image.size
        except OSError:
            return None
        self._size_cache[path] = (stamp, size)
        return size

    def nearest_sample_index(self, background_size):
        candidates = []
        background_width, background_height = background_size
        for index, path in enumerate(self.samples):
            label = sample_label_path(path)
            try:
                if not label.exists() or not label.read_text(encoding="utf-8-sig").strip():
                    continue
            except (OSError, UnicodeError):
                continue
            size = self.image_size(path)
            if size is None:
                continue
            width, height = size
            score = (abs(math.log(width / background_width))
                     + abs(math.log(height / background_height))
                     + abs(math.log((width / height) / (background_width / background_height))))
            candidates.append((score, path.name.casefold(), index))
        return min(candidates)[2] if candidates else None

    def write_sample_labels(self):
        atomic_write(sample_label_path(self.sample_path),
                     encode(self.sample_boxes, *self.sample_image.size))

    def save_sample_boxes(self):
        if self.sample_image is None or self.sample_path is None:
            return
        try:
            self.write_sample_labels()
            self.status.set(f"Đã lưu {len(self.sample_boxes)} box mẫu")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không lưu được box", str(exc))

    def autosave_sample_boxes(self):
        if self.sample_image is None or self.sample_path is None:
            return
        try:
            self.write_sample_labels()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không tự động lưu được box mẫu", str(exc))
        self.update_code_row(self.sample_index)
        if self.batch_dialog is not None:
            self.batch_dialog.schedule_refresh()

    def write_region_label(self):
        path = region_path(self.background_path)
        if self.region is None:
            if path.exists():
                atomic_write(path, "")
        else:
            atomic_write(path, encode([self.region], *self.background_image.size))

    def save_region(self):
        if self.region is None or self.background_path is None:
            messagebox.showinfo("Thiếu vùng", "Hãy vẽ vùng đặt date trên background.")
            return
        try:
            self.write_region_label()
            self.status.set("Đã lưu vùng đặt date")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không lưu được vùng đặt", str(exc))

    def autosave_region(self):
        if self.background_image is None or self.background_path is None:
            return
        try:
            self.write_region_label()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không tự động lưu được vùng đặt", str(exc))
        self.update_background_row(self.background_index)
        if self.batch_dialog is not None:
            self.batch_dialog.schedule_refresh()

    def autosave_out_lines(self):
        if self.background_image is None or self.background_path is None:
            return
        path = out_line_path(self.background_path)
        try:
            if self.out_lines:
                atomic_write(path, encode_out_lines(self.out_lines, self.background_image.size))
            elif path.exists():
                path.unlink()
        except OSError as exc:
            messagebox.showerror("Không tự động lưu được đường out-line", str(exc))
        self.update_background_row(self.background_index)
        if self.batch_dialog is not None:
            self.batch_dialog.schedule_refresh()

    def create_background_from_boxes(self):
        """Remove the boxed characters of every checked code and save each
        result as a new background ``<code>_background_NNNNNN.png``."""
        paths = [path for path in self.checked_codes() if self.code_info(path)["boxes"]]
        if not paths and self.sample_path is not None and self.sample_boxes:
            paths = [self.sample_path]  # nothing checked: use the open code
        if not paths:
            messagebox.showinfo(
                "Thiếu box ký tự",
                "Hãy tick (☑) các code có box ký tự, hoặc vẽ box quanh các ký tự "
                "của ảnh đang mở trước khi tạo background.")
            return
        created, failures = [], []
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            BACKGROUND_ROOT.mkdir(parents=True, exist_ok=True)
            # Any extension counts: labels are keyed on the stem only.
            taken = {path.stem.casefold() for path in BACKGROUND_ROOT.iterdir() if path.is_file()}
            for path in paths:
                try:
                    code = (self.current_code() if path == self.sample_path
                            else load_code(path, sample_label_path(path)))
                    result = erase_boxes(code.image, code.boxes)
                    index = 0
                    while f"{path.stem}_background_{index:06d}".casefold() in taken:
                        index += 1
                    destination = BACKGROUND_ROOT / f"{path.stem}_background_{index:06d}.png"
                    result.save(destination)
                    taken.add(destination.stem.casefold())
                    created.append(destination)
                except (OSError, ValueError) as exc:
                    failures.append(f"{path.name}: {exc}")
        finally:
            self.root.configure(cursor="")
        self.refresh_files()
        if len(created) == 1:
            lookup = {path.name: position for position, path in enumerate(self.backgrounds)}
            if created[0].name in lookup:
                self.load_background(lookup[created[0].name])
                self.set_view("preview")
        self.status.set(f"Đã tạo {len(created)} background")
        lines = [f"Đã tạo {len(created)} background từ {len(paths)} code:"]
        lines += [f"  {path.name}" for path in created]
        if created:
            lines.append("Hãy vẽ vùng đặt date cho chúng (nút 'Vẽ vùng lần lượt').")
        if failures:
            lines.append("Lỗi:\n" + "\n".join(failures))
        (messagebox.showwarning if failures else messagebox.showinfo)(
            "Tạo background từ box", "\n".join(lines))
    # ------------------------------------------------------------- edit state

    def _sync_view(self):
        target = {"cut_line": "preview", "out_line": "preview", "adjust_horizontal": "preview",
                  "adjust_arc": "preview"}.get(self.draw_mode.get())
        # "edit" draws/edits in whichever view is shown, like the toolbar button.
        if target:
            self.set_view(target)

    def enable_box_edit(self):
        self.draw_mode.set("edit")
        self.selected_box = None
        self.status.set("Kéo ở chỗ trống để vẽ · bấm vào box để chọn, kéo để di chuyển, "
                        "kéo góc để resize, kéo nút tròn hoặc Q/E để xoay")
        self.render()

    def rotate_selected(self, delta):
        """Rotate the selected sample box / region by ``delta`` radians (Q/E keys)."""
        boxes = self.editable_boxes()
        index = self.selected_box
        if self.gesture or index is None or not 0 <= index < len(boxes):
            if not self.gesture:
                self.status.set("Hãy bấm chọn một box hoặc vùng trước khi xoay bằng Q/E")
            return
        image = self.current_image()
        box = copy.deepcopy(boxes[index])
        box.angle = normalize_angle(box.angle + delta)
        if image is None or not box.valid(*image.size):
            self.status.set("Không xoay được: box sẽ vượt ra ngoài ảnh")
            return
        # Holding Q/E (or tapping it quickly) on the same box is one undo step.
        key, now = (self.view.get(), index), time.monotonic()
        if self._rotate_key != key or now - self._rotate_time > 1.0:
            self.snapshot()
        self._rotate_key, self._rotate_time = key, now
        self.replace_editable_box(index, box)
        if self.view.get() == "sample":
            self.autosave_sample_boxes()
        else:
            self.autosave_region()
        self.show_preview()
        self.status.set(f"Góc {math.degrees(box.angle):.1f}° · Q/E xoay 1°, Shift+Q/E xoay 5°")

    def on_mode_change(self, refresh=True):
        mode = self.mode.get()
        if mode != "delete_objects":
            self.last_effect_mode = mode
        if mode == "spacing":
            self.spacing_panel.pack(fill="x", pady=8)
        else:
            self.spacing_panel.pack_forget()
        if mode == "delete_objects":
            self.delete_panel.pack(fill="x", pady=8)
            self.set_view("preview")
        else:
            self.delete_panel.pack_forget()
        if refresh and hasattr(self, "canvas"):
            self.show_preview()

    def set_mode(self, value, refresh=True):
        """Change the error mode exactly as the radio buttons do."""
        self.mode.set(value)
        self.on_mode_change(refresh)

    def parse_group_indices(self, text):
        indices = []
        for part in text.replace(" ", "").split(","):
            if not part:
                continue
            try:
                if "-" in part:
                    fields = part.split("-")
                    if len(fields) != 2:
                        raise ValueError
                    start, end = map(int, fields)
                    if end < start:
                        start, end = end, start
                    indices.extend(range(start, end + 1))
                else:
                    indices.append(int(part))
            except ValueError:
                raise ValueError(f"Không hiểu '{part}'. Ví dụ hợp lệ: 0-3,5,7") from None
        unique = list(dict.fromkeys(indices))
        if not unique:
            raise ValueError("Cụm phải chứa ít nhất một box.")
        invalid = [index for index in unique if not 0 <= index < len(self.sample_boxes)]
        if invalid:
            raise ValueError(
                f"Box ngoài phạm vi 0-{max(0, len(self.sample_boxes) - 1)}: {invalid}")
        return unique

    def add_spacing_group(self):
        try:
            indices = self.parse_group_indices(self.group_indices_text.get())
        except ValueError as exc:
            messagebox.showerror("Cụm box không hợp lệ", str(exc))
            return
        used = {index for group in self.spacing_groups for index in group}
        duplicated = [index for index in indices if index in used]
        if duplicated:
            messagebox.showerror(
                "Box đã thuộc cụm khác",
                f"Mỗi box chỉ được thuộc một cụm. Bị trùng: {duplicated}")
            return
        self.snapshot()
        self.spacing_groups.append(indices)
        self.refresh_spacing_groups()
        self.spacing_group_list.selection_set(len(self.spacing_groups) - 1)
        self.set_group_factor(1.0)
        self.show_preview()

    def refresh_spacing_groups(self):
        if not hasattr(self, "spacing_group_list"):
            return
        self.spacing_group_list.delete(0, "end")
        for number, indices in enumerate(self.spacing_groups, 1):
            self.spacing_group_list.insert(
                "end", f"Cụm {number}: {','.join(map(str, indices))}")

    def choose_spacing_group(self, _event=None):
        chosen = self.spacing_group_list.curselection()
        if chosen:
            self.group_indices_text.set(
                ",".join(map(str, self.spacing_groups[chosen[0]])))

    def set_group_factor(self, factor):
        """Set the spacing factor and its label without recomposing."""
        self.group_x_factor.set(factor)
        self.group_factor_label.configure(text=f"{factor:.2f}×")

    def change_group_factor(self, value):
        self.group_factor_label.configure(text=f"{float(value):.2f}×")
        self.show_preview()

    def delete_spacing_group(self):
        chosen = self.spacing_group_list.curselection()
        if not chosen:
            return
        self.snapshot()
        self.spacing_groups.pop(chosen[0])
        self.refresh_spacing_groups()
        self.show_preview()

    def restore_deleted_objects(self):
        if not self.deleted_indices:
            return
        self.snapshot()
        self.deleted_indices.clear()
        self.show_preview()

    def delete_placed_object(self, point):
        for preview_index in range(len(self.preview_boxes) - 1, -1, -1):
            if not self.preview_boxes[preview_index].contains(*point):
                continue
            if preview_index >= len(self.preview_source_indices):
                return
            source_index = self.preview_source_indices[preview_index]
            to_delete = {source_index}
            if self.delete_scope.get() == "group":
                for group in self.spacing_groups:
                    if source_index in group:
                        to_delete = set(group)
                        break
            self.snapshot()
            self.deleted_indices.update(to_delete)
            self.show_preview()
            self.status.set(
                f"Đã xóa {len(to_delete)} object khỏi preview; Ctrl+Z để hoàn tác")
            return

    def forget_sample_index(self, removed):
        """Keep index-based references valid after sample box ``removed`` is gone."""
        def shifted(index):
            return index - 1 if index > removed else index

        groups = [[shifted(index) for index in group if index != removed]
                  for group in self.spacing_groups]
        self.spacing_groups = [group for group in groups if group]
        self.deleted_indices = {shifted(index) for index in self.deleted_indices
                                if index != removed}
        self.refresh_spacing_groups()
        # Saved modes are re-matched by box position (box_refs) when used.

    def history_entry(self):
        return {
            "sample_boxes": copy.deepcopy(self.sample_boxes),
            "region": copy.deepcopy(self.region),
            "cut_lines": copy.deepcopy(self.cut_lines),
            "out_lines": copy.deepcopy(self.out_lines),
            "deleted_indices": set(self.deleted_indices),
            "spacing_groups": copy.deepcopy(self.spacing_groups),
        }

    def push_history(self, entry=None):
        self._rotate_key = None  # any other change ends a Q/E rotation run
        self.history.append(entry if entry is not None else self.history_entry())
        del self.history[:-100]

    def snapshot(self):
        self.push_history()

    def undo(self):
        if not self.history:
            self.status.set("Không còn thao tác để hoàn tác")
            return
        entry = self.history.pop()
        sample_changed = entry["sample_boxes"] != self.sample_boxes
        region_changed = entry["region"] != self.region
        lines_changed = entry["out_lines"] != self.out_lines
        self.sample_boxes = entry["sample_boxes"]
        self.region = entry["region"]
        self.cut_lines = entry["cut_lines"]
        self.out_lines = entry["out_lines"]
        self.deleted_indices = entry["deleted_indices"]
        self.spacing_groups = entry["spacing_groups"]
        self.refresh_spacing_groups()
        self.selected_box = None
        if sample_changed:
            self.autosave_sample_boxes()
        if region_changed:
            self.autosave_region()
        if lines_changed:
            self.autosave_out_lines()
        self.show_preview()
        self.status.set("Đã hoàn tác thao tác gần nhất")

    def delete_selected(self):
        if self.selected_box is None:
            self.status.set("Hãy chọn box hoặc vùng cần xóa")
            return
        index, self.selected_box = self.selected_box, None
        if self.view.get() == "sample":
            if not 0 <= index < len(self.sample_boxes):
                return
            self.snapshot()
            self.sample_boxes.pop(index)
            self.forget_sample_index(index)
            self.autosave_sample_boxes()
        else:
            if self.region is None or index != 0:
                return
            self.snapshot()
            self.region = None
            self.autosave_region()
        self.show_preview()
        self.status.set("Đã xóa; nhấn Ctrl+Z để hoàn tác")

    def set_view(self, value):
        if self.view.get() != value:
            # selected_box means "sample box N" in one view and "the region"
            # in the other, so it must not survive a view change.
            self.selected_box = None
            self.gesture = None
        self.view.set(value)
        self.fit()

    def current_image(self):
        if self.view.get() == "sample":
            return self.sample_image
        return self.preview_cache or self.background_image

    # ------------------------------------------------------------- composing

    def current_code(self):
        """The open sample as a Code; shares the box list and asset cache."""
        return Code(self.sample_path, self.sample_image, self.sample_boxes, self._asset_cache)

    def current_target(self):
        return Target(self.background_path, self.background_image, self.region, self.out_lines)

    def on_size_mode_change(self):
        self.generator.size_mode = "region" if self.size_by_region.get() else "image"
        self.batch.size_mode = self.generator.size_mode
        self.show_preview()
        if self.batch_dialog is not None:
            self.batch_dialog.schedule_refresh()

    def dimension_scale(self):
        """Sample → background pixel ratio of the open pair (see Generator.scale)."""
        if self.sample_image is None or self.background_image is None:
            return 1.0
        return self.generator.scale(self.current_code(), self.current_target())

    def asset_for(self, box):
        """character_asset() for one sample box, cached until the sample changes."""
        return self.current_code().asset(box)

    def live_state(self):
        """The effect settings currently shown in the editor."""
        state = default_state(self.mode.get())
        state.update({
            "last_effect_mode": self.last_effect_mode,
            "stack_size": read_int(self.stack_size, 3, 1, 5),
            "arc_height": read_float(self.arc_height, 35.0, -100.0, 100.0),
            "blur_radius": read_float(self.blur_radius, 2.0, .2, 8.0),
            "spacing_groups": copy.deepcopy(self.spacing_groups),
            "group_x_factor": read_float(self.group_x_factor, 1.0, .1, 4.0),
            "cut_lines": copy.deepcopy(self.cut_lines),
            "cut_side": self.cut_side.get(),
            "deleted_indices": set(self.deleted_indices),
        })
        return state

    def compose(self, state, region):
        """Render ``state`` for the open pair with the date placed in ``region``."""
        return self.generator.compose(self.current_code(), self.current_target(), state, region)
    def show_preview(self):
        result = self.compose(self.live_state(), self.region)
        self.preview_cache = result.image
        self.preview_boxes = result.boxes
        self.preview_source_indices = result.sources
        self.preview_blank = result.blank
        self.update_info()
        self.render()

    def preview_fits(self):
        return self.region is None or all(
            box_inside_box(box, self.region) for box in self.preview_boxes)

    def preview_out_hits(self):
        """Preview boxes touching an out-line (the date would be "out")."""
        return out_line_hits(self.preview_boxes, self.out_lines)

    def update_info(self):
        parts = [f"Mẫu: {len(self.sample_boxes)} box",
                 "Background: " + ("đã có vùng đặt" if self.region else "chưa có vùng đặt"),
                 f"Dạng: {self.mode.get()}"]
        if self.sample_image is not None and self.background_image is not None:
            parts.append(f"Đồng bộ size {self.dimension_scale():.3f}×")
        if self.preview_blank:
            parts.append(f"CẢNH BÁO: box {self.preview_blank} không tách được nét in")
        if self.region and self.preview_boxes and not self.preview_fits():
            parts.append("CẢNH BÁO: date vượt vùng chứa")
        if self.out_lines:
            parts.append(f"{len(self.out_lines)} đường out-line")
            if self.preview_out_hits():
                parts.append("CẢNH BÁO: date chạm đường out-line (= lỗi out)")
        self.info.set(" | ".join(parts))

    # ---------------------------------------------------------------- canvas

    def fit(self):
        image = self.current_image()
        if image:
            cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            self.scale = max(.001, min((cw - 30) / image.width, (ch - 30) / image.height))
            self.ox = (cw - image.width * self.scale) / 2
            self.oy = (ch - image.height * self.scale) / 2
        self.render()

    def point(self, x, y):
        return ((x - self.ox) / self.scale, (y - self.oy) / self.scale)

    def screen(self, x, y):
        return x * self.scale + self.ox, y * self.scale + self.oy

    def rotation_handle(self, box):
        distance = box.h / 2 + 28 / self.scale
        return (box.cx + math.sin(box.angle) * distance,
                box.cy - math.cos(box.angle) * distance)

    def draw_edit_handles(self, box, color):
        points = [self.screen(x, y) for x, y in box.corners()]
        for x, y in points:
            self.canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5,
                                         fill=color, outline="white")
        rx, ry = self.screen(*self.rotation_handle(box))
        mx, my = ((points[0][0] + points[1][0]) / 2,
                  (points[0][1] + points[1][1]) / 2)
        self.canvas.create_line(mx, my, rx, ry, fill=color)
        self.canvas.create_oval(rx - 6, ry - 6, rx + 6, ry + 6,
                                fill=color, outline="white")

    def editable_boxes(self):
        if self.view.get() == "sample":
            return self.sample_boxes
        return [self.region] if self.region is not None else []

    def replace_editable_box(self, index, box):
        if self.view.get() == "sample":
            self.sample_boxes[index] = box
        else:
            self.region = box

    def render(self):
        self.canvas.delete("all")
        image = self.current_image()
        if image is None:
            return
        display = image.resize((max(1, round(image.width * self.scale)),
                                max(1, round(image.height * self.scale))), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.create_image(self.ox, self.oy, image=self.photo, anchor="nw")
        if self.view.get() == "sample":
            for index, box in enumerate(self.sample_boxes):
                points = [self.screen(x, y) for x, y in box.corners()]
                self.canvas.create_polygon(*[v for point in points for v in point], fill="",
                                           outline=self.COLORS[index % len(self.COLORS)], width=2)
                self.canvas.create_text(*points[0], text=str(index), anchor="sw", fill="white")
                if self.selected_box == index:
                    self.draw_edit_handles(box, self.COLORS[index % len(self.COLORS)])
            return
        for line in self.out_lines:
            start, end = (self.screen(*point) for point in line)
            self.canvas.create_line(*start, *end, fill="#ff40ff", width=3)
        for index in self.preview_out_hits():
            points = [self.screen(x, y) for x, y in self.preview_boxes[index].corners()]
            self.canvas.create_polygon(*[value for point in points for value in point],
                                       fill="", outline="#ff5252", width=2)
        if self.region:
            region_color = ("#00e5ff" if self.preview_fits() and not self.preview_out_hits()
                            else "#ff5252")
            points = [self.screen(x, y) for x, y in self.region.corners()]
            self.canvas.create_polygon(*[value for point in points for value in point],
                                       fill="", outline=region_color, width=2, dash=(6, 4))
            if self.selected_box == 0:
                self.draw_edit_handles(self.region, region_color)
            scale = self.dimension_scale()
            for line in self.cut_lines:
                start, end = (self.screen(*local_to_image(point, self.region, scale))
                              for point in line)
                self.canvas.create_line(*start, *end, fill="#ffca28", width=3)

    def press(self, event):
        image = self.current_image()
        if image is None:
            return
        point = self.point(event.x, event.y)
        if not (0 <= point[0] <= image.width and 0 <= point[1] <= image.height):
            return
        if self.mode.get() == "delete_objects" and self.view.get() != "sample":
            self.delete_placed_object(point)
            return
        kind = self.draw_mode.get()
        if kind == "edit":
            boxes = self.editable_boxes()
            # The undo entry is pushed only once the gesture really changes
            # something, so plain clicks do not fill the history.
            before = self.history_entry()
            if self.selected_box is not None and self.selected_box < len(boxes):
                box = boxes[self.selected_box]
                if math.dist(point, self.rotation_handle(box)) * self.scale < 12:
                    self.gesture = {"kind": "edit_rotate", "start": point,
                                    "original": copy.deepcopy(box), "index": self.selected_box,
                                    "snapshot": before, "changed": False}
                    return
                for corner_index, corner in enumerate(box.corners()):
                    if math.dist(point, corner) * self.scale < 12:
                        self.gesture = {"kind": "edit_resize", "start": point,
                                        "original": copy.deepcopy(box),
                                        "index": self.selected_box, "corner": corner_index,
                                        "snapshot": before, "changed": False}
                        return
            self.selected_box = next(
                (index for index in reversed(range(len(boxes)))
                 if boxes[index].contains(*point)), None)
            if self.selected_box is not None:
                self.gesture = {"kind": "edit_move", "start": point,
                                "original": copy.deepcopy(boxes[self.selected_box]),
                                "index": self.selected_box, "snapshot": before,
                                "changed": False}
            else:
                # Empty spot: start drawing a new sample box, or a new region
                # (replacing the old one) on the background.
                self.gesture = {"kind": "sample_box" if self.view.get() == "sample"
                                else "target_region", "start": point}
            self.render()
            return
        needed = {"cut_line": "preview", "out_line": "preview"}.get(kind)
        if needed and self.view.get() != needed:
            self.status.set(
                "Thao tác này dùng trên " + ("ảnh mẫu: bấm 'Xem ảnh mẫu'" if needed == "sample"
                                             else "background: bấm 'Xem background/preview'"))
            return
        self.gesture = {"kind": kind, "start": point}

    def delete_sample_box(self, event):
        if self.view.get() != "sample" or self.sample_image is None:
            return
        point = self.point(event.x, event.y)
        for index in range(len(self.sample_boxes) - 1, -1, -1):
            if self.sample_boxes[index].contains(*point):
                self.selected_box = index
                self.delete_selected()
                return

    def drag(self, event):
        gesture = self.gesture
        if not gesture:
            return
        image = self.current_image()
        if image is None:
            return
        kind, start = gesture["kind"], gesture["start"]
        point = self.point(event.x, event.y)
        if kind.startswith("edit_"):
            original, index = gesture["original"], gesture["index"]
            box = copy.deepcopy(original)
            if kind == "edit_move":
                box.cx += point[0] - start[0]
                box.cy += point[1] - start[1]
            elif kind == "edit_rotate":
                # Keep the angle in the same range a reload would give, so the
                # box/region does not flip 180° after saving.
                box.angle = normalize_angle(
                    original.angle
                    + math.atan2(point[1] - box.cy, point[0] - box.cx)
                    - math.atan2(start[1] - box.cy, start[0] - box.cx))
            elif kind == "edit_resize":
                corner_index = gesture["corner"]
                anchor = original.corners()[(corner_index + 2) % 4]
                cosine, sine = math.cos(box.angle), math.sin(box.angle)
                dx, dy = point[0] - anchor[0], point[1] - anchor[1]
                local_x = dx * cosine + dy * sine
                local_y = -dx * sine + dy * cosine
                signs = ((-1, -1), (1, -1), (1, 1), (-1, 1))[corner_index]
                if local_x * signs[0] < 1 or local_y * signs[1] < 1:
                    return
                box.w, box.h = abs(local_x), abs(local_y)
                box.cx, box.cy = (anchor[0] + point[0]) / 2, (anchor[1] + point[1]) / 2
            if box.valid(*image.size):
                if not gesture["changed"]:
                    self.push_history(gesture["snapshot"])
                    gesture["changed"] = True
                self.replace_editable_box(index, box)
                self.show_preview()
        elif kind == "adjust_horizontal":
            delta = point[0] - start[0]
            if self.mode.get() == "spacing":
                self.set_group_factor(max(.1, min(4.0, 1.0 + delta / 100)))
            elif self.mode.get() == "overlap":
                self.stack_size.set(max(1, min(5, 1 + round(abs(delta) / 25))))
            self.show_preview()
        elif kind == "adjust_arc":
            self.arc_height.set(max(-100, min(100, point[1] - start[1])))
            self.show_preview()
        else:
            self.canvas.delete("gesture_preview")
            if kind in ("cut_line", "out_line"):
                self.canvas.create_line(*self.screen(*start), *self.screen(*point),
                                        fill="white" if kind == "cut_line" else "#ff40ff",
                                        width=2, dash=(5, 3), tags="gesture_preview")
            else:
                self.canvas.create_rectangle(*self.screen(*start), *self.screen(*point),
                                             outline="white", dash=(5, 3), tags="gesture_preview")

    def release(self, event):
        gesture, self.gesture = self.gesture, None
        if not gesture:
            return
        kind, start = gesture["kind"], gesture["start"]
        point = self.point(event.x, event.y)
        if kind in ("sample_box", "target_region") and min(
                abs(point[0] - start[0]), abs(point[1] - start[1])) * self.scale < 4:
            # A click (or a tiny slip of the mouse) on an empty spot only
            # deselects; it must not create a box or replace the region.
            self.render()
            return
        if kind == "sample_box" and self.view.get() == "sample" and self.sample_image is not None:
            box = Box(0, (start[0] + point[0]) / 2, (start[1] + point[1]) / 2,
                      abs(point[0] - start[0]), abs(point[1] - start[1]))
            if box.valid(*self.sample_image.size):
                self.snapshot()
                self.sample_boxes.append(box)
                self.selected_box = len(self.sample_boxes) - 1
                self.autosave_sample_boxes()
        elif (kind == "target_region" and self.view.get() == "preview"
              and self.background_image is not None):
            region = Box(0, (start[0] + point[0]) / 2,
                         (start[1] + point[1]) / 2,
                         abs(point[0] - start[0]), abs(point[1] - start[1]))
            if region.valid(*self.background_image.size):
                self.snapshot()
                self.region = region
                self.selected_box = 0
                self.autosave_region()
                if self.region_queue is not None:
                    # Region queue: the rectangle is saved, show it briefly
                    # and move on to the next background.
                    self.show_preview()
                    self.root.after(350, lambda: self.queue_step(1)
                                    if self.region_queue is not None else None)
                    return
        elif kind == "cut_line" and self.view.get() == "preview":
            self.add_cut_line(start, point)
        elif kind == "out_line" and self.view.get() == "preview":
            self.add_out_line(start, point)
        elif kind.startswith("edit_") and gesture["changed"]:
            if self.view.get() == "sample":
                self.autosave_sample_boxes()
            else:
                self.autosave_region()
        self.show_preview()

    def add_cut_line(self, start, end):
        if math.dist(start, end) * self.scale < 4:
            self.status.set("Đường cắt quá ngắn: hãy giữ chuột và kéo để kẻ")
            return
        if self.region is None or self.sample_image is None or self.background_image is None:
            self.status.set("Hãy chọn ảnh mẫu và vẽ vùng đặt trước khi kẻ đường cắt")
            return
        scale = self.dimension_scale()
        line = oriented_line([image_to_local(point, self.region, scale)
                              for point in (start, end)])
        self.snapshot()
        self.cut_lines.append(line)
        if self.cut_side.get() == "none":
            self.cut_side.set("above")
        self.set_mode("cut", refresh=False)
        self.status.set(f"Đã kẻ {len(self.cut_lines)} đường cắt; chọn phía cần xóa")

    def add_out_line(self, start, end):
        """Add an out-line to the open background (saved at once)."""
        if self.background_image is None or self.background_path is None:
            self.status.set("Hãy chọn background trước khi kẻ đường out-line")
            return
        if math.dist(start, end) * self.scale < 4:
            self.status.set("Đường out-line quá ngắn: hãy giữ chuột và kéo để kẻ")
            return
        width, height = self.background_image.size
        line = tuple((max(0.0, min(width, x)), max(0.0, min(height, y))) for x, y in (start, end))
        self.snapshot()
        self.out_lines.append(line)
        self.autosave_out_lines()
        self.status.set(f"Background có {len(self.out_lines)} đường out-line: date chạm "
                        "đường = lỗi out (dạng auto out_line); Ctrl+Z để hoàn tác")

    def clear_out_lines(self):
        if not self.out_lines:
            self.status.set("Background đang mở chưa có đường out-line")
            return
        self.snapshot()
        self.out_lines = []
        self.autosave_out_lines()
        self.show_preview()
        self.status.set("Đã xóa các đường out-line của background này; Ctrl+Z để hoàn tác")

    def clear_cut_lines(self):
        if not self.cut_lines:
            return
        self.snapshot()
        self.cut_lines = []
        self.show_preview()
        self.status.set("Đã xóa các đường cắt; Ctrl+Z để hoàn tác")

    def zoom(self, event):
        image = self.current_image()
        if image:
            point = self.point(event.x, event.y)
            self.scale = max(.02, min(20, self.scale * (1.15 if event.delta > 0 else 1 / 1.15)))
            self.ox, self.oy = event.x - point[0] * self.scale, event.y - point[1] * self.scale
            self.render()

    def side_mousewheel(self, event):
        widget = event.widget
        while widget is not None:
            if widget is self.side_panel or widget is self.side_canvas:
                units = -1 if event.delta > 0 else 1
                self.side_canvas.yview_scroll(units * 3, "units")
                return "break"
            widget = getattr(widget, "master", None)

    # ------------------------------------------------------------ saved modes

    def saved_state_from_live(self):
        """Capture the editor state in the portable format stored in the config."""
        state = self.live_state()
        mode = state["mode"]
        effect = state["last_effect_mode"] if mode == "delete_objects" else mode
        if effect != "spacing":
            state["spacing_groups"] = []
        if mode != "delete_objects":
            state["deleted_indices"] = set()
        if effect != "cut":
            state["cut_lines"], state["cut_side"] = [], "none"
        state["sample"] = self.sample_path.name if self.sample_path else ""
        state["box_refs"] = [[box.cx, box.cy] for box in self.sample_boxes]
        return state

    def effect_problem(self, state):
        """Return why ``state`` would not produce a visible error, or None."""
        return self.generator.effect_problem(self.current_code(), self.current_target(), state)

    def resolve_saved_state(self, state):
        """Map a saved state's box indices onto the current sample boxes."""
        return self.generator.resolve_saved(self.current_code(), state)

    @staticmethod
    def saved_mode_label(saved):
        state = saved["state"]
        if state_uses_indices(state) and state["sample"]:
            return f"{saved['name']}  [{state['sample']}]"
        return saved["name"]

    def save_current_mode(self):
        if (self.sample_image is None or self.background_image is None
                or not self.sample_boxes or self.region is None):
            messagebox.showinfo("Thiếu dữ liệu",
                                "Hãy chọn ảnh mẫu có box và background có vùng đặt trước khi lưu mode.")
            return
        state = self.saved_state_from_live()
        problem = self.effect_problem(state)
        if problem:
            messagebox.showwarning("Mode chưa tạo ra lỗi", problem)
            return
        effective = state["mode"]
        if effective == "delete_objects" and state["last_effect_mode"] != "normal":
            effective = f"{state['last_effect_mode']}_delete"
        names = {item["name"] for item in self.saved_modes}
        while True:
            self.saved_mode_serial += 1
            name = f"mode_{self.saved_mode_serial:02d}_{effective}"
            if name not in names:
                break
        saved = {"name": name, "state": state}
        self.saved_modes.append(saved)
        self.saved_mode_list.insert("end", self.saved_mode_label(saved))
        self.saved_mode_list.selection_clear(0, "end")
        self.saved_mode_list.selection_set("end")
        self.save_config()
        self.rebuild_batch_dialog()
        self.status.set(f"Đã lưu {name}; có thể làm mới và tạo dạng tiếp theo")

    def rebuild_batch_dialog(self):
        """Show a changed list of saved modes in an open batch dialog."""
        if self.batch_dialog is not None:
            self.batch_dialog.close()
            self.batch_dialog = BatchDialog(self)

    def delete_saved_mode(self):
        selection = self.saved_mode_list.curselection()
        if not selection:
            return
        index = selection[0]
        self.saved_modes.pop(index)
        self.saved_mode_list.delete(index)
        self.save_config()
        self.rebuild_batch_dialog()
        self.status.set(f"Còn {len(self.saved_modes)} mode đã lưu")

    def reset_effect_editor(self):
        self.snapshot()
        self.last_effect_mode = "normal"
        self.mode.set("normal")
        self.stack_size.set(3)
        self.arc_height.set(35.0)
        self.blur_radius.set(2.0)
        self.spacing_groups = []
        self.refresh_spacing_groups()
        self.set_group_factor(1.0)
        self.cut_lines = []
        self.cut_side.set("none")
        self.deleted_indices = set()
        self.on_mode_change()
        self.status.set("Đã làm mới; hãy tạo dạng lỗi tiếp theo rồi bấm Lưu mode hiện tại")

    # ----------------------------------------------------------------- export

    def choose_synthetic_output(self, parent=None):
        directory = filedialog.askdirectory(
            parent=parent or self.root,
            title="Chọn thư mục lưu data giả",
            initialdir=self.synthetic_output_dir.get() or str(OUTPUT_ROOT))
        if directory:
            self.synthetic_output_dir.set(directory)
            self.save_config()
        return bool(directory)

    def synthetic_target(self):
        value = self.synthetic_output_dir.get().strip()
        if not value:
            self.choose_synthetic_output()
            value = self.synthetic_output_dir.get().strip()
        return Path(value).resolve() if value else None

    def validate_synthetic_source(self):
        if self.sample_image is None or self.background_image is None:
            messagebox.showinfo("Thiếu ảnh", "Hãy chọn cả ảnh mẫu và background.")
            return False
        if not self.sample_boxes or self.region is None:
            messagebox.showinfo("Thiếu nhãn", "Hãy vẽ box date và vùng đặt date.")
            return False
        blank = self.current_code().blank_boxes()
        if blank:
            messagebox.showerror(
                "Box không có nét in",
                f"Không tách được nét in khỏi nền giấy ở box {blank}, nên các object này "
                "sẽ có nhãn nhưng không nhìn thấy. Hãy vẽ lại các box này sát ký tự.")
            return False
        return True

    def export_seed(self):
        text = self.seed_text.get().strip()
        if not text:
            return random.SystemRandom().randrange(2 ** 31)
        try:
            return int(text)
        except ValueError:
            messagebox.showerror("Seed không hợp lệ", "Seed phải là số nguyên hoặc để trống.")
            return None

    # The open pair, through the engine (kept for the editor and scripts).

    def auto_arc_range(self):
        return self.generator.auto_arc_range(self.current_code(), self.current_target())

    def auto_mode_problem(self, mode):
        return self.generator.mode_problem(self.current_code(), self.current_target(), mode)

    def automatic_cut_lines(self, mode, rng):
        return self.generator.automatic_cut_lines(self.current_code(), mode, rng)

    def automatic_error_state(self, mode, variant, rng):
        return self.generator.auto_state(self.current_code(), self.current_target(),
                                         mode, variant, rng)

    def export_jobs(self):
        """Return ``(jobs, skipped)`` for the open pair: automatic modes plus
        the saved modes that apply. ``skipped`` holds "name: reason" texts."""
        jobs, skipped = self.generator.jobs(self.current_code(), self.current_target(),
                                            self.saved_modes)
        return list(jobs.values()), [f"{name}: {reason}" for name, reason in skipped.items()]

    def jittered_region(self, rng):
        return self.generator.jittered_region(self.region, rng)

    def generate_unique(self, job, variant, rng, fingerprints):
        return self.generator.generate(self.current_code(), self.current_target(),
                                       job, variant, rng, fingerprints)

    def save_synthetic_previews(self):
        """Save one image per job for the open pair: exactly what a batch run
        with the same seed makes as variant 0 of this pair."""
        if self.busy or not self.validate_synthetic_source():
            return
        seed = self.export_seed()
        if seed is None:
            return
        target = self.synthetic_target()
        if target is None:
            return
        preview_dir = target / "preview"
        jobs, skipped = self.export_jobs()
        created, failures = [], []
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            preview_dir.mkdir(parents=True, exist_ok=True)
            background_name = self.background_path.stem
            for job in jobs:
                rng = task_rng(seed, self.sample_path.name, self.background_path.name, job.name, 0)
                try:
                    result = self.generate_unique(job, 0, rng, set())
                except ValueError as exc:
                    failures.append(f"{job.name}: {exc}")
                    continue
                prefix = f"preview_{background_name}_{job.name}"
                key = f"{prefix}_{first_free_number(prefix, ((preview_dir, '.png'),)):06d}"
                result.image.save(preview_dir / f"{key}.png")
                created.append(key)
        except OSError as exc:
            failures.append(str(exc))
        finally:
            self.root.configure(cursor="")
        self.status.set(f"Đã tạo {len(created)} ảnh kiểm tra tại {preview_dir}")
        lines = [f"Đã lưu {len(created)} ảnh (mỗi dạng lỗi một ảnh) tại:\n{preview_dir}",
                 f"Seed: {seed} — nhập seed này vào ô Seed để xuất đúng các biến thể vừa xem."]
        if skipped:
            lines.append("Bỏ qua:\n" + "\n".join(skipped))
        if failures:
            lines.append("Lỗi:\n" + "\n".join(failures))
        (messagebox.showwarning if failures else messagebox.showinfo)(
            "Ảnh kiểm tra", "\n\n".join(lines))

    # ------------------------------------------------- auto-propose characters

    def auto_propose_boxes(self):
        """Run best_char_obb.pt on the open code and add its detections as
        normal, editable boxes."""
        if self.busy or self.sample_image is None or self.sample_path is None:
            messagebox.showinfo("Thiếu ảnh mẫu", "Hãy chọn ảnh mẫu (code) trước.")
            return
        if not MODEL_PATH.is_file():
            messagebox.showerror("Thiếu model", f"Không tìm thấy model: {MODEL_PATH}")
            return
        replace = True
        if self.sample_boxes:
            answer = messagebox.askyesnocancel(
                "Tự tìm ký tự",
                f"Ảnh đang có {len(self.sample_boxes)} box.\n\n"
                "Yes: thay toàn bộ bằng box tìm được\n"
                "No: chỉ thêm box mới không trùng box cũ\n"
                "Cancel: thôi")
            if answer is None:
                return
            replace = answer
        self.status.set("Đang chạy model tìm ký tự… (lần đầu sẽ chậm vì phải nạp thư viện)")
        self.root.configure(cursor="watch")
        self.root.update()
        try:
            found = propose_character_boxes(self.sample_image, MODEL_PATH)
        except ImportError as exc:
            messagebox.showerror(
                "Thiếu thư viện",
                f"Cần cài ultralytics để tự tìm ký tự ({exc}).\n"
                "Vẫn có thể vẽ box bằng tay.")
            return
        except Exception as exc:  # noqa: BLE001  (model/device errors vary)
            messagebox.showerror("Không chạy được model", f"{type(exc).__name__}: {exc}")
            return
        finally:
            self.root.configure(cursor="")
        if not found:
            self.status.set("Model không tìm thấy ký tự nào")
            messagebox.showinfo("Tự tìm ký tự", "Model không tìm thấy ký tự nào; hãy vẽ box bằng tay.")
            return
        self.snapshot()
        if replace:
            self.sample_boxes = found
            self.spacing_groups, self.deleted_indices = [], set()
            self.refresh_spacing_groups()
            added = len(found)
        else:
            new = [box for box in found
                   if not any(box.contains(old.cx, old.cy) or old.contains(box.cx, box.cy)
                              for old in self.sample_boxes)]
            self.sample_boxes.extend(new)
            added = len(new)
        self.selected_box = None
        self.autosave_sample_boxes()
        self.set_view("sample")
        self.draw_mode.set("edit")
        self.show_preview()
        self.status.set(f"Model đề xuất {added} box: hãy kiểm tra, sửa/xóa box sai "
                        "(chuột phải để xóa) và vẽ thêm box còn thiếu; Ctrl+Z để hoàn tác")

    # -------------------------------------------------------- region helpers

    def start_region_queue(self):
        """Open, one after another, every checked background without a region."""
        paths = [path for path in self.checked_backgrounds()
                 if self.background_region(path) is None]
        if not paths:
            messagebox.showinfo("Vẽ vùng lần lượt",
                                "Mọi background đang được chọn (☑) đều đã có vùng đặt date.")
            return
        self.region_queue = {"paths": paths, "position": 0}
        self.open_queue_item()

    def open_queue_item(self):
        queue = self.region_queue
        path = queue["paths"][queue["position"]]
        if path not in self.backgrounds:
            self.stop_region_queue("Background trong hàng đợi không còn tồn tại")
            return
        if self.backgrounds.index(path) != self.background_index:
            self.load_background(self.backgrounds.index(path))
        self.mode.set("normal")
        self.on_mode_change(refresh=False)
        self.draw_mode.set("edit")
        self.set_view("preview")
        self.show_preview()
        self.status.set(
            f"Vùng {queue['position'] + 1}/{len(queue['paths'])}: kéo chuột vẽ vùng đặt date "
            f"trên {path.name} · Enter/→ tiếp · ← lùi · Esc dừng")

    def queue_step(self, step):
        queue = self.region_queue
        if queue is None:
            return
        position = queue["position"] + step
        if position < 0:
            position = 0
        if position >= len(queue["paths"]):
            done = sum(1 for path in queue["paths"] if self.background_region(path) is not None)
            self.stop_region_queue(f"Đã xong hàng đợi: {done}/{len(queue['paths'])} "
                                   "background có vùng")
            return
        queue["position"] = position
        self.open_queue_item()

    def stop_region_queue(self, message="Đã dừng vẽ vùng lần lượt"):
        self.region_queue = None
        self.draw_mode.set("edit")
        self.status.set(message)

    def on_queue_key(self, event, step):
        if self.region_queue is None or self._typing(event) or self.busy:
            return None
        if step == 0:
            self.stop_region_queue()
        else:
            self.queue_step(step)
        return "break"

    def copy_region_to_checked(self):
        """Copy the open background's region to the other checked backgrounds,
        scaled by the image size ratio."""
        if self.region is None or self.background_path is None:
            messagebox.showinfo("Thiếu vùng", "Hãy mở một background đã có vùng đặt date.")
            return
        targets = [path for path in self.checked_backgrounds() if path != self.background_path]
        if not targets:
            messagebox.showinfo("Áp vùng", "Hãy tick (☑) các background cần áp vùng này.")
            return
        with_region = [path for path in targets if self.background_region(path) is not None]
        overwrite = False
        if with_region:
            answer = messagebox.askyesnocancel(
                "Ghi đè vùng?",
                f"{len(with_region)}/{len(targets)} background đã chọn đã có vùng:\n"
                + "\n".join(path.name for path in with_region[:8])
                + ("\n…" if len(with_region) > 8 else "")
                + "\n\nYes: ghi đè cả các vùng này\nNo: chỉ áp cho background chưa có vùng"
                  "\nCancel: thôi")
            if answer is None:
                return
            overwrite = answer
        source_size = self.background_image.size
        copied, failures = 0, []
        for path in targets:
            if path in with_region and not overwrite:
                continue
            size = self.image_size(path)
            if size is None:
                failures.append(f"{path.name}: không đọc được ảnh")
                continue
            region = scale_region(self.region, source_size, size)
            try:
                atomic_write(region_path(path), encode([region], *size))
                if self.out_lines:
                    # Stored relative to the image size, so the same text fits.
                    atomic_write(out_line_path(path),
                                 encode_out_lines(self.out_lines, source_size))
            except (OSError, ValueError) as exc:
                failures.append(f"{path.name}: {exc}")
                continue
            copied += 1
        self.update_rows()
        what = "vùng và đường out-line" if self.out_lines else "vùng"
        self.status.set(f"Đã áp {what} cho {copied} background")
        message = f"Đã áp {what} cho {copied} background."
        if failures:
            message += "\n\nLỗi:\n" + "\n".join(failures)
        (messagebox.showwarning if failures else messagebox.showinfo)("Áp vùng", message)

    def show_region_overview(self):
        """A thumbnail grid of every background with its region drawn."""
        if not self.backgrounds:
            messagebox.showinfo("Tổng quan vùng", "Chưa có background.")
            return
        window = tk.Toplevel(self.root)
        window.title("Tổng quan vùng đặt date — click để mở")
        window.geometry("1250x720")
        canvas = tk.Canvas(window, background="#20242b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        window.photos = []
        cell_w, cell_h, columns = 300, 150, 4
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            for index, path in enumerate(self.backgrounds):
                x0, y0 = (index % columns) * cell_w + 10, (index // columns) * (cell_h + 30) + 10
                try:
                    with Image.open(path) as opened:
                        image = opened.convert("RGB")
                except OSError:
                    canvas.create_text(x0, y0, text=f"{path.name}: lỗi đọc", anchor="nw",
                                       fill="#ff5252")
                    continue
                factor = min((cell_w - 12) / image.width, cell_h / image.height)
                thumb = image.resize((max(1, round(image.width * factor)),
                                      max(1, round(image.height * factor))),
                                     Image.Resampling.BILINEAR)
                region = self.background_region(path)
                if region is not None:
                    ImageDraw.Draw(thumb).polygon(
                        [(x * factor, y * factor) for x, y in region.corners()],
                        outline=(0, 229, 255), width=2)
                for line in self.background_out_lines(path):
                    ImageDraw.Draw(thumb).line(
                        [(x * factor, y * factor) for x, y in line], fill=(255, 64, 255), width=2)
                photo = ImageTk.PhotoImage(thumb)
                window.photos.append(photo)
                tag = f"item{index}"
                canvas.create_image(x0, y0, image=photo, anchor="nw", tags=tag)
                checked = CHECK_ON if self.background_checked(path) else CHECK_OFF
                text = f"{checked} {path.name[:34]}" + ("" if region else "  ✗ chưa có vùng")
                canvas.create_text(x0, y0 + thumb.height + 3, text=text, anchor="nw",
                                   fill="#ffffff" if region else "#ffca28", tags=tag)
                canvas.tag_bind(tag, "<Button-1>",
                                lambda _event, index=index: self._overview_open(window, index))
        finally:
            self.root.configure(cursor="")
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(
            -1 if event.delta > 0 else 1, "units"))

    def _overview_open(self, window, index):
        window.destroy()
        self.load_background(index)
        self.set_view("preview")

    # ----------------------------------------------------------------- batch

    def open_batch_dialog(self):
        if self.busy:
            return
        if self.batch_dialog is not None:
            self.batch_dialog.window.lift()
            self.batch_dialog.refresh()
            return
        self.batch_dialog = BatchDialog(self)

    def batch_inputs(self):
        """Load the checked codes and backgrounds; returns ``(codes, targets, errors)``.

        The open code and background come from the editor (their boxes and
        region are autosaved, so they match the files)."""
        cache = self.__dict__.setdefault("_batch_cache", {})
        codes, targets, errors = [], [], []
        for path in self.checked_codes():
            if path == self.sample_path and self.sample_image is not None:
                codes.append(self.current_code())
                continue
            label = sample_label_path(path)
            stamp = self._stamp(path, label)
            cached = cache.get(("code", path))
            if cached and cached[0] == stamp:
                codes.append(cached[1])
                continue
            try:
                code = load_code(path, label)
            except (OSError, ValueError) as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            cache[("code", path)] = (stamp, code)
            codes.append(code)
        for path in self.checked_backgrounds():
            if path == self.background_path and self.background_image is not None:
                targets.append(self.current_target())
                continue
            label, lines = region_path(path), out_line_path(path)
            stamp = self._stamp(path, label, lines)
            cached = cache.get(("target", path))
            if cached and cached[0] == stamp:
                targets.append(cached[1])
                continue
            try:
                target = load_target(path, label, lines)
            except (OSError, ValueError) as exc:
                errors.append(f"{path.name}: {exc}")
                continue
            cache[("target", path)] = (stamp, target)
            targets.append(target)
        return codes, targets, errors

    def build_plan(self, seed, out=None):
        codes, targets, errors = self.batch_inputs()
        self.batch.size_mode = self.generator.size_mode
        self.batch.count = read_int(self.export_count, 10, 1, 100000)
        previous = previous_split(read_manifest(out)) if out is not None else None
        plan = plan_batch(codes, targets, self.saved_modes, self.batch, seed,
                          self.generator, previous)
        plan.warnings.extend(f"Không đọc được {error}" for error in errors)
        return plan


class BatchDialog:
    """"Tạo data hàng loạt": every checked code × every checked background."""

    def __init__(self, app):
        self.app = app
        self._pending = None
        settings = app.batch
        window = self.window = tk.Toplevel(app.root)
        window.title("Tạo data hàng loạt")
        window.transient(app.root)
        window.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(window, padding=10)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        row = 0

        def label(text):
            ttk.Label(frame, text=text).grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=2)

        self.code_text, self.background_text = tk.StringVar(), tk.StringVar()
        label("Code:")
        ttk.Label(frame, textvariable=self.code_text, wraplength=560).grid(
            row=row, column=1, sticky="w")
        row += 1
        label("Background:")
        ttk.Label(frame, textvariable=self.background_text, wraplength=560).grid(
            row=row, column=1, sticky="w")
        row += 1
        ttk.Label(frame, text="(tick ☑ code/background ở danh sách bên trái cửa sổ chính)",
                  foreground="#666").grid(row=row, column=1, sticky="w")
        row += 1

        self.pairing = tk.StringVar(value=settings.pairing)
        self.limit_on = tk.BooleanVar(value=settings.max_backgrounds_per_code > 0)
        self.limit = tk.IntVar(value=settings.max_backgrounds_per_code or 3)
        label("Ghép cặp:")
        pairing = ttk.Frame(frame)
        pairing.grid(row=row, column=1, sticky="w")
        ttk.Radiobutton(pairing, text="mọi cặp hợp lệ", variable=self.pairing,
                        value="all").pack(side="left")
        ttk.Radiobutton(pairing, text="chỉ ảnh cùng kích thước", variable=self.pairing,
                        value="same_size").pack(side="left", padx=(10, 0))
        row += 1
        limit = ttk.Frame(frame)
        limit.grid(row=row, column=1, sticky="w")
        ttk.Checkbutton(limit, text="tối đa", variable=self.limit_on).pack(side="left")
        ttk.Spinbox(limit, from_=1, to=1000, textvariable=self.limit, width=5).pack(
            side="left", padx=4)
        ttk.Label(limit, text="background ngẫu nhiên cho mỗi code").pack(side="left")
        row += 1

        label("Dạng lỗi:")
        modes = ttk.Frame(frame)
        modes.grid(row=row, column=1, sticky="w")
        available = all_mode_names(app.saved_modes)
        selected = set(settings.selected_modes(available))
        saved_names = {saved["name"] for saved in app.saved_modes}
        self.mode_vars = {}
        for index, name in enumerate(available):
            variable = tk.BooleanVar(value=name in selected)
            self.mode_vars[name] = variable
            text = name[5:] if name.startswith("auto_") else f"{name} (đã lưu)"
            ttk.Checkbutton(modes, text=text, variable=variable).grid(
                row=index // 4, column=index % 4, sticky="w", padx=(0, 10))
        if not saved_names:
            ttk.Label(modes, text="(chưa có mode tự tạo đã lưu)", foreground="#666").grid(
                row=len(available) // 4 + 1, column=0, columnspan=4, sticky="w")
        row += 1

        self.count_mode = tk.StringVar(value=settings.count_mode)
        label("Số lượng:")
        count = ttk.Frame(frame)
        count.grid(row=row, column=1, sticky="w")
        ttk.Spinbox(count, from_=1, to=100000, textvariable=app.export_count, width=8).pack(
            side="left")
        ttk.Label(count, text="ảnh").pack(side="left", padx=(4, 10))
        ttk.Radiobutton(count, text="cho mỗi dạng lỗi, chia đều các cặp",
                        variable=self.count_mode, value="per_mode").pack(side="left")
        row += 1
        ttk.Radiobutton(frame, text="cho mỗi dạng lỗi × mỗi cặp", variable=self.count_mode,
                        value="per_mode_per_pair").grid(row=row, column=1, sticky="w",
                                                        padx=(108, 0))
        row += 1

        self.val = tk.IntVar(value=settings.val_percent)
        label("Tập val:")
        val = ttk.Frame(frame)
        val.grid(row=row, column=1, sticky="w")
        ttk.Spinbox(val, from_=0, to=90, textvariable=self.val, width=5).pack(side="left")
        ttk.Label(val, text="%  tách theo code (0 % = images/ + labels/ phẳng như cũ)").pack(
            side="left", padx=4)
        row += 1

        label("Cỡ ký tự:")
        ttk.Checkbutton(frame, text="theo vùng (date cao 85 % vùng) thay vì theo tỉ lệ kích "
                                    "thước ảnh", variable=app.size_by_region,
                        command=app.on_size_mode_change).grid(row=row, column=1, sticky="w")
        row += 1

        self.angle = tk.StringVar(value=f"{settings.angle_jitter_deg:g}")
        label("Góc xoay:")
        angle = ttk.Frame(frame)
        angle.grid(row=row, column=1, sticky="w")
        ttk.Label(angle, text="ngẫu nhiên ±").pack(side="left")
        angle_spin = ttk.Spinbox(angle, from_=0, to=MAX_JITTER_ANGLE_DEG, increment=.5,
                                 textvariable=self.angle, width=6)
        angle_spin.pack(side="left", padx=4)
        angle_spin.bind("<FocusOut>", lambda _event: self.angle.set(
            f"{read_float(self.angle, settings.angle_jitter_deg, 0, MAX_JITTER_ANGLE_DEG):g}"))
        ttk.Label(angle, text=f"° quanh góc vùng đặt (mặc định {JITTER_ANGLE_DEG:g}°; góc lớn "
                              "cần vùng đặt rộng hơn)").pack(side="left")
        row += 1

        label("Seed:")
        seed = ttk.Frame(frame)
        seed.grid(row=row, column=1, sticky="w")
        ttk.Entry(seed, textvariable=app.seed_text, width=14).pack(side="left")
        ttk.Label(seed, text="trống = ngẫu nhiên; cùng seed + cùng thư mục → chạy tiếp "
                             "phần còn thiếu").pack(side="left", padx=4)
        row += 1

        label("Thư mục:")
        out = ttk.Frame(frame)
        out.grid(row=row, column=1, sticky="we")
        ttk.Entry(out, textvariable=app.synthetic_output_dir).pack(side="left", fill="x",
                                                                   expand=True)
        ttk.Button(out, text="Chọn…", command=lambda: app.choose_synthetic_output(
            self.window)).pack(side="left", padx=(4, 0))
        row += 1

        self.estimate = tk.StringVar(value="Dự kiến: …")
        ttk.Label(frame, textvariable=self.estimate, font=("TkDefaultFont", 10, "bold"),
                  wraplength=640).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 4))
        row += 1
        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=2, sticky="e", pady=(6, 0))
        for text, command in (("Kiểm tra", self.check), ("Xem trước", self.preview),
                              ("Tạo", self.run), ("Đóng", self.close)):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=3)

        for variable in (self.pairing, self.limit_on, self.limit, self.count_mode, self.val,
                         self.angle, app.export_count, *self.mode_vars.values()):
            variable.trace_add("write", lambda *_args: self.schedule_refresh())
        self.refresh()

    # ------------------------------------------------------------- settings

    def read_settings(self):
        settings = self.app.batch
        settings.pairing = self.pairing.get()
        settings.max_backgrounds_per_code = (
            read_int(self.limit, 3, 1, 10000) if self.limit_on.get() else 0)
        settings.count_mode = self.count_mode.get()
        settings.val_percent = read_int(self.val, 20, 0, 90)
        settings.count = read_int(self.app.export_count, 10, 1, 100000)
        settings.modes_off = [name for name, variable in self.mode_vars.items()
                              if not variable.get()]
        settings.modes = None
        settings.size_mode = self.app.generator.size_mode
        settings.angle_jitter_deg = read_float(self.angle, settings.angle_jitter_deg,
                                               0.0, MAX_JITTER_ANGLE_DEG)
        self.app.generator.angle_jitter_deg = settings.angle_jitter_deg
        return settings

    def schedule_refresh(self):
        if self._pending is not None:
            try:
                self.window.after_cancel(self._pending)
            except tk.TclError:
                pass
        try:
            self._pending = self.window.after(300, self.refresh)
        except tk.TclError:
            self._pending = None

    def plan(self, seed, out=None):
        self.read_settings()
        return self.app.build_plan(seed, out)

    def refresh(self):
        self._pending = None
        if self.app.busy:
            return
        try:
            seed = int(self.app.seed_text.get().strip() or 0)
        except ValueError:
            seed = 0
        plan = self.plan(seed)
        codes = sorted({pair.code.name for pair in plan.pairs})
        used_codes = sorted({pair.code.stem for pair in plan.valid_pairs if pair.total})
        self.code_text.set(f"{len(codes)} đã chọn" + (
            f" — dùng được: {', '.join(used_codes)}" if used_codes else ""))
        targets = {pair.target.name: pair.target for pair in plan.pairs}
        missing = sum(1 for target in targets.values() if target.region is None)
        self.background_text.set(
            f"{len(targets)} đã chọn" + (f" · {missing} thiếu vùng → bỏ qua" if missing else ""))
        images, size, seconds = plan.estimate()
        pairs = len({(pair.code.name, pair.target.name) for pair in plan.valid_pairs
                     if pair.total})
        text = (f"Dự kiến: {pairs} cặp · {len(plan.modes)} dạng → {images} ảnh · "
                f"~{size / 1e6:.0f} MB · ~{_duration(seconds)}")
        skipped = sum(1 for pair in plan.pairs if pair.problem)
        if skipped:
            text += f"   ({skipped} cặp bị bỏ qua — bấm Kiểm tra)"
        if plan.warnings:
            text += "\n" + "\n".join(plan.warnings)
        self.estimate.set(text)
        self.app.save_config()

    # -------------------------------------------------------------- actions

    def check(self):
        """Show pairs × modes with the number of images, without composing."""
        try:
            seed = int(self.app.seed_text.get().strip() or 0)
        except ValueError:
            seed = 0
        out = self.app.synthetic_target()
        plan = self.plan(seed, out)
        window = tk.Toplevel(self.window)
        window.title("Kiểm tra: cặp × dạng lỗi")
        window.geometry("1400x520")
        columns = ["pair", *plan.modes, "reason"]
        tree = ttk.Treeview(window, columns=columns, show="headings")
        tree.heading("pair", text="Code × background")
        tree.column("pair", width=440, stretch=False)
        for mode in plan.modes:
            tree.heading(mode, text=mode[5:] if mode.startswith("auto_") else mode)
            tree.column(mode, width=72, anchor="center", stretch=False)
        tree.heading("reason", text="Lý do bỏ qua")
        tree.column("reason", width=420, stretch=True)
        tree.tag_configure("skip", foreground="#9e9e9e")
        for pair in plan.pairs:
            name = f"{pair.code.stem} × {pair.target.stem}"
            if pair.problem:
                tree.insert("", "end", values=[name, *["–"] * len(plan.modes), pair.problem],
                            tags=("skip",))
                continue
            cells = [str(pair.counts[mode]) if pair.counts.get(mode) else "–"
                     for mode in plan.modes]
            reasons = "; ".join(f"{mode}: {reason}" for mode, reason in pair.skipped.items())
            tree.insert("", "end", values=[name, *cells, reasons])
        totals = ["TỔNG", *[str(sum(pair.counts.get(mode, 0) for pair in plan.pairs))
                            for mode in plan.modes], f"{plan.total} ảnh"]
        tree.insert("", "end", values=totals)
        xscroll = ttk.Scrollbar(window, orient="horizontal", command=tree.xview)
        yscroll = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
        tree.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        lines = []
        if plan.split_by:
            val = sorted(name for name, split in plan.split.items() if split == "val")
            train = sorted(name for name, split in plan.split.items() if split == "train")
            if plan.split_by in ("code", "background"):
                lines.append(f"Tách theo {plan.split_by}: val = {', '.join(val) or '—'}"
                             f" | train = {', '.join(train) or '—'}")
            else:
                lines.append("Tách val ngẫu nhiên theo ảnh")
        else:
            lines.append("Không tách val (dạng phẳng images/ + labels/)")
        lines.extend(plan.warnings)
        ttk.Label(window, text="\n".join(lines), padding=6, wraplength=1250).pack(
            side="bottom", fill="x")
        xscroll.pack(side="bottom", fill="x")
        yscroll.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

    def preview(self):
        """Save one image per mode for a few pairs as one contact sheet."""
        app = self.app
        seed = app.export_seed()
        if seed is None:
            return
        out = app.synthetic_target()
        if out is None:
            return
        plan = self.plan(seed, out)
        if not plan.total:
            messagebox.showinfo("Xem trước", "Không có ảnh nào để tạo — bấm Kiểm tra để xem "
                                "lý do.", parent=self.window)
            return
        app.root.configure(cursor="watch")
        self.window.configure(cursor="watch")
        self.window.update_idletasks()
        try:
            sheet, failures = contact_sheet(plan, app.generator)
            path = out / "preview" / f"sheet_{seed}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(path)
        except OSError as exc:
            messagebox.showerror("Xem trước", str(exc), parent=self.window)
            return
        finally:
            app.root.configure(cursor="")
            self.window.configure(cursor="")
        _open_file(path)
        lines = [f"Đã lưu ảnh xem trước:\n{path}",
                 f"Seed: {seed} — nhập seed này vào ô Seed để Tạo đúng các ảnh vừa xem."]
        if failures:
            lines.append("Lỗi:\n" + "\n".join(failures))
        (messagebox.showwarning if failures else messagebox.showinfo)(
            "Xem trước", "\n\n".join(lines), parent=self.window)

    def run(self):
        app = self.app
        if app.busy:
            return
        seed_was_blank = not app.seed_text.get().strip()
        seed = app.export_seed()
        if seed is None:
            return
        out = app.synthetic_target()
        if out is None:
            return
        settings = self.read_settings()
        while True:
            conflict = layout_conflict(out, settings.val_percent)
            if conflict is None:
                break
            if not messagebox.askyesno("Thư mục đã có data khác dạng",
                                       conflict + "\n\nChọn thư mục mới?", parent=self.window):
                return
            if not app.choose_synthetic_output(self.window):
                return
            out = app.synthetic_target()
        plan = self.plan(seed, out)
        if not plan.total:
            messagebox.showerror("Không có ảnh để tạo",
                                 "Không cặp/dạng lỗi nào dùng được — bấm Kiểm tra để xem lý do."
                                 + ("\n\n" + "\n".join(plan.warnings) if plan.warnings else ""),
                                 parent=self.window)
            return
        images, size, seconds = plan.estimate()
        if images > CONFIRM_ABOVE and not messagebox.askyesno(
                "Nhiều ảnh", f"Sẽ tạo {images} ảnh (~{size / 1e9:.1f} GB, "
                             f"~{_duration(seconds)}). Tiếp tục?", parent=self.window):
            return
        app.save_config()
        app.busy = True
        progress = app.progress = ProgressDialog(app.root, "Đang tạo data hàng loạt", plan.total)

        def report(done, _total, text):
            progress.step(done, text)
            return not progress.cancelled

        try:
            result = run_batch(plan, out, app.generator, report)
        finally:
            progress.close()
            app.progress = None
            app.busy = False
        if result.cancelled and seed_was_blank:
            # Resuming needs the same seed.
            app.seed_text.set(str(seed))
        app.save_config()
        if app.close_requested:
            app.close()
            return
        app.show_preview()
        lines = [f"Đã ghi {result.total} ảnh và label tại:\n{out}", f"Seed: {seed}"]
        if result.resumed:
            lines.append(f"Bỏ qua {result.resumed} ảnh đã có từ lần chạy trước (cùng seed).")
        if result.split_counts:
            lines.append("Theo tập: " + ", ".join(f"{split}={count}" for split, count
                                                  in sorted(result.split_counts.items())))
        lines.append("\n".join(f"  {name}: {made}" for name, made in result.made.items()))
        if result.cancelled:
            lines.append(f"Đã hủy giữa chừng. Bấm Tạo lại với seed {seed} và cùng thư mục "
                         "để chạy tiếp phần còn thiếu.")
        if plan.warnings:
            lines.append("\n".join(plan.warnings))
        if result.failures:
            lines.append("Lỗi:\n" + "\n".join(result.failures))
        if result.error:
            lines.append("Dừng vì lỗi ghi file: " + result.error)
        problem = result.failures or result.error or result.cancelled
        (messagebox.showwarning if problem else messagebox.showinfo)(
            "Kết quả tạo data", "\n\n".join(lines), parent=self.window)

    def close(self):
        self.read_settings()
        self.app.save_config()
        self.app.batch_dialog = None
        self.window.destroy()


def _duration(seconds):
    if seconds < 90:
        return f"{max(1, round(seconds))} giây"
    if seconds < 5400:
        return f"{seconds / 60:.0f} phút"
    return f"{seconds / 3600:.1f} giờ"


def _open_file(path):
    """Open a file with the system viewer when the platform supports it."""
    try:
        if hasattr(os, "startfile"):
            os.startfile(path)  # noqa: S606  (local file the tool just wrote)
    except OSError:
        pass


# ------------------------------------------------------------- command line

def run_batch_cli(argv=None):
    """Run "Tạo data hàng loạt" from the saved config without opening a window."""
    for stream in (sys.stdout, sys.stderr):
        try:
            # A redirected Windows console may not encode Vietnamese text.
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(
        description="Tạo data hàng loạt từ cấu hình đã lưu (label_tool_config.json).")
    parser.add_argument("--batch", action="store_true", help="chạy tạo data hàng loạt")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--out", help="thư mục xuất (mặc định: theo cấu hình)")
    parser.add_argument("--seed", type=int, help="seed (mặc định: theo cấu hình, trống = ngẫu nhiên)")
    parser.add_argument("--count", type=int, help="số ảnh (mặc định: theo cấu hình)")
    parser.add_argument("--angle", type=float,
                        help="góc xoay ngẫu nhiên ± độ (mặc định: theo cấu hình)")
    parser.add_argument("--check", action="store_true", help="chỉ in bảng kiểm tra, không tạo ảnh")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    data = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    batch = data.get("batch") if isinstance(data.get("batch"), dict) else {}
    settings = BatchSettings.from_config(batch)
    if args.count:
        settings.count = max(1, args.count)
    if args.angle is not None and math.isfinite(args.angle):
        settings.angle_jitter_deg = max(0.0, min(MAX_JITTER_ANGLE_DEG, args.angle))
    saved_modes = []
    for item in data.get("saved_modes", []) or []:
        try:
            saved_modes.append({"name": str(item["name"]),
                                "state": normalize_state(item["state"])})
        except (KeyError, TypeError, ValueError) as exc:
            print(f"bỏ qua mode đã lưu {item!r:.40}: {exc}", file=sys.stderr)
    checks_on = set(batch.get("codes", []) or [])
    checks_off = set(batch.get("codes_unchecked", []) or [])
    backgrounds_off = set(batch.get("backgrounds_unchecked", []) or [])
    codes, targets = [], []
    for path in image_files(SAMPLE_ROOT)[0]:
        if path.name in checks_off:
            continue
        try:
            code = load_code(path, sample_label_path(path))
        except (OSError, ValueError) as exc:
            print(f"bỏ qua code {path.name}: {exc}", file=sys.stderr)
            continue
        if code.boxes or path.name in checks_on:
            codes.append(code)
    for path in image_files(BACKGROUND_ROOT)[0]:
        if path.name in backgrounds_off:
            continue
        try:
            targets.append(load_target(path, region_path(path), out_line_path(path)))
        except (OSError, ValueError) as exc:
            print(f"bỏ qua background {path.name}: {exc}", file=sys.stderr)
    seed = args.seed
    if seed is None:
        text = str(data.get("seed", "")).strip()
        seed = int(text) if text.lstrip("-").isdigit() else random.SystemRandom().randrange(2 ** 31)
    out = Path(args.out or data.get("synthetic_output_dir") or OUTPUT_ROOT).resolve()
    generator = Generator.from_settings(settings)
    plan = plan_batch(codes, targets, saved_modes, settings, seed, generator,
                      previous_split(read_manifest(out)))
    print(f"seed={seed}  out={out}  angle=±{settings.angle_jitter_deg:g}°")
    for pair in plan.pairs:
        name = f"{pair.code.stem} × {pair.target.stem}"
        if pair.problem:
            print(f"  –  {name}: {pair.problem}")
        else:
            counts = " ".join(f"{mode}={count}" for mode, count in pair.counts.items() if count)
            skipped = "; ".join(f"{mode}: {reason}" for mode, reason in pair.skipped.items())
            print(f"  ✓  {name}: {counts}" + (f"  (bỏ: {skipped})" if skipped else ""))
    for warning in plan.warnings:
        print("CẢNH BÁO:", warning)
    images, size, seconds = plan.estimate()
    print(f"Dự kiến {images} ảnh · ~{size / 1e6:.0f} MB · ~{_duration(seconds)}")
    if args.check or not images:
        return 0 if images else 1
    conflict = layout_conflict(out, settings.val_percent)
    if conflict:
        print("LỖI:", conflict, file=sys.stderr)
        return 2
    last = [0.0]

    def report(done, total, text):
        if done == total or done - last[0] >= max(1, total // 50):
            last[0] = done
            print(f"[{done}/{total}] {text}", flush=True)
        return True

    try:
        result = run_batch(plan, out, generator, report)
    except KeyboardInterrupt:
        print(f"Đã dừng. Chạy lại với --seed {seed} để tiếp tục.", file=sys.stderr)
        return 130
    print(f"Đã ghi {result.total} ảnh, bỏ qua {result.resumed} ảnh đã có. "
          + ", ".join(f"{split}={count}" for split, count in sorted(result.split_counts.items())))
    for failure in result.failures:
        print("LỖI:", failure, file=sys.stderr)
    if result.error:
        print("LỖI GHI FILE:", result.error, file=sys.stderr)
    return 1 if result.failures or result.error else 0


if __name__ == "__main__":
    if "--batch" in sys.argv[1:]:
        sys.exit(run_batch_cli())
    app_root = tk.Tk()
    SyntheticLabelTool(app_root)
    app_root.mainloop()
