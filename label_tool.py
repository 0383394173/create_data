from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import shutil
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageStat, ImageTk

from label_obb import Box, atomic_write, decode, encode


BASE = Path(__file__).resolve().parent
SAMPLE_ROOT = BASE / "group_samples" /"images"
BACKGROUND_ROOT = BASE / "group_background"
OUTPUT_ROOT = BASE / "synthetic_output"
CONFIG_PATH = BASE / "label_tool_config.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODES = ("normal", "spacing", "overlap", "arc", "blur", "cut", "delete_objects")
AUTO_ERROR_MODES = (
    "spacing", "overlap", "arc", "blur",
    "cut_row_1", "cut_row_2", "cut_both_rows",
    "delete_objects",
)


def image_files(folder):
    folder.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in folder.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def sample_label_path(path):
    return SAMPLE_ROOT / "labels" / f"{path.stem}.txt"


def region_path(path):
    return BACKGROUND_ROOT / "labels" / f"{path.stem}.txt"


def otsu_threshold(image, mask):
    histogram = image.histogram(mask)
    total = sum(histogram)
    if not total:
        return 127
    weighted_total = sum(value * count for value, count in enumerate(histogram))
    background_weight = background_sum = 0
    best_threshold, best_variance = 127, -1
    for threshold, count in enumerate(histogram):
        background_weight += count
        if not background_weight:
            continue
        foreground_weight = total - background_weight
        if not foreground_weight:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_threshold, best_variance = threshold, variance
    return best_threshold


def character_asset(source, box):
    """Return the print inside one OBB as a paper-relative shade map.

    The map is 255 on paper and sample/paper on ink, so multiplying it onto a
    background keeps the real ink tone, dot texture and soft edges.
    """
    corners = box.corners()
    padding = 12
    left = max(0, math.floor(min(x for x, _ in corners)) - padding)
    top = max(0, math.floor(min(y for _, y in corners)) - padding)
    right = min(source.width, math.ceil(max(x for x, _ in corners)) + padding + 1)
    bottom = min(source.height, math.ceil(max(y for _, y in corners)) + padding + 1)
    crop = source.crop((left, top, right, bottom)).convert("RGB")
    gray = crop.convert("L")
    polygon = [(x - left, y - top) for x, y in corners]
    roi = Image.new("L", crop.size, 0)
    ImageDraw.Draw(roi).polygon(polygon, fill=255)
    # Ignore two pixels next to the OBB edge, where label borders or shadows
    # touching the box are usually not part of the printed character.
    inner = roi.filter(ImageFilter.MinFilter(5))
    if inner.getbbox():
        roi = inner
    threshold = otsu_threshold(gray, roi)
    paper_mask = ImageChops.multiply(
        gray.point([0 if value <= threshold else 255 for value in range(256)]), roi)
    ink_mask = ImageChops.multiply(
        gray.point([255 if value <= threshold else 0 for value in range(256)]), roi)
    white = Image.new("RGB", crop.size, (255, 255, 255))
    if not paper_mask.getbbox() or not ink_mask.getbbox():
        return white, (left, top)
    paper = ImageStat.Stat(crop, paper_mask).mean
    paper_level = ImageStat.Stat(gray, paper_mask).mean[0]
    ink_level = ImageStat.Stat(gray, ink_mask).mean[0]
    shade = Image.merge("RGB", [
        channel.point([min(255, round(value * 255 / max(1.0, level)))
                       for value in range(256)])
        for channel, level in zip(crop.split(), paper)])
    # The soft gate only removes paper noise. The actual darkness comes from
    # shade, preserving dot texture and anti-aliased print edges.
    contrast = max(1.0, paper_level - ink_level)
    start, end = paper_level - contrast * .15, paper_level - contrast * .3
    table = [255 if value <= end else 0 if value >= start
             else round((start - value) * 255 / (start - end)) for value in range(256)]
    gate = gray.point(table).filter(ImageFilter.MaxFilter(3))
    gate = ImageChops.multiply(gate.filter(ImageFilter.GaussianBlur(.6)), roi)
    return Image.composite(shade, white, gate), (left, top)


def split_rows(boxes):
    """Split into two visual rows when useful; preserve left-to-right order."""
    if len(boxes) < 2:
        return [list(boxes)]
    ordered_y = sorted(boxes, key=lambda box: box.cy)
    best = None
    for index in range(1, len(ordered_y)):
        groups = (ordered_y[:index], ordered_y[index:])
        means = [sum(box.cy for box in group) / len(group) for group in groups]
        score = sum((box.cy - mean) ** 2 for group, mean in zip(groups, means)
                    for box in group)
        candidate = (score, index, groups)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    rows = best[2]
    overall_height = max(box.cy for box in boxes) - min(box.cy for box in boxes)
    mean_height = sum(box.h for box in boxes) / len(boxes)
    if overall_height < mean_height * .55:
        return [sorted(boxes, key=lambda box: box.cx)]
    return [sorted(row, key=lambda box: box.cx) for row in rows]


def layout_boxes(boxes, region, mode, stack_size, arc_height,
                 dimension_scale=1.0, spacing_groups=(), group_x_factor=1.0):
    """Return translated boxes at original size; never scale source characters."""
    placed = copy.deepcopy(boxes)
    source_left = min(x for box in placed for x, _ in box.corners())
    source_right = max(x for box in placed for x, _ in box.corners())
    source_top = min(y for box in placed for _, y in box.corners())
    source_bottom = max(y for box in placed for _, y in box.corners())
    source_center_x = (source_left + source_right) / 2
    source_center_y = (source_top + source_bottom) / 2
    for box in placed:
        box.cx = source_center_x + (box.cx - source_center_x) * dimension_scale
        box.cy = source_center_y + (box.cy - source_center_y) * dimension_scale
        box.w *= dimension_scale
        box.h *= dimension_scale
    rows = split_rows(placed)
    if mode == "spacing":
        # Mỗi cụm là một khối cứng. Hệ số X chung chỉ thay đổi khoảng cách giữa
        # tâm các cụm; vị trí tương đối của box bên trong cụm không đổi.
        configured = [list(dict.fromkeys(
            index for index in indices if 0 <= index < len(placed)))
            for indices in spacing_groups]
        configured = [indices for indices in configured if indices]
        assigned = {index for indices in configured for index in indices}
        groups = configured + [[index] for index in range(len(placed)) if index not in assigned]
        if len(groups) > 1:
            group_centers = [sum(placed[index].cx for index in indices) / len(indices)
                             for indices in groups]
            global_center = sum(group_centers) / len(group_centers)
            for indices, old_center in zip(groups, group_centers):
                new_center = global_center + (old_center - global_center) * group_x_factor
                shift = new_center - old_center
                for index in indices:
                    placed[index].cx += shift
    elif mode == "overlap":
        for row in rows:
            cursor = min(box.cx - box.w / 2 for box in row)
            for offset in range(0, len(row), max(1, stack_size)):
                stack = row[offset:offset + max(1, stack_size)]
                width = max(box.w for box in stack)
                center_x = cursor + width / 2
                for box in stack:
                    box.cx = center_x
                cursor += width + 1
    elif mode == "arc":
        left = min(box.cx for box in placed)
        right = max(box.cx for box in placed)
        middle = (left + right) / 2
        half_width = max((right - left) / 2, 1)
        for box in placed:
            normalized_x = (box.cx - middle) / half_width
            box.cy += arc_height * normalized_x ** 2

    left = min(x for box in placed for x, _ in box.corners())
    right = max(x for box in placed for x, _ in box.corners())
    top = min(y for box in placed for _, y in box.corners())
    bottom = max(y for box in placed for _, y in box.corners())
    target_x, target_y = region.cx, region.cy
    source_x, source_y = (left + right) / 2, (top + bottom) / 2
    for box in placed:
        dx, dy = box.cx - source_x, box.cy - source_y
        cosine, sine = math.cos(region.angle), math.sin(region.angle)
        box.cx = target_x + cosine * dx - sine * dy
        box.cy = target_y + sine * dx + cosine * dy
        box.angle += region.angle
    return placed


def box_inside_box(inner, outer, tolerance=1e-6):
    cosine, sine = math.cos(outer.angle), math.sin(outer.angle)
    for x, y in inner.corners():
        dx, dy = x - outer.cx, y - outer.cy
        local_x = dx * cosine + dy * sine
        local_y = -dx * sine + dy * cosine
        if abs(local_x) > outer.w / 2 + tolerance or abs(local_y) > outer.h / 2 + tolerance:
            return False
    return True


def line_side_mask(size, line, remove_side):
    mask = Image.new("L", size, 255)
    if not line or remove_side == "none":
        return mask
    (x1, y1), (x2, y2) = line
    pixels = mask.load()
    for y in range(size[1]):
        for x in range(size[0]):
            cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
            remove = cross < 0 if remove_side == "above" else cross > 0
            if remove:
                pixels[x, y] = 0
    return mask


def line_crosses_box(line, box, tolerance=1e-6):
    """Return True only when the drawn segment cuts through an OBB."""
    if not line:
        return False
    (x1, y1), (x2, y2) = line
    dx, dy = x2 - x1, y2 - y1
    if math.hypot(dx, dy) <= tolerance:
        return False

    # The infinite line must split the box. This excludes touches along an edge
    # or at a corner, where there is no meaningful upper/lower part to remove.
    sides = [dx * (y - y1) - dy * (x - x1) for x, y in box.corners()]
    if not (min(sides) < -tolerance and max(sides) > tolerance):
        return False

    # Also require the segment the user actually drew (not its infinite
    # extension) to enter the rotated box. Transform it to box-local space and
    # clip it against the resulting axis-aligned rectangle.
    cosine, sine = math.cos(box.angle), math.sin(box.angle)

    def local(x, y):
        offset_x, offset_y = x - box.cx, y - box.cy
        return (offset_x * cosine + offset_y * sine,
                -offset_x * sine + offset_y * cosine)

    start_x, start_y = local(x1, y1)
    end_x, end_y = local(x2, y2)
    segment_x, segment_y = end_x - start_x, end_y - start_y
    enter, leave = 0.0, 1.0
    for start, delta, minimum, maximum in (
            (start_x, segment_x, -box.w / 2, box.w / 2),
            (start_y, segment_y, -box.h / 2, box.h / 2)):
        if abs(delta) <= tolerance:
            if start < minimum - tolerance or start > maximum + tolerance:
                return False
            continue
        near, far = (minimum - start) / delta, (maximum - start) / delta
        if near > far:
            near, far = far, near
        enter, leave = max(enter, near), min(leave, far)
        if enter > leave + tolerance:
            return False
    return True


class SyntheticLabelTool:
    COLORS = ("#00e5ff", "#ffca28", "#ff66aa", "#76ff03")

    def __init__(self, root):
        self.root = root
        root.title("Synthetic Date OBB Tool")
        root.geometry("1600x960")
        root.minsize(1100, 700)
        SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
        BACKGROUND_ROOT.mkdir(parents=True, exist_ok=True)
        self.samples, self.backgrounds = [], []
        self.sample_index = self.background_index = -1
        self.sample_image = self.background_image = None
        self.sample_boxes = []
        self.region = None
        self.selected_box = None
        self.history = []
        self.cut_line = None
        self.cut_lines = []
        self.cut_side = tk.StringVar(value="none")
        self.mode = tk.StringVar(value="normal")
        self.last_effect_mode = "normal"
        self.saved_modes = []
        self.saved_mode_serial = 0
        self.synthetic_output_dir = tk.StringVar(value=str(OUTPUT_ROOT))
        self.export_count = tk.IntVar(value=10)
        self.stack_size = tk.IntVar(value=3)
        self.arc_height = tk.DoubleVar(value=35.0)
        self.blur_radius = tk.DoubleVar(value=2.0)
        self.spacing_groups = []
        self.group_indices_text = tk.StringVar(value="0-2")
        self.group_x_factor = tk.DoubleVar(value=1.0)
        self.delete_scope = tk.StringVar(value="box")
        self.deleted_indices = set()
        self.preview_source_indices = []
        self.view = tk.StringVar(value="sample")
        self.draw_mode = tk.StringVar(value="sample_box")
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self.photo = None
        self.gesture = None
        self.preview_cache = None
        self.preview_boxes = []
        self.status = tk.StringVar(value="Sẵn sàng")
        self._build_ui()
        self.load_config()
        self.refresh_files()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def config_data(self):
        """Return JSON-safe settings that should survive an application restart."""
        def state_data(state):
            result = copy.deepcopy(state)
            result["deleted_indices"] = sorted(result.get("deleted_indices", ()))
            return result

        return {
            "synthetic_output_dir": self.synthetic_output_dir.get(),
            "export_count": self.export_count.get(),
            "stack_size": self.stack_size.get(),
            "arc_height": self.arc_height.get(),
            "blur_radius": self.blur_radius.get(),
            "group_indices_text": self.group_indices_text.get(),
            "delete_scope": self.delete_scope.get(),
            "saved_mode_serial": self.saved_mode_serial,
            "saved_modes": [
                {"name": item["name"], "state": state_data(item["state"])}
                for item in self.saved_modes
            ],
        }

    def save_config(self):
        try:
            atomic_write(CONFIG_PATH, json.dumps(
                self.config_data(), ensure_ascii=False, indent=2) + "\n")
        except (OSError, TypeError, ValueError):
            # Closing the UI must not fail just because settings cannot be saved.
            pass

    def load_config(self):
        if not CONFIG_PATH.exists():
            return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            self.synthetic_output_dir.set(
                data.get("synthetic_output_dir", str(OUTPUT_ROOT)))
            self.export_count.set(max(1, min(10000, int(data.get("export_count", 10)))))
            self.stack_size.set(max(1, min(5, int(data.get("stack_size", 3)))))
            self.arc_height.set(float(data.get("arc_height", 35.0)))
            self.blur_radius.set(float(data.get("blur_radius", 2.0)))
            self.group_indices_text.set(data.get("group_indices_text", "0-2"))
            self.delete_scope.set(data.get("delete_scope", "box"))
            self.saved_mode_serial = max(0, int(data.get("saved_mode_serial", 0)))
            for item in data.get("saved_modes", []):
                state = item["state"]
                state["deleted_indices"] = set(state.get("deleted_indices", ()))
                saved = {"name": str(item["name"]), "state": state}
                self.saved_modes.append(saved)
                self.saved_mode_list.insert("end", saved["name"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.status.set("Không đọc được cấu hình cũ; đang dùng cấu hình mặc định")

    def close(self):
        self.save_config()
        self.root.destroy()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill="x")
        for text, command in (("Thêm ảnh mẫu", self.add_samples),
                              ("Thêm background", self.add_backgrounds),
                              ("Sửa box/vùng", self.enable_box_edit),
                              ("Undo Ctrl+Z", self.undo),
                              ("Xóa Delete", self.delete_selected),
                              ("Lưu box mẫu", self.save_sample_boxes),
                              ("Tạo background từ box", self.create_background_from_boxes),
                              ("Lưu vùng đặt", self.save_region),
                              ("TỰ ĐỘNG TẠO TẤT CẢ LỖI", self.export_synthetic)):
            ttk.Button(top, text=text, command=command).pack(side="left", padx=3)
        ttk.Button(top, text="Vừa ảnh", command=self.fit).pack(side="left", padx=3)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)
        side_host = ttk.Frame(body, width=360)
        body.add(side_host, weight=0)
        self.side_canvas = tk.Canvas(side_host, width=340, highlightthickness=0)
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
        ttk.Label(side, text="1. Ảnh lấy date — group_samples").pack(anchor="w")
        self.sample_list = tk.Listbox(side, width=34, height=9, exportselection=False)
        self.sample_list.pack(fill="x")
        self.sample_list.bind("<<ListboxSelect>>", self.choose_sample)
        ttk.Label(side, text="2. Ảnh chứa date — group_background").pack(anchor="w", pady=(10, 0))
        self.background_list = tk.Listbox(side, width=34, height=9, exportselection=False)
        self.background_list.pack(fill="x")
        self.background_list.bind("<<ListboxSelect>>", self.choose_background)

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="Vẽ / thao tác chuột").pack(anchor="w")
        for text, value in (("Chọn/sửa box hoặc vùng", "select"),
                            ("Vẽ box date trên ảnh mẫu", "sample_box"),
                            ("Vẽ vùng đặt trên background", "target_region"),
                            ("Kéo ngang chỉnh spacing/overlap", "adjust_horizontal"),
                            ("Kéo dọc chỉnh cong", "adjust_arc"),
                            ("Kẻ đường cắt", "cut_line")):
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
        ttk.Spinbox(side, from_=1, to=5, textvariable=self.stack_size, width=8,
                    command=self.show_preview).pack(anchor="w")
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
        ttk.Button(side, text="Xem ảnh mẫu", command=lambda: self.set_view("sample")).pack(fill="x", pady=(8, 2))
        ttk.Button(side, text="Xem background/preview", command=lambda: self.set_view("preview")).pack(fill="x")
        output_panel = ttk.LabelFrame(side, text="Thư mục data giả / xem trước", padding=6)
        output_panel.pack(fill="x", pady=(8, 0))
        ttk.Entry(output_panel, textvariable=self.synthetic_output_dir).pack(fill="x")
        count_row = ttk.Frame(output_panel)
        count_row.pack(fill="x", pady=(4, 0))
        ttk.Label(count_row, text="Số ảnh cho mỗi dạng lỗi").pack(side="left")
        ttk.Spinbox(count_row, from_=1, to=10000, textvariable=self.export_count,
                    width=9).pack(side="right")
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
        ttk.Label(self.root, textvariable=self.status, padding=6).pack(fill="x")
        self.canvas.bind("<Configure>", lambda event: self.render())
        self.canvas.bind("<Button-1>", self.press)
        self.canvas.bind("<Button-3>", self.delete_sample_box)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.root.bind_all("<MouseWheel>", self.side_mousewheel, add="+")
        self.root.bind("<Control-s>", lambda _event: (
            self.save_sample_boxes() if self.view.get() == "sample" else self.save_region()))
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Delete>", lambda _event: self.delete_selected())

    def refresh_files(self):
        self.samples, self.backgrounds = image_files(SAMPLE_ROOT), image_files(BACKGROUND_ROOT)
        self.sample_list.delete(0, "end")
        self.background_list.delete(0, "end")
        for path in self.samples:
            self.sample_list.insert("end", path.name)
        for path in self.backgrounds:
            self.background_list.insert("end", path.name)
        if self.samples and self.sample_index < 0:
            self.load_sample(0)
        if self.backgrounds and self.background_index < 0:
            self.load_background(0)

    def _copy_images(self, target):
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        added_names = []
        for value in paths:
            source = Path(value)
            destination = target / source.name
            if destination.exists() and source.resolve() != destination.resolve():
                messagebox.showwarning("Trùng tên", f"Đã bỏ qua: {source.name}")
                continue
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            added_names.append(destination.name)
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
        chosen = self.sample_list.curselection()
        if chosen:
            self.load_sample(chosen[0])

    def choose_background(self, _event=None):
        chosen = self.background_list.curselection()
        if chosen:
            self.load_background(chosen[0])

    def load_sample(self, index):
        try:
            image = Image.open(self.samples[index]).convert("RGB")
            label = sample_label_path(self.samples[index])
            boxes = decode(label.read_text(encoding="utf-8-sig"), *image.size, 1) if label.exists() else []
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không mở được ảnh mẫu", str(exc))
            return
        self.sample_index, self.sample_image, self.sample_boxes = index, image, boxes
        self.selected_box = None
        self.history.clear()
        self.spacing_groups.clear()
        self.deleted_indices.clear()
        self.refresh_spacing_groups()
        self.sample_list.selection_clear(0, "end")
        self.sample_list.selection_set(index)
        self.show_preview()

    def load_background(self, index):
        try:
            image = Image.open(self.backgrounds[index]).convert("RGB")
            saved = region_path(self.backgrounds[index])
            saved_boxes = decode(saved.read_text(encoding="utf-8-sig"), *image.size, 1) if saved.exists() else []
            if len(saved_boxes) > 1:
                raise ValueError("Nhãn background chỉ được chứa một OBB vùng đặt date.")
            region = saved_boxes[0] if saved_boxes else None
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("Không mở được background", str(exc))
            return
        self.background_index, self.background_image, self.region = index, image, region
        self.selected_box = None
        self.history.clear()
        self.background_list.selection_clear(0, "end")
        self.background_list.selection_set(index)
        nearest = self.select_nearest_sample(image.size)
        self.show_preview()
        if nearest is not None:
            self.status.set(
                f"Đã tự chọn mẫu gần kích thước background nhất: {self.samples[nearest].name}")

    def select_nearest_sample(self, background_size):
        if not self.samples:
            return None
        candidates = []
        for index, path in enumerate(self.samples):
            label = sample_label_path(path)
            try:
                if not label.exists() or not label.read_text(encoding="utf-8-sig").strip():
                    continue
                with Image.open(path) as image:
                    width, height = image.size
            except (OSError, UnicodeError):
                continue
            background_width, background_height = background_size
            score = (abs(math.log(width / background_width))
                     + abs(math.log(height / background_height))
                     + abs(math.log((width / height) / (background_width / background_height))))
            candidates.append((score, path.name.casefold(), index))
        if not candidates:
            return None
        nearest = min(candidates)[2]
        if nearest != self.sample_index:
            self.load_sample(nearest)
        return nearest

    def save_sample_boxes(self):
        if self.sample_image is None or self.sample_index < 0:
            return
        try:
            atomic_write(sample_label_path(self.samples[self.sample_index]),
                         encode(self.sample_boxes, *self.sample_image.size))
            self.status.set(f"Đã lưu {len(self.sample_boxes)} box mẫu")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không lưu được box", str(exc))

    def autosave_sample_boxes(self):
        if self.sample_image is None or self.sample_index < 0:
            return
        try:
            atomic_write(sample_label_path(self.samples[self.sample_index]),
                         encode(self.sample_boxes, *self.sample_image.size))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không tự động lưu được box mẫu", str(exc))

    def create_background_from_boxes(self):
        """Remove boxed characters and save a new paper/background image."""
        if self.sample_image is None or self.sample_index < 0:
            messagebox.showinfo("Thiếu ảnh mẫu", "Hãy chọn ảnh mẫu trước.")
            return
        if not self.sample_boxes:
            messagebox.showinfo(
                "Thiếu box ký tự",
                "Hãy vẽ box quanh các ký tự cần xóa trước khi tạo background.")
            return

        source = self.sample_image.convert("RGB")
        result = source.copy()
        gray = source.convert("L")
        try:
            for box in self.sample_boxes:
                polygon = box.corners()
                character_mask = Image.new("L", source.size, 0)
                ImageDraw.Draw(character_mask).polygon(polygon, fill=255)

                # Cover ink touching the OBB edge, while sampling paper from a
                # separate outer ring around the character.
                inner_size = max(5, min(21, (round(min(box.w, box.h) * .12) | 1)))
                outer_size = max(inner_size + 4, min(41, inner_size * 2 - 1))
                if outer_size % 2 == 0:
                    outer_size += 1
                removal_mask = character_mask.filter(ImageFilter.MaxFilter(inner_size))
                outer_mask = character_mask.filter(ImageFilter.MaxFilter(outer_size))
                sample_ring = ImageChops.subtract(outer_mask, removal_mask)

                threshold = otsu_threshold(gray, sample_ring)
                bright_paper = gray.point(
                    [0 if value <= threshold else 255 for value in range(256)])
                paper_mask = ImageChops.multiply(sample_ring, bright_paper)
                if not paper_mask.getbbox():
                    paper_mask = sample_ring
                if not paper_mask.getbbox():
                    raise ValueError("Không tìm thấy vùng giấy quanh một box ký tự.")

                paper_stat = ImageStat.Stat(source, paper_mask)
                paper_color = tuple(round(value) for value in paper_stat.mean)
                paper_patch = Image.new("RGB", source.size, paper_color)
                feathered = removal_mask.filter(ImageFilter.GaussianBlur(1.2))
                result = Image.composite(paper_patch, result, feathered)

            source_name = self.samples[self.sample_index].stem
            index = 0
            while True:
                destination = BACKGROUND_ROOT / f"{source_name}_background_{index:06d}.png"
                if not destination.exists():
                    break
                index += 1
            BACKGROUND_ROOT.mkdir(parents=True, exist_ok=True)
            result.save(destination)
            self.refresh_files()
            lookup = {path.name: position for position, path in enumerate(self.backgrounds)}
            if destination.name in lookup:
                self.load_background(lookup[destination.name])
                self.set_view("preview")
            self.status.set(f"Đã tạo background: {destination.name}")
            messagebox.showinfo(
                "Đã tạo background",
                f"Đã xóa {len(self.sample_boxes)} vùng ký tự và lưu ảnh mới:\n{destination}")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không tạo được background", str(exc))

    def save_region(self):
        if self.region is None or self.background_index < 0:
            messagebox.showinfo("Thiếu vùng", "Hãy vẽ vùng đặt date trên background.")
            return
        atomic_write(region_path(self.backgrounds[self.background_index]),
                     encode([self.region], *self.background_image.size))
        self.status.set("Đã lưu vùng đặt date")

    def _sync_view(self):
        self.set_view("sample" if self.draw_mode.get() == "sample_box" else "preview")

    def enable_box_edit(self):
        self.draw_mode.set("select")
        self.selected_box = None
        self.status.set("Chế độ sửa: kéo box để di chuyển, kéo góc để resize, kéo nút tròn để xoay")
        self.render()

    def on_mode_change(self):
        if self.mode.get() != "delete_objects":
            self.last_effect_mode = self.mode.get()
        if self.mode.get() == "spacing":
            self.spacing_panel.pack(fill="x", pady=8)
        else:
            self.spacing_panel.pack_forget()
        if self.mode.get() == "delete_objects":
            self.delete_panel.pack(fill="x", pady=8)
            self.set_view("preview")
        else:
            self.delete_panel.pack_forget()
        if hasattr(self, "canvas"):
            self.show_preview()

    def parse_group_indices(self, text):
        indices = []
        for part in text.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                fields = part.split("-")
                if len(fields) != 2:
                    raise ValueError(f"Khoảng không hợp lệ: {part}")
                start, end = map(int, fields)
                if end < start:
                    start, end = end, start
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        unique = []
        for index in indices:
            if index not in unique:
                unique.append(index)
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
        self.spacing_groups.append(indices)
        self.refresh_spacing_groups()
        index = len(self.spacing_groups) - 1
        self.spacing_group_list.selection_set(index)
        self.group_x_factor.set(1.0)
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

    def change_group_factor(self, value):
        factor = float(value)
        self.group_factor_label.configure(text=f"{factor:.2f}×")
        self.show_preview()

    def delete_spacing_group(self):
        chosen = self.spacing_group_list.curselection()
        if not chosen:
            return
        index = chosen[0]
        self.spacing_groups.pop(index)
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

    def snapshot(self):
        self.history.append((copy.deepcopy(self.sample_boxes),
                             copy.deepcopy(self.region),
                             copy.deepcopy(self.cut_line),
                             copy.deepcopy(self.deleted_indices)))
        self.history = self.history[-100:]

    def undo(self):
        if not self.history:
            self.status.set("Không còn thao tác để hoàn tác")
            return
        self.sample_boxes, self.region, self.cut_line, self.deleted_indices = self.history.pop()
        self.selected_box = None
        self.autosave_sample_boxes()
        self.show_preview()
        self.status.set("Đã hoàn tác thao tác gần nhất")

    def delete_selected(self):
        if self.selected_box is None:
            self.status.set("Hãy chọn box hoặc vùng cần xóa")
            return
        if self.view.get() == "sample":
            if not 0 <= self.selected_box < len(self.sample_boxes):
                return
            self.snapshot()
            self.sample_boxes.pop(self.selected_box)
        elif self.region is not None and self.selected_box == 0:
            self.snapshot()
            self.region = None
        self.selected_box = None
        self.autosave_sample_boxes()
        self.show_preview()
        self.status.set("Đã xóa; nhấn Ctrl+Z để hoàn tác")

    def set_view(self, value):
        self.view.set(value)
        self.fit()

    def current_image(self):
        if self.view.get() == "sample":
            return self.sample_image
        return self.preview_cache or self.background_image

    def compose(self, effect_mode=None):
        if self.sample_image is None or self.background_image is None or not self.sample_boxes or self.region is None:
            return self.background_image, []
        selected_mode = self.mode.get() if effect_mode is None else effect_mode
        mode = (self.last_effect_mode if selected_mode == "delete_objects"
                else selected_mode)
        dimension_scale = min(
            self.background_image.width / self.sample_image.width,
            self.background_image.height / self.sample_image.height,
        )
        placed = layout_boxes(self.sample_boxes, self.region, mode,
                              max(1, min(5, int(self.stack_size.get()))),
                              float(self.arc_height.get()), dimension_scale,
                              self.spacing_groups, float(self.group_x_factor.get()))
        canvas = self.background_image.copy()
        active_cut_lines = (self.cut_lines or ([self.cut_line] if self.cut_line else []))
        active = [(index, source_box, target_box)
                  for index, (source_box, target_box)
                  in enumerate(zip(self.sample_boxes, placed))
                  if index not in self.deleted_indices]
        self.preview_source_indices = [index for index, _, _ in active]
        for _, source_box, target_box in active:
            shade, origin = character_asset(self.sample_image, source_box)
            object_scale = target_box.w / source_box.w
            if not math.isclose(object_scale, 1.0, abs_tol=1e-9):
                resized = (max(1, round(shade.width * object_scale)),
                           max(1, round(shade.height * object_scale)))
                shade = shade.resize(resized, Image.Resampling.BICUBIC)
            angle_delta = target_box.angle - source_box.angle
            crop_center_offset = (
                (origin[0] + (shade.width / object_scale) / 2 - source_box.cx)
                * object_scale,
                (origin[1] + (shade.height / object_scale) / 2 - source_box.cy)
                * object_scale,
            )
            cosine, sine = math.cos(angle_delta), math.sin(angle_delta)
            rotated_offset = (
                cosine * crop_center_offset[0] - sine * crop_center_offset[1],
                sine * crop_center_offset[0] + cosine * crop_center_offset[1],
            )
            if abs(angle_delta) > 1e-9:
                degrees = -math.degrees(angle_delta)
                shade = shade.rotate(degrees, resample=Image.Resampling.BICUBIC,
                                     expand=True, fillcolor=(255, 255, 255))
            if mode == "blur":
                shade = shade.filter(ImageFilter.GaussianBlur(
                    max(.2, float(self.blur_radius.get()))))
            x = round(target_box.cx + rotated_offset[0] - shade.width / 2)
            y = round(target_box.cy + rotated_offset[1] - shade.height / 2)
            object_cut_mask = Image.new("L", canvas.size, 255)
            if mode == "cut":
                for cut_line in active_cut_lines:
                    if line_crosses_box(cut_line, target_box):
                        cut_mask = line_side_mask(
                            canvas.size, cut_line, self.cut_side.get())
                        object_cut_mask = ImageChops.multiply(object_cut_mask, cut_mask)
            # Multiply the paper-relative shade onto the target background.
            # White pixels leave it untouched; ink retains source tone/texture.
            layer = Image.new("RGB", canvas.size, (255, 255, 255))
            layer.paste(shade, (x, y))
            canvas.paste(ImageChops.multiply(canvas, layer),
                         (0, 0), object_cut_mask)
        return canvas, [target_box for _, _, target_box in active]

    def show_preview(self):
        self.preview_cache, self.preview_boxes = self.compose()
        self.render()

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
        elif self.region:
            fits = all(box_inside_box(box, self.region) for box in self.preview_boxes)
            region_color = "#00e5ff" if fits else "#ff5252"
            points = [self.screen(x, y) for x, y in self.region.corners()]
            self.canvas.create_polygon(*[value for point in points for value in point],
                                       fill="", outline=region_color, width=2, dash=(6, 4))
            if self.selected_box == 0:
                self.draw_edit_handles(self.region, region_color)
            if self.cut_line:
                self.canvas.create_line(*self.screen(*self.cut_line[0]),
                                        *self.screen(*self.cut_line[1]), fill="#ffca28", width=3)
        self.status.set(
            f"Mẫu: {len(self.sample_boxes)} box | Background: "
            f"{'đã có vùng đặt' if self.region else 'chưa có vùng đặt'} | Dạng: {self.mode.get()}"
            + (f" | Đồng bộ size {min(self.background_image.width / self.sample_image.width, self.background_image.height / self.sample_image.height):.3f}×"
               if self.sample_image and self.background_image else "")
            + (" | CẢNH BÁO: date vượt vùng chứa" if self.region and self.preview_boxes
               and not all(box_inside_box(box, self.region) for box in self.preview_boxes) else ""))

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
        if self.draw_mode.get() == "select":
            boxes = self.editable_boxes()
            if self.selected_box is not None and self.selected_box < len(boxes):
                box = boxes[self.selected_box]
                if math.dist(point, self.rotation_handle(box)) * self.scale < 12:
                    self.snapshot()
                    self.gesture = ("edit_rotate", point, copy.deepcopy(box), self.selected_box)
                    return
                for corner_index, corner in enumerate(box.corners()):
                    if math.dist(point, corner) * self.scale < 12:
                        self.snapshot()
                        self.gesture = ("edit_resize", point, copy.deepcopy(box),
                                        self.selected_box, corner_index)
                        return
            self.selected_box = next(
                (index for index in reversed(range(len(boxes)))
                 if boxes[index].contains(*point)), None)
            if self.selected_box is not None:
                self.snapshot()
                self.gesture = ("edit_move", point,
                                copy.deepcopy(boxes[self.selected_box]), self.selected_box)
            self.render()
            return
        self.gesture = (self.draw_mode.get(), point)

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
        if not self.gesture:
            return
        image = self.current_image()
        if image is None:
            return
        kind, start = self.gesture[:2]
        point = self.point(event.x, event.y)
        if kind.startswith("edit_"):
            original, index = self.gesture[2], self.gesture[3]
            box = copy.deepcopy(original)
            if kind == "edit_move":
                box.cx += point[0] - start[0]
                box.cy += point[1] - start[1]
            elif kind == "edit_rotate":
                box.angle += (math.atan2(point[1] - box.cy, point[0] - box.cx)
                              - math.atan2(start[1] - box.cy, start[0] - box.cx))
            elif kind == "edit_resize":
                corner_index = self.gesture[4]
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
                self.replace_editable_box(index, box)
                self.show_preview()
        elif kind == "adjust_horizontal":
            delta = point[0] - start[0]
            if self.mode.get() == "spacing":
                factor = max(.1, min(4.0, 1.0 + delta / 100))
                self.group_x_factor.set(factor)
                self.change_group_factor(factor)
            elif self.mode.get() == "overlap":
                self.stack_size.set(max(1, min(5, 1 + round(abs(delta) / 25))))
            self.show_preview()
        elif kind == "adjust_arc":
            self.arc_height.set(max(-100, min(100, point[1] - start[1])))
            self.show_preview()
        else:
            self.canvas.delete("gesture_preview")
            if kind == "cut_line":
                self.canvas.create_line(*self.screen(*start), *self.screen(*point),
                                        fill="white", width=2, dash=(5, 3), tags="gesture_preview")
            else:
                self.canvas.create_rectangle(*self.screen(*start), *self.screen(*point),
                                             outline="white", dash=(5, 3), tags="gesture_preview")

    def release(self, event):
        if not self.gesture:
            return
        kind, start = self.gesture[:2]
        point = self.point(event.x, event.y)
        if kind == "sample_box" and self.view.get() == "sample":
            box = Box(0, (start[0] + point[0]) / 2, (start[1] + point[1]) / 2,
                      abs(point[0] - start[0]), abs(point[1] - start[1]))
            if box.valid(*self.sample_image.size):
                self.snapshot()
                self.sample_boxes.append(box)
                self.selected_box = len(self.sample_boxes) - 1
        elif kind == "target_region" and self.background_image is not None:
            region = Box(0, (start[0] + point[0]) / 2,
                         (start[1] + point[1]) / 2,
                         abs(point[0] - start[0]), abs(point[1] - start[1]))
            if region.valid(*self.background_image.size):
                self.snapshot()
                self.region = region
                self.selected_box = 0
        elif kind == "cut_line":
            self.snapshot()
            self.cut_line = (start, point)
            self.mode.set("cut")
        if kind == "sample_box" or (kind.startswith("edit_") and self.view.get() == "sample"):
            self.autosave_sample_boxes()
        self.gesture = None
        self.show_preview()

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

    def capture_effect_state(self):
        return {
            "mode": self.mode.get(),
            "last_effect_mode": self.last_effect_mode,
            "stack_size": self.stack_size.get(),
            "arc_height": self.arc_height.get(),
            "blur_radius": self.blur_radius.get(),
            "spacing_groups": copy.deepcopy(self.spacing_groups),
            "group_x_factor": self.group_x_factor.get(),
            "cut_line": copy.deepcopy(self.cut_line),
            "cut_lines": copy.deepcopy(self.cut_lines),
            "cut_side": self.cut_side.get(),
            "deleted_indices": set(self.deleted_indices),
        }

    def apply_effect_state(self, state):
        self.mode.set(state["mode"])
        self.last_effect_mode = state["last_effect_mode"]
        self.stack_size.set(state["stack_size"])
        self.arc_height.set(state["arc_height"])
        self.blur_radius.set(state["blur_radius"])
        self.spacing_groups = copy.deepcopy(state["spacing_groups"])
        self.group_x_factor.set(state["group_x_factor"])
        self.cut_line = copy.deepcopy(state["cut_line"])
        self.cut_lines = copy.deepcopy(state.get("cut_lines", []))
        self.cut_side.set(state["cut_side"])
        self.deleted_indices = set(state["deleted_indices"])

    def compose_saved_mode(self, saved_mode):
        current = self.capture_effect_state()
        try:
            self.apply_effect_state(saved_mode["state"])
            return self.compose()
        finally:
            self.apply_effect_state(current)

    def save_current_mode(self):
        effective = (self.last_effect_mode if self.mode.get() == "delete_objects"
                     else self.mode.get())
        self.saved_mode_serial += 1
        name = f"mode_{self.saved_mode_serial:02d}_{effective}"
        self.saved_modes.append({"name": name, "state": self.capture_effect_state()})
        self.saved_mode_list.insert("end", name)
        self.saved_mode_list.selection_clear(0, "end")
        self.saved_mode_list.selection_set("end")
        self.save_config()
        self.status.set(f"Đã lưu {name}; có thể làm mới và tạo dạng tiếp theo")

    def delete_saved_mode(self):
        selection = self.saved_mode_list.curselection()
        if not selection:
            return
        index = selection[0]
        self.saved_modes.pop(index)
        self.saved_mode_list.delete(index)
        self.save_config()
        self.status.set(f"Còn {len(self.saved_modes)} mode đã lưu")

    def reset_effect_editor(self):
        self.mode.set("normal")
        self.last_effect_mode = "normal"
        self.stack_size.set(3)
        self.arc_height.set(35.0)
        self.blur_radius.set(2.0)
        self.spacing_groups = []
        self.spacing_group_list.delete(0, "end")
        self.group_x_factor.set(1.0)
        self.group_factor_label.configure(text="1.00×")
        self.cut_line = None
        self.cut_side.set("none")
        self.deleted_indices.clear()
        self.on_mode_change()
        self.status.set("Đã làm mới; hãy tạo dạng lỗi tiếp theo rồi bấm Lưu mode hiện tại")

    def choose_synthetic_output(self):
        directory = filedialog.askdirectory(
            title="Chọn thư mục lưu data giả",
            initialdir=self.synthetic_output_dir.get() or str(OUTPUT_ROOT))
        if directory:
            self.synthetic_output_dir.set(directory)
            self.save_config()

    def synthetic_target(self):
        value = self.synthetic_output_dir.get().strip()
        if not value:
            self.choose_synthetic_output()
            value = self.synthetic_output_dir.get().strip()
        return Path(value).resolve() if value else None

    @staticmethod
    def next_output_key(directory, prefix, suffix=".png"):
        """Return the first unused numbered key, preserving every previous export."""
        index = 0
        while (directory / f"{prefix}_{index:06d}{suffix}").exists():
            index += 1
        return f"{prefix}_{index:06d}"

    def validate_synthetic_source(self):
        if self.sample_image is None or self.background_image is None:
            messagebox.showinfo("Thiếu ảnh", "Hãy chọn cả ảnh mẫu và background.")
            return False
        if not self.sample_boxes or self.region is None:
            messagebox.showinfo("Thiếu nhãn", "Hãy vẽ box date và vùng đặt date.")
            return False
        if not AUTO_ERROR_MODES:
            messagebox.showinfo(
                "Chưa lưu mode",
                "Hãy tạo một dạng lỗi rồi bấm 'Lưu mode hiện tại' trước khi xuất.")
            return False
        return True

    def automatic_error_state(self, mode):
        """Build one randomized state for an automatic error type."""
        state = self.capture_effect_state()
        effect_mode = "cut" if mode.startswith("cut_") else mode
        state.update({
            "mode": effect_mode,
            "last_effect_mode": "normal" if mode == "delete_objects" else effect_mode,
            "spacing_groups": [],
            "cut_line": None,
            "cut_lines": [],
            "cut_side": "none",
            "deleted_indices": set(),
        })
        if mode == "spacing":
            state["group_x_factor"] = random.choice((
                random.uniform(.35, .78), random.uniform(1.25, 2.1)))
        elif mode == "overlap":
            state["stack_size"] = random.randint(2, max(2, min(5, len(self.sample_boxes))))
        elif mode == "arc":
            state["arc_height"] = random.choice((-1, 1)) * random.uniform(18.0, 90.0)
        elif mode == "blur":
            state["blur_radius"] = random.uniform(1.2, 7.5)
        elif mode == "delete_objects":
            count = len(self.sample_boxes)
            delete_count = random.randint(1, max(1, count - 1))
            state["deleted_indices"] = set(random.sample(range(count), delete_count))
        return state

    def compose_automatic_error(self, mode, variant=0):
        """Compose an error plus safe placement jitter for non-duplicate output."""
        original_region = self.region
        jittered = copy.deepcopy(original_region)
        jittered.cx += random.uniform(-.018, .018) * original_region.w
        jittered.cy += random.uniform(-.018, .018) * original_region.h
        jittered.angle += random.uniform(-.025, .025)
        state = self.automatic_error_state(mode)
        if mode.startswith("cut_"):
            normal_boxes = layout_boxes(
                self.sample_boxes, jittered, "normal", 1, 0,
                min(self.background_image.width / self.sample_image.width,
                    self.background_image.height / self.sample_image.height))
            cosine, sine = math.cos(jittered.angle), math.sin(jittered.angle)

            # Split by the largest gap on the region-local vertical axis. This
            # remains correct when the two-row date or its target region is rotated.
            ordered = sorted(
                normal_boxes,
                key=lambda box: -(box.cx - jittered.cx) * sine
                                + (box.cy - jittered.cy) * cosine)
            local_y = [-(box.cx - jittered.cx) * sine
                       + (box.cy - jittered.cy) * cosine for box in ordered]
            if len(ordered) < 2:
                rows = [ordered]
            else:
                split_at = max(range(1, len(ordered)),
                               key=lambda index: local_y[index] - local_y[index - 1])
                rows = [ordered[:split_at], ordered[split_at:]]

            requested_rows = ([0] if mode == "cut_row_1" else
                              [1] if mode == "cut_row_2" else
                              list(range(len(rows))))
            if any(index >= len(rows) for index in requested_rows):
                raise ValueError("Date phải có đủ 2 hàng để tạo lỗi cắt từng hàng.")
            half = jittered.w * .65
            lines = []
            for row_index in requested_rows:
                row = rows[row_index]
                row_y = sum(-(box.cx - jittered.cx) * sine
                            + (box.cy - jittered.cy) * cosine
                            for box in row) / len(row)
                row_y += random.uniform(-.12, .12) * min(box.h for box in row)
                center_x = jittered.cx - sine * row_y
                center_y = jittered.cy + cosine * row_y
                lines.append(((center_x - cosine * half, center_y - sine * half),
                              (center_x + cosine * half, center_y + sine * half)))
            state["cut_line"] = lines[0]
            state["cut_lines"] = lines
            state["cut_side"] = "above" if variant % 2 == 0 else "below"
        self.region = jittered
        try:
            return self.compose_saved_mode({"name": mode, "state": state})
        finally:
            self.region = original_region

    def save_synthetic_previews(self):
        if not self.validate_synthetic_source():
            return
        target = self.synthetic_target()
        if target is None:
            return
        preview_dir = target / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        created = []
        try:
            background_name = self.backgrounds[self.background_index].stem
            for saved_mode in self.saved_modes:
                image, boxes = self.compose_saved_mode(saved_mode)
                if not all(box_inside_box(box, self.region) for box in boxes):
                    raise ValueError(f"{saved_mode['name']}: date vượt khỏi vùng chứa.")
                prefix = f"preview_{background_name}_{saved_mode['name']}"
                key = self.next_output_key(preview_dir, prefix)
                image.save(preview_dir / f"{key}.png")
                created.append(key)
            self.status.set(f"Đã tạo {len(created)} ảnh kiểm tra tại {preview_dir}")
            messagebox.showinfo("Ảnh kiểm tra",
                                f"Đã lưu {len(created)} ảnh, mỗi dạng một ảnh tại:\n{preview_dir}")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không tạo được ảnh kiểm tra", str(exc))

    def export_synthetic(self):
        if not self.validate_synthetic_source():
            return
        count = simpledialog.askinteger("Số lượng", "Số ảnh cho mỗi dạng lỗi:",
                                        initialvalue=self.export_count.get(),
                                        minvalue=1, maxvalue=10000)
        if count is None:
            return
        self.export_count.set(count)
        self.save_config()
        target = self.synthetic_target()
        if target is None:
            return
        images_dir, labels_dir = target / "images", target / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        modes = list(AUTO_ERROR_MODES)
        total = 0
        try:
            background_name = self.backgrounds[self.background_index].stem
            sample_name = self.samples[self.sample_index].stem
            for mode in modes:
                prefix = f"synthetic_{background_name}_{sample_name}_auto_{mode}"
                fingerprints = set()
                for variant in range(count):
                    for _attempt in range(100):
                        image, boxes = self.compose_automatic_error(mode, variant)
                        fingerprint = hashlib.sha256(image.tobytes()).digest()
                        if (fingerprint not in fingerprints
                                and all(box_inside_box(box, self.region) for box in boxes)):
                            fingerprints.add(fingerprint)
                            break
                    else:
                        raise ValueError(
                            f"Không thể tạo đủ {count} ảnh khác nhau cho lỗi {mode}. "
                            "Hãy tăng vùng đặt date hoặc giảm số lượng.")
                    key = self.next_output_key(images_dir, prefix)
                    # Check both destinations in case a previous export was interrupted.
                    while ((images_dir / f"{key}.png").exists()
                           or (labels_dir / f"{key}.txt").exists()):
                        number = int(key.rsplit("_", 1)[1]) + 1
                        key = f"{prefix}_{number:06d}"
                    image.save(images_dir / f"{key}.png")
                    atomic_write(labels_dir / f"{key}.txt",
                                 encode(boxes, *self.background_image.size))
                    total += 1
            # Preserve an existing dataset's metadata as well as all images/labels.
            if not (target / "classes.txt").exists():
                atomic_write(target / "classes.txt", "date\n")
            if not (target / "data.yaml").exists():
                atomic_write(target / "data.yaml",
                             "path: " + json.dumps(target.as_posix())
                             + "\ntrain: images\nnc: 1\nnames:\n  0: date\n")
            self.save_config()
            messagebox.showinfo(
                "Hoàn tất",
                f"Đã ghi tiếp {total} ảnh ({count} ảnh × {len(modes)} dạng) và label tại:\n{target}")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không xuất được", str(exc))
        finally:
            self.show_preview()


if __name__ == "__main__":
    app_root = tk.Tk()
    SyntheticLabelTool(app_root)
    app_root.mainloop()
