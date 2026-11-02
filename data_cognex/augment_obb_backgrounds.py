
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ================= CẤU HÌNH ĐƯỜNG DẪN TẠI ĐÂY =================
SOURCE_DATASET = r"D:\data_cognex\data_create"
# Thư mục cha lưu các lần chạy dataset tăng cường.
OUTPUT_FOLDER = r"D:\data_cognex\data_cognex_create_bg"

COPY_SOURCE_DATASET_FIRST = True
# True: chép cả ảnh gốc vào dataset mới và bỏ nhãn line1/line2.
INCLUDE_ORIGINAL_DATA = True
DEFAULT_COUNT = 9
DEFAULT_SEED = 1234
SPLIT_RATIOS = {"train": 1.0, "val": 0.0, "test": 0.0}
# Giữ nguyên tỷ lệ pixel và kích thước ký tự của ảnh mẫu.
FIXED_IMAGE_SCALE = 1.0
FIXED_ROW_GROUPS = (
    ((0, 1, 2), (3, 4, 5, 6, 7, 8), (9, 10, 11), (12, 13, 14)),
    ((15, 16, 17), (18, 19, 20, 21, 22, 23), (24, 25, 26, 27, 28)),
)
SPACING_BOX_GAP_PIXELS = 2
SPACING_GROUP_GAP_PIXELS = 2
MIN_OVERLAP_STACK_BOXES = 1
MAX_OVERLAP_STACK_BOXES = 5
# Độ võng so với chiều cao box trung bình. Biên độ tách khỏi 0 để cung luôn
# nhìn rõ; hướng âm/dương tương ứng hai dạng cong lên và cong xuống.
MIN_ARC_HEIGHT_RATIO = 0.65
MAX_ARC_HEIGHT_RATIO = 1.10
ARC_DIRECTIONS = (-1, 1)
# Mỗi ảnh output chỉ dùng đúng một trường hợp. Danh sách này cũng quyết định
# thứ tự luân phiên, giúp số lượng giữa các trường hợp cân bằng.
AUGMENTATION_CASES = ("group_spacing", "box_overlap", "vertical_arc")
# Chỉ dùng khi tâm char nằm hơi ngoài line do sai số/nhãn sát mép.
MAX_CENTER_OUTSIDE_RATIO = 0.10
# Tách foreground trong từng box char. Dữ liệu hiện tại là nét tối trên nền sáng.
# Có thể đổi thành "light" hoặc "auto" nếu nguồn có cực tính khác.
FOREGROUND_POLARITY = "dark"
# Giữ nguyên mọi mảng tối, kể cả dấu gạch ngang rộng; chỉ nền ngoài bị trong suốt.
FOREGROUND_COLOR = (0, 0, 0)
FOREGROUND_EDGE_FEATHER = 6
MORPHOLOGY_SIZE = 3
# =============================================================

BASE = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_NAMES = ["char"]


@dataclass(frozen=True)
class Obb:
    class_id: int
    points: tuple[tuple[float, float], ...]

    @property
    def center(self):
        return (sum(point[0] for point in self.points) / 4,
                sum(point[1] for point in self.points) / 4)


@dataclass(frozen=True)
class LineGroup:
    image_path: Path
    image_size: tuple[int, int]
    line: Obb
    chars: tuple[Obb, ...]


@dataclass(frozen=True)
class Placement:
    source_center: tuple[float, float]
    destination_center: tuple[float, float]
    scale: float
    angle: float
    bounds: tuple[float, float, float, float]


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else BASE / path).resolve()


def read_names(source, yaml_module):
    config_path = source / "data.yaml"
    classes_path = source / "classes.txt"
    if config_path.is_file():
        config = yaml_module.safe_load(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict):
            raise ValueError("data.yaml phải là một YAML mapping.")
        value = config.get("names")
    elif classes_path.is_file():
        value = [line.strip() for line in classes_path.read_text(
            encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        raise FileNotFoundError(
            f"Dataset nguồn cần data.yaml hoặc classes.txt: {source}")
    if isinstance(value, dict):
        indexed = {int(key): str(name) for key, name in value.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("Class ID trong data.yaml phải liên tiếp từ 0.")
        names = [indexed[index] for index in range(len(indexed))]
    elif isinstance(value, list):
        names = [str(name) for name in value]
    else:
        raise ValueError("data.yaml thiếu names hợp lệ.")
    if not names or len(names) != len(set(names)) or any(not name.strip() for name in names):
        raise ValueError("Tên lớp phải duy nhất và không được trống.")
    has_lines = names.count("line1") == 1 and names.count("line2") == 1
    if ("line1" in names or "line2" in names) and not has_lines:
        raise ValueError("Dataset có nhãn dòng phải có đủ đúng một lớp line1 và line2.")
    return names


def dataset_split_roots(source):
    """Nhận cả YOLO chuẩn images/train lẫn dataset phẳng images/."""
    train_images = source / "images" / "train"
    train_labels = source / "labels" / "train"
    if train_images.is_dir() or train_labels.is_dir():
        if not train_images.is_dir() or not train_labels.is_dir():
            raise FileNotFoundError(
                "Dataset nguồn phải có đồng thời images/train và labels/train.")
        return train_images, train_labels
    image_root, label_root = source / "images", source / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            "Dataset nguồn cần images + labels hoặc images/train + labels/train.")
    return image_root, label_root


def inferred_line_groups(chars):
    """Chia nhãn char-only thành hai dòng theo khoảng cách tâm theo trục dọc."""
    expected_rows = tuple(tuple(class_id for group in groups for class_id in group)
                          for groups in FIXED_ROW_GROUPS)
    expected_count = sum(len(row) for row in expected_rows)
    # qq1.txt dùng đúng một class 0 cho toàn bộ ký tự. Gán ID nội bộ theo thứ tự
    # dòng trong file để cấu hình box 0-28 không bị nhầm thành class huấn luyện.
    if len(chars) == expected_count and all(box.class_id == 0 for box in chars):
        indexed = tuple(Obb(index, box.points) for index, box in enumerate(chars))
        first_count = len(expected_rows[0])
        return indexed[:first_count], indexed[first_count:]
    by_id = {box.class_id: box for box in chars}
    expected = {class_id for rows in FIXED_ROW_GROUPS for group in rows
                for class_id in group}
    if set(by_id) == expected and len(by_id) == len(chars):
        return (tuple(by_id[class_id] for group in FIXED_ROW_GROUPS[0]
                      for class_id in group),
                tuple(by_id[class_id] for group in FIXED_ROW_GROUPS[1]
                      for class_id in group))
    if len(chars) < 2:
        raise ValueError("Ảnh char-only cần ít nhất hai box để suy ra hai dòng.")
    ordered = sorted(chars, key=lambda box: (box.center[1], box.center[0]))
    best = None
    for index in range(1, len(ordered)):
        first, second = ordered[:index], ordered[index:]
        means = [sum(box.center[1] for box in group) / len(group)
                 for group in (first, second)]
        score = sum((box.center[1] - mean) ** 2
                    for group, mean in zip((first, second), means)
                    for box in group)
        candidate = (score, index, first, second)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2], best[3]


def enclosing_line(chars, class_id=-1):
    points = [point for box in chars for point in box.points]
    left, right = min(point[0] for point in points), max(point[0] for point in points)
    top, bottom = min(point[1] for point in points), max(point[1] for point in points)
    return Obb(class_id, ((left, top), (right, top), (right, bottom), (left, bottom)))


def parse_label(path, width, height, class_count):
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 9:
                raise ValueError("cần class_id và 8 tọa độ")
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            fixed_ids = {item for rows in FIXED_ROW_GROUPS for group in rows
                         for item in group}
            if not (0 <= class_id < class_count or class_id in fixed_ids):
                raise ValueError(f"class ID {class_id} nằm ngoài data.yaml")
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError("tọa độ phải hữu hạn và nằm trong [0, 1]")
            points = tuple((values[index] * width, values[index + 1] * height)
                           for index in range(0, 8, 2))
            if abs(polygon_area(points)) < 1:
                raise ValueError("box có diện tích quá nhỏ")
            boxes.append(Obb(class_id, points))
        except ValueError as exc:
            raise ValueError(f"{path}, dòng {line_number}: {exc}") from exc
    return boxes


def polygon_area(points):
    return sum(points[index][0] * points[(index + 1) % len(points)][1]
               - points[(index + 1) % len(points)][0] * points[index][1]
               for index in range(len(points))) / 2


def polygon_contains(outer, inner):
    """Kiểm tra mọi đỉnh inner nằm trong/trên một OBB outer lồi."""
    winding = 1 if polygon_area(outer) >= 0 else -1
    for point in inner:
        for index, start in enumerate(outer):
            end = outer[(index + 1) % len(outer)]
            cross = ((end[0] - start[0]) * (point[1] - start[1])
                     - (end[1] - start[1]) * (point[0] - start[0]))
            # Dung sai nhỏ cho tọa độ nhãn đã làm tròn.
            if winding * cross < -0.02:
                return False
    return True


def expand_line_to_chars(line, chars):
    """Mở rộng OBB dòng theo hai trục cũ để bao trọn mọi box char được gán."""
    first, second, _, fourth = line.points
    axis_x = (second[0] - first[0], second[1] - first[1])
    length = math.hypot(*axis_x)
    if length < 1e-9:
        raise ValueError("Box dòng có cạnh bằng 0.")
    axis_x = (axis_x[0] / length, axis_x[1] / length)
    axis_y = (-axis_x[1], axis_x[0])
    if ((fourth[0] - first[0]) * axis_y[0]
            + (fourth[1] - first[1]) * axis_y[1]) < 0:
        axis_y = (-axis_y[0], -axis_y[1])
    points = [point for box in (line, *chars) for point in box.points]
    projected_x = [point[0] * axis_x[0] + point[1] * axis_x[1] for point in points]
    projected_y = [point[0] * axis_y[0] + point[1] * axis_y[1] for point in points]
    minimum_x, maximum_x = min(projected_x), max(projected_x)
    minimum_y, maximum_y = min(projected_y), max(projected_y)

    def restore(x, y):
        return (x * axis_x[0] + y * axis_y[0],
                x * axis_x[1] + y * axis_y[1])

    return Obb(line.class_id, (restore(minimum_x, minimum_y),
                               restore(maximum_x, minimum_y),
                               restore(maximum_x, maximum_y),
                               restore(minimum_x, maximum_y)))


def normalized_center_distance(line, point):
    """Khoảng cách ngoài OBB theo tỷ lệ cạnh; 0 nghĩa là tâm đang ở bên trong."""
    first, second, _, fourth = line.points
    horizontal = (second[0] - first[0], second[1] - first[1])
    vertical = (fourth[0] - first[0], fourth[1] - first[1])
    horizontal_squared = horizontal[0] ** 2 + horizontal[1] ** 2
    vertical_squared = vertical[0] ** 2 + vertical[1] ** 2
    if horizontal_squared < 1e-9 or vertical_squared < 1e-9:
        return math.inf
    relative = (point[0] - first[0], point[1] - first[1])
    x = (relative[0] * horizontal[0] + relative[1] * horizontal[1]) / horizontal_squared
    y = (relative[0] * vertical[0] + relative[1] * vertical[1]) / vertical_squared
    outside_x = max(-x, 0.0, x - 1.0)
    outside_y = max(-y, 0.0, y - 1.0)
    return math.hypot(outside_x, outside_y)


def load_groups(source, names, image_module):
    image_root, label_root = dataset_split_roots(source)
    images = sorted((path for path in image_root.rglob("*")
                     if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda path: path.relative_to(image_root).as_posix())
    if not images:
        raise ValueError("Không tìm thấy ảnh nguồn trong images/train.")

    has_line_labels = "line1" in names and "line2" in names
    line_ids = ({"line1": names.index("line1"), "line2": names.index("line2")}
                if has_line_labels else {})
    groups = {"line1": [], "line2": []}
    seen_labels = set()
    for image_path in images:
        relative = image_path.relative_to(image_root)
        label_path = label_root / relative.with_suffix(".txt")
        if label_path in seen_labels:
            raise ValueError(f"Hai ảnh ánh xạ tới cùng nhãn: {label_path}")
        seen_labels.add(label_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Ảnh thiếu file nhãn: {label_path}")
        try:
            with image_module.open(image_path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Không đọc được ảnh {image_path}: {exc}") from exc
        boxes = parse_label(label_path, width, height, len(names))
        if not has_line_labels:
            first, second = inferred_line_groups(boxes)
            for kind, chars in (("line1", first), ("line2", second)):
                groups[kind].append(LineGroup(
                    image_path, (width, height), enclosing_line(chars), tuple(chars)))
            continue
        line_boxes = [(kind, box) for kind, class_id in line_ids.items()
                      for box in boxes if box.class_id == class_id]
        chars = [box for box in boxes if box.class_id not in line_ids.values()]
        assignments = {id(line_box): [] for _, line_box in line_boxes}
        for box in chars:
            # Nhãn thực tế có thể để góc char nhô nhẹ khỏi line; tâm char mới là
            # tín hiệu ổn định để xác định dòng. OBB dòng sẽ được nới sau đó.
            owners = [(kind, line_box) for kind, line_box in line_boxes
                      if polygon_contains(line_box.points, (box.center,))]
            if not owners:
                distances = [(normalized_center_distance(line_box, box.center), kind, line_box)
                             for kind, line_box in line_boxes]
                distance, kind, line_box = min(distances, default=(math.inf, None, None),
                                                key=lambda item: item[0])
                if distance <= MAX_CENTER_OUTSIDE_RATIO:
                    owners = [(kind, line_box)]
            if len(owners) != 1:
                raise ValueError(
                    f"{label_path}: tâm mỗi box char phải thuộc đúng một line1/line2; "
                    f"tâm box hiện thuộc {len(owners)} dòng và không nằm gần duy nhất một dòng.")
            assignments[id(owners[0][1])].append(box)
        for kind, line_box in line_boxes:
            contained = tuple(assignments[id(line_box)])
            if not contained:
                raise ValueError(f"{label_path}: box {kind} không chứa tâm box char nào.")
            expanded_line = expand_line_to_chars(line_box, contained)
            if not all(polygon_contains(expanded_line.points, box.points) for box in contained):
                raise ValueError(f"{label_path}: không thể nới box {kind} để chứa các box char.")
            groups[kind].append(LineGroup(
                image_path, (width, height), expanded_line, contained))
    for kind in ("line1", "line2"):
        if not groups[kind]:
            raise ValueError(f"Không có nhóm {kind} hợp lệ trong labels/train.")
    return groups, len(images)


def transform_point(point, placement):
    source_x, source_y = placement.source_center
    target_x, target_y = placement.destination_center
    cosine, sine = math.cos(placement.angle), math.sin(placement.angle)
    x, y = point[0] - source_x, point[1] - source_y
    return (target_x + placement.scale * (cosine * x - sine * y),
            target_y + placement.scale * (sine * x + cosine * y))


def fixed_pair_placements(selected, canvas_size, rng, case):
    """Apply exactly one configured layout case to the rigid two-row pair."""
    if case not in AUGMENTATION_CASES:
        raise ValueError(f"Unknown augmentation case: {case}")
    rows = (selected["line1"].chars, selected["line2"].chars)
    expected_rows = tuple(tuple(class_id for group in groups for class_id in group)
                          for groups in FIXED_ROW_GROUPS)
    actual_rows = tuple(tuple(box.class_id for box in row) for row in rows)
    if actual_rows != expected_rows:
        raise ValueError(
            "Fixed layout requires boxes 0-14 on row 1 and 15-28 on row 2.")

    # Mặc định giữ nguyên tâm box. Từng nhánh dưới chỉ thay đổi đúng một đặc tính.
    target_centers = {
        (row_index, box.class_id): box.center
        for row_index, boxes in enumerate(rows) for box in boxes
    }

    if case == "group_spacing":
        # Mỗi hàng chia đúng hai nhóm. Khoảng cách giữa từng box là 1 px và
        # khoảng cách giữa hai nhóm cũng là 1 px trên ảnh output.
        box_gap = SPACING_BOX_GAP_PIXELS
        group_gap = SPACING_GROUP_GAP_PIXELS
        for row_index, boxes in enumerate(rows):
            split_index = (len(boxes) + 1) // 2
            two_groups = (boxes[:split_index], boxes[split_index:])
            cursor = min(point[0] for box in boxes for point in box.points)
            previous_right = None
            for group_index, group_boxes in enumerate(two_groups):
                if group_index:
                    cursor = previous_right + group_gap
                for box_index, box in enumerate(group_boxes):
                    width = (max(point[0] for point in box.points)
                             - min(point[0] for point in box.points))
                    if box_index:
                        cursor = previous_right + box_gap
                    target_centers[(row_index, box.class_id)] = (
                        cursor + width / 2, box.center[1])
                    previous_right = cursor + width

    elif case == "box_overlap":
        # Chia tuần tự thành các chồng ngẫu nhiên gồm 1-5 box. Mọi box trong
        # cùng chồng dùng chung tâm X; các chồng kề nhau cách nhau 1 px.
        for row_index, boxes in enumerate(rows):
            cursor = min(point[0] for box in boxes for point in box.points)
            offset = 0
            while offset < len(boxes):
                stack_size = rng.randint(
                    MIN_OVERLAP_STACK_BOXES,
                    min(MAX_OVERLAP_STACK_BOXES, len(boxes) - offset),
                )
                stack = boxes[offset:offset + stack_size]
                stack_width = max(
                    max(point[0] for point in box.points)
                    - min(point[0] for point in box.points)
                    for box in stack)
                stack_center_x = cursor + stack_width / 2
                for box in stack:
                    target_centers[(row_index, box.class_id)] = (
                        stack_center_x, box.center[1])
                cursor += stack_width + SPACING_BOX_GAP_PIXELS
                offset += stack_size

    elif case == "vertical_arc":
        # Không thay đổi X. Cùng một hàm Y được áp cho cả hai hàng, vì vậy chúng
        # luôn cong đồng thời và không bị đặt ngẫu nhiên tách rời nhau.
        all_boxes = tuple(box for row in rows for box in row)
        left = min(box.center[0] for box in all_boxes)
        right = max(box.center[0] for box in all_boxes)
        middle_x = (left + right) / 2
        half_width = max((right - left) / 2, 1.0)
        mean_height = sum(
            max(point[1] for point in box.points)
            - min(point[1] for point in box.points)
            for box in all_boxes) / len(all_boxes)
        arc_height = (rng.choice(ARC_DIRECTIONS)
                      * rng.uniform(MIN_ARC_HEIGHT_RATIO, MAX_ARC_HEIGHT_RATIO)
                      * mean_height)
        for key, (x, y) in tuple(target_centers.items()):
            normalized_x = (x - middle_x) / half_width
            target_centers[key] = (x, y + arc_height * normalized_x ** 2)

    source_center = (
        sum(center[0] for center in target_centers.values()) / len(target_centers),
        sum(center[1] for center in target_centers.values()) / len(target_centers),
    )
    canvas_center = (canvas_size[0] / 2, canvas_size[1] / 2)
    placements = {"line1": [], "line2": []}
    for row_index, (kind, boxes) in enumerate(zip(("line1", "line2"), rows)):
        for box in boxes:
            local_center = target_centers[(row_index, box.class_id)]
            destination = (
                canvas_center[0] + local_center[0] - source_center[0],
                canvas_center[1] + local_center[1] - source_center[1],
            )
            placements[kind].append(Placement(
                box.center, destination, FIXED_IMAGE_SCALE, 0.0, (0, 0, 0, 0)))
    return placements


def paste_fixed_group(canvas, group, placements, image_module, draw_module,
                      chops_module, filter_module):
    """Transform each character so its pixels and OBB remain synchronized."""
    with image_module.open(group.image_path) as source_file:
        original = source_file.convert("RGB")
    for box, placement in zip(group.chars, placements):
        mask = char_foreground_mask(
            original, (box,), image_module, draw_module, chops_module, filter_module)
        ink = image_module.new("RGB", original.size, FOREGROUND_COLOR)
        coefficients = affine_coefficients(placement)
        options = dict(method=image_module.Transform.AFFINE, data=coefficients)
        warped = ink.transform(canvas.size, resample=image_module.Resampling.BICUBIC,
                               fillcolor=(0, 0, 0), **options)
        warped_mask = mask.transform(canvas.size, resample=image_module.Resampling.BILINEAR,
                                     fillcolor=0, **options)
        canvas.paste(warped, (0, 0), warped_mask)


def make_background(size, rng, image_module, draw_module, filter_module):
    """Sinh nền không chứa đối tượng để tránh tạo false-negative label."""
    width, height = size
    base = tuple(rng.randint(25, 230) for _ in range(3))
    canvas = image_module.new("RGB", size, base)
    draw = draw_module.Draw(canvas, "RGB")
    mode = rng.choice(("gradient", "bands", "tiles", "noise", "soft_shapes", "plain"))
    accent = tuple(max(0, min(255, channel + rng.randint(-70, 70))) for channel in base)
    if mode == "gradient":
        horizontal = rng.choice((True, False))
        length = width if horizontal else height
        for position in range(length):
            ratio = position / max(length - 1, 1)
            color = tuple(round(base[index] * (1 - ratio) + accent[index] * ratio)
                          for index in range(3))
            if horizontal:
                draw.line((position, 0, position, height), fill=color)
            else:
                draw.line((0, position, width, position), fill=color)
    elif mode == "bands":
        step = rng.randint(6, max(7, min(width, height) // 5))
        for position in range(-height, width, step * 2):
            draw.polygon(((position, 0), (position + step, 0),
                          (position + height + step, height), (position + height, height)),
                         fill=accent)
    elif mode == "tiles":
        tile = rng.randint(8, max(9, min(width, height) // 4))
        for row, y in enumerate(range(0, height, tile)):
            for column, x in enumerate(range(0, width, tile)):
                if (row + column) % 2:
                    draw.rectangle((x, y, x + tile, y + tile), fill=accent)
    elif mode == "noise":
        small_size = (max(2, width // 12), max(2, height // 12))
        noise = image_module.new("L", small_size)
        noise.putdata([rng.randrange(256) for _ in range(small_size[0] * small_size[1])])
        noise = noise.resize(size, image_module.Resampling.BILINEAR).filter(
            filter_module.GaussianBlur(rng.uniform(1.0, 3.0)))
        tint = image_module.new("RGB", size, accent)
        canvas = image_module.composite(tint, canvas, noise.point(lambda value: value // 3))
    elif mode == "soft_shapes":
        overlay = image_module.new("RGB", size, base)
        overlay_draw = draw_module.Draw(overlay, "RGB")
        for _ in range(rng.randint(4, 10)):
            lower = max(12, min(width, height) // 5)
            upper = max(lower, min(width, height))
            diameter = rng.randint(lower, upper)
            x, y = rng.randint(-diameter, width), rng.randint(-diameter, height)
            color = tuple(max(0, min(255, channel + rng.randint(-35, 35)))
                          for channel in base)
            overlay_draw.ellipse((x, y, x + diameter, y + diameter), fill=color)
        radius = max(2, min(width, height) / 20)
        overlay = overlay.filter(filter_module.GaussianBlur(radius))
        canvas = image_module.blend(canvas, overlay, 0.65)
    return canvas


def affine_coefficients(placement):
    """Pillow cần ánh xạ nghịch từ pixel đích về pixel nguồn."""
    cosine, sine = math.cos(placement.angle), math.sin(placement.angle)
    inverse_scale = 1 / placement.scale
    a, b = cosine * inverse_scale, sine * inverse_scale
    d, e = -sine * inverse_scale, cosine * inverse_scale
    source_x, source_y = placement.source_center
    target_x, target_y = placement.destination_center
    return (a, b, source_x - a * target_x - b * target_y,
            d, e, source_y - d * target_x - e * target_y)


def foreground_alpha(image, threshold, feather, polarity):
    """Đổi mức xám thành alpha; vùng tối được giữ nguyên khi polarity=dark."""
    low = max(0, threshold - feather)
    high = min(255, threshold + feather)
    if high <= low:
        high = min(255, low + 1)
    if polarity == "dark":
        table = [255 if value <= low else 0 if value >= high
                 else round((high - value) * 255 / (high - low)) for value in range(256)]
    else:
        table = [0 if value <= low else 255 if value >= high
                 else round((value - low) * 255 / (high - low)) for value in range(256)]
    return image.point(table)


def otsu_threshold(image, mask):
    histogram = image.histogram(mask)
    total = sum(histogram)
    if total == 0:
        return 0
    weighted_total = sum(value * count for value, count in enumerate(histogram))
    background_weight = 0
    background_sum = 0
    best_variance = -1.0
    best_threshold = 0
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def histogram_percentile(histogram, ratio):
    target = sum(histogram) * ratio
    cumulative = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return value
    return 255


def char_foreground_mask(source, boxes, image_module, draw_module, chops_module,
                         filter_module):
    """Tạo alpha mask chỉ giữ nét chữ/số, loại nền cục bộ trong từng OBB char."""
    if FOREGROUND_POLARITY not in {"dark", "light", "auto"}:
        raise ValueError("FOREGROUND_POLARITY chỉ nhận dark, light hoặc auto.")
    grayscale = source.convert("L")
    combined = image_module.new("L", source.size, 0)
    for box_index, box in enumerate(boxes, 1):
        xs = [point[0] for point in box.points]
        ys = [point[1] for point in box.points]
        padding = MORPHOLOGY_SIZE + 2
        left = max(0, math.floor(min(xs)) - padding)
        top = max(0, math.floor(min(ys)) - padding)
        right = min(source.width, math.ceil(max(xs)) + padding + 1)
        bottom = min(source.height, math.ceil(max(ys)) + padding + 1)
        crop = grayscale.crop((left, top, right, bottom))
        polygon = tuple((x - left, y - top) for x, y in box.points)
        roi = image_module.new("L", crop.size, 0)
        draw_module.Draw(roi).polygon(polygon, fill=255)

        polarity = FOREGROUND_POLARITY
        if polarity == "auto":
            histogram = crop.histogram(roi)
            low_value = histogram_percentile(histogram, 0.10)
            middle = histogram_percentile(histogram, 0.50)
            high_value = histogram_percentile(histogram, 0.90)
            polarity = "dark" if middle - low_value >= high_value - middle else "light"
        threshold = otsu_threshold(crop, roi)
        alpha = foreground_alpha(crop, threshold, FOREGROUND_EDGE_FEATHER, polarity)
        if MORPHOLOGY_SIZE >= 3:
            size = MORPHOLOGY_SIZE if MORPHOLOGY_SIZE % 2 else MORPHOLOGY_SIZE + 1
            # Closing nối nét đứt mà không khôi phục mảng nền hình chữ nhật.
            alpha = alpha.filter(filter_module.MaxFilter(size)).filter(
                filter_module.MinFilter(size))
        alpha = chops_module.multiply(alpha, roi).filter(filter_module.GaussianBlur(0.45))
        alpha = chops_module.multiply(alpha, roi)
        if alpha.getbbox() is None:
            raise ValueError(
                f"Không tách được nét đen ở char {box_index}; hãy kiểm tra nhãn nguồn.")
        layer = image_module.new("L", source.size, 0)
        layer.paste(alpha, (left, top))
        combined = chops_module.lighter(combined, layer)
    return combined


def paste_group(canvas, group, placement, image_module, draw_module,
                chops_module, filter_module):
    with image_module.open(group.image_path) as source_file:
        source = source_file.convert("RGB")
    mask = char_foreground_mask(
        source,
        group.chars,
        image_module,
        draw_module,
        chops_module,
        filter_module,
    )
    source = image_module.new("RGB", source.size, FOREGROUND_COLOR)
    coefficients = affine_coefficients(placement)
    options = dict(method=image_module.Transform.AFFINE, data=coefficients)
    warped = source.transform(
        canvas.size,
        resample=image_module.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
        **options,
    )
    warped_mask = mask.transform(
        canvas.size,
        resample=image_module.Resampling.BILINEAR,
        fillcolor=0,
        **options,
    )
    canvas.paste(warped, (0, 0), warped_mask)


def encode_box(class_id, points, width, height):
    values = []
    for x, y in points:
        if not (-1e-6 <= x <= width + 1e-6 and -1e-6 <= y <= height + 1e-6):
            raise ValueError("Phép biến đổi tạo box vượt biên ảnh.")
        values.extend((max(0.0, min(1.0, x / width)),
                       max(0.0, min(1.0, y / height))))
    return str(class_id) + " " + " ".join(f"{value:.8f}" for value in values)


def output_labels(group, placement, canvas_size):
    width, height = canvas_size
    return [
        encode_box(
            0,
            tuple(
                transform_point(point, placement)
                for point in box.points
            ),
            width,
            height,
        )
        for box in group.chars
    ]


def split_for_index(index, counts):
    offset = 0
    for split in ("train", "val", "test"):
        offset += counts[split]
        if index < offset:
            return split
    raise AssertionError("Chỉ số split không hợp lệ.")


def allocate_splits(total):
    if total < sum(ratio > 0 for ratio in SPLIT_RATIOS.values()):
        raise ValueError("Số ảnh output quá ít để mỗi split đang bật có ít nhất một ảnh.")
    exact = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: int(value) for name, value in exact.items()}
    for name, ratio in SPLIT_RATIOS.items():
        if ratio > 0 and counts[name] == 0:
            counts[name] = 1
    while sum(counts.values()) < total:
        name = max(SPLIT_RATIOS, key=lambda item: exact[item] - counts[item])
        counts[name] += 1
    while sum(counts.values()) > total:
        candidates = [name for name in SPLIT_RATIOS if counts[name] > int(SPLIT_RATIOS[name] > 0)]
        if not candidates:
            raise ValueError("Không thể phân bổ split với cấu hình hiện tại.")
        name = max(candidates, key=lambda item: counts[item] - exact[item])
        counts[name] -= 1
    return counts


def partition_donors(groups, rng):
    """Giữ mọi group của một ảnh nguồn trong đúng một split để tránh leakage."""
    active = [name for name, ratio in SPLIT_RATIOS.items() if ratio > 0]
    paths = sorted({group.image_path for items in groups.values() for group in items},
                   key=lambda path: path.as_posix())
    if len(paths) < len(active):
        raise ValueError(f"Cần ít nhất {len(active)} ảnh nguồn để tách donor cho "
                         f"{len(active)} split mà không rò rỉ dữ liệu.")
    path_counts = allocate_splits(len(paths))
    for _ in range(500):
        rng.shuffle(paths)
        owners, offset = {}, 0
        for split in ("train", "val", "test"):
            for path in paths[offset:offset + path_counts[split]]:
                owners[path] = split
            offset += path_counts[split]
        pools = {split: {"line1": [], "line2": []} for split in active}
        for kind, items in groups.items():
            for group in items:
                pools[owners[group.image_path]][kind].append(group)
        if all(pools[split][kind] for split in active for kind in ("line1", "line2")):
            return pools, owners
    raise ValueError("Không thể chia ảnh nguồn thành các split mà mỗi split đều có "
                     "line1 và line2. Hãy bổ sung dữ liệu nguồn hoặc đổi SPLIT_RATIOS.")


def char_only_label_text(path, line_ids, class_count):
    """Bỏ line1/line2 khỏi nhãn gốc, ánh xạ mọi lớp ký tự về class 0."""
    lines = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 9:
                raise ValueError("cần class_id và 8 tọa độ")
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            fixed_ids = {item for rows in FIXED_ROW_GROUPS for group in rows
                         for item in group}
            if not (0 <= class_id < class_count or class_id in fixed_ids):
                raise ValueError(f"class ID {class_id} nằm ngoài data.yaml")
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError("tọa độ phải hữu hạn và nằm trong [0, 1]")
            if class_id not in line_ids:
                lines.append("0 " + " ".join(fields[1:]))
        except ValueError as exc:
            raise ValueError(f"{path}, dòng {line_number}: {exc}") from exc
    return "\n".join(lines) + ("\n" if lines else "")


def collect_original_records(source, names, donor_owners):
    """Preflight ảnh/nhãn gốc trước khi tạo output và xác định split đích."""
    line_ids = ({names.index("line1"), names.index("line2")}
                if "line1" in names and "line2" in names else set())
    records = []
    flat_layout = not (source / "images" / "train").is_dir()
    source_splits = ("flat",) if flat_layout else ("train", "val", "test")
    for source_split in source_splits:
        if source_split == "flat":
            image_root, label_root = source / "images", source / "labels"
        else:
            image_root = source / "images" / source_split
            label_root = source / "labels" / source_split
        if not image_root.is_dir() and not label_root.is_dir():
            continue
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(
                f"Split {source_split} phải có cả images/{source_split} và labels/{source_split}.")
        images = sorted((path for path in image_root.rglob("*")
                         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                        key=lambda path: path.relative_to(image_root).as_posix())
        for image_path in images:
            relative = image_path.relative_to(image_root)
            label_path = label_root / relative.with_suffix(".txt")
            if not label_path.is_file():
                raise FileNotFoundError(f"Ảnh gốc thiếu nhãn: {label_path}")
            text = char_only_label_text(label_path, line_ids, len(names))
            # Ảnh train dùng làm donor phải theo đúng split donor; ảnh val/test
            # giữ nguyên split nguồn. Ảnh train không làm donor được giữ ở train.
            destination_split = (donor_owners.get(image_path, "train")
                                 if source_split in {"train", "flat"} else source_split)
            records.append((image_path, relative, source_split, destination_split, text))
    if not records:
        raise ValueError("Không tìm thấy ảnh gốc để đưa vào dataset output.")
    return records


def copy_original_records(records, run):
    """Chép ảnh gốc trực tiếp cạnh ảnh augment, không tạo thư mục original."""
    counts = Counter()
    boxes = 0
    for image_path, relative, source_split, destination_split, text in records:
        # Ghép các thành phần đường dẫn để vẫn phẳng nhưng không trùng tên khi
        # dataset nguồn có thư mục con hoặc cùng stem ở các split khác nhau.
        flat_relative_name = "__".join(relative.parts)
        image_name = f"original_{source_split}_{flat_relative_name}"
        image_target = run / "images" / destination_split / image_name
        label_target = (
            run / "labels" / destination_split /
            Path(image_name).with_suffix(".txt")
        )
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_target)
        label_target.write_text(text, encoding="utf-8")
        counts[destination_split] += 1
        boxes += len(text.splitlines())
    return dict(counts), boxes


def prepare_output_copy(source, run):
    """Copy dataset nguồn rồi chuẩn bị images/labels của chính bản copy."""
    if COPY_SOURCE_DATASET_FIRST:
        shutil.copytree(source, run)
        # Bản copy ban đầu còn nhãn line1/line2. Chỉ xóa hai cây dữ liệu trong
        # bản copy rồi dựng lại bằng nhãn char-only; mọi file phụ khác của
        # dataset nguồn vẫn được giữ nguyên.
        for name in ("images", "labels"):
            copied_data = run / name
            if copied_data.exists():
                shutil.rmtree(copied_data)
    else:
        run.mkdir(parents=True, exist_ok=False)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Tách nét char, ghép lên nhiều background và xuất YOLO OBB chỉ có lớp char.")
    parser.add_argument("--source", default=SOURCE_DATASET,
                        help=("Dataset nguồn; nhận data.yaml/classes.txt và cấu trúc "
                              "images+labels hoặc images/train+labels/train"))
    parser.add_argument("--output", default=OUTPUT_FOLDER,
                        help="Thư mục cha chứa dataset mới; không được trùng/lồng source")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="Số ảnh tổng hợp cần tạo")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Seed để tái lập kết quả")
    args = parser.parse_args()

    # Import trễ: chỉ import module này sẽ không đọc ảnh hoặc tạo output.
    import yaml
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    source, output_parent = resolve_path(args.source), resolve_path(args.output)
    if args.count <= 0:
        raise ValueError("--count phải lớn hơn 0.")
    if not source.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy dataset nguồn: {source}. Hãy sửa SOURCE_DATASET hoặc --source.")
    if output_parent == source or output_parent.is_relative_to(source) or source.is_relative_to(output_parent):
        raise ValueError("Output và source phải tách biệt, không được lồng nhau.")
    if any(ratio < 0 for ratio in SPLIT_RATIOS.values()) or not math.isclose(
            sum(SPLIT_RATIOS.values()), 1.0, abs_tol=1e-9):
        raise ValueError("SPLIT_RATIOS phải không âm và có tổng bằng 1.")
    if not AUGMENTATION_CASES or len(set(AUGMENTATION_CASES)) != len(AUGMENTATION_CASES):
        raise ValueError("AUGMENTATION_CASES phải có các trường hợp duy nhất.")

    # Preflight toàn bộ nguồn trước khi tạo bất kỳ thư mục output nào.
    names = read_names(source, yaml)
    groups, source_image_count = load_groups(source, names, Image)
    split_counts = allocate_splits(args.count)
    rng = random.Random(args.seed)
    donor_pools, donor_owners = partition_donors(groups, rng)
    original_records = (collect_original_records(source, names, donor_owners)
                        if INCLUDE_ORIGINAL_DATA else [])
    schedule = [split_for_index(index, split_counts) for index in range(args.count)]
    rng.shuffle(schedule)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = output_parent / f"{source.name}_augmented_{timestamp}"
    prepare_output_copy(source, run)
    output_splits = {split for split, count in split_counts.items() if count}
    output_splits.update(record[3] for record in original_records)
    for split in ("train", "val", "test"):
        if split in output_splits:
            (run / "images" / split).mkdir(parents=True)
            (run / "labels" / split).mkdir(parents=True)
    config = {"path": run.as_posix(),
              **{split: f"images/{split}" for split in ("train", "val", "test")
                 if split in output_splits},
              "nc": len(OUTPUT_NAMES), "names": dict(enumerate(OUTPUT_NAMES))}
    (run / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (run / "classes.txt").write_text("\n".join(OUTPUT_NAMES) + "\n", encoding="utf-8")
    report = {
        "status": "running", "source": str(source), "output": str(run),
        "seed": args.seed, "requested": args.count, "source_train_images": source_image_count,
        "source_groups": {kind: len(items) for kind, items in groups.items()},
        "donor_groups_per_split": {
            split: {kind: len(items) for kind, items in kinds.items()}
            for split, kinds in donor_pools.items()
        },
        "split_counts": split_counts, "classes": OUTPUT_NAMES, "generated": 0,
        "include_original_data": INCLUDE_ORIGINAL_DATA,
        "copy_source_dataset_first": COPY_SOURCE_DATASET_FIRST,
        "original_images": 0,
        "original_split_counts": {},
        "box_counts": {name: 0 for name in OUTPUT_NAMES},
        "foreground_removal": {
            "polarity": FOREGROUND_POLARITY,
            "color": FOREGROUND_COLOR,
            "edge_feather": FOREGROUND_EDGE_FEATHER,
            "adaptive_threshold": "otsu_grayscale_per_char",
            "morphology_size": MORPHOLOGY_SIZE,
        },
        "image_box_scale": {
            "mode": "fixed_original_size",
            "value": FIXED_IMAGE_SCALE,
            "pixels_and_obb_transformed_together": True,
        },
        "flat_split_layout": True,
        "augmentation_cases": {
            "mode": "exactly_one_case_per_generated_image",
            "enabled": list(AUGMENTATION_CASES),
            "generated_counts": {case: 0 for case in AUGMENTATION_CASES},
        },
    }
    report_path = run / "augmentation_report.json"
    write_json(report_path, report)
    counts = Counter()
    try:
        if INCLUDE_ORIGINAL_DATA:
            original_counts, original_boxes = copy_original_records(original_records, run)
            report["original_images"] = len(original_records)
            report["original_split_counts"] = original_counts
            counts[0] += original_boxes
            write_json(report_path, report)
        for index, split in enumerate(schedule):
            pool = donor_pools[split]
            case = AUGMENTATION_CASES[index % len(AUGMENTATION_CASES)]
            pairs = [(first, second) for first in pool["line1"]
                     for second in pool["line2"]
                     if first.image_path == second.image_path]
            if not pairs:
                raise ValueError("Không có cặp line1/line2 thuộc cùng một ảnh nguồn.")
            first, second = rng.choice(pairs)
            selected = {"line1": first, "line2": second}
            canvas_size = first.image_size
            placements = fixed_pair_placements(
                selected, canvas_size, rng, case)

            # Kích thước canvas lấy từ dữ liệu thật; pixel nền luôn được tạo mới.
            canvas = make_background(canvas_size, rng, Image, ImageDraw, ImageFilter)
            label_lines = []
            for kind in ("line1", "line2"):
                group, box_placements = selected[kind], placements[kind]
                paste_fixed_group(
                    canvas,
                    group,
                    box_placements,
                    Image,
                    ImageDraw,
                    ImageChops,
                    ImageFilter,
                )
                lines = [encode_box(
                    0,
                    tuple(transform_point(point, placement) for point in box.points),
                    canvas_size[0],
                    canvas_size[1],
                ) for box, placement in zip(group.chars, box_placements)]
                label_lines.extend(lines)
                counts[0] += len(group.chars)
            key = f"aug_{case}_{index:08d}"
            canvas.save(run / "images" / split / f"{key}.png")
            (run / "labels" / split / f"{key}.txt").write_text(
                "\n".join(label_lines) + "\n", encoding="utf-8")
            report["generated"] = index + 1
            report["augmentation_cases"]["generated_counts"][case] += 1
            if (index + 1) % 100 == 0:
                report["box_counts"] = {name: counts[class_id]
                                        for class_id, name in enumerate(OUTPUT_NAMES)}
                write_json(report_path, report)
            print(f"[{index + 1}/{args.count}] {split}/{key}.png", flush=True)
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="incomplete", error=str(exc))
        raise
    finally:
        report["box_counts"] = {name: counts[class_id]
                                for class_id, name in enumerate(OUTPUT_NAMES)}
        write_json(report_path, report)
    print(f"Đã tạo dataset: {run}")


if __name__ == "__main__":
    main()
