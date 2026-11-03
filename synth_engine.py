"""Image-generation engine for synthetic date OBB data.

Nothing in this module uses Tk, so the same code serves the label tool's
editor, its batch dialog and the ``label_tool.py --batch`` command line.

Terms: a *code* is a sample image plus its character boxes, a *target* is a
background image plus the one region where the date is placed, a *pair* is
one code placed on one target, and a *mode* is one error type.
"""
from __future__ import annotations

import copy
import csv
import functools
import hashlib
import json
import math
import os
import random
import re
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

from label_obb import Box, atomic_write, decode, encode


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EFFECT_MODES = ("normal", "spacing", "overlap", "arc", "blur", "cut", "delete_objects")
CUT_SIDES = ("above", "below")
AUTO_ERROR_MODES = (
    "spacing", "overlap", "arc", "blur",
    "cut_row_1", "cut_row_2", "cut_both_rows",
    "delete_objects", "out_line",
)
# Saved effect states use format 2: cut lines are stored relative to the date
# (see local_to_image) and index-based data is tied to a sample via box_refs.
STATE_VERSION = 2
JITTER_SHIFT = .018      # placement jitter, fraction of the region size
JITTER_ANGLE = .025      # default placement jitter, radians (±1.43°)
JITTER_ANGLE_DEG = round(math.degrees(JITTER_ANGLE), 2)
MAX_JITTER_ANGLE_DEG = 45.0
MAX_ATTEMPTS = 100
MIN_PRINT_CONTRAST = 10  # grey levels between paper and ink
REGION_FILL = .85        # "Cỡ ký tự theo vùng": date fills this much of the region
# Measured on 1920x480 exports: about 0.44 MB and 60 ms per image.
BYTES_PER_PIXEL = .44e6 / (1920 * 480)
SECONDS_PER_PIXEL = .060 / (1920 * 480)
CONFIRM_ABOVE = 20000    # ask before generating more images than this
MANIFEST_FIELDS = ("image", "split", "code", "background", "mode", "variant", "seed")
MAX_PATH_LENGTH = 259    # Windows MAX_PATH without the terminating NUL
OUT_LINES_VERSION = 1


def angle_jitter_radians(degrees):
    """The ± rotation range in radians for a setting in degrees.

    The default (shown rounded as 1.43°) maps to exactly JITTER_ANGLE, so runs
    with the default keep making the same images for the same seed.
    """
    if abs(degrees - JITTER_ANGLE_DEG) < 1e-9:
        return JITTER_ANGLE
    return math.radians(degrees)


def auto_job_name(mode):
    return f"auto_{mode}"


def image_files(folder):
    """Return ``(images, skipped_names)`` for one folder.

    Labels are stored as ``labels/<stem>.txt``, so two images that only differ
    by extension would share one label file. Only the oldest one is used: it
    is the one the existing label was most likely made for.
    """
    folder.mkdir(parents=True, exist_ok=True)
    by_stem = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            by_stem.setdefault(path.stem.casefold(), []).append(path)

    def age(path):
        try:
            return path.stat().st_mtime_ns, path.name
        except OSError:
            return math.inf, path.name

    images, skipped = [], []
    for paths in by_stem.values():
        paths.sort(key=age)
        images.append(paths[0])
        skipped.extend(path.name for path in paths[1:])
    return sorted(images), sorted(skipped)


# ------------------------------------------------------------------ print assets

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


def print_is_light(gray, roi):
    """Return True when the print is lighter than the paper around the box.

    The thin ring just outside the OBB is almost entirely paper, so its median
    tells which side of the Otsu split inside the box is the paper.
    """
    ring = ImageChops.subtract(roi.filter(ImageFilter.MaxFilter(7)), roi)
    if not ring.getbbox() or not roi.getbbox():
        return False
    return ImageStat.Stat(gray, ring).median[0] <= otsu_threshold(gray, roi)


def character_asset(source, box, padding=12):
    """Return ``(shade, origin, inverted)`` for the print inside one OBB.

    ``shade`` is 255 on paper and sample/paper on ink, so multiplying it onto a
    background keeps the real ink tone, dot texture and soft edges. Light print
    on a darker surface is processed on the inverted crop and ``inverted`` is
    True; the caller must then blend on the inverted background. ``shade`` is
    None when no print can be separated from the paper, so the caller never
    labels an object that would be invisible.
    """
    corners = box.corners()
    left = max(0, math.floor(min(x for x, _ in corners)) - padding)
    top = max(0, math.floor(min(y for _, y in corners)) - padding)
    right = min(source.width, math.ceil(max(x for x, _ in corners)) + padding + 1)
    bottom = min(source.height, math.ceil(max(y for _, y in corners)) + padding + 1)
    crop = source.crop((left, top, right, bottom)).convert("RGB")
    gray = crop.convert("L")
    polygon = [(x - left, y - top) for x, y in corners]
    roi = Image.new("L", crop.size, 0)
    ImageDraw.Draw(roi).polygon(polygon, fill=255)
    inverted = print_is_light(gray, roi)
    if inverted:
        crop, gray = ImageOps.invert(crop), ImageOps.invert(gray)
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
    if not paper_mask.getbbox() or not ink_mask.getbbox():
        return None, (left, top), inverted
    paper = ImageStat.Stat(crop, paper_mask).mean
    paper_level = ImageStat.Stat(gray, paper_mask).mean[0]
    ink_level = ImageStat.Stat(gray, ink_mask).mean[0]
    if paper_level - ink_level < MIN_PRINT_CONTRAST:
        return None, (left, top), inverted
    shade = Image.merge("RGB", [
        channel.point([min(255, round(value * 255 / max(1.0, level)))
                       for value in range(256)])
        for channel, level in zip(crop.split(), paper)])
    # The soft gate only removes paper noise. The actual darkness comes from
    # shade, preserving dot texture and anti-aliased print edges.
    contrast = paper_level - ink_level
    start, end = paper_level - contrast * .15, paper_level - contrast * .3
    table = [255 if value <= end else 0 if value >= start
             else round((start - value) * 255 / (start - end)) for value in range(256)]
    gate = gray.point(table).filter(ImageFilter.MaxFilter(3))
    gate = ImageChops.multiply(gate.filter(ImageFilter.GaussianBlur(.6)), roi)
    if not gate.getbbox():
        return None, (left, top), inverted
    white = Image.new("RGB", crop.size, (255, 255, 255))
    return Image.composite(shade, white, gate), (left, top), inverted


def erase_boxes(source, boxes):
    """Return a copy of ``source`` with the print inside ``boxes`` painted over.

    Raises ValueError when no paper can be found around a box.
    """
    source = source.convert("RGB")
    result = source.copy()
    gray = source.convert("L")
    for box in boxes:
        polygon = box.corners()
        character_mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(character_mask).polygon(polygon, fill=255)
        # Cover ink touching the OBB edge, while sampling paper from a
        # separate outer ring around the character. Both sizes are odd.
        inner_size = max(5, min(21, (round(min(box.w, box.h) * .12) | 1)))
        outer_size = max(inner_size + 4, min(41, inner_size * 2 - 1))
        removal_mask = character_mask.filter(ImageFilter.MaxFilter(inner_size))
        outer_mask = character_mask.filter(ImageFilter.MaxFilter(outer_size))
        sample_ring = ImageChops.subtract(outer_mask, removal_mask)
        threshold = otsu_threshold(gray, sample_ring)
        bright_paper = gray.point([0 if value <= threshold else 255 for value in range(256)])
        paper_mask = ImageChops.multiply(sample_ring, bright_paper)
        if not paper_mask.getbbox():
            paper_mask = sample_ring
        if not paper_mask.getbbox():
            raise ValueError("Không tìm thấy vùng giấy quanh một box ký tự.")
        paper_color = tuple(round(value) for value in ImageStat.Stat(source, paper_mask).mean)
        paper_patch = Image.new("RGB", source.size, paper_color)
        feathered = removal_mask.filter(ImageFilter.GaussianBlur(1.2))
        result = Image.composite(paper_patch, result, feathered)
    return result


# ---------------------------------------------------------------------- geometry

def text_angle(boxes):
    """Dominant box angle (period 180°), i.e. the text direction."""
    sine = sum(math.sin(2 * box.angle) for box in boxes)
    cosine = sum(math.cos(2 * box.angle) for box in boxes)
    if abs(sine) < 1e-12 and abs(cosine) < 1e-12:
        return 0.0
    return math.atan2(sine, cosine) / 2


def split_rows(boxes):
    """Group boxes into text rows, top row first, each row left to right.

    Rows are measured across the text direction. A new row starts only where
    consecutive centres jump by more than half the median box height, so a
    slanted or jittered single row stays one row.
    """
    boxes = list(boxes)
    if len(boxes) < 2:
        return [boxes] if boxes else []
    angle = text_angle(boxes)
    cosine, sine = math.cos(angle), math.sin(angle)

    def across(box):
        return -sine * box.cx + cosine * box.cy

    def along(box):
        return cosine * box.cx + sine * box.cy

    ordered = sorted(boxes, key=across)
    heights = sorted(box.h for box in boxes)
    threshold = heights[len(heights) // 2] * .5
    rows = [[ordered[0]]]
    for previous, box in zip(ordered, ordered[1:]):
        if across(box) - across(previous) > threshold:
            rows.append([box])
        else:
            rows[-1].append(box)
    return [sorted(row, key=along) for row in rows]


def boxes_bounds(boxes):
    xs = [x for box in boxes for x, _ in box.corners()]
    ys = [y for box in boxes for _, y in box.corners()]
    return min(xs), min(ys), max(xs), max(ys)


def overlapping_boxes(boxes, limit=.3):
    """Index pairs whose bounding rectangles overlap by more than ``limit``
    of the smaller one."""
    rects = []
    for box in boxes:
        corners = box.corners()
        rects.append((min(x for x, _ in corners), min(y for _, y in corners),
                      max(x for x, _ in corners), max(y for _, y in corners)))
    pairs = []
    for first in range(len(rects)):
        a = rects[first]
        for second in range(first + 1, len(rects)):
            b = rects[second]
            width = min(a[2], b[2]) - max(a[0], b[0])
            height = min(a[3], b[3]) - max(a[1], b[1])
            if width <= 0 or height <= 0:
                continue
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if smaller > 0 and width * height / smaller > limit:
                pairs.append((first, second))
    return pairs


def layout_boxes(boxes, region, mode, stack_size, arc_height,
                 dimension_scale=1.0, spacing_groups=(), group_x_factor=1.0):
    """Return copies of ``boxes`` laid out for ``mode`` and moved into ``region``.

    Characters are resized only by ``dimension_scale`` (sample → background
    pixels); the error effects move boxes but never resize them.
    """
    placed = copy.deepcopy(boxes)
    source_left, source_top, source_right, source_bottom = boxes_bounds(placed)
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

    left, top, right, bottom = boxes_bounds(placed)
    target_x, target_y = region.cx, region.cy
    source_x, source_y = (left + right) / 2, (top + bottom) / 2
    cosine, sine = math.cos(region.angle), math.sin(region.angle)
    for box in placed:
        dx, dy = box.cx - source_x, box.cy - source_y
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


def oriented_line(line, axis_angle=0.0):
    """Return the segment pointing along ``axis_angle``.

    "Above"/"below" are defined relative to this direction, so they must not
    depend on which end the user started drawing from.
    """
    (x1, y1), (x2, y2) = line
    axis_x, axis_y = math.cos(axis_angle), math.sin(axis_angle)
    dx, dy = x2 - x1, y2 - y1
    along = dx * axis_x + dy * axis_y
    # A line perpendicular to the axis is pointed along the axis' +90° side.
    if along < 0 or (along == 0 and axis_x * dy - axis_y * dx < 0):
        return (x2, y2), (x1, y1)
    return (x1, y1), (x2, y2)


def line_side_mask(size, line, remove_side, axis_angle=0.0, origin=(0, 0)):
    """Return an L mask (255 keep, 0 remove) for one side of a cut line.

    ``remove_side`` is "above" or "below" relative to ``axis_angle`` (image Y
    grows downwards). ``origin`` is the image position of the mask's top-left
    pixel, so the mask can cover just the area of one object.
    """
    mask = Image.new("L", size, 255)
    if not line or remove_side not in CUT_SIDES:
        return mask
    (x1, y1), (x2, y2) = oriented_line(line, axis_angle)
    if math.hypot(x2 - x1, y2 - y1) <= 1e-9:
        return mask
    # cross = (x2-x1)*(y-y1) - (y2-y1)*(x-x1) is < 0 above and > 0 below a
    # segment that points along the axis; pixels on the line are kept. Per
    # row, sign*cross = constant + slope*(x-x1) is linear in x, so each row
    # is cleared with one paste instead of a Python loop over every pixel.
    sign = -1 if remove_side == "above" else 1
    width, height = size
    left, top = origin
    slope = -sign * (y2 - y1)
    for row in range(height):
        constant = sign * (x2 - x1) * (row + top - y1)
        if slope == 0:
            if constant > 0:
                mask.paste(0, (0, row, width, row + 1))
            continue
        boundary = x1 - constant / slope
        if slope > 0:  # removed where x > boundary
            start = max(0, math.floor(boundary) + 1 - left)
            if start < width:
                mask.paste(0, (start, row, width, row + 1))
        else:          # removed where x < boundary
            end = min(width, math.ceil(boundary) - left)
            if end > 0:
                mask.paste(0, (0, row, end, row + 1))
    return mask


def clip_box_to_side(box, line, remove_side, axis_angle=0.0):
    """Return the part of ``box`` that a cut keeps, as a box with the same angle.

    Returns None when less than one pixel remains.
    """
    if remove_side not in CUT_SIDES:
        return box
    (x1, y1), (x2, y2) = oriented_line(line, axis_angle)
    sign = 1 if remove_side == "above" else -1

    def kept_side(point):
        return sign * ((x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1))

    corners = box.corners()
    kept = []
    for current, following in zip(corners, corners[1:] + corners[:1]):
        a, b = kept_side(current), kept_side(following)
        if a >= 0:
            kept.append(current)
        if (a >= 0) != (b >= 0):
            t = a / (a - b)
            kept.append((current[0] + (following[0] - current[0]) * t,
                         current[1] + (following[1] - current[1]) * t))
    if not kept:
        return None
    cosine, sine = math.cos(box.angle), math.sin(box.angle)
    xs = [(x - box.cx) * cosine + (y - box.cy) * sine for x, y in kept]
    ys = [-(x - box.cx) * sine + (y - box.cy) * cosine for x, y in kept]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if width < 1 or height < 1:
        return None
    middle_x, middle_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return Box(box.cls, box.cx + middle_x * cosine - middle_y * sine,
               box.cy + middle_x * sine + middle_y * cosine, width, height, box.angle)


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
    # extension) to enter the rotated box.
    return segment_hits_box(line, box, tolerance)


def segment_hits_box(line, box, tolerance=1e-6):
    """Return True when a segment touches or enters an OBB.

    The segment is moved to box-local space and clipped against the resulting
    axis-aligned rectangle (Liang–Barsky).
    """
    if not line:
        return False
    (x1, y1), (x2, y2) = line
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


# Cut lines are stored in "date-local" coordinates: region axes, origin at the
# region centre (where the date is centred), in sample pixels. They follow the
# region when it moves or is jittered, and keep their position relative to the
# characters on any background.
def out_line_hits(boxes, lines):
    """Indices of the boxes that touch an out-line.

    This is the "out" rule: a date is out when any of its characters overlaps
    a line drawn on the background. Lines are in image pixels.
    """
    if not lines:
        return []
    return [index for index, box in enumerate(boxes)
            if any(segment_hits_box(line, box) for line in lines)]


def read_out_lines(path, size):
    """Out-lines stored for an image of ``size``, in pixels ([] if none).

    The file holds ``{"version": 1, "lines": [[[x1, y1], [x2, y2]], ...]}``
    with coordinates relative to the image size (0–1), like YOLO labels, so
    it can be copied to another image with the same framing.
    """
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("lines"), list):
        raise ValueError("file đường out-line phải có danh sách 'lines'")
    width, height = size
    lines = []
    for item in data["lines"]:
        try:
            (x1, y1), (x2, y2) = item
            values = [float(value) for value in (x1, y1, x2, y2)]
        except (TypeError, ValueError):
            raise ValueError(f"đường out-line không hợp lệ: {item!r:.60}") from None
        if not all(math.isfinite(value) and -1e-6 <= value <= 1 + 1e-6 for value in values):
            raise ValueError("tọa độ đường out-line phải nằm trong 0–1")
        lines.append(((values[0] * width, values[1] * height),
                      (values[2] * width, values[3] * height)))
    return lines


def encode_out_lines(lines, size):
    """JSON text for ``read_out_lines``; pixel lines → 0–1 coordinates."""
    width, height = size
    return json.dumps({"version": OUT_LINES_VERSION, "lines": [
        [[round(max(0.0, min(1.0, x / width)), 8), round(max(0.0, min(1.0, y / height)), 8)]
         for x, y in line] for line in lines]}, indent=2) + "\n"


def local_to_image(point, region, scale):
    x, y = point[0] * scale, point[1] * scale
    cosine, sine = math.cos(region.angle), math.sin(region.angle)
    return region.cx + cosine * x - sine * y, region.cy + sine * x + cosine * y


def image_to_local(point, region, scale):
    dx, dy = point[0] - region.cx, point[1] - region.cy
    cosine, sine = math.cos(region.angle), math.sin(region.angle)
    return (cosine * dx + sine * dy) / scale, (-sine * dx + cosine * dy) / scale


def scale_region(region, old_size, new_size):
    """Map a region onto an image of another size (same framing assumed)."""
    sx, sy = new_size[0] / old_size[0], new_size[1] / old_size[1]
    return Box(region.cls, region.cx * sx, region.cy * sy, region.w * sx, region.h * sy,
               region.angle)


# ------------------------------------------------------------------ effect state

def default_state(mode="normal"):
    return {
        "version": STATE_VERSION,
        "mode": mode,
        "last_effect_mode": "normal" if mode == "delete_objects" else mode,
        "stack_size": 3,
        "arc_height": 35.0,
        "blur_radius": 2.0,
        "spacing_groups": [],
        "group_x_factor": 1.0,
        "cut_lines": [],
        "cut_side": "none",
        "deleted_indices": set(),
        # Only saved modes fill these: the sample the box indices refer to and
        # each sample box centre at save time, to re-identify the boxes later.
        "sample": "",
        "box_refs": [],
    }


def state_uses_indices(state):
    return bool(state["spacing_groups"] or state["deleted_indices"])


def normalize_state(raw):
    """Validate an effect state read from the config file.

    Raises ValueError (or TypeError/KeyError) when the state cannot be used,
    e.g. cut lines stored by an older version in background pixels.
    """
    if not isinstance(raw, dict):
        raise ValueError("state phải là object")
    if raw.get("version") != STATE_VERSION and (raw.get("cut_line") or raw.get("cut_lines")):
        raise ValueError("đường cắt kiểu cũ (pixel background) không dùng lại được")
    mode = raw.get("mode", "normal")
    if mode not in EFFECT_MODES:
        raise ValueError(f"mode không hợp lệ: {mode}")
    last = raw.get("last_effect_mode", "normal") if mode == "delete_objects" else mode
    if last not in EFFECT_MODES or last == "delete_objects":
        raise ValueError(f"last_effect_mode không hợp lệ: {last}")
    cut_side = raw.get("cut_side", "none")
    if cut_side not in CUT_SIDES + ("none",):
        raise ValueError(f"cut_side không hợp lệ: {cut_side}")

    def number(value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("số không hợp lệ")
        return value

    def index(value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"chỉ số box không hợp lệ: {value!r}")
        return value

    def point(value):
        x, y = value
        return number(x), number(y)

    cut_lines = [tuple(point(item) for item in line) for line in raw.get("cut_lines", [])]
    if any(len(line) != 2 for line in cut_lines):
        raise ValueError("đường cắt phải có 2 điểm")
    state = default_state(mode)
    state.update({
        "last_effect_mode": last,
        "stack_size": max(1, min(5, int(raw.get("stack_size", 3)))),
        "arc_height": number(raw.get("arc_height", 35.0)),
        "blur_radius": max(.2, number(raw.get("blur_radius", 2.0))),
        "spacing_groups": [[index(item) for item in group]
                           for group in raw.get("spacing_groups", [])],
        "group_x_factor": number(raw.get("group_x_factor", 1.0)),
        "cut_lines": cut_lines,
        "cut_side": cut_side,
        "deleted_indices": {index(item) for item in raw.get("deleted_indices", [])},
        "sample": str(raw.get("sample", "")),
        "box_refs": [list(point(item)) for item in raw.get("box_refs", [])],
    })
    return state


def state_to_json(state):
    result = copy.deepcopy(state)
    result["deleted_indices"] = sorted(result["deleted_indices"])
    result["cut_lines"] = [[list(item) for item in line] for line in result["cut_lines"]]
    return result


def first_free_number(prefix, targets):
    """Return the number after the highest ``<prefix>_NNNNNN`` in ``targets``.

    ``targets`` holds ``(directory, suffix)`` pairs. Each directory is listed
    once, instead of probing every number from 0 for every new file.
    """
    start, highest = prefix + "_", -1
    for directory, suffix in targets:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            name = path.name
            if name.startswith(start) and name.endswith(suffix):
                number = name[len(start):len(name) - len(suffix)]
                if number.isdigit():
                    highest = max(highest, int(number))
    return highest + 1


Job = namedtuple("Job", "name make_state")


@dataclass
class Composition:
    """One rendered image plus the labels of the objects drawn on it."""
    image: object = None
    boxes: list = field(default_factory=list)    # exported labels
    sources: list = field(default_factory=list)  # sample-box index of each label
    blank: list = field(default_factory=list)    # sample boxes without visible print
    cut_count: int = 0                           # objects crossed by a cut line
    out_hits: list = field(default_factory=list)  # labels touching an out-line


# ------------------------------------------------------------------ code, target

@dataclass(eq=False)
class Code:
    """One sample image and its character boxes."""
    path: Path | None
    image: Image.Image | None
    boxes: list
    assets: dict = field(default_factory=dict)  # character_asset() cache

    @property
    def name(self):
        return self.path.name if self.path else ""

    @property
    def stem(self):
        return self.path.stem if self.path else ""

    def asset(self, box):
        """character_asset() for one box, cached by the box geometry."""
        key = (box.cx, box.cy, box.w, box.h, box.angle)
        cached = self.assets.get(key)
        if cached is None:
            if len(self.assets) > 512:
                self.assets.clear()
            cached = self.assets[key] = character_asset(self.image, box)
        return cached

    def blank_boxes(self):
        return [index for index, box in enumerate(self.boxes) if self.asset(box)[0] is None]

    def rows(self):
        return split_rows(self.boxes)


@dataclass(eq=False)
class Target:
    """One background image and the region where the date is placed."""
    path: Path | None
    image: Image.Image | None
    region: Box | None
    out_lines: list = field(default_factory=list)  # ((x1, y1), (x2, y2)) in pixels

    @property
    def name(self):
        return self.path.name if self.path else ""

    @property
    def stem(self):
        return self.path.stem if self.path else ""


def load_code(path, label_path):
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    boxes = (decode(label_path.read_text(encoding="utf-8-sig"), *image.size, 1)
             if label_path.exists() else [])
    return Code(path, image, boxes)


def read_region(label_path, size):
    """Return the region stored for an image of ``size``, or None."""
    if not label_path.exists():
        return None
    boxes = decode(label_path.read_text(encoding="utf-8-sig"), *size, 1)
    if len(boxes) > 1:
        raise ValueError("Nhãn background chỉ được chứa một OBB vùng đặt date.")
    return boxes[0] if boxes else None


def load_target(path, label_path, lines_path=None):
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    lines = read_out_lines(lines_path, image.size) if lines_path is not None else []
    return Target(path, image, read_region(label_path, image.size), lines)


def code_mode_problem(code, mode):
    """Why an automatic mode cannot make a real error with this code, or None.

    Only the checks that do not depend on the background.
    """
    boxes = code.boxes
    if not boxes:
        return "chưa có box"
    rows = split_rows(boxes)
    if mode in ("spacing", "arc", "delete_objects") and len(boxes) < 2:
        return "cần ít nhất 2 box"
    if mode == "overlap" and max(map(len, rows)) < 2:
        return "không hàng nào có từ 2 box trở lên"
    if mode == "arc" and max(box.cx for box in boxes) - min(box.cx for box in boxes) < 1:
        return "các box thẳng hàng dọc, không uốn cong được"
    if mode in ("cut_row_2", "cut_both_rows") and len(rows) < 2:
        return "date chỉ có 1 hàng (đã có cut_row_1)"
    return None


def task_rng(seed, code_name, background_name, mode, variant):
    """The random generator of one image: (seed, code, background, mode, variant).

    Each image has its own generator, so adding a background or a mode, or
    resuming a cancelled run, never changes the other images.
    """
    return random.Random(f"{seed}:{code_name}:{background_name}:{mode}:{variant}")


# --------------------------------------------------------------------- generator

class Generator:
    """Composes synthetic images for any code × target pair."""

    def __init__(self, size_mode="image", region_fill=REGION_FILL,
                 angle_jitter_deg=JITTER_ANGLE_DEG):
        self.size_mode = size_mode
        self.region_fill = region_fill
        self.angle_jitter_deg = angle_jitter_deg

    @classmethod
    def from_settings(cls, settings):
        return cls(settings.size_mode, angle_jitter_deg=settings.angle_jitter_deg)

    @property
    def angle_jitter(self):
        """Largest random rotation of the date, in radians."""
        return angle_jitter_radians(self.angle_jitter_deg)

    # ---------------------------------------------------------------- sizing

    def scale(self, code, target):
        """Sample → background pixel ratio for this pair.

        "image": the whole-image size ratio, which assumes the code and the
        background were captured with the same camera framing.
        "region": the date fills ``region_fill`` of the region (height, or
        width when the date is relatively wider than the region).
        """
        if code is None or target is None or code.image is None or target.image is None:
            return 1.0
        if self.size_mode == "region" and target.region is not None and code.boxes:
            left, top, right, bottom = boxes_bounds(code.boxes)
            return min(target.region.h * self.region_fill / max(1.0, bottom - top),
                       target.region.w * self.region_fill / max(1.0, right - left))
        return min(target.image.width / code.image.width,
                   target.image.height / code.image.height)

    def fit_problem(self, code, target):
        """Why the date in normal layout does not fit the region, or None."""
        if target.region is None:
            return "chưa có vùng"
        if not code.boxes:
            return "chưa có box"
        scale = self.scale(code, target)
        region = target.region
        placed = layout_boxes(code.boxes, region, "normal", 3, 0.0, scale)
        if all(box_inside_box(box, region) for box in placed):
            return None
        # Measure in region axes so a rotated region is handled too.
        cosine, sine = math.cos(region.angle), math.sin(region.angle)
        xs, ys = [], []
        for box in placed:
            for x, y in box.corners():
                dx, dy = x - region.cx, y - region.cy
                xs.append(dx * cosine + dy * sine)
                ys.append(-dx * sine + dy * cosine)
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        parts = []
        if width > region.w:
            parts.append(f"date rộng hơn vùng ({width:.0f} > {region.w:.0f} px)")
        if height > region.h:
            parts.append(f"date cao hơn vùng ({height:.0f} > {region.h:.0f} px)")
        return "; ".join(parts) or "date vượt vùng"

    # ------------------------------------------------------------- composing

    def placed_boxes(self, code, target, state, region):
        mode = state["mode"]
        effect = state["last_effect_mode"] if mode == "delete_objects" else mode
        return layout_boxes(code.boxes, region, effect, state["stack_size"],
                            state["arc_height"], self.scale(code, target),
                            state["spacing_groups"], state["group_x_factor"])

    def cut_lines_in_image(self, code, target, state, region):
        mode = state["mode"]
        effect = state["last_effect_mode"] if mode == "delete_objects" else mode
        if effect != "cut" or state["cut_side"] not in CUT_SIDES:
            return []
        scale = self.scale(code, target)
        return [tuple(local_to_image(point, region, scale) for point in line)
                for line in state["cut_lines"]]

    def cut_count(self, code, target, state, region=None):
        """Number of objects a state's cut lines cross; no image is composed."""
        region = region or target.region
        if region is None or not code.boxes:
            return 0
        lines = self.cut_lines_in_image(code, target, state, region)
        if not lines:
            return 0
        placed = self.placed_boxes(code, target, state, region)
        deleted = state["deleted_indices"] if state["mode"] == "delete_objects" else ()
        return sum(1 for index, box in enumerate(placed)
                   if index not in deleted and any(line_crosses_box(line, box) for line in lines))

    def compose(self, code, target, state, region):
        """Render ``state`` with the date placed in ``region``.

        Has no side effects: the caller decides what to keep from the result.
        """
        if (code is None or target is None or code.image is None or target.image is None
                or not code.boxes or region is None):
            return Composition(target.image if target is not None else None)
        mode = state["mode"]
        effect = state["last_effect_mode"] if mode == "delete_objects" else mode
        placed = self.placed_boxes(code, target, state, region)
        # Deletions belong to the delete mode only; other modes show every object.
        deleted = state["deleted_indices"] if mode == "delete_objects" else ()
        cut_side = state["cut_side"]
        cut_lines = self.cut_lines_in_image(code, target, state, region)
        canvas = target.image.copy()
        result = Composition(canvas)
        for index, (source_box, target_box) in enumerate(zip(code.boxes, placed)):
            if index in deleted:
                continue
            shade, origin, inverted = code.asset(source_box)
            if shade is None:
                result.blank.append(index)
                continue
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
            if effect == "blur":
                radius = max(.2, state["blur_radius"])
                # Pad first so the blur halo is not clipped at the crop edge.
                shade = ImageOps.expand(shade, border=math.ceil(radius * 3) + 1,
                                        fill=(255, 255, 255))
                shade = shade.filter(ImageFilter.GaussianBlur(radius))
            x = round(target_box.cx + rotated_offset[0] - shade.width / 2)
            y = round(target_box.cy + rotated_offset[1] - shade.height / 2)
            area = (max(0, x), max(0, y),
                    min(canvas.width, x + shade.width), min(canvas.height, y + shade.height))
            visible = area[2] > area[0] and area[3] > area[1]
            label_box, mask = target_box, None
            crossing = [line for line in cut_lines if line_crosses_box(line, target_box)]
            if crossing:
                result.cut_count += 1
            for line in crossing:
                if visible:
                    part = line_side_mask((area[2] - area[0], area[3] - area[1]), line,
                                          cut_side, region.angle, area[:2])
                    mask = part if mask is None else ImageChops.multiply(mask, part)
                if label_box is not None:
                    # The label covers only the part of the object left visible.
                    label_box = clip_box_to_side(label_box, line, cut_side, region.angle)
            if visible:
                # Multiply the paper-relative shade onto the target background.
                # White pixels leave it untouched; ink retains source tone/texture.
                patch = canvas.crop(area)
                ink = shade.crop((area[0] - x, area[1] - y, area[2] - x, area[3] - y))
                if inverted:
                    blended = ImageOps.invert(ImageChops.multiply(ImageOps.invert(patch), ink))
                else:
                    blended = ImageChops.multiply(patch, ink)
                canvas.paste(blended, area[:2], mask)
            if label_box is not None:
                result.boxes.append(label_box)
                result.sources.append(index)
        return result

    # ----------------------------------------------------------- user states

    def effect_problem(self, code, target, state):
        """Return why ``state`` would not produce a visible error, or None."""
        boxes = code.boxes
        mode = state["mode"]
        if mode == "delete_objects":
            if not state["deleted_indices"]:
                return "Chưa xóa object nào: hãy click vào object trên preview."
            if len(state["deleted_indices"]) >= len(boxes):
                return "Không được xóa hết object."
            return None
        if mode == "spacing":
            valid = [[i for i in group if 0 <= i < len(boxes)] for group in state["spacing_groups"]]
            valid = [group for group in valid if group]
            assigned = {i for group in valid for i in group}
            if len(valid) + len(boxes) - len(assigned) < 2:
                return "Cần ít nhất 2 cụm/box để dãn cách."
            if abs(state["group_x_factor"] - 1) < .01:
                return "Hệ số dãn đang là 1.00× nên chưa tạo lỗi."
        elif mode == "overlap":
            if state["stack_size"] < 2:
                return "Số box/chồng phải từ 2 trở lên."
            if max(map(len, split_rows(boxes))) < 2:
                return "Không hàng nào có từ 2 box để chồng lên nhau."
        elif mode == "arc":
            if abs(state["arc_height"]) < .5:
                return "Độ cong đang bằng 0."
            if len(boxes) < 2:
                return "Cần ít nhất 2 box để uốn cong."
        elif mode == "cut":
            if state["cut_side"] not in CUT_SIDES:
                return "Hãy chọn xóa phía trên hoặc phía dưới đường cắt."
            if not state["cut_lines"]:
                return "Hãy kẻ đường cắt trên preview."
            if self.cut_count(code, target, state) == 0:
                return "Đường cắt không cắt qua object nào."
        return None

    @staticmethod
    def resolve_saved(code, saved_state):
        """Map a saved state's box indices onto ``code``'s boxes.

        Returns ``(state, None)`` or ``(None, reason)``. Boxes are matched by
        the centre recorded at save time, so the state stays correct when other
        boxes were deleted, and is rejected when it belongs to another sample.
        """
        state = copy.deepcopy(saved_state)
        used = sorted({i for group in state["spacing_groups"] for i in group}
                      | state["deleted_indices"])
        if not used:
            return state, None
        if code.path is None or state["sample"] != code.name:
            return None, f"dùng chỉ số box của ảnh mẫu {state['sample'] or '(không rõ)'}"
        mapping, taken = {}, set()
        for old in used:
            if not 0 <= old < len(state["box_refs"]):
                return None, "thiếu thông tin vị trí box"
            x, y = state["box_refs"][old]
            candidates = [index for index, box in enumerate(code.boxes)
                          if index not in taken and box.contains(x, y)]
            if not candidates:
                return None, f"box {old} lúc lưu mode đã bị xóa hoặc di chuyển"
            new = min(candidates, key=lambda index: math.dist(
                (x, y), (code.boxes[index].cx, code.boxes[index].cy)))
            mapping[old] = new
            taken.add(new)
        state["spacing_groups"] = [[mapping[i] for i in group]
                                   for group in state["spacing_groups"]]
        state["deleted_indices"] = {mapping[i] for i in state["deleted_indices"]}
        if (state["mode"] == "delete_objects"
                and len(state["deleted_indices"]) >= len(code.boxes)):
            return None, "sẽ xóa hết object"
        return state, None

    def saved_problem(self, code, target, saved_state):
        """``(state, None)`` when a saved mode can run on this pair, else
        ``(None, reason)``. Composes no image."""
        state, problem = self.resolve_saved(code, saved_state)
        if problem:
            return None, problem
        mode = state["mode"]
        effect = state["last_effect_mode"] if mode == "delete_objects" else mode
        if effect in ("overlap", "spacing", "arc") and len(code.boxes) < 2:
            return None, "cần ít nhất 2 box"
        if effect == "cut" and target.region is not None and \
                self.cut_count(code, target, state) == 0:
            return None, "đường cắt không cắt qua object nào"
        return state, None

    # -------------------------------------------------------- automatic modes

    def auto_arc_range(self, code, target):
        """Arc heights (background px) that are visible and still fit the region."""
        scale = self.scale(code, target)
        heights = sorted(box.h for box in code.boxes)
        character_height = heights[len(heights) // 2] * scale
        ys = [y for box in code.boxes for _, y in box.corners()]
        date_height = (max(ys) - min(ys)) * scale
        slack = target.region.h * (1 - 2 * JITTER_SHIFT) - date_height
        high = min(1.5 * character_height, .9 * slack)
        if high < .15 * character_height:
            return None
        return min(.35 * character_height, .5 * high), high

    def max_spacing_factor(self, code, target):
        """Largest spacing factor whose date still fits the region width."""
        left, _top, right, _bottom = boxes_bounds(code.boxes)
        width = (right - left) * self.scale(code, target)
        if width <= 0:
            return 2.1
        return target.region.w * (1 - 2 * JITTER_SHIFT) / width * .97

    def mode_problem(self, code, target, mode):
        """Return why an automatic mode cannot make a real error here, or None."""
        problem = code_mode_problem(code, mode)
        if problem:
            return problem
        if mode == "arc" and target.region is not None and \
                self.auto_arc_range(code, target) is None:
            return "vùng đặt quá thấp để uốn cong date"
        if mode == "out_line" and not target.out_lines:
            return "background chưa có đường out-line"
        return None

    @staticmethod
    def automatic_cut_lines(code, mode, rng):
        """Cut lines (date-local coordinates) through the requested rows."""
        boxes = code.boxes
        rows = split_rows(boxes)
        chosen = (rows[:1] if mode == "cut_row_1" else
                  rows[1:2] if mode == "cut_row_2" else rows)
        if not chosen:
            raise ValueError("date không có hàng phù hợp để cắt")
        left, top, right, bottom = boxes_bounds(boxes)
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        angle = text_angle(boxes)
        cosine, sine = math.cos(angle), math.sin(angle)
        half = (right - left) / 2 + max(box.w for box in boxes)
        lines = []
        for row in chosen:
            offset = rng.uniform(-.12, .12) * min(box.h for box in row)
            row_x = sum(box.cx for box in row) / len(row) - center_x - sine * offset
            row_y = sum(box.cy for box in row) / len(row) - center_y + cosine * offset
            lines.append(((row_x - cosine * half, row_y - sine * half),
                          (row_x + cosine * half, row_y + sine * half)))
        return lines

    def auto_state(self, code, target, mode, variant, rng):
        """Build one randomized state for an automatic error type."""
        effect = "cut" if mode.startswith("cut_") else mode
        state = default_state(effect)
        boxes = code.boxes
        if mode == "spacing":
            widest = min(2.1, self.max_spacing_factor(code, target))
            if widest >= 1.25:
                state["group_x_factor"] = rng.choice((
                    rng.uniform(.35, .78), rng.uniform(1.25, widest)))
            else:
                # No room to spread the date in this region: squeeze it instead.
                state["group_x_factor"] = rng.uniform(.35, .78)
        elif mode == "overlap":
            longest = max(map(len, split_rows(boxes)))
            state["stack_size"] = rng.randint(2, min(5, longest))
        elif mode == "arc":
            low, high = self.auto_arc_range(code, target)
            state["arc_height"] = rng.choice((-1, 1)) * rng.uniform(low, high)
        elif mode == "blur":
            state["blur_radius"] = rng.uniform(1.2, 7.5)
        elif mode == "delete_objects":
            count = len(boxes)
            state["deleted_indices"] = set(rng.sample(range(count), rng.randint(1, count - 1)))
        elif effect == "cut":
            state["cut_lines"] = self.automatic_cut_lines(code, mode, rng)
            state["cut_side"] = "above" if variant % 2 == 0 else "below"
        elif mode == "out_line":
            # Where the date sits on the line, as fractions; see out_line_region.
            state["out_line"] = {"line": rng.randrange(len(target.out_lines)),
                                 "t": rng.random(), "across": rng.random(),
                                 "along": rng.random()}
        return state

    def out_line_region(self, code, target, state, base):
        """Placement region for an "out_line" state.

        Takes the point at fraction ``t`` of the chosen line and moves the date
        (normal layout, angle of ``base``) so that this point lies inside the
        date's outline: ``across`` sets how deep the line cuts into the date,
        ``along`` where along it. The caller still checks that a character
        really touches the line and that the date stays inside the image.
        """
        params = state["out_line"]
        (x1, y1), (x2, y2) = target.out_lines[params["line"] % len(target.out_lines)]
        point_x, point_y = x1 + (x2 - x1) * params["t"], y1 + (y2 - y1) * params["t"]
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = ((x2 - x1) / length, (y2 - y1) / length) if length > 1e-9 else (1.0, 0.0)
        nx, ny = -uy, ux
        probe = Box(base.cls, 0.0, 0.0, base.w, base.h, base.angle)
        placed = layout_boxes(code.boxes, probe, "normal", 3, 0.0, self.scale(code, target))
        corners = [corner for box in placed for corner in box.corners()]
        across = [x * nx + y * ny for x, y in corners]
        along = [x * ux + y * uy for x, y in corners]
        s = min(across) + (max(across) - min(across)) * params["across"]
        k = min(along) + (max(along) - min(along)) * params["along"]
        center = [point_x - nx * s - ux * k, point_y - ny * s - uy * k]
        # Slide the date back into the image. A line near the image edge then
        # still cuts the date's edge instead of the attempt being wasted.
        for axis, limit in enumerate(target.image.size):
            low = -min(corner[axis] for corner in corners) + .5
            high = limit - max(corner[axis] for corner in corners) - .5
            if low <= high:
                center[axis] = max(low, min(high, center[axis]))
        return Box(base.cls, center[0], center[1], base.w, base.h, base.angle)

    def jobs(self, code, target, saved_modes, selected=None):
        """Return ``(jobs, skipped)`` for one pair.

        ``jobs`` maps a mode name to its Job; ``skipped`` maps a mode name to
        the reason it cannot make a real error on this pair. ``selected``
        limits the names considered (None = every mode).
        """
        jobs, skipped = {}, {}
        for mode in AUTO_ERROR_MODES:
            name = auto_job_name(mode)
            if selected is not None and name not in selected:
                continue
            problem = self.mode_problem(code, target, mode)
            if problem:
                skipped[name] = problem
                continue
            jobs[name] = Job(name, functools.partial(self.auto_state, code, target, mode))
        for saved in saved_modes:
            name = saved["name"]
            if selected is not None and name not in selected:
                continue
            state, problem = self.saved_problem(code, target, saved["state"])
            if problem:
                skipped[name] = problem
                continue
            jobs[name] = Job(name, lambda _variant, _rng, state=state: copy.deepcopy(state))
        return jobs, skipped

    # ------------------------------------------------------------ generating

    def jittered_region(self, region, rng):
        jittered = copy.deepcopy(region)
        jittered.cx += rng.uniform(-JITTER_SHIFT, JITTER_SHIFT) * region.w
        jittered.cy += rng.uniform(-JITTER_SHIFT, JITTER_SHIFT) * region.h
        # Always draw the angle, even at 0°, so the other draws stay the same.
        limit = self.angle_jitter
        jittered.angle += rng.uniform(-limit, limit)
        return jittered

    def generate(self, code, target, job, variant, rng, fingerprints):
        """Compose one new image for ``job``.

        Every mode except "out_line" must fit the region and must not touch an
        out-line, since that would also be an "out" error. "out_line" must touch
        a line with at least one character and stay inside the image.
        """
        out_mode, touched = False, 0
        for _attempt in range(MAX_ATTEMPTS):
            # The jitter is drawn before the state, as before out-lines existed,
            # so the other modes still make the same images for the same seed.
            region = self.jittered_region(target.region, rng)
            state = job.make_state(variant, rng)
            out_mode = state["mode"] == "out_line"
            if out_mode:
                region = self.out_line_region(code, target, state, region)
            result = self.compose(code, target, state, region)
            if result.blank:
                raise ValueError(f"box {result.blank} không có nét in")
            if not result.boxes:
                continue
            result.out_hits = out_line_hits(result.boxes, target.out_lines)
            if out_mode:
                if not result.out_hits or not all(
                        box.valid(*target.image.size) for box in result.boxes):
                    continue
            elif not all(box_inside_box(box, target.region) for box in result.boxes):
                continue
            elif result.out_hits:
                touched += 1
                continue
            fingerprint = hashlib.sha256(result.image.tobytes()).digest()
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            return result
        if out_mode:
            raise ValueError(f"sau {MAX_ATTEMPTS} lần thử không đặt được date chạm đường "
                             "out-line mà vẫn nằm trọn trong ảnh; hãy kẻ đường xa mép ảnh hơn")
        if touched > MAX_ATTEMPTS // 2:
            raise ValueError(f"sau {MAX_ATTEMPTS} lần thử date vẫn chạm đường out-line; "
                             "hãy dời vùng đặt ra xa đường out-line")
        raise ValueError(f"sau {MAX_ATTEMPTS} lần thử không tạo được ảnh mới nằm trọn "
                         "trong vùng đặt; hãy nới rộng vùng đặt hoặc giảm số lượng")


# ------------------------------------------------------------------- batch plan

@dataclass
class BatchSettings:
    pairing: str = "all"                 # "all" | "same_size"
    max_backgrounds_per_code: int = 0    # 0 = no limit
    count: int = 10
    count_mode: str = "per_mode"         # "per_mode" | "per_mode_per_pair"
    modes: list | None = None            # mode names; None = every mode
    # Unticked mode names. When set it wins over ``modes``, so a mode saved
    # after the last batch is included by default.
    modes_off: list | None = None
    val_percent: int = 20
    size_mode: str = "image"             # "image" | "region"
    angle_jitter_deg: float = JITTER_ANGLE_DEG  # date rotated by up to ± this

    @classmethod
    def from_config(cls, data):
        """Build settings from the ``batch`` config object; bad values fall back."""
        settings = cls()
        if not isinstance(data, dict):
            return settings
        if data.get("pairing") in ("all", "same_size"):
            settings.pairing = data["pairing"]
        if data.get("count_mode") in ("per_mode", "per_mode_per_pair"):
            settings.count_mode = data["count_mode"]
        if data.get("size_mode") in ("image", "region"):
            settings.size_mode = data["size_mode"]
        for name, low, high in (("max_backgrounds_per_code", 0, 10000), ("count", 1, 100000),
                                ("val_percent", 0, 90)):
            try:
                setattr(settings, name, max(low, min(high, int(data[name]))))
            except (KeyError, TypeError, ValueError):
                pass
        try:
            angle = float(data["angle_jitter_deg"])
            if math.isfinite(angle):
                settings.angle_jitter_deg = max(0.0, min(MAX_JITTER_ANGLE_DEG, angle))
        except (KeyError, TypeError, ValueError):
            pass
        if isinstance(data.get("modes"), list):
            settings.modes = [str(name) for name in data["modes"]]
        if isinstance(data.get("modes_unchecked"), list):
            settings.modes_off = [str(name) for name in data["modes_unchecked"]]
        return settings

    def selected_modes(self, available):
        if self.modes_off is not None:
            off = set(self.modes_off)
            return [name for name in available if name not in off]
        if self.modes is not None:
            wanted = set(self.modes)
            return [name for name in available if name in wanted]
        return list(available)


@dataclass(eq=False)
class PairPlan:
    code: Code
    target: Target
    problem: str | None = None                  # the whole pair is skipped
    jobs: dict = field(default_factory=dict)    # mode name → Job
    skipped: dict = field(default_factory=dict)  # mode name → reason
    counts: dict = field(default_factory=dict)  # mode name → images to make

    @property
    def total(self):
        return sum(self.counts.values())


@dataclass
class BatchPlan:
    seed: int
    settings: BatchSettings
    modes: list                                  # column order
    pairs: list                                  # every pair, valid or not
    split_by: str | None = None                  # "code" | "background" | "image" | None
    split: dict = field(default_factory=dict)    # code/background name → "train"/"val"
    warnings: list = field(default_factory=list)

    @property
    def valid_pairs(self):
        return [pair for pair in self.pairs if pair.problem is None]

    @property
    def total(self):
        return sum(pair.total for pair in self.pairs)

    def tasks(self):
        """Yield ``(pair, mode, count)`` in run order: code → background → mode."""
        for pair in self.valid_pairs:
            for mode in self.modes:
                count = pair.counts.get(mode, 0)
                if count:
                    yield pair, mode, count

    def split_of(self, pair, mode, variant):
        if self.split_by is None:
            return None
        if self.split_by == "code":
            return self.split.get(pair.code.name, "train")
        if self.split_by == "background":
            return self.split.get(pair.target.name, "train")
        rng = task_rng(self.seed, pair.code.name, pair.target.name, f"split:{mode}", variant)
        return "val" if rng.random() * 100 < self.settings.val_percent else "train"

    def estimate(self):
        """``(images, bytes, seconds)`` expected for this plan."""
        pixels = sum(pair.target.image.width * pair.target.image.height * pair.total
                     for pair in self.valid_pairs)
        return self.total, pixels * BYTES_PER_PIXEL, pixels * SECONDS_PER_PIXEL

    def mode_names(self):
        return list(self.modes)


def all_mode_names(saved_modes):
    return [auto_job_name(mode) for mode in AUTO_ERROR_MODES] + [
        saved["name"] for saved in saved_modes]


def _split_names(weights, percent, seed, salt, previous=None):
    """Assign each name to "train" or "val".

    ``weights`` maps a name (code or background) to its number of images.
    Names are taken in a seeded random order and put in val while that brings
    the val share of the images closer to ``percent``; each split gets at
    least one name. ``previous`` holds assignments from earlier runs into the
    same folder, which are kept so an image's split never changes.
    """
    names = sorted(weights)
    split = {name: value for name, value in (previous or {}).items() if name in weights}
    fresh = [name for name in names if name not in split]
    random.Random(f"{seed}:{salt}").shuffle(fresh)
    total = sum(weights.values()) or 1
    goal = total * percent / 100
    val = sum(weights[name] for name, value in split.items() if value == "val")
    for name in fresh:
        if "val" not in split.values() or abs(val + weights[name] - goal) < abs(val - goal):
            split[name] = "val"
            val += weights[name]
        else:
            split[name] = "train"
    if "train" not in split.values():
        # Everything ended in val: move the smallest fresh name back to train.
        movable = [name for name in fresh if split[name] == "val"]
        if movable:
            split[min(movable, key=lambda name: (weights[name], name))] = "train"
    return split


def plan_batch(codes, targets, saved_modes, settings, seed, generator=None,
               previous_split=None):
    """Plan which images a batch makes, without composing any image.

    ``previous_split`` is ``(split_by, {name: split})`` read from an earlier
    run's manifest in the same output folder.
    """
    generator = generator or Generator.from_settings(settings)
    modes = settings.selected_modes(all_mode_names(saved_modes))
    plan = BatchPlan(seed, settings, modes, [])
    if not codes:
        plan.warnings.append("Chưa chọn code nào.")
    if not targets:
        plan.warnings.append("Chưa chọn background nào.")
    if not modes:
        plan.warnings.append("Chưa chọn dạng lỗi nào.")

    for target in sorted(targets, key=lambda item: item.name.casefold()):
        if target.region is not None and out_line_hits([target.region], target.out_lines):
            plan.warnings.append(
                f"Vùng đặt của {target.name} cắt đường out-line: các dạng lỗi khác chỉ "
                "đặt date ở phần vùng không chạm đường, có thể không tạo được ảnh.")

    code_problems = {}
    for code in codes:
        if not code.boxes:
            code_problems[code.name] = "code chưa có box"
            continue
        blank = code.blank_boxes()
        if blank:
            code_problems[code.name] = f"box {blank} của code không có nét in"

    for code in sorted(codes, key=lambda item: item.name.casefold()):
        candidates = []
        for target in sorted(targets, key=lambda item: item.name.casefold()):
            pair = PairPlan(code, target)
            plan.pairs.append(pair)
            if code.name in code_problems:
                pair.problem = code_problems[code.name]
            elif target.region is None:
                pair.problem = "chưa có vùng"
            elif settings.pairing == "same_size" and code.image.size != target.image.size:
                pair.problem = (f"khác kích thước ({code.image.width}×{code.image.height} / "
                                f"{target.image.width}×{target.image.height})")
            else:
                pair.problem = generator.fit_problem(code, target)
            if pair.problem is None:
                candidates.append(pair)
        limit = settings.max_backgrounds_per_code
        if limit and len(candidates) > limit:
            keep = set(random.Random(f"{seed}:pick:{code.name}").sample(
                range(len(candidates)), limit))
            for position, pair in enumerate(candidates):
                if position not in keep:
                    pair.problem = f"không được chọn (tối đa {limit} background/code)"

    for pair in plan.valid_pairs:
        pair.jobs, pair.skipped = generator.jobs(pair.code, pair.target, saved_modes, set(modes))

    count = max(1, settings.count)
    for mode in modes:
        eligible = [pair for pair in plan.valid_pairs if mode in pair.jobs]
        if not eligible:
            continue
        if settings.count_mode == "per_mode_per_pair":
            for pair in eligible:
                pair.counts[mode] = count
            continue
        # Cycle through the pairs in a shuffled order: each pair gets
        # ⌊N/P⌋ or ⌊N/P⌋+1 images of this mode.
        random.Random(f"{seed}:spread:{mode}").shuffle(eligible)
        base, extra = divmod(count, len(eligible))
        for position, pair in enumerate(eligible):
            if base + (position < extra):
                pair.counts[mode] = base + (position < extra)

    _plan_split(plan, previous_split)
    return plan


def _plan_split(plan, previous_split):
    percent = plan.settings.val_percent
    used = [pair for pair in plan.valid_pairs if pair.total]
    if percent <= 0 or not used:
        return
    previous_by, previous = previous_split or (None, {})
    codes, backgrounds = {}, {}
    for pair in used:
        codes[pair.code.name] = codes.get(pair.code.name, 0) + pair.total
        backgrounds[pair.target.name] = backgrounds.get(pair.target.name, 0) + pair.total
    if len(codes) >= 2:
        plan.split_by = "code"
        plan.split = _split_names(codes, percent, plan.seed, "split:code",
                                  previous if previous_by == "code" else None)
    elif len(backgrounds) >= 2:
        plan.split_by = "background"
        plan.split = _split_names(backgrounds, percent, plan.seed, "split:background",
                                  previous if previous_by == "background" else None)
        plan.warnings.append("Chỉ có 1 code: tập val được tách theo background, nên val "
                             "vẫn chứa ký tự của code dùng để train.")
    else:
        plan.split_by = "image"
        plan.warnings.append("Chỉ có 1 code và 1 background: tập val được tách ngẫu nhiên "
                             "theo ảnh, nên chỉ đo được mô hình trên dữ liệu rất giống train.")
        return
    weights = codes if plan.split_by == "code" else backgrounds
    share = 100 * sum(weight for name, weight in weights.items()
                      if plan.split.get(name) == "val") / max(1, sum(weights.values()))
    if abs(share - percent) > 10:
        plan.warnings.append(f"Tập val chiếm {share:.0f} % số ảnh (yêu cầu {percent} %) vì "
                             f"phải giữ trọn mỗi {plan.split_by} trong một tập.")


# ---------------------------------------------------------------------- running

def existing_layout(out):
    """``"flat"``, ``"split"`` or None for what an output folder already holds."""
    images = Path(out) / "images"
    if not images.is_dir():
        return None
    if (images / "train").is_dir() or (images / "val").is_dir():
        return "split"
    if any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
           for path in images.iterdir()):
        return "flat"
    return None


def layout_conflict(out, val_percent):
    """Return why ``out`` cannot take this export's layout, or None."""
    current = existing_layout(out)
    wanted = "split" if val_percent > 0 else "flat"
    if current is None or current == wanted:
        return None
    if current == "flat":
        return ("Thư mục đã có data dạng phẳng (images/ + labels/); không trộn với dạng "
                "train/val. Hãy chọn thư mục mới hoặc đặt tập val = 0 %.")
    return ("Thư mục đã có data dạng train/val; không trộn với dạng phẳng. "
            "Hãy chọn thư mục mới hoặc đặt tập val > 0 %.")


def read_manifest(out):
    path = Path(out) / "manifest.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle)]


def previous_split(rows):
    """``(split_by, {name: split})`` recorded in an earlier manifest."""
    by_code, by_background = {}, {}
    for row in rows:
        if row.get("split") in ("train", "val"):
            by_code.setdefault(row.get("code", ""), set()).add(row["split"])
            by_background.setdefault(row.get("background", ""), set()).add(row["split"])
    if by_code and all(len(splits) == 1 for splits in by_code.values()):
        return "code", {name: next(iter(splits)) for name, splits in by_code.items()}
    if by_background and all(len(splits) == 1 for splits in by_background.values()):
        return "background", {name: next(iter(splits)) for name, splits in by_background.items()}
    return None, {}


@dataclass
class BatchResult:
    out: Path
    seed: int
    made: dict = field(default_factory=dict)       # mode → images written
    resumed: int = 0                               # images already written, skipped
    failures: list = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False
    split_counts: dict = field(default_factory=dict)

    @property
    def total(self):
        return sum(self.made.values())


def write_dataset_metadata(out, split):
    if not (out / "classes.txt").exists():
        atomic_write(out / "classes.txt", "date\n")
    yaml_path = out / "data.yaml"
    if split:
        atomic_write(yaml_path, "path: " + json.dumps(out.as_posix())
                     + "\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: date\n")
        return
    if not yaml_path.exists():
        atomic_write(yaml_path, "path: " + json.dumps(out.as_posix())
                     + "\ntrain: images\nval: images\nnc: 1\nnames:\n  0: date\n")
        return
    # Keep an existing dataset's metadata, but Ultralytics needs a val split.
    text = yaml_path.read_text(encoding="utf-8-sig")
    if not re.search(r"^val\s*:", text, re.MULTILINE):
        if re.search(r"^train\s*:.*$", text, re.MULTILINE):
            text = re.sub(r"^(train\s*:.*)$", r"\1\nval: images", text,
                          count=1, flags=re.MULTILINE)
        else:
            text = text.rstrip("\n") + "\nval: images\n"
        atomic_write(yaml_path, text)


def write_out_lines_index(out, plan):
    """Record each background's out-lines in ``out/out_lines.json``.

    Generated images keep their background's pixel frame, so an automatic
    "out" check can apply ``out_line_hits`` with these lines to the detected
    boxes of any image whose manifest row names that background. Entries of
    earlier runs are kept.
    """
    targets = {pair.target.name: pair.target for pair, _mode, _count in plan.tasks()
               if pair.target.out_lines}
    if not targets:
        return
    path = out / "out_lines.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    for name, target in sorted(targets.items()):
        data[name] = {"size": list(target.image.size),
                      "lines": [[[round(x, 2), round(y, 2)] for x, y in line]
                                for line in target.out_lines]}
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _fingerprint_file(path):
    with Image.open(path) as opened:
        return hashlib.sha256(opened.convert("RGB").tobytes()).digest()


def run_batch(plan, out, generator=None, progress=None):
    """Write every image of ``plan`` into ``out``.

    ``progress(done, total, text)`` is called after each image; returning
    False cancels the run. Images already listed in ``manifest.csv`` for the
    same seed are skipped, so a cancelled run can be resumed.
    """
    out = Path(out)
    generator = generator or Generator.from_settings(plan.settings)
    split_layout = plan.settings.val_percent > 0
    result = BatchResult(out, plan.seed)
    rows = read_manifest(out)
    written = {}
    for row in rows:
        if row.get("seed") == str(plan.seed):
            key = (row.get("code"), row.get("background"), row.get("mode"), row.get("variant"))
            written[key] = row
    total, done = plan.total, 0
    manifest_path = out / "manifest.csv"
    handle = None
    splits = ("train", "val") if split_layout else (None,)
    folders = {}
    for split in splits:
        folders[split] = (out / "images" / split if split else out / "images",
                          out / "labels" / split if split else out / "labels")
    longest = max((len(str(labels_dir / f"synthetic_{pair.target.stem}_{pair.code.stem}"
                                         f"_{mode}_000000.txt.tmp"))
                   for pair, mode, _count in plan.tasks()
                   for _images_dir, labels_dir in folders.values()), default=0)
    if os.name == "nt" and longest >= MAX_PATH_LENGTH:
        result.error = (f"đường dẫn file sẽ dài {longest} ký tự, vượt giới hạn "
                        f"{MAX_PATH_LENGTH} của Windows; hãy chọn thư mục xuất ngắn hơn "
                        "(ví dụ D:\\out) hoặc đổi tên ảnh ngắn hơn")
        return result
    try:
        for images_dir, labels_dir in folders.values():
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
        new_manifest = not manifest_path.exists()
        handle = open(manifest_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if new_manifest:
            writer.writeheader()
        for pair, mode, count in plan.tasks():
            if result.cancelled:
                break
            code, target = pair.code, pair.target
            job = pair.jobs[mode]
            prefix = f"synthetic_{target.stem}_{code.stem}_{mode}"
            number = first_free_number(prefix, [
                (directory, suffix) for images_dir, labels_dir in folders.values()
                for directory, suffix in ((images_dir, ".png"), (labels_dir, ".txt"))])
            fingerprints, pending_fingerprints = set(), []
            made = 0
            try:
                for variant in range(count):
                    key = (code.name, target.name, mode, str(variant))
                    row = written.get(key)
                    if row and (out / row["image"]).exists():
                        pending_fingerprints.append(out / row["image"])
                        result.resumed += 1
                        done += 1
                        continue
                    if pending_fingerprints:
                        # Resuming: the earlier variants must still count as
                        # taken, exactly as in an uninterrupted run.
                        fingerprints.update(_fingerprint_file(path) for path in pending_fingerprints)
                        pending_fingerprints = []
                    rng = task_rng(plan.seed, code.name, target.name, mode, variant)
                    composition = generator.generate(code, target, job, variant, rng, fingerprints)
                    split = plan.split_of(pair, mode, variant)
                    images_dir, labels_dir = folders[split]
                    name = f"{prefix}_{number:06d}"
                    number += 1
                    composition.image.save(images_dir / f"{name}.png")
                    atomic_write(labels_dir / f"{name}.txt",
                                 encode(composition.boxes, *target.image.size))
                    writer.writerow({
                        "image": (images_dir / f"{name}.png").relative_to(out).as_posix(),
                        "split": split or "", "code": code.name, "background": target.name,
                        "mode": mode, "variant": variant, "seed": plan.seed})
                    handle.flush()
                    made += 1
                    done += 1
                    result.split_counts[split or "all"] = result.split_counts.get(
                        split or "all", 0) + 1
                    if progress is not None and progress(
                            done, total, f"{code.stem} × {target.stem} · {mode}: "
                                         f"{variant + 1}/{count}") is False:
                        result.cancelled = True
                        break
            except ValueError as exc:
                result.failures.append(f"{code.stem} × {target.stem} · {mode}: {exc}")
                done += count - made
            result.made[mode] = result.made.get(mode, 0) + made
        write_dataset_metadata(out, split_layout)
        write_out_lines_index(out, plan)
        append_export_log(out, plan, result)
    except OSError as exc:
        result.error = str(exc)
    finally:
        if handle is not None:
            handle.close()
    return result


def append_export_log(out, plan, result):
    pairs = sorted({(pair.code.name, pair.target.name) for pair, _mode, _count in plan.tasks()})
    line = "\t".join([
        f"{datetime.now():%Y-%m-%d %H:%M:%S}", f"seed={plan.seed}",
        f"count={plan.settings.count}", f"count_mode={plan.settings.count_mode}",
        f"val={plan.settings.val_percent}%", f"split_by={plan.split_by or '-'}",
        f"size={plan.settings.size_mode}",
        f"angle=±{plan.settings.angle_jitter_deg:g}°",
        "pairs=" + ";".join(f"{code}+{background}" for code, background in pairs),
        ",".join(f"{name}={made}" for name, made in result.made.items()),
        f"resumed={result.resumed}" + (" cancelled" if result.cancelled else ""),
    ])
    with open(out / "export_log.txt", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# --------------------------------------------------------------- contact sheet

def _font(size):
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def contact_sheet(plan, generator=None, max_pairs=4, cell=(360, 150)):
    """Render variant 0 of every mode for a few pairs as one image.

    Each cell is cropped around the region and shows the label boxes. The
    images are exactly what the run makes as variant 0 of each pair and mode.
    Returns ``(image, failures)``.
    """
    generator = generator or Generator.from_settings(plan.settings)
    pairs = [pair for pair in plan.valid_pairs if pair.jobs]
    # Prefer different codes first, then different backgrounds.
    chosen, seen_codes = [], set()
    for pair in pairs:
        if pair.code.name not in seen_codes:
            chosen.append(pair)
            seen_codes.add(pair.code.name)
    for pair in pairs:
        if pair not in chosen:
            chosen.append(pair)
    chosen = chosen[:max_pairs]
    modes = [mode for mode in plan.modes if any(mode in pair.jobs for pair in chosen)]
    label_width, header = 190, 26
    cell_w, cell_h = cell
    sheet = Image.new("RGB", (label_width + cell_w * max(1, len(modes)),
                              header + cell_h * max(1, len(chosen))), (32, 36, 43))
    draw = ImageDraw.Draw(sheet)
    font, small = _font(13), _font(11)
    for column, mode in enumerate(modes):
        draw.text((label_width + column * cell_w + 6, 6), mode, fill=(255, 255, 255), font=font)
    failures = []
    for row, pair in enumerate(chosen):
        top = header + row * cell_h
        draw.text((6, top + 6), pair.code.stem[:26], fill=(0, 229, 255), font=small)
        draw.text((6, top + 24), "× " + pair.target.stem[:24], fill=(255, 202, 40), font=small)
        region = pair.target.region
        for column, mode in enumerate(modes):
            x0 = label_width + column * cell_w
            if mode not in pair.jobs:
                draw.text((x0 + cell_w // 2 - 4, top + cell_h // 2 - 8), "–",
                          fill=(150, 150, 150), font=font)
                continue
            rng = task_rng(plan.seed, pair.code.name, pair.target.name, mode, 0)
            try:
                composition = generator.generate(pair.code, pair.target, pair.jobs[mode], 0,
                                                 rng, set())
            except ValueError as exc:
                failures.append(f"{pair.code.stem} × {pair.target.stem} · {mode}: {exc}")
                draw.text((x0 + 6, top + cell_h // 2 - 8), "lỗi", fill=(255, 82, 82), font=font)
                continue
            # Crop around the region and the date: an out-line date may sit
            # outside the region.
            corners = region.corners() + [corner for label in composition.boxes
                                          for corner in label.corners()]
            left, right = min(x for x, _ in corners), max(x for x, _ in corners)
            upper, lower = min(y for _, y in corners), max(y for _, y in corners)
            margin_x, margin_y = (right - left) * .15, (lower - upper) * .35
            box = tuple(round(value) for value in (
                max(0, left - margin_x), max(0, upper - margin_y),
                min(pair.target.image.width, right + margin_x),
                min(pair.target.image.height, lower + margin_y)))
            crop = composition.image.crop(box)
            overlay = ImageDraw.Draw(crop)
            for line in pair.target.out_lines:
                overlay.line([(x - box[0], y - box[1]) for x, y in line],
                             fill=(255, 64, 255), width=3)
            hits = set(composition.out_hits)
            for index, label in enumerate(composition.boxes):
                overlay.polygon([(x - box[0], y - box[1]) for x, y in label.corners()],
                                outline=(255, 64, 64) if index in hits else (0, 230, 0))
            factor = min((cell_w - 8) / crop.width, (cell_h - 8) / crop.height)
            thumb = crop.resize((max(1, round(crop.width * factor)),
                                 max(1, round(crop.height * factor))),
                                Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x0 + (cell_w - thumb.width) // 2,
                                top + (cell_h - thumb.height) // 2))
    return sheet, failures


# ------------------------------------------------------- auto-propose characters

_MODEL_CACHE = {}


def propose_character_boxes(image, model_path, conf=.25, sizes=None, iou_limit=.4):
    """Detect characters with a YOLO OBB model and return them as boxes.

    Ultralytics is imported only here, so the tool still starts quickly. The
    model is run at a few input sizes and the detections are merged, because
    a single size misses characters on large sample images.
    """
    from ultralytics import YOLO  # noqa: PLC0415  (slow import, only on demand)

    model_path = Path(model_path)
    key = (str(model_path), model_path.stat().st_mtime_ns)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = YOLO(str(model_path))
        if model.task != "obb":
            raise ValueError(f"Model phải thuộc task obb; model hiện tại: {model.task}")
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = model
    image = image.convert("RGB")
    if sizes is None:
        longest = max(image.size)
        sizes = sorted({640, 1024, max(640, min(1920, (longest + 31) // 32 * 32))})
    detections = []
    for size in sizes:
        result = model.predict(source=image, conf=conf, imgsz=size, verbose=False)[0]
        if result.obb is None:
            continue
        for (cx, cy, width, height, angle), confidence in zip(
                result.obb.xywhr.cpu().tolist(), result.obb.conf.cpu().tolist()):
            # Keep the box's width along the text: a box reported as rotated
            # by ~90° is the same rectangle with width and height swapped.
            if abs(angle) > math.pi / 4:
                width, height = height, width
                angle -= math.copysign(math.pi / 2, angle)
            box = Box(0, cx, cy, width, height, angle)
            if box.valid(*image.size):
                detections.append((float(confidence), box))
    detections.sort(key=lambda item: -item[0])
    kept = []
    for _confidence, box in detections:
        if all(_rect_iou(box, other) <= iou_limit for other in kept):
            kept.append(box)
    rows = split_rows(kept)
    return [box for row in rows for box in row]


def _rect_iou(first, second):
    a = [*map(min, zip(*first.corners())), *map(max, zip(*first.corners()))]
    b = [*map(min, zip(*second.corners())), *map(max, zip(*second.corners()))]
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
