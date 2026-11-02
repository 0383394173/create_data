from __future__ import annotations

import copy
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from PIL import Image, ImageTk
import yaml


def normalize_angle(angle):
    return (angle + math.pi/2) % math.pi - math.pi/2

@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0

    def __post_init__(self):
        self.angle = normalize_angle(self.angle)

    def corners(self):
        c, s = math.cos(self.angle), math.sin(self.angle)
        return [(self.cx + x*c - y*s, self.cy + x*s + y*c)
                for x, y in [(-self.w/2, -self.h/2), (self.w/2, -self.h/2),
                             (self.w/2, self.h/2), (-self.w/2, self.h/2)]]

    def contains(self, x, y):
        c, s = math.cos(self.angle), math.sin(self.angle)
        dx, dy = x-self.cx, y-self.cy
        return abs(dx*c+dy*s) <= self.w/2 and abs(-dx*s+dy*c) <= self.h/2

    def valid(self, width, height):
        return self.w >= 1 and self.h >= 1 and all(
            -1e-6 <= x <= width+1e-6 and -1e-6 <= y <= height+1e-6
            for x, y in self.corners())


def encode(boxes, width, height):
    lines = []
    for box in boxes:
        if not box.valid(width, height):
            raise ValueError('Hộp quá nhỏ hoặc nằm ngoài ảnh.')
        coords = [max(0.0, min(1.0, v)) for x, y in box.corners()
                  for v in (x/width, y/height)]
        lines.append(str(box.cls)+' '+' '.join(f'{v:.8f}' for v in coords))
    return '\n'.join(lines) + ('\n' if lines else '')

def decode(text, width, height, class_count):
    boxes = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 9:
                raise ValueError('cần 9 trường')
            cls = int(fields[0])
            values = list(map(float, fields[1:]))
            if not 0 <= cls < class_count or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError('class ID hoặc tọa độ không hợp lệ')
            pts = [(values[i]*width, values[i+1]*height) for i in range(0, 8, 2)]
            # Ultralytics may emit the opposite winding to the editor's corners.
            signed_area = sum(pts[i][0]*pts[(i+1) % 4][1] - pts[(i+1) % 4][0]*pts[i][1]
                              for i in range(4))
            if signed_area < 0:
                pts = [pts[0], pts[3], pts[2], pts[1]]
            dx, dy = pts[1][0]-pts[0][0], pts[1][1]-pts[0][1]
            box = Box(cls, sum(x for x, _ in pts)/4, sum(y for _, y in pts)/4,
                      math.hypot(dx, dy), math.dist(pts[1], pts[2]), math.atan2(dy, dx))
            tolerance = max(width, height)*1e-5
            if not box.valid(width, height) or any(
                    min(math.dist(point, corner) for corner in box.corners()) > tolerance
                    for point in pts):
                raise ValueError('bốn đỉnh phải là hình chữ nhật theo thứ tự liên tiếp')
            boxes.append(box)
        except ValueError as exc:
            raise ValueError(f'Dòng {number}: {exc}') from exc
    return boxes


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def read_classes(folder, dataset_mode=False):
    """Use dataset names as authoritative, with classes.txt as a fallback."""
    yaml_path = folder/'data.yaml'
    classes_path = folder/'classes.txt'
    if dataset_mode and yaml_path.exists():
        config = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
        if not isinstance(config, dict):
            raise ValueError('data.yaml phải chứa cấu hình dataset.')
        names = config.get('names')
        if isinstance(names, dict):
            names = {int(key): str(value) for key, value in names.items()}
            if sorted(names) != list(range(len(names))):
                raise ValueError('ID lớp trong data.yaml phải liên tiếp từ 0.')
            classes = [names[i] for i in range(len(names))]
        elif isinstance(names, list):
            classes = [str(name) for name in names]
        else:
            raise ValueError('data.yaml thiếu danh sách names hợp lệ.')
        if 'nc' in config and int(config['nc']) != len(classes):
            raise ValueError('nc không khớp với names trong data.yaml.')
    else:
        classes = classes_path.read_text(encoding='utf-8-sig').splitlines() if classes_path.exists() else []
    if any(not c.strip() or '\n' in c or '\r' in c for c in classes) or len(set(classes)) != len(classes):
        raise ValueError('Tên lớp không được trống, trùng nhau hoặc chứa xuống dòng.')
    if dataset_mode and not classes:
        raise ValueError('Dataset cần names trong data.yaml hoặc danh sách lớp trong classes.txt.')
    return classes


class App:
    COLORS = ['#00e5ff', '#ffca28', '#ff66aa', '#76ff03', '#b388ff', '#ff8a65']

    def __init__(self, root):
        self.root = root
        root.title('YOLO OBB Label — Gán nhãn hộp xoay')
        root.geometry('1280x820')
        self.folder = None
        self.image_root = None
        self.dataset_mode = False
        self.images = []
        self.index = -1
        self.photo = None
        self.original = None
        self.boxes = []
        self.classes = []
        self.selected = None
        self.history, self.future = [], []
        self.dirty = False
        self.scale, self.ox, self.oy = 1., 0., 0.
        self.gesture = None
        self.mode = tk.StringVar(value='select')
        self.status = tk.StringVar(value='Mở thư mục ảnh để bắt đầu.')
        bar = ttk.Frame(root, padding=6)
        bar.pack(fill='x')
        for label, command in [('Mở ảnh', self.open_folder), ('Mở dataset', self.open_dataset), ('Lưu Ctrl+S', self.save),
                               ('◀ A', lambda: self.navigate(-1)), ('D ▶', lambda: self.navigate(1)),
                               ('Vừa ảnh F', self.fit), ('Xuất dataset', self.export)]:
            ttk.Button(bar, text=label, command=command).pack(side='left', padx=3)
        ttk.Radiobutton(bar, text='Chọn / sửa (V)', variable=self.mode, value='select').pack(side='left')
        ttk.Radiobutton(bar, text='Vẽ hộp (B)', variable=self.mode, value='draw').pack(side='left')
        body = ttk.Panedwindow(root, orient='horizontal')
        body.pack(fill='both', expand=True)
        side = ttk.Frame(body, padding=8)
        body.add(side, weight=0)
        ttk.Label(side, text='Ảnh').pack(anchor='w')
        self.image_list = tk.Listbox(side, width=30, height=12, exportselection=False)
        self.image_list.pack(fill='x')
        self.image_list.bind('<<ListboxSelect>>', self.choose_image)
        ttk.Label(side, text='Lớp nhãn (ID giữ cố định)').pack(anchor='w', pady=(12, 0))
        self.class_list = tk.Listbox(side, height=7, exportselection=False)
        self.class_list.pack(fill='x')
        for label, command in [('Thêm lớp', self.add_class), ('Đổi tên lớp', self.rename_class),
                               ('Gán lớp cho hộp', self.assign_class)]:
            ttk.Button(side, text=label, command=command).pack(fill='x', pady=2)
        ttk.Label(side, text='Các hộp trong ảnh').pack(anchor='w', pady=(12, 0))
        self.box_list = tk.Listbox(side, height=8, exportselection=False)
        self.box_list.pack(fill='both', expand=True)
        self.box_list.bind('<<ListboxSelect>>', self.choose_box)
        for label, command in [('Xóa hộp (Delete)', self.delete), ('Nhân bản Ctrl+D', self.duplicate),
                               ('Hoàn tác Ctrl+Z', self.undo), ('Làm lại Ctrl+Y', self.redo)]:
            ttk.Button(side, text=label, command=command).pack(fill='x', pady=2)
        ttk.Label(side, text='Kéo trong hộp: di chuyển\nKéo góc: đổi kích thước\nKéo nút tròn: xoay\nQ/E: xoay ±1°; Shift: ±15°\nCon lăn: zoom; chuột giữa: pan', justify='left').pack(pady=10)
        self.canvas = tk.Canvas(body, background='#20242b', highlightthickness=0)
        body.add(self.canvas, weight=1)
        ttk.Label(root, textvariable=self.status, padding=6).pack(fill='x')
        self.canvas.bind('<Configure>', lambda e: self.render())
        self.canvas.bind('<Button-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.canvas.bind('<MouseWheel>', self.zoom)
        self.canvas.bind('<Button-2>', lambda e: setattr(self, 'pan', (e.x, e.y)))
        self.canvas.bind('<B2-Motion>', self.pan_move)
        for key, command in [('<Control-s>', self.save), ('<Control-z>', self.undo),
                             ('<Control-y>', self.redo), ('<Control-d>', self.duplicate),
                             ('<Delete>', self.delete), ('a', lambda: self.navigate(-1)),
                             ('d', lambda: self.navigate(1)), ('f', self.fit),
                             ('b', lambda: self.mode.set('draw')), ('v', lambda: self.mode.set('select'))]:
            root.bind(key, lambda e, fn=command: fn())
        root.bind('q', lambda e: self.rotate(-15 if e.state & 1 else -1))
        root.bind('e', lambda e: self.rotate(15 if e.state & 1 else 1))
        root.bind('Q', lambda e: self.rotate(-15))
        root.bind('E', lambda e: self.rotate(15))
        root.protocol('WM_DELETE_WINDOW', self.close)

    def label_path(self, image):
        image_root = getattr(self, 'image_root', None) or self.folder
        return self.folder/'labels'/image.relative_to(image_root).with_suffix('.txt')

    def open_folder(self):
        directory = filedialog.askdirectory(title='Chọn thư mục chứa ảnh (tìm cả thư mục con)')
        if directory:
            folder = Path(directory)
            self.open_project(folder, dataset_mode=(folder/'images').is_dir() and (folder/'labels').is_dir())

    def open_dataset(self):
        directory = filedialog.askdirectory(title='Chọn dataset_<thời_gian> chứa images, labels và data.yaml')
        if directory:
            self.open_project(Path(directory), dataset_mode=True)

    def open_project(self, folder, dataset_mode=False):
        if not self.save():
            return
        folder = folder.resolve()
        image_root = folder/'images' if dataset_mode else folder
        try:
            if dataset_mode and not image_root.is_dir():
                raise ValueError('Hãy chọn thư mục dataset chứa images/ và labels/, không chọn thư mục predict/.')
            classes = read_classes(folder, dataset_mode)
            images = sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in
                            {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
                            and 'labels' not in p.relative_to(image_root).parts)
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            messagebox.showerror('Không mở được dataset', str(exc))
            return
        if not images:
            messagebox.showwarning('Không có ảnh', 'Thư mục không chứa ảnh được hỗ trợ.')
            return
        labels = [p.relative_to(image_root).with_suffix('.txt').as_posix().casefold() for p in images]
        if len(set(labels)) != len(labels):
            messagebox.showerror('Trùng tên', 'Có ảnh cùng tên nhưng khác đuôi trong cùng thư mục. Hãy đổi tên để tránh trùng nhãn.')
            return
        self.folder, self.images, self.classes = folder, images, classes
        self.image_root, self.dataset_mode = image_root, dataset_mode
        self.original, self.photo, self.selected = None, None, None
        self.index, self.boxes, self.dirty = -1, [], False
        self.history, self.future, self.gesture = [], [], None
        self.canvas.delete('all')
        self.box_list.delete(0, 'end')
        self.status.set(f'Đã mở {folder}; chọn ảnh để xem nhãn.')
        self.refresh_classes()
        self.image_list.delete(0, 'end')
        for p in images:
            self.image_list.insert('end', p.relative_to(folder).as_posix())
        self.load(0)

    def save_classes(self, classes):
        yaml_path = self.folder/'data.yaml'
        if self.dataset_mode and yaml_path.exists():
            config = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
            if not isinstance(config, dict):
                raise ValueError('data.yaml không hợp lệ.')
            config['names'] = dict(enumerate(classes))
            config['nc'] = len(classes)
            atomic_write(yaml_path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        atomic_write(self.folder/'classes.txt', '\n'.join(classes)+ ('\n' if classes else ''))

    def refresh_classes(self):
        self.class_list.delete(0, 'end')
        for i, name in enumerate(self.classes):
            self.class_list.insert('end', f'{i}: {name}')
        if self.classes:
            self.class_list.selection_set(0)

    def edit_class(self, rename=False):
        if self.folder is None:
            return
        chosen = self.class_list.curselection()
        if rename and not chosen:
            return
        name = simpledialog.askstring('Tên nhãn', 'Nhập tên lớp:', initialvalue=self.classes[chosen[0]] if rename else '')
        if not name or not name.strip():
            return
        name = name.strip()
        if '\n' in name or '\r' in name or name in self.classes:
            messagebox.showerror('Tên không hợp lệ', 'Tên lớp phải duy nhất và nằm trên một dòng.')
            return
        updated = self.classes.copy()
        if rename:
            updated[chosen[0]] = name
        else:
            updated.append(name)
        try:
            self.save_classes(updated)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            messagebox.showerror('Không lưu được lớp', str(exc))
            return
        self.classes = updated
        self.refresh_classes()
        self.class_list.selection_clear(0, 'end')
        self.class_list.selection_set(chosen[0] if rename else len(self.classes)-1)
        self.render()

    def add_class(self):
        self.edit_class()

    def rename_class(self):
        self.edit_class(True)

    def load(self, index):
        try:
            with Image.open(self.images[index]) as img:
                original = img.convert('RGB')
            path = self.label_path(self.images[index])
            boxes = decode(path.read_text(encoding='utf-8-sig'), *original.size, len(self.classes)) if path.exists() else []
        except (OSError, ValueError) as exc:
            messagebox.showerror('Không mở được ảnh/nhãn', str(exc))
            self.image_list.selection_clear(0, 'end')
            if self.index >= 0:
                self.image_list.selection_set(self.index)
            return
        self.index, self.original, self.boxes = index, original, boxes
        self.selected = None
        self.history, self.future = [], []
        self.dirty = False
        self.gesture = None
        self.image_list.selection_clear(0, 'end')
        self.image_list.selection_set(index)
        self.image_list.see(index)
        self.fit()

    def choose_image(self, event):
        chosen = self.image_list.curselection()
        if chosen and chosen[0] != self.index and self.save():
            self.load(chosen[0])

    def navigate(self, step):
        if self.images and self.save():
            self.load(max(0, min(len(self.images)-1, self.index+step)))

    def save(self):
        if self.original is None:
            return True
        try:
            atomic_write(self.label_path(self.images[self.index]), encode(self.boxes, *self.original.size))
            self.save_classes(self.classes)
            self.dirty = False
            self.render()
            return True
        except (OSError, ValueError, yaml.YAMLError) as exc:
            messagebox.showerror('Không lưu được', str(exc))
            return False

    def snapshot(self):
        self.history.append(copy.deepcopy(self.boxes))
        self.history = self.history[-100:]
        self.future.clear()
        self.dirty = True

    def undo(self):
        if self.history:
            self.future.append(copy.deepcopy(self.boxes))
            self.boxes = self.history.pop()
            self.selected = None
            self.dirty = True
            self.render()

    def redo(self):
        if self.future:
            self.history.append(copy.deepcopy(self.boxes))
            self.boxes = self.future.pop()
            self.selected = None
            self.dirty = True
            self.render()

    def choose_box(self, event):
        chosen = self.box_list.curselection()
        if chosen:
            self.selected = chosen[0]
            self.render()

    def assign_class(self):
        chosen = self.class_list.curselection()
        if self.selected is not None and chosen:
            self.snapshot()
            self.boxes[self.selected].cls = chosen[0]
            self.render()

    def delete(self):
        if self.selected is not None:
            self.snapshot()
            self.boxes.pop(self.selected)
            self.selected = None
            self.render()

    def duplicate(self):
        if self.selected is not None:
            self.snapshot()
            box = copy.deepcopy(self.boxes[self.selected])
            box.cx += 10
            box.cy += 10
            if not box.valid(*self.original.size):
                box = copy.deepcopy(self.boxes[self.selected])
            self.boxes.append(box)
            self.selected = len(self.boxes)-1
            self.render()

    def rotate(self, degrees):
        if self.selected is not None:
            box = copy.deepcopy(self.boxes[self.selected])
            box.angle = normalize_angle(box.angle + math.radians(degrees))
            if box.valid(*self.original.size):
                self.snapshot()
                self.boxes[self.selected] = box
                self.render()

    def point(self, x, y):
        return ((x-self.ox)/self.scale, (y-self.oy)/self.scale)

    def screen(self, x, y):
        return (x*self.scale+self.ox, y*self.scale+self.oy)

    def rotation_handle(self, box):
        distance = box.h/2+28/self.scale
        return (box.cx+math.sin(box.angle)*distance, box.cy-math.cos(box.angle)*distance)

    def fit(self):
        if self.original:
            w, h = self.original.size
            cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            self.scale = max(.001, min((cw-30)/w, (ch-30)/h))
            self.ox, self.oy = (cw-w*self.scale)/2, (ch-h*self.scale)/2
            self.render()

    def render(self):
        self.canvas.delete('all')
        if self.original is None:
            return
        w, h = self.original.size
        # Crop to the visible viewport before resizing, keeping zoom memory bounded.
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        left, top = max(0, int(-self.ox/self.scale)), max(0, int(-self.oy/self.scale))
        right, bottom = min(w, math.ceil((cw-self.ox)/self.scale)), min(h, math.ceil((ch-self.oy)/self.scale))
        if right > left and bottom > top:
            cropped = self.original.crop((left, top, right, bottom))
            self.photo = ImageTk.PhotoImage(cropped.resize((max(1, round((right-left)*self.scale)),
                                                           max(1, round((bottom-top)*self.scale))), Image.Resampling.BILINEAR))
            self.canvas.create_image(*self.screen(left, top), image=self.photo, anchor='nw')
        self.box_list.delete(0, 'end')
        for i, box in enumerate(self.boxes):
            color = self.COLORS[box.cls % len(self.COLORS)]
            pts = [self.screen(x, y) for x, y in box.corners()]
            self.canvas.create_polygon(*[v for p in pts for v in p], fill='', outline=color,
                                       width=3 if i == self.selected else 2)
            self.canvas.create_text(pts[0][0], pts[0][1]-10, text=f'{i+1}: {self.classes[box.cls]}', fill=color, anchor='sw')
            self.box_list.insert('end', f'{i+1}. {self.classes[box.cls]} ({math.degrees(box.angle):.1f}°)')
            if i == self.selected:
                self.box_list.selection_set(i)
                for x, y in pts:
                    self.canvas.create_rectangle(x-5, y-5, x+5, y+5, fill=color, outline='white')
                rx, ry = self.screen(*self.rotation_handle(box))
                mx, my = (pts[0][0]+pts[1][0])/2, (pts[0][1]+pts[1][1])/2
                self.canvas.create_line(mx, my, rx, ry, fill=color)
                self.canvas.create_oval(rx-6, ry-6, rx+6, ry+6, fill=color, outline='white')
        self.status.set(f'{self.index+1}/{len(self.images)} • {self.images[self.index].relative_to(self.folder)} • {w}×{h} • '
                        f'{len(self.boxes)} hộp • Zoom {self.scale:.0%}'+ (' • Chưa lưu' if self.dirty else ''))

    def press(self, event):
        if self.original is None:
            return
        self.canvas.focus_set()
        p = self.point(event.x, event.y)
        if self.mode.get() == 'draw':
            if not self.class_list.curselection():
                messagebox.showinfo('Chọn lớp', 'Thêm và chọn một lớp trước khi vẽ.')
                return
            if not (0 <= p[0] <= self.original.width and 0 <= p[1] <= self.original.height):
                return
            self.gesture = ('draw', p, None)
            return
        if self.selected is not None:
            box = self.boxes[self.selected]
            if math.dist(p, self.rotation_handle(box))*self.scale < 12:
                self.gesture = ('rotate', p, copy.deepcopy(box))
                self.snapshot()
                return
            for i, corner in enumerate(box.corners()):
                if math.dist(p, corner)*self.scale < 12:
                    self.gesture = ('resize', p, copy.deepcopy(box), i)
                    self.snapshot()
                    return
        self.selected = next((i for i in reversed(range(len(self.boxes))) if self.boxes[i].contains(*p)), None)
        if self.selected is not None:
            self.gesture = ('move', p, copy.deepcopy(self.boxes[self.selected]))
            self.snapshot()
        self.render()

    def drag(self, event):
        if not self.gesture:
            return
        kind, start, original = self.gesture[:3]
        x, y = self.point(event.x, event.y)
        if kind == 'draw':
            x, y = max(0, min(self.original.width, x)), max(0, min(self.original.height, y))
            self.canvas.delete('preview')
            self.canvas.create_rectangle(*self.screen(*start), *self.screen(x, y), outline='white', dash=(4, 4), tags='preview')
            return
        box = copy.deepcopy(original)
        if kind == 'move':
            box.cx += x-start[0]
            box.cy += y-start[1]
        elif kind == 'rotate':
            box.angle = normalize_angle(
                box.angle + math.atan2(y-box.cy, x-box.cx)
                - math.atan2(start[1]-box.cy, start[0]-box.cx))
        elif kind == 'resize':
            corner = self.gesture[3]
            anchor = original.corners()[(corner+2) % 4]
            c, s = math.cos(box.angle), math.sin(box.angle)
            dx, dy = x-anchor[0], y-anchor[1]
            u, v = dx*c+dy*s, -dx*s+dy*c
            signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)][corner]
            if u*signs[0] < 1 or v*signs[1] < 1:
                return
            box.w, box.h = abs(u), abs(v)
            box.cx, box.cy = (anchor[0]+x)/2, (anchor[1]+y)/2
        if box.valid(*self.original.size):
            self.boxes[self.selected] = box
            self.render()

    def release(self, event):
        if self.gesture and self.gesture[0] == 'draw':
            start = self.gesture[1]
            x, y = self.point(event.x, event.y)
            x, y = max(0, min(self.original.width, x)), max(0, min(self.original.height, y))
            box = Box(self.class_list.curselection()[0], (x+start[0])/2, (y+start[1])/2, abs(x-start[0]), abs(y-start[1]))
            if box.valid(*self.original.size):
                self.snapshot()
                self.boxes.append(box)
                self.selected = len(self.boxes)-1
        self.gesture = None
        self.render()

    def zoom(self, event):
        if self.original:
            p = self.point(event.x, event.y)
            self.scale = max(.01, min(30, self.scale*(1.15 if event.delta > 0 else 1/1.15)))
            self.ox, self.oy = event.x-p[0]*self.scale, event.y-p[1]*self.scale
            self.render()

    def pan_move(self, event):
        if hasattr(self, 'pan'):
            self.ox += event.x-self.pan[0]
            self.oy += event.y-self.pan[1]
            self.pan = (event.x, event.y)
            self.render()

    def export(self):
        if not self.images or not self.classes or not self.save():
            return
        ratio = simpledialog.askfloat('Train / val', 'Tỷ lệ validation (0 < tỷ lệ < 1):', initialvalue=.2, minvalue=.01, maxvalue=.99)
        if ratio is None:
            return
        directory = filedialog.askdirectory(title='Chọn thư mục đích trống')
        if not directory:
            return
        target = Path(directory).resolve()
        if any(target.iterdir()):
            messagebox.showerror('Thư mục đích', 'Chọn thư mục trống để tránh ghi đè dữ liệu.')
            return
        records, missing = [], []
        try:
            for path in self.images:
                label = self.label_path(path)
                if not label.exists():
                    missing.append(path.name)
                    continue
                with Image.open(path) as img:
                    text = encode(decode(label.read_text(encoding='utf-8-sig'), *img.size, len(self.classes)), *img.size)
                records.append((path, text))
            if missing:
                raise ValueError(f'{len(missing)} ảnh chưa được gán nhãn/xác nhận. Mở từng ảnh và lưu; ảnh âm tính lưu file nhãn rỗng. Ví dụ: {missing[0]}')
            if len(records) < 2:
                raise ValueError('Cần ít nhất 2 ảnh đã lưu để tạo train và val.')
            random.Random(42).shuffle(records)
            nval = max(1, min(len(records)-1, round(len(records)*ratio)))
            for i, (path, text) in enumerate(records):
                split = 'val' if i < nval else 'train'
                name = f'{i:06d}_{path.stem}'
                destination = target/'images'/split/(name+path.suffix.lower())
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                atomic_write(target/'labels'/split/(name+'.txt'), text)
            yaml = 'path: .\ntrain: images/train\nval: images/val\nnames:\n'
            yaml += ''.join(f'  {i}: {json.dumps(name, ensure_ascii=False)}\n' for i, name in enumerate(self.classes))
            # Absolute root lets Ultralytics resolve this dataset from any working directory.
            yaml = yaml.replace('path: .', 'path: '+json.dumps(target.as_posix(), ensure_ascii=False), 1)
            atomic_write(target/'data.yaml', yaml)
            atomic_write(target/'classes.txt', '\n'.join(self.classes)+'\n')
            messagebox.showinfo('Xuất thành công', f'{len(records)-nval} train, {nval} val\n{target / "data.yaml"}')
        except (OSError, ValueError) as exc:
            messagebox.showerror('Không xuất được dataset', str(exc))

    def close(self):
        if self.dirty:
            answer = messagebox.askyesnocancel('Chưa lưu', 'Lưu nhãn trước khi thoát?')
            if answer is None or (answer and not self.save()):
                return
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
