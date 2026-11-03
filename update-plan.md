# Update plan: synthetic data from many codes × many backgrounds

Status: **implemented** (phases 1–6), with the notes below. The engine lives in
`synth_engine.py` (no Tk); `label_tool.py` holds the GUI and `--batch` command line.

Implementation notes (decisions from §6, and where the build differs from the plan):

- §6 decisions used the proposed defaults: N per mode spread over the pairs; all valid pairs;
  val 20 % split by code; size by image ratio, with "Cỡ ký tự theo vùng" as an opt-in checkbox
  (date fills 85 % of the region); auto-propose with `best_char_obb.pt` included.
- Random generator: one per **(seed, code, background, mode, variant)** instead of per
  (seed, code, background, mode). The output is still fully reproducible from the seed, and a
  resumed run produces byte-identical images without regenerating the variants already written.
- Val split by code picks val codes so their **share of images** is close to the %, not their
  count (a 1-box code makes few images). Each code still sits in one split, and a warning appears
  when the share is off by more than 10 points. Later runs into the same folder keep earlier codes'
  splits (read from `manifest.csv`).
- Checkbox state is stored as overrides (`codes` / `codes_unchecked`, `backgrounds` /
  `backgrounds_unchecked`, `modes` / `modes_unchecked`). Anything not listed uses the default, so a
  newly labeled code or newly saved mode is included automatically.
- Auto-propose has no Otsu/connected-component fallback. Without ultralytics it shows an error and
  you draw boxes by hand. The model is run at 640, 1024 and the image's own size and the
  detections are merged, because one size misses many characters on the 1920×480 samples. Its
  boxes are taller than the hand labels, so always review them.
- Windows path limit: a run whose longest file path would exceed 259 characters stops before
  writing anything and asks for a shorter output folder.
- CLI: `python label_tool.py --batch [--check] [--seed N] [--count N] [--out DIR]`.

## 1. Where things stand

The "TỰ ĐỘNG TẠO TẤT CẢ LỖI" export uses exactly **one code and one background**: whichever pair is
open in the editor.

| Input | Read from | Used by |
|---|---|---|
| Code (the characters) | `self.sample_image`, `self.sample_boxes` | `compose`, `asset_for`, `automatic_error_state`, `auto_mode_problem`, `auto_arc_range`, `automatic_cut_lines`, `resolve_saved_state` |
| Background + region | `self.background_image`, `self.region` | `compose`, `dimension_scale`, `jittered_region`, `generate_unique`, `export_synthetic` |

To cover 5 codes × 10 backgrounds today, you would open each of the 50 pairs by hand and export
50 times. Regions are drawn one background at a time, and labeling a code means drawing every
character box by hand (29 boxes for `Image_0030`).

Terms used below:

- **code**: an image in `group_samples/images/` plus its character boxes (`labels/<stem>.txt`).
- **background**: an image in `group_background/`.
- **region**: the one rectangle on a background where the date is placed (`group_background/labels/<stem>.txt`).
- **pair**: one code placed on one background.
- **mode**: one error type (the 8 automatic modes plus the saved modes).

## 2. Target workflow

```
 ① Take codes             ② Make backgrounds         ③ Draw regions              ④ Generate
 ──────────────────       ─────────────────────      ──────────────────────      ──────────────────────
 add images, let the      "Tạo background từ box"    region queue: one           choose codes, backgrounds,
 model propose the        for all checked codes      rectangle per background,   modes and count
 character boxes, fix     at once, or import         autosave, jump to next;     → Kiểm tra → Xem trước
 them, tick ✓             backgrounds                copy to same-size images    → Tạo
          │                                                                               ▲
          └──────────────────── every checked code × every background with a region ──────┘
```

File formats on disk stay the same: same folders, same YOLO OBB label files. Only the tool's
workflow and the export engine change.

## 3. Design

### 3.1 Engine refactor (prerequisite)

Move image generation out of the Tk class so that it can run on any pair, not just the open one:

```python
@dataclass
class Code:              # one sample image
    path: Path
    image: Image.Image
    boxes: list[Box]
    assets: dict          # character_asset() cache for this code

@dataclass
class Target:            # one background
    path: Path
    image: Image.Image
    region: Box

class Generator:          # no Tk inside: usable from the GUI and from a command line
    def compose(self, code, target, state, region) -> Composition
    def auto_state(self, code, target, mode, variant, rng) -> dict
    def mode_problem(self, code, target, mode) -> str | None
    def resolve_saved(self, code, saved_state) -> tuple[dict | None, str | None]
    def generate(self, code, target, job, variant, rng, fingerprints) -> Composition
```

The current methods move almost unchanged. `self.sample_*`, `self.background_*` and `self.region`
become parameters, and `dimension_scale()` becomes a function of `(code, target)`. The editor keeps
working by calling `Generator` with the code and background that are open.

Done when the single-pair export still works and the existing smoke test passes.

### 3.2 Step ① Take codes

The sample list shows each code's status and gets a checkbox:

```
1. Ảnh lấy date
 [x] Image_0030_20261029203310.png   29 box · 2 hàng
 [x] Image_0050_20261101194708.png   29 box · 2 hàng
 [ ] Image_0060_20261101195212.bmp    1 box · 1 hàng   ⚠ chỉ chạy được blur / cut_row_1
 [ ] <ảnh mới>.png                    chưa có box
```

- **Checkbox** means "use in batch" and is saved in the config. It defaults to checked when the code
  has at least one box.
- **"Tự tìm ký tự"** (auto-propose boxes): runs `best_char_obb.pt` on the open code, loading the
  model the same way `predict_obb.py` does, and turns the detections into normal editable boxes that
  you then fix. Ultralytics is only imported when you press the button, so the tool still starts
  quickly.
  - Fallback without the model: draw one rectangle around the date and split it with Otsu plus
    connected components. The dates are dot-matrix print, so the dots have to be merged (dilated)
    first, which makes this less reliable. The model is the recommended path.
- **Checks shown next to each code:** boxes with no print found, the number of rows, and boxes that
  overlap.

### 3.3 Step ② Make backgrounds

- "Tạo background từ box" processes **all checked codes in one click** (today it handles only the
  open code). Names stay `<code>_background_NNNNNN.png`, so you can still see which code each
  background came from.
- "Thêm background" (import) is unchanged.

### 3.4 Step ③ One rectangle per background

The background list also shows its status and gets a checkbox:

```
2. Ảnh chứa date
 [x] Image_0052_20261029203505.png               ✓ vùng 522×92
 [x] Image_0048_20261029203500.png               ✗ chưa có vùng
 [x] <code>_background_000000.png                ✗ chưa có vùng
```

- **Region queue ("Vẽ vùng lần lượt")**
  - Opens the first checked background that has no region, in preview view with the region tool
    active.
  - After you draw the rectangle it is autosaved (this already happens today) and the next
    background opens.
  - Keys: `Enter` / `→` next, `←` previous, `Esc` stop. The status bar shows e.g. "Vùng 3/12".
- **Copy region ("Áp vùng này cho…")**
  - Copies the current region to the checked backgrounds that have none, scaled by the size ratio
    when the image size differs.
  - Asks before overwriting an existing region.
  - Backgrounds from the same camera put the date in the same place, so usually you draw once and
    copy.
- **Overview (optional):** a thumbnail grid with every region drawn, so a misplaced rectangle is easy
  to spot.

### 3.5 Step ④ Batch generate

A new dialog **"Tạo data hàng loạt"** replaces the plain "how many images" prompt of
"TỰ ĐỘNG TẠO TẤT CẢ LỖI":

```
┌ Tạo data hàng loạt ───────────────────────────────────────────────────┐
│ Code:        2 đã chọn  (Image_0030, Image_0050)                        │
│ Background:  3 đã chọn · 1 thiếu vùng → bỏ qua                          │
│ Ghép cặp:    (•) mọi cặp hợp lệ    ( ) chỉ ảnh cùng kích thước          │
│              [ ] tối đa [ 3 ] background ngẫu nhiên cho mỗi code        │
│ Dạng lỗi:    [x] spacing  [x] overlap  [x] arc  [x] blur                │
│              [x] cut_row_1  [x] cut_row_2  [x] cut_both_rows            │
│              [x] delete_objects  [x] mode_01_cut_delete (đã lưu)        │
│ Số lượng:    [ 10 ] ảnh  (•) cho mỗi dạng lỗi, chia đều các cặp         │
│                          ( ) cho mỗi dạng lỗi × mỗi cặp                 │
│ Tập val:     [ 20 ] %  tách theo code                                   │
│ Seed:        [        ]                                                 │
│ Dự kiến: 4 cặp · 9 dạng → 90 ảnh · ~40 MB · ~6 giây                     │
│                  [Kiểm tra]   [Xem trước]   [Tạo]   [Đóng]              │
└────────────────────────────────────────────────────────────────────────┘
```

**Pairs.** Every checked code is paired with every checked background that has a region. A pair is
skipped (and the reason is reported) when:

- the code's date, in normal layout, does not fit inside the region ("date rộng/cao hơn vùng");
- "chỉ ảnh cùng kích thước" is on and the image sizes differ (this protects the size assumption, see
  §5).

Optionally, "tối đa K background cho mỗi code" picks K random backgrounds per code instead of the
full cross product.

**Count.** There are two possible meanings. The dialog shows the resulting total before you run:

| Option | Total images | Use it when |
|---|---|---|
| N per mode, spread over the pairs (proposed default) | modes × N | you want a fixed dataset size: more pairs add variety, not volume |
| N per mode for every pair | pairs × modes × N | you want today's behavior, repeated for each pair |

When spreading, the pairs are cycled in shuffled order, so each pair gets ⌊N/P⌋ or ⌊N/P⌋+1 images of
each mode.

**Modes.** These are the same jobs as today's `export_jobs()`, but checked for each pair:

- An automatic mode is skipped only for the pairs where it cannot produce a real error (one row,
  one box, a region too short for the arc, …).
- A saved mode that uses box indices (spacing groups, deleted objects) runs only on the code it was
  made on. Saved modes without index data (blur, arc, overlap, cut lines) run on every pair.

**"Kiểm tra" (check).** This applies all the rules above **without composing any image** and shows a
table of pairs × modes, so problems surface before a long run instead of in the middle of it:

```
                    spacing overlap arc blur cut_r1 cut_r2 cut_all delete mode_01
0030 × 0052            3      3     3    3     3      3      3      3      3
0050 × 0052            3      3     3    3     3      3      3      3      –  dùng box của 0030
0060 × 0052            –      –     –    3     3      –      –      –      –  1 box
0030 × 0048            –      –     –    –     –      –      –      –      –  chưa có vùng
```

**"Xem trước" (preview).** Renders one image per mode for a few pairs and saves them as a single
contact sheet (`preview/sheet_<seed>.png`) with the label boxes drawn, so you can check a whole batch
at a glance.

**"Tạo" (run).**

- **Loop order** is `code → background → mode → variant`, so each code's character crops and each
  background image are loaded only once.
- **One random generator per (seed, code, background, mode).** Adding a background later does not
  change the images of the other pairs.
- **Progress:** uses the existing progress window with its Cancel button.

**Output folder**

```
synthetic_output/
  images/train/   images/val/
  labels/train/   labels/val/
  data.yaml         train: images/train · val: images/val
  manifest.csv      image, split, code, background, mode, variant, seed
  export_log.txt
```

- **File names** stay `synthetic_<background>_<code>_<mode>_NNNNNN.png`, which is already unique per
  pair.
- **Val split by code.** All images made from one code go to the same split, so val measures how the
  model does on codes it has never seen, not on copies of its training characters. This is the same
  idea as `partition_donors()` in `augment_obb_backgrounds.py`.
  - It needs at least 2 codes. With a single code it falls back to splitting by background and shows
    a warning.
  - `0 %` keeps today's flat `images/` + `labels/` layout.
- **Existing exports:** a folder that already holds the flat layout is not mixed with the
  train/val layout. The dialog asks for a new folder instead.
- **Resuming:** `manifest.csv` makes a cancelled run resumable. With the same seed, images already
  written are skipped on restart.

### 3.6 Config additions (`label_tool_config.json`)

```jsonc
"batch": {
  "codes": ["Image_0030_20261029203310.png", "Image_0050_20261101194708.png"],
  "backgrounds": ["Image_0052_20261029203505.png"],
  "pairing": "all",                // "all" | "same_size"
  "max_backgrounds_per_code": 0,   // 0 = no limit
  "count": 10,
  "count_mode": "per_mode",        // "per_mode" | "per_mode_per_pair"
  "modes": ["auto_spacing", "auto_overlap", "...", "mode_01_cut_delete"],
  "val_percent": 20
}
```

Names that no longer exist (deleted files or modes) are ignored when the config is loaded.

### 3.7 Optional: command line

`python label_tool.py --batch` would run §3.5 from the saved config without opening the window.
This works because the `Generator` from §3.1 has no Tk dependency. It is useful for large runs or for
running on another machine.

## 4. Phases

| # | Work | What you get | Size |
|---|---|---|---|
| 1 | Engine refactor (§3.1) | nothing visible; same output | M |
| 2 | Status and checkboxes in both lists, region queue, copy region (§3.2–3.4) | fast region drawing | M |
| 3 | Batch dialog: pairs, count options, Kiểm tra, run, manifest (§3.5) | export over many pairs | L |
| 4 | Val split by code, `data.yaml` with train/val | a ready train/val dataset | S |
| 5 | Batch background creation, contact-sheet preview (§3.3, §3.5) | fewer clicks, faster checking | S |
| 6 | Auto-propose character boxes with the model, command line (§3.2, §3.7) | faster code labeling | M |

Phases 1 → 3 deliver the main request. Phases 4–6 can follow in any order.

**Testing for each phase:** extend the scripted smoke test (scratch copy of the data, dialogs
stubbed) with:

- same seed → identical output;
- pair filtering;
- how counts are spread across pairs;
- split by code;
- resuming a cancelled run.

## 5. Risks and limits

- **Dataset size grows fast.** The total is pairs × modes × N, e.g. 5 codes × 10 backgrounds ×
  9 modes × 10 = 4,500 images.
  - Measured on the current code: about 0.44 MB and 60 ms per 1920×480 image, so 4,500 images take
    about 2 GB and about 5 minutes.
  - The dialog shows this estimate and asks for confirmation above about 20,000 images.
- **Character size across cameras.** Characters are scaled by the ratio of the whole image sizes,
  which assumes the code and the background were captured with the same camera framing. Mixing
  backgrounds from different cameras would give characters of the wrong size. Two ways to handle it:
  - the "chỉ ảnh cùng kích thước" pairing option;
  - a new option "Cỡ ký tự theo vùng": scale the date to fill e.g. 85 % of the region height. The
    rectangle you draw would then also set the character size.
- **One region per background**, as you asked. The label format could hold several regions, but
  `load_background()` rejects more than one today.
- **Saved modes that use box indices** do not transfer between codes. This is by design (§3.5).

## 6. Decisions needed before implementing

1. **Count default:** N per mode spread over all pairs, or N per mode for every pair?
2. **Pairing default:** all valid pairs, or only images of the same size?
3. **Val split:** by code (recommended)? What percentage?
4. **Character size:** keep scaling by the image size ratio, or add "Cỡ ký tự theo vùng"?
5. **Auto-proposing character boxes with `best_char_obb.pt`:** include it in this round, or later?
