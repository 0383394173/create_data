# Logic problems in `label_tool.py`

Review of `label_tool.py` (plus the `Box` / `encode` / `decode` helpers it imports from `label_obb.py`).
Line numbers refer to the current file. Items marked **(verified)** were reproduced with a small script; the
others come from reading the code paths.

## Summary

| #  | Severity | Area | Problem |
|----|----------|------|---------|
| 1  | High   | Export | The real export ignores saved modes; "Tạo ảnh kiểm tra" previews something different from what gets exported |
| 2  | High   | Key bindings | `Delete` / `Ctrl+Z` fire while typing in text fields → deletes a box and autosaves the label file |
| 3  | High   | Selection | `selected_box` is not reset when the view changes → `Delete` removes the wrong object |
| 4  | High   | Box indices | Deleting a sample box doesn't remap `spacing_groups` / `deleted_indices` / saved modes |
| 5  | High   | Cut | The removed side of a cut line depends on which direction the line was drawn **(verified)** |
| 6  | Medium | Delete mode | `compose()` overwrites `preview_source_indices` during preview export → clicks delete the wrong object or raise `IndexError` |
| 7  | Medium | Drawing | Region and cut-line gestures accept sample-image coordinates when the sample view is shown |
| 8  | Medium | Auto cut | A single-row date is always split into "2 rows"; the "needs 2 rows" error never fires |
| 9  | Medium | Loading | Picking a background silently replaces the chosen sample and wipes spacing groups and deletions |
| 10 | Medium | Persistence | Region edits are never autosaved (sample boxes are), so they're lost on switch or close |
| 11 | Medium | Robustness | Invalid text in a spinbox raises `TclError`: the preview breaks and the window can't be closed |
| 12 | Medium | Saved modes | Saved modes store indices and pixel coordinates but aren't tied to a sample or background |
| 13 | Medium | Mode state | Drawing a cut line changes the mode without `on_mode_change()` → stale panels and `last_effect_mode` |
| 14 | Medium | Mode state | `deleted_indices` applies in every mode, so "Không lỗi" (normal) can still have objects missing |
| 15 | Medium | Region angle | A region rotated past ±90° flips 180° after save/reload **(verified)** |
| 16 | Low    | Asset | `character_asset` can return an all-white crop → the object is labeled but invisible |
| 17 | Low    | Paths | Label paths use only the file stem → `a.png` and `a.bmp` share one label file |
| 18 | Low    | Auto params | Some automatic variants produce no error, an empty image, or a size-independent arc |
| 19 | Low    | Rows | `split_rows` splits one slightly slanted row into two rows **(verified)** |
| 20 | Low    | Undo | Every click in select mode adds a no-op undo entry; undo doesn't cover spacing groups |
| 21 | Low    | Status bar | `render()` overwrites status messages right after they are set |
| 22 | Low    | UI sync | Adding a spacing group resets the factor but not its label |
| 23 | Low    | Indices | `sample_index` can point to a different file after `refresh_files()` → wrong label overwritten |
| 24 | Low    | Dataset | The generated `data.yaml` has no `val:` key |
| 25 | Low    | Labels | Cut objects keep full-size labels, and manual "cut" with side "none" is identical to normal |
| 26 | Low    | Blur | Large blur radii get clipped by the 12 px crop padding |
| 27 | Perf   | Speed | Per-pixel Python loop in `line_side_mask`, O(n²) file-name search, everything runs on the UI thread |
| 28 | Minor  | Misc | Dead code, misleading docstring, uncaught errors, `_sync_view` quirks |

---

## High

### 1. The export ignores saved modes; the preview and the export don't match
`export_synthetic` (`label_tool.py:1441`), `validate_synthetic_source` (`label_tool.py:1309`), `save_synthetic_previews` (`label_tool.py:1411`)

- The "Các mode lỗi tự tạo đã lưu" panel (spacing groups, cut lines, deleted objects, blur and arc values) is only
  used by **"Tạo ảnh kiểm tra"** (preview images).
- The dataset export (`TỰ ĐỘNG TẠO TẤT CẢ LỖI`) always loops over the hard-coded `AUTO_ERROR_MODES` with random
  parameters (`automatic_error_state`). It never reads `self.saved_modes`, and it drops the user's `spacing_groups`,
  `cut_line` and `deleted_indices` (`label_tool.py:1323-1328`).
- So the preview images show the saved modes, but the exported dataset contains different, random errors.
- `if not AUTO_ERROR_MODES:` tests a non-empty constant tuple, so it is always false. The message inside ("Hãy tạo một dạng
  lỗi rồi bấm 'Lưu mode hiện tại' trước khi xuất") shows the intended check was `if not self.saved_modes:`. As a
  result, "Tạo ảnh kiểm tra" with no saved modes reports success with **0 images**.

**Fix direction:** decide which source drives the export (saved modes, automatic modes, or both). Make the preview
use the same source, and check `self.saved_modes` where saved modes are required.

### 2. Global `Delete` / `Ctrl+Z` bindings fire while typing in text fields
`label_tool.py:533-536`

`self.root.bind(...)` puts the handler on the toplevel's bindtag, which every child widget also carries (Entry,
Spinbox, and so on). For example, when you edit the group text `0-3,5,7` or the output folder and press **Delete**:
1. The Entry deletes the character, **and**
2. `delete_selected()` also runs. Right after you draw a box, `selected_box` points to that new box
   (`label_tool.py:1170`), so the box gets deleted and `autosave_sample_boxes()` **writes the label file to disk**.

`Ctrl+Z` in a text field has the same problem: it runs the app's undo and autosaves.

**Fix direction:** bind on the canvas, or ignore the event when `event.widget` is an Entry or Spinbox.

### 3. `selected_box` keeps its old meaning after a view change
`set_view` (`label_tool.py:905-907`), `delete_selected` (`label_tool.py:888-903`), `press` (`label_tool.py:1072-1084`)

`selected_box` is an index into the sample boxes in the sample view, and `0` means "the region" in the preview view.
`set_view()` doesn't clear it:
- Select sample box 0 → click "Xem background/preview" → press Delete → **the region is deleted**.
- Select the region (`selected_box = 0`) → click "Xem ảnh mẫu" → press Delete → **sample box 0 is deleted and the
  label file is autosaved**.

**Fix direction:** set `self.selected_box = None` in `set_view()`.

### 4. Deleting a sample box breaks every index-based reference
`delete_selected` (`label_tool.py:896`), `delete_sample_box` (`label_tool.py:1096-1104`)

`spacing_groups`, `deleted_indices` and every saved mode's state refer to boxes by **list position**. After
`self.sample_boxes.pop(k)`, every box after `k` moves down one position, but none of those structures are updated:
- `spacing_groups = [[5, 6]]` now groups the boxes that used to be 6 and 7. Out-of-range indices are dropped without
  any warning (`layout_boxes`, `label_tool.py:160-163`).
- `deleted_indices` now hides a different character.
- Undo restores `sample_boxes` and `deleted_indices`, but not `spacing_groups` (see #20).

**Fix direction:** remap or clear these structures when a box is removed, or identify boxes by a stable ID.

### 5. The removed side of a cut line depends on drawing direction (verified)
`line_side_mask` (`label_tool.py:220-232`)

The side is chosen from the sign of the 2D cross product. That sign flips when the segment is reversed, so
"Phía trên" (remove above) only works when the line is drawn **left → right**:

```
left->right 'above' keeps pixel at y=2: False | keeps y=17: True    (correct)
right->left 'above' keeps pixel at y=2: True  | keeps y=17: False   (removes the BOTTOM instead)
```
Automatic cuts always build the line in the region's +x direction, so they're fine. Lines the user draws by hand
are not.

**Fix direction:** normalize the line so that `x2 >= x1` (or orient it with the region's x axis) before computing the side.

---

## Medium

### 6. Preview export leaves `preview_source_indices` out of sync with `preview_boxes`
`compose` (`label_tool.py:934`), `save_synthetic_previews` (`label_tool.py:1400-1423`), `delete_placed_object` (`label_tool.py:857`)

`compose()` writes `self.preview_source_indices` as a side effect, including when it is called through
`compose_saved_mode()`. `save_synthetic_previews()` never calls `show_preview()` afterwards, so
`preview_source_indices` belongs to the **last saved mode** while `preview_boxes` still belongs to the live preview.
In "Xóa object" mode, a click then deletes a different source box, or raises `IndexError` when the last saved mode
had deleted objects (the list is shorter).

**Fix direction:** make `compose()` return the indices instead of storing them, or call `show_preview()` after previews.

### 7. Region and cut-line gestures don't check which view is shown
`release` (`label_tool.py:1171-1182`)

`sample_box` checks `self.view.get() == "sample"`, but `target_region` and `cut_line` don't check the view.
If you choose "Vẽ vùng đặt" or "Kẻ đường cắt" and then click "Xem ảnh mẫu" (draw mode stays the same),
dragging stores sample-image coordinates as a **background** region or cut line.

### 8. The automatic row split always finds two rows
`compose_automatic_error` (`label_tool.py:1367-1378`)

`split_at = max(range(1, n), key=gap)` always returns a split point when there are 2 or more boxes, so `rows` always
has two entries. For a **single-row** date, `cut_row_1` / `cut_row_2` cut an arbitrary subset (the boxes on either side of the
largest small vertical jitter). The guard `"Date phải có đủ 2 hàng..."` only triggers when there is a single box.
This is also inconsistent with `split_rows()`, which does have a single-row threshold.

**Fix direction:** require a minimum gap (for example, a fraction of the median box height) before accepting a split.

### 9. Picking a background silently replaces the chosen sample
`load_background` (`label_tool.py:623`), `select_nearest_sample` (`label_tool.py:650-651`), `load_sample` (`label_tool.py:599-601`)

Every background load calls `select_nearest_sample()`, which loads the "closest size" sample whenever it differs from
the current one. `load_sample()` clears `spacing_groups`, `deleted_indices` and undo history. So choosing a background:
- discards the sample the user picked by hand, and
- discards the spacing groups and deleted objects the user set up.

The only notice is a status-bar message, which is easily overwritten (#21).

### 10. Region edits are never autosaved
`save_region` (`label_tool.py:739-745`), `release` (`label_tool.py:1171-1178`), `delete_selected` (`label_tool.py:897-899`), `close` (`label_tool.py:381-383`)

Sample boxes autosave after every edit, but the target region is only written on "Lưu vùng đặt" / `Ctrl+S`.
Drawing, moving, rotating or deleting a region and then switching background or closing the app throws the change away
without asking. Deleting the region also doesn't persist, so it comes back on reload.

### 11. Invalid spinbox text raises `TclError`; the preview breaks and the app can't close
`compose` (`label_tool.py:925`), `capture_effect_state` (`label_tool.py:1209`), `save_config` (`label_tool.py:354`), `close` (`label_tool.py:381-383`)

`IntVar.get()` raises `tk.TclError` when its spinbox holds `""` or non-numeric text. Consequences:
- `stack_size` invalid → every `show_preview()` fails, including in modes that don't use the stack size, because
  `compose()` always reads it.
- `export_count` invalid → `save_config()` only catches `OSError/TypeError/ValueError`, so `close()` raises before
  `root.destroy()` and **the window can't be closed with ×**.

Typing into the stack-size spinbox also doesn't refresh the preview: `command=` only fires on the arrow buttons.

### 12. Saved modes aren't tied to a sample or background
`capture_effect_state` (`label_tool.py:1205-1218`), `config_data` / `load_config` (`label_tool.py:335-377`)

A saved state contains box indices (`spacing_groups`, `deleted_indices`) and **background pixel coordinates**
(`cut_line`). It is persisted to `label_tool_config.json` and later applied to whatever sample and background are
currently loaded. On another pair, the indices point to different characters or are silently dropped, and the cut line
lands somewhere arbitrary.

### 13. Drawing a cut line changes the mode without `on_mode_change()`
`release` (`label_tool.py:1182`)

`self.mode.set("cut")` doesn't run the radio button's command, so:
- the spacing panel stays visible, and the delete panel stays hidden or visible according to the previous mode;
- `last_effect_mode` keeps the previous mode. If you then pick "Xóa object", the deletion is composed on top of the
  **previous** effect instead of the cut.

### 14. Deletions leak into every mode
`compose` (`label_tool.py:930-933`)

`deleted_indices` is filtered out regardless of mode. After deleting objects and switching to "Không lỗi" (normal) or any
other mode, those objects are still missing, and saving that mode stores the deletions too. This contradicts the panel
title "Dạng lỗi — mỗi lần chỉ chọn một" (one error type at a time).

### 15. A region rotated past ±90° flips after save/reload (verified)
`drag` / `edit_rotate` (`label_tool.py:1120-1122`), `Box.__post_init__` (`label_obb.py:29-30`)

`Box.angle` is only normalized to `[-90°, 90°)` in `__post_init__`. Rotating in the editor changes `box.angle`
directly, so in the session the region can be at 170°, which puts the date upside down. After save and reload,
`decode()` builds a new `Box` and the angle becomes -10°:
```
angle in session: 170.0 -> after reload: -10.0
```
The preview and placement (including arc direction and the cut "above"/"below" sides) differ before and after reloading.

**Fix direction:** normalize after each rotate, or keep the full angle range consistently.

---

## Low

### 16. `character_asset` can return a blank crop that is still labeled
`label_tool.py:99-100`

If Otsu can't separate paper from ink (a uniform or very low-contrast crop), the function returns pure white. The box
is still placed and **exported as a label**, but nothing is visible. The function also assumes dark ink on light
paper. With light-on-dark print, the "paper" is the ink, and the result is a dark rectangle with a character-shaped hole.

### 17. Label paths are keyed on the stem only
`sample_label_path` / `region_path` (`label_tool.py:39-44`)

`IMAGE_EXTENSIONS` allows several formats, but the label is `labels/<stem>.txt`. `Image_0060.bmp` and
`Image_0060.png` in the same folder read and overwrite the same label file.

### 18. Automatic parameters can produce non-errors or degenerate samples
`automatic_error_state` (`label_tool.py:1332-1341`)

- **overlap:** with 1 box, or with 1 box per row, `stack_size` is 2 but nothing stacks, so the "overlap" image is
  identical to normal.
- **delete_objects:** with 1 box, `randint(1, max(1, 0))` deletes it. The result is an empty background with an empty label file.
- **arc:** `arc_height` is ±18–90 **pixels** whatever the character size or `dimension_scale`. Small dates go out of
  the region (100 retries, then failure); large dates barely curve.

### 19. `split_rows` splits a slightly slanted single row (verified)
`label_tool.py:134-138`

The single-row test is `overall cy spread < 0.55 * mean box height`. Ten 40 px characters with a 2.6 px/char slope
(about 3°) are split into two "rows", which breaks overlap mode. Rows are also measured on image Y, not on the text
direction.

### 20. Undo history issues
`press` (`label_tool.py:1088-1091`), `snapshot` (`label_tool.py:871-876`)

- Every click on a box in select mode calls `snapshot()` even if nothing moves. `Ctrl+Z` then appears to do nothing,
  and real edits get pushed out of the 100-entry history.
- `spacing_groups` (and mode parameters) aren't in the snapshot, so adding or deleting groups can't be undone, and
  undoing a box deletion doesn't restore matching groups (see #4).
- `reset_effect_editor()` clears `deleted_indices` without a snapshot.

### 21. `render()` overwrites status messages
`render` (`label_tool.py:1053`)

`render()` always rewrites the status bar. Messages set **before** a render are lost immediately:
- `enable_box_edit()` sets the help text and then calls `render()` (`label_tool.py:753-754`).
- The `load_config()` error ("Không đọc được cấu hình cũ…", `label_tool.py:379`) is overwritten by
  `refresh_files() → load_sample() → show_preview()` at startup, so a corrupt config is never reported.
- Any zoom or resize also wipes messages such as "Đã tự chọn mẫu…".

### 22. `add_spacing_group` resets the factor without updating its label
`label_tool.py:815`

`group_x_factor.set(1.0)` doesn't run the Scale command, so `group_factor_label` keeps showing the old value.

### 23. `sample_index` can go stale after `refresh_files()`
`refresh_files` (`label_tool.py:538-549`), `load_sample` (`label_tool.py:590-596`)

`refresh_files()` re-sorts `self.samples` but keeps `self.sample_index`. If the follow-up `load_sample()` fails (an
unreadable image added through "Thêm ảnh mẫu"), or the folder changed outside the tool, `samples[sample_index]` now
names a **different file**. The next autosave then writes the current boxes into that file's label.

### 24. The generated `data.yaml` has no `val:` key
`label_tool.py:1474-1477`

Ultralytics requires both `train` and `val`, so the exported dataset can't be used directly for training without
editing the YAML (the root `data.yaml` has the same gap).

### 25. Cut labels and the "none" cut side
`compose` (`label_tool.py:963-969`)

- A cut removes half of a character's pixels, but its exported OBB still covers the full original character.
- Manual "Cắt ngang" mode with "Không cắt" (the default `cut_side`) changes nothing, yet it can be saved as a "cut" mode.
- Only one manual cut line is supported (`cut_line`); `cut_lines` is only filled by the automatic path, and
  `reset_effect_editor()` doesn't clear it.

### 26. Large blur radii are clipped by the crop padding
`character_asset` padding (`label_tool.py:78`), blur (`label_tool.py:958-960`)

The crop has 12 px of padding around the OBB. A Gaussian blur of radius 7.5 spreads about 20 px, so the halo is
cut off at the crop rectangle and leaves visible straight edges.

---

## Performance (these affect usability)

### 27. Slow paths, all on the UI thread
- `line_side_mask` (`label_tool.py:220-232`) loops over **every canvas pixel in Python**. It took about 0.26 s for
  1600×1200 (measured), and it runs once per crossed box per cut line, on every cut preview and every exported cut
  image (`cut_both_rows` crosses every box).
  Build the mask by drawing a polygon (`ImageDraw.polygon`) for the half-plane instead.
- `compose()` recomputes `character_asset()` for every box on every preview refresh and every drag event
  (`drag` → `change_group_factor` → `show_preview` → `show_preview` again in spacing mode, so it composes **twice** per motion event).
- `next_output_key` (`label_tool.py:1295-1300`) scans from `000000` for every image, which is O(n²) file checks
  over an export of thousands of images.
- `select_nearest_sample` opens every sample image and label file on every background change.
- Export runs synchronously in the Tk callback with no progress feedback, so the window shows "Not responding" for large counts.

---

## Minor / code hygiene

### 28. Miscellaneous
- `MODES` (`label_tool.py:25`) is unused, and the `effect_mode` parameter of `compose()` (`label_tool.py:914`) is never passed.
- The `layout_boxes` docstring says "never scale source characters", but it scales by `dimension_scale` (`label_tool.py:143-155`).
- `dimension_scale` is the min ratio of whole-image sizes, not anything related to the region. This assumes identical
  camera framing between the sample and the background.
- `outer_size % 2` (`label_tool.py:697-698`) is dead code: `outer_size` is always odd already.
- `save_region()` (`label_tool.py:743`) has no `try/except`, so a write error becomes an uncaught callback exception with no message box.
- A malformed saved state from the config (missing `last_effect_mode`, etc.) raises `KeyError` inside
  `compose_saved_mode`, which `save_synthetic_previews` doesn't catch (`OSError, ValueError` only).
- `_sync_view` (`label_tool.py:747-748`) sends **"Chọn/sửa box hoặc vùng"** to the preview view, so the radio
  meant for editing sample boxes hides the sample image. The toolbar "Sửa box/vùng" button doesn't switch views,
  so the two entry points behave differently.
- A plain click in cut-line mode (no drag) creates a zero-length cut line, adds an undo entry, and forces the mode to "cut".
- `random` is never seeded, so exports can't be reproduced.
