# Issue: Rotation reduces how far the date can shift

| | |
|---|---|
| **Status** | Known limitation, won't fix (accepted as by design on 2026-11-06) |
| **Area** | Batch generation: date placement |
| **Settings** | Batch dialog "Góc xoay" (`angle_jitter_deg`) and "Lệch vị trí" (`shift_percent`) |
| **Code** | `Generator.jittered_region`, `Generator.shifted_region` in `synth_engine.py` |

This issue is independent. It is not caused by, and does not depend on, any other known defect.

## What you see

With both rotation and position shift turned on, the two settings are not independent. The more the date is
rotated, the less it moves, mainly across the region (vertically for a horizontal region). Across a batch:

- dates that are rotated more stay closer to the middle of the region in height;
- "Lệch vị trí" moves the date less than its value suggests whenever the rotation is large;
- once a rotated date no longer fits the region's height, that axis has no room left at all. The date stays
  centred across the region, and images that would stick out of the region are rejected and drawn again.
  In practice, the larger angles of the configured range never appear in the output.

## Measured on the current data

Code `Image_0000_20261105021332` on background `Image_0000_20261105021332_background_000000` (region 1916 × 226 px,
date 556 × 62 px unrotated), "normal" layout, no bend:

| Date rotation | Date size in region axes | Free room along (x) | Free room across (y) |
|---:|---:|---:|---:|
| 0° | 556 × 62 px | 1360 px | 164 px |
| 5° | 560 × 91 px | 1357 px | 135 px |
| 10° | 559 × 139 px | 1358 px | 88 px |
| 15° | 553 × 185 px | 1363 px | 41 px |
| 30° | 513 × 315 px | 1404 px | **−89 px** (does not fit) |

With the current setting of ±15°, a date at the edge of the range has **75 % less room across the region** than
an unrotated one (41 px vs 164 px). Above about 17° the rotated date is taller than the 226 px region and can't
be placed at all. Background `…_000001` gives the same numbers to within 1 px.

## Cause

Placement happens in this order for every image:

1. `jittered_region` draws a random rotation within ± "Góc xoay".
2. `shifted_region` measures the free room around the date **as it will be drawn**: rotation, bend and the
   error mode's layout included, in the target region's axes. It then moves the date by up to
   "Lệch vị trí" of that room on each side.
3. `generate` keeps the image only if every character lies inside the region, and otherwise tries again
   (up to 100 attempts).

A rotated date's bounding box is taller (and, for long dates, slightly shorter), so the room in step 2 shrinks as the
angle grows. The shift is a percentage of that smaller room, so the actual offset shrinks with it.

## Why it is not changed

No change fixes this without causing a different problem:

| Alternative | Problem it causes |
|---|---|
| Measure the room on the unrotated date | Rotated dates are shifted past the region edge and then rejected, so the retries rise and more batches fail with "không tạo được ảnh mới nằm trọn trong vùng đặt" |
| Shift first, then rotate around the new centre | Same as above: a date shifted near the edge sticks out once rotated |
| Shrink the date so it always has full room | The character size then depends on the angle, so size stops being controlled by "Cỡ ký tự" |
| Allow the date to leave the region | Breaks the rule that the date stays inside the placement region, which the out-line and cut modes rely on |
| Pick angle and offset together from all valid placements | Large angles then need small offsets and the reverse, so the coupling stays, only in a different form |

## Workarounds

- **More shift across the region:** draw a taller placement region, or lower "Góc xoay".
- **Larger angles:** keep "Góc xoay" below the angle at which the date still fits the region's height
  (about 17° for the data above). Anything larger only adds rejected attempts.
- **Even coverage of positions:** generate separate batches with different "Góc xoay" values into the same output
  folder. File numbers continue, so batches don't overwrite each other.
