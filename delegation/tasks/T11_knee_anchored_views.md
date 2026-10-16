# T11: Knee-anchored close-up views (corpus builder + training + Kaggle inference)

Read `delegation/PREAMBLE.md` first and obey it. Exceptions under "Where you work". Time-box: 6 hours.

## Why
Our corpus cuts one 140 mm square from the IMAGE centre (`build_corpus96_stream.py` `mm_crop_and_resize`, 384 px,
~0.36 mm/px). The kneecap often sits at its edge and each joint half gets few pixels. Our weakest findings on gold-58
are Lateral OA (0.76-0.83) and PF OA (~0.87). heliosli's public R6 model (the ConvNeXt in the best public blend)
first finds the knee, then takes 100 mm close-ups in millimetres around it (summary in
`heliosli_vs_our_approach.md` in the project root, read it). We want the same idea for OUR CoAtNet, taken from the
ORIGINAL DICOM pixels (a crop of our 384 px corpus would only be interpolation, no new detail).
Write your own code. Do not copy heliosli's code; the idea and the mm offsets below are enough.

## Views (new slots appended after the 5 normal slots of the 192 layout)
Base layout stays `40,40,36,36,40` (Sag fluid, Sag non-fluid, Cor fluid, Cor non-fluid, Axial), 140 mm centre crop.
Extra close-up slots, each 100 mm square, 384 px (~0.26 mm/px), same slice picks as their source slot:
| new slot | source series | close-up centre (mm, patient space) |
|---|---|---|
| 5 sag_patella | Sag fluid (slot 0) | joint centre + 40 mm ANTERIOR + 25 mm SUPERIOR |
| 6 sag_posterior | Sag fluid (slot 0) | joint centre + 35 mm POSTERIOR + 5 mm INFERIOR |
| 7 cor_side_a | Cor fluid (slot 2) | joint centre + 30 mm toward patient LEFT (+x, LPS) |
| 8 cor_side_b | Cor fluid (slot 2) | joint centre + 30 mm toward patient RIGHT (-x) |
| 9 ax_patella | Axial (slot 4) | joint centre + 40 mm ANTERIOR |
| 10 ax_posterior | Axial (slot 4) | joint centre + 35 mm POSTERIOR |
Slot sizes of the new slots = their source slot size (40, 40, 36, 36, 40, 40): total 192 + 232 = 424 slices/study.

## Joint centre (no external model)
`knee_anchor(volume_slices, pixel_spacing)`: on the middle third of the series' slices, threshold tissue
(e.g. > 20th-percentile of non-zero pixels), take the largest connected component, return its centroid in pixels;
convert to mm with PixelSpacing. If it fails (empty), use the image centre. Directions (anterior, superior, left)
must come from `ImageOrientationPatient` (row/column direction cosines, LPS: +x left, +y posterior, +z superior),
NOT from assumed image orientation. Crops that leave the image are zero-padded.

## Build
1. `build_corpus96_stream.py`: flag `--views anchored_v1` adds the 6 close-up slots using the table above; reads the
   needed header tags (IOP, PixelSpacing) it already parses; writes `views.json` next to the corpus with the slot
   list (name, source slot, size, offsets). Default (no flag): byte-identical output to today.
2. `train_knee.py`: support more than 5 slots. `--slot_sizes` already exists (T10, asserts 5 entries): allow any
   length; `mirror_volume` must treat the new slots by their PLANE (sagittal slots reversed, coronal/axial flipped)
   and, under mirror, SWAP the contents of slot 7 and slot 8 (side a <-> side b). Save `"views": "anchored_v1"`
   in checkpoints.
3. `kaggle_base954_fleet/fleet_v6.py`: when `ck.get("views") == "anchored_v1"`, build the same 424-slice volume
   from the test DICOMs with the SAME functions (copy them into fleet_v6; it must stay one self-contained file) and a
   test that the copies are source-identical. Old checkpoints: unchanged behaviour.

## Tests (`tests/test_anchored_views.py`, plain functions + `__main__` runner, no pytest; synthetic data only)
- `knee_anchor` finds the centroid of a synthetic bright disc within 2 px; falls back to the centre on an empty image.
- direction handling: for a synthetic sagittal IOP and a coronal IOP, "anterior" / "left" offsets move the crop centre
  the right way in pixel space (hand-computed expectation).
- crop: a 100 mm crop at 0.3 mm/px is 333 px before resize; off-image parts are zero.
- default builder path (no `--views`) produces the same arrays as before on a synthetic series.
- `mirror_volume` with 11 slots swaps slot 7 and 8 contents and mirroring twice is identity.
- train/fleet copies identical; old fleet path unchanged (reference `git show HEAD:kaggle_base954_fleet/fleet_v6.py`).
CPU only. Do NOT run training, the full corpus build, or GPU jobs (training runs on the PC and node1).

## Where you work
`git worktree add ../knee-views -b views` from the project root; edit only there. Do not commit.
Results: `delegation/results/T11_knee_anchored_views.md` in the MAIN folder; first 15 lines = SUMMARY.
