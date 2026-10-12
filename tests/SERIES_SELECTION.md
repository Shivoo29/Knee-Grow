# Series→slot selection, 96-slice pipeline

Canonical rule: `pick_series_for_slot` (rsna_preproc.py:111), driven by the caller loop at
build_corpus96_stream.py:131 / sovereign_fast_dual_fleet.py:138.
1. Slots fill greedily in fixed order Sag-FS(26) → Sag-nonFS(22) → Cor-FS(18) → Cor-nonFS(12) → Ax(18); each chosen series is removed from the pool (`used_series`).
2. Slot pool = not-yet-used series whose `Anatomical_Plane` matches the slot plane (case-normalized in rsna_preproc; the corpus clone at build_corpus96_stream.py:110-112 matches exactly, no normalization).
3. If the slot wants fluid 0/1, the pool subset with matching `Fluid_Sensitive` is preferred; the winner is the FIRST such row (`pref[0]`/`cands[0]`) — ties break purely by input row order (CSV order in every caller), never by slice count.
4. Plane missing → pool empty → pick returns `None` → caller skips the slot's k slices, leaving them all-zero (mask bit 0); this is silent (no log/warning) in the preproc and fleet paths.
5. Fluid-sensitive series missing → silent fallback to ANY series of that plane: the Sag-FS slot can be filled by a T1/PD series, consuming it so the Sag-nonFS slot then goes empty (order-dependent "steal"). The Ax slot wants fluid=-1, so any Axial series qualifies.
6. The same greedy rule is cloned at sovereign_fast_dual_fleet.py:120 (exact copy) and build_corpus96_stream.py:110 (diverges on dirty values); the champion notebook adds three more dialects: _KE pick (same greedy but NaN-safe `flag()`), A5 pick (exact `Fat_Suppression` match, no fallback → slot skipped), and recovered/public `pick_slots` (max `n_slices` tie-break).
