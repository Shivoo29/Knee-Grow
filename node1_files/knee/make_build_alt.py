"""Derive build_alt.py from build_axial2.py: same 5-slot layout as corpus192, each slot refilled with the next-best
UNUSED series of the same plane + fluid class (axial: prefer fluid-sensitive); slots without one keep corpus192's."""
src = open("build_axial2.py").read()
def rep(a, b):
    global src
    assert src.count(a) == 1, a[:60]
    src = src.replace(a, b)
rep('SLOTS = [("Axial", -1, 40)]   # axial2 add-on: one more axial series per study, never one corpus192 already used',
    'SLOTS = [("Sagittal", 1, 40), ("Sagittal", 0, 40), ("Coronal", 1, 36), ("Coronal", 0, 36), ("Axial", -1, 40)]  # alt: corpus192 layout')
rep('''    cands = [r for r in series_rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used_series]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None''',
'''    # alt: strict same contrast (never swap FS <-> plain inside a slot); skip scouts (< 12 images)
    cands = [r for r in series_rows if r['Anatomical_Plane'] == plane and str(r['SeriesInstanceUID']) not in used_series
             and len(_FILE_MAP.get(str(r['SeriesInstanceUID']), ())) >= 12]
    if fluid in (0, 1):
        cands = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
    else:
        cands.sort(key=lambda r: -int(r.get('Fluid_Sensitive', 0) or 0))
    return cands[0] if cands else None''')
rep('''    parser.add_argument("--limit"''', '''    parser.add_argument("--base_dir", default="ds/corpus192", help="corpus whose slots fill the ones without an alternate")
    parser.add_argument("--limit"''')
rep('''    # Flush memmapped vols to disk
    vols_mmap.flush()''', '''    # alt: slots without an alternate series keep corpus192's series, so the input there is identical
    base_ids = {str(s): i for i, s in enumerate(np.load(os.path.join(args.base_dir, "all_ids.npy"), allow_pickle=True))}
    base_v = np.load(os.path.join(args.base_dir, "all_vols.npy"), mmap_mode="r")
    base_m = np.load(os.path.join(args.base_dir, "all_masks.npy"))
    off = np.cumsum([0] + [s[2] for s in SLOTS])
    pos = {sid: i for i, sid in enumerate(study_ids)}
    for rec in all_slots_records:
        if not rec["filled"]:
            i, j, s = pos[rec["StudyInstanceUID"]], base_ids[rec["StudyInstanceUID"]], rec["slot_idx"]
            vols_mmap[i, off[s]:off[s + 1]] = base_v[j, off[s]:off[s + 1]]
            masks_arr[i, off[s]:off[s + 1]] = base_m[j, off[s]:off[s + 1]]
    alt = {r["StudyInstanceUID"] for r in all_slots_records if r["filled"]}
    print(f"[alt] {len(alt)}/{len(study_ids)} studies got at least one alternate series", flush=True)

    # Flush memmapped vols to disk
    vols_mmap.flush()''')
open("build_alt.py", "w").write(src)
print("wrote build_alt.py")
