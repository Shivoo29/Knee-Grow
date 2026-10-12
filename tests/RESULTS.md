# RESULTS — series-count robustness of 96-slot selection (2026-10-12)

Run: `env -u PYTHONPATH -u PYTHONHOME .venv/bin/python tests/test_series_selection.py`
Metadata only: `data/train_series.csv` pinned columns (`StudyInstanceUID, SeriesInstanceUID,
Fluid_Sensitive, Fat_Suppression, Anatomical_Plane`); `train.csv`/`Report` never opened; no
DICOM, pixels, writes, or network. Selection logic under test: canonical
`rsna_preproc.pick_series_for_slot` driven by the caller loop (SLOTS order, greedy, used-set),
cross-checked against the corpus-builder clone. Scope: 4407 studies / 24371 series
(3–14 series/study, mean 5.53).

## 1. Empty slot groups (selection returns nothing for a slot)

| slot | empty studies | % | filled with wrong fluid contrast |
|---|---|---|---|
| Sag-FS(26) | 0 | 0.00% | 257 |
| Sag-nonFS(22) | 119 | 2.70% | 170 |
| Cor-FS(18) | 0 | 0.00% | 159 |
| Cor-nonFS(12) | 1124 | 25.50% | 104 |
| Ax(18) | 0 | 0.00% | 0 |

- 1241/4407 studies (28.2%) have ≥1 empty group; histogram empty-groups/study = {0: 3166, 1: 1239, 2: 2}; no study has all 5 empty (a study with zero matching series would still flow through inference as an all-zero volume — the fleet path has no per-study guard, see Bug 4).
- FS/Ax slots are never empty because of the fallback to any same-plane series — that same fallback is what starves the non-FS slots: Cor-nonFS is empty in 1124 studies (study owns ≤1 coronal series, consumed by Cor-FS — data reality, correctly masked), Sag-nonFS in 119.
- The single 3-series study shows the greedy steal end-to-end: `study=707857 n=3 | Sag-FS=711191(fs0) Sag-nonFS=EMPTY Cor-FS=406014(fs1) Cor-nonFS=EMPTY Ax=939540(fs1)` — the only sagittal series is non-FS, the Sag-FS slot's fallback consumes it, leaving Sag-nonFS empty: 34+12=46 of 96 slices are zero for this study.

## 2. Order non-determinism (shuffled series list, same study)

- two independent shuffles disagree on the chosen UIDs: **1388/4407 (31.5%)**
- any shuffled run differs from the CSV-order pick: **1750/4407 (39.7%)**

Winner = first CSV row among qualifying candidates (`pref[0]`/`cands[0]`), so which physical
series lands in which slot is decided by row order wherever a study has >1 same-plane series.
The hidden test's `test_series.csv` row order is unknown → slot assignment there is arbitrary
among equally-qualifying series.

## 3. Train/inference skew: corpus clone vs canonical picker (clean data)

**210/4407 studies (4.8%) get different series chosen** — all differences on the two fluid=0
slots (Sag-nonFS 142, Cor-nonFS 70); in every case the corpus clone (training data) picks a
true non-FS series while the canonical picker (inference) picks a fluid-sensitive one.
Root cause is Bug 1 below. Examples: `study=622764 Cor-nonFS preproc=905795(fs1) corpus=627242(fs0)`,
`study=036863 Sag-nonFS preproc=407274(fs1) corpus=930115(fs0)`.

## 4. Dirty-value probes (synthetic rows)

| probe | canonical preproc | corpus clone |
|---|---|---|
| NaN Fluid_Sensitive, slot wants fluid 0/1 | **ValueError crash** | **ValueError crash** |
| NaN Fluid_Sensitive, Ax slot (fluid=-1) | fine (no preference filter) | fine |
| slot wants fluid=0, rows=[FS, nonFS] | picks the **FS** row | picks the nonFS row |
| Fluid_Sensitive = `''` vs real 0 | `''` row via fallback | `''` row (treated as 0) |
| plane = `sagittal` (lowercase) | picks it (capitalize) | **None → silent empty slot** |

Train metadata is clean (0 NaN, planes exact, fluid ∈ {0,1}), but the champion notebook's own
preflight counts `missing_fluid_flags` on test_series.csv — NaNs are anticipated in the hidden test.

## 5. Extreme study sizes

- 3-series studies: exactly 1 in train (dumped above).
- ≥10-series studies: 103 — full dump in the Appendix (chosen UID tail-6 + fluid flag per slot).

## Bugs found (fix proposed, NOT applied)

1. **`or -1` idiom destroys the fluid=0 preference** — `rsna_preproc.py:120`
   (`int(float(r.get('Fluid_Sensitive', -1) or -1))`: a legit `0` is falsy → becomes `-1` → never
   matches fluid=0), same at `sovereign_fast_dual_fleet.py:124`. The corpus builder
   (`build_corpus96_stream.py:114`, `or 0`) and the notebook's `flag()` are correct.
   Impact: at inference the non-FS slots are filled with fluid-sensitive series even when a true
   non-FS series is available (170 Sag / 104 Cor wrong-contrast fills; 210 studies of train skew
   between what models saw and what they see).
   Fix: NaN-safe flag without the `or` idiom, e.g.
   `def _fluid_flag(v):\n    try: f = float(v)\n    except (TypeError, ValueError): return None\n    return None if f != f else int(f)`
   then `pref = [r for r in cands if _fluid_flag(r.get('Fluid_Sensitive')) == fluid]`.
   Apply to both files; changing selection changes inference inputs, so re-score gold before shipping.
2. **NaN Fluid_Sensitive crashes selection** — `rsna_preproc.py:120`, `sovereign_fast_dual_fleet.py:124`
   (`int(float(nan))` → ValueError), also `build_corpus96_stream.py:114` (`or 0` variant crashes too).
   Impact: one NaN in hidden-test metadata kills the whole fleet run (no try/except around
   `build_study_volumes`, `sovereign_fast_dual_fleet.py:368`) or silently drops the study from the corpus.
   Fix: the same `_fluid_flag` helper (NaN → None → plane-level fallback, matching the notebook's `flag()`).
3. **Order-dependent tie-break** — `rsna_preproc.py:122-123`, `sovereign_fast_dual_fleet.py:126-127`,
   `build_corpus96_stream.py:116-117`; callers feed CSV row order (`build_corpus96_stream.py:280-281`,
   `sovereign_fast_dual_fleet.py:305`). Impact: 39.7% of studies pick differently under reordering.
   Fix: content-based deterministic tie-break before taking `[0]` — sort candidates by
   `SeriesInstanceUID` (no extra metadata needed) or `n_slices` desc (as the notebook's `pick_slots`
   does) — applied identically to all three sites.
4. **Missing test_series.csv → silent all-zero inference** — `sovereign_fast_dual_fleet.py:299-300`
   fabricates an empty metadata frame; every study then zero-fills and predicts ~0.5 (the NaN guard at
   `:385-386` is the only net). This is the past silent-0.500 bug class. The notebook hard-fails instead
   (its root probe requires `test_series*`) and logs `empty_study_rows` events.
   Fix: raise `FileNotFoundError` when no test_series.csv is found; additionally log/flag any study whose
   `series_map` is empty or whose `mask96.sum() == 0`.
5. **Corpus clone matches plane exactly** — `build_corpus96_stream.py:112` (no `strip().capitalize()`,
   vs `rsna_preproc.py:113-116`); a lowercase plane string silently empties the slot in the training path.
   Fix: have `build_corpus96_stream.py` import `pick_series_for_slot` from `rsna_preproc` instead of
   keeping a diverging copy (its own docstring claims SSOT parity).

Caveat (metadata-only): a chosen series whose DICOMs are missing/unreadable also zero-fills silently
(`build_corpus96_stream.py:164-166`, `sovereign_fast_dual_fleet.py:146-148`); not countable here —
the training-time slots.csv audit has no inference-time counterpart.

## Appendix — ≥10-series studies (103), CSV-order pick, UIDs tail-6

Legend: `Sag-FS/Sag-nonFS/Cor-FS/Cor-nonFS/Ax = <uid tail-6>(fs<flag>)`, `EMPTY` = no series found.
study=622764 n=10 | Sag-FS=097711(fs1) Sag-nonFS=778633(fs0) Cor-FS=111543(fs1) Cor-nonFS=905795(fs1) Ax=650204(fs0)
study=323922 n=10 | Sag-FS=572598(fs1) Sag-nonFS=956182(fs0) Cor-FS=548426(fs1) Cor-nonFS=538256(fs0) Ax=234850(fs0)
study=084725 n=10 | Sag-FS=499257(fs1) Sag-nonFS=836493(fs1) Cor-FS=330221(fs1) Cor-nonFS=809821(fs0) Ax=520295(fs1)
study=467179 n=10 | Sag-FS=575130(fs1) Sag-nonFS=843243(fs0) Cor-FS=023100(fs1) Cor-nonFS=538141(fs1) Ax=594122(fs0)
study=646590 n=10 | Sag-FS=066510(fs1) Sag-nonFS=783516(fs0) Cor-FS=903569(fs1) Cor-nonFS=912109(fs1) Ax=611379(fs1)
study=277080 n=10 | Sag-FS=036454(fs1) Sag-nonFS=596541(fs0) Cor-FS=942701(fs1) Cor-nonFS=991193(fs1) Ax=595509(fs0)
study=084306 n=10 | Sag-FS=490901(fs1) Sag-nonFS=769648(fs0) Cor-FS=294458(fs1) Cor-nonFS=632432(fs0) Ax=732717(fs0)
study=564425 n=10 | Sag-FS=344735(fs1) Sag-nonFS=168879(fs0) Cor-FS=298133(fs1) Cor-nonFS=752642(fs0) Ax=651947(fs1)
study=109258 n=10 | Sag-FS=464366(fs1) Sag-nonFS=136998(fs0) Cor-FS=151813(fs1) Cor-nonFS=423062(fs0) Ax=920455(fs1)
study=283241 n=10 | Sag-FS=997587(fs1) Sag-nonFS=445458(fs0) Cor-FS=029516(fs1) Cor-nonFS=719772(fs0) Ax=832511(fs0)
study=188597 n=10 | Sag-FS=599360(fs1) Sag-nonFS=533288(fs0) Cor-FS=301617(fs1) Cor-nonFS=598580(fs0) Ax=448498(fs0)
study=612197 n=10 | Sag-FS=647315(fs1) Sag-nonFS=689557(fs0) Cor-FS=414329(fs1) Cor-nonFS=309674(fs0) Ax=195981(fs0)
study=922967 n=10 | Sag-FS=477187(fs1) Sag-nonFS=342732(fs0) Cor-FS=300286(fs1) Cor-nonFS=866878(fs0) Ax=051774(fs0)
study=737855 n=10 | Sag-FS=478171(fs1) Sag-nonFS=983756(fs0) Cor-FS=260625(fs1) Cor-nonFS=227897(fs0) Ax=203520(fs1)
study=037991 n=10 | Sag-FS=896185(fs1) Sag-nonFS=728064(fs0) Cor-FS=878506(fs1) Cor-nonFS=896161(fs0) Ax=061274(fs1)
study=756615 n=10 | Sag-FS=798324(fs1) Sag-nonFS=369071(fs0) Cor-FS=361964(fs1) Cor-nonFS=203235(fs0) Ax=711045(fs0)
study=664817 n=10 | Sag-FS=276161(fs1) Sag-nonFS=541107(fs0) Cor-FS=862850(fs1) Cor-nonFS=526788(fs0) Ax=081707(fs1)
study=384269 n=10 | Sag-FS=706113(fs1) Sag-nonFS=658122(fs0) Cor-FS=112235(fs1) Cor-nonFS=640638(fs0) Ax=023979(fs0)
study=829266 n=10 | Sag-FS=272441(fs1) Sag-nonFS=401555(fs0) Cor-FS=413117(fs1) Cor-nonFS=106372(fs0) Ax=781380(fs1)
study=399656 n=10 | Sag-FS=365292(fs1) Sag-nonFS=714670(fs0) Cor-FS=310252(fs1) Cor-nonFS=565674(fs0) Ax=106304(fs0)
study=455084 n=10 | Sag-FS=721992(fs1) Sag-nonFS=336061(fs0) Cor-FS=505002(fs1) Cor-nonFS=214963(fs0) Ax=468143(fs1)
study=807158 n=10 | Sag-FS=039148(fs1) Sag-nonFS=896452(fs0) Cor-FS=775886(fs1) Cor-nonFS=366544(fs0) Ax=384058(fs1)
study=808226 n=10 | Sag-FS=067639(fs1) Sag-nonFS=789895(fs0) Cor-FS=451818(fs1) Cor-nonFS=571370(fs0) Ax=454234(fs0)
study=667314 n=10 | Sag-FS=458604(fs1) Sag-nonFS=628441(fs0) Cor-FS=639618(fs1) Cor-nonFS=947309(fs0) Ax=773661(fs0)
study=363680 n=10 | Sag-FS=368133(fs1) Sag-nonFS=619591(fs0) Cor-FS=447206(fs1) Cor-nonFS=527656(fs0) Ax=308576(fs1)
study=241297 n=10 | Sag-FS=523204(fs1) Sag-nonFS=407064(fs0) Cor-FS=083302(fs1) Cor-nonFS=117499(fs0) Ax=763817(fs1)
study=263817 n=10 | Sag-FS=965959(fs1) Sag-nonFS=597548(fs0) Cor-FS=049548(fs1) Cor-nonFS=517755(fs0) Ax=604452(fs1)
study=773871 n=10 | Sag-FS=975634(fs1) Sag-nonFS=980366(fs0) Cor-FS=922926(fs1) Cor-nonFS=972150(fs0) Ax=263467(fs1)
study=520487 n=10 | Sag-FS=184811(fs1) Sag-nonFS=101573(fs0) Cor-FS=019399(fs1) Cor-nonFS=019770(fs0) Ax=014352(fs0)
study=658070 n=10 | Sag-FS=417800(fs1) Sag-nonFS=661057(fs0) Cor-FS=010218(fs1) Cor-nonFS=184461(fs0) Ax=647744(fs1)
study=769100 n=10 | Sag-FS=280253(fs1) Sag-nonFS=606201(fs0) Cor-FS=144307(fs1) Cor-nonFS=577072(fs0) Ax=897948(fs1)
study=670071 n=10 | Sag-FS=433667(fs1) Sag-nonFS=501037(fs0) Cor-FS=618340(fs1) Cor-nonFS=865301(fs0) Ax=661072(fs0)
study=589893 n=10 | Sag-FS=636136(fs1) Sag-nonFS=075217(fs0) Cor-FS=636075(fs1) Cor-nonFS=861436(fs0) Ax=347242(fs1)
study=604853 n=10 | Sag-FS=904588(fs1) Sag-nonFS=767289(fs0) Cor-FS=918173(fs1) Cor-nonFS=641941(fs0) Ax=966038(fs1)
study=287611 n=10 | Sag-FS=473889(fs1) Sag-nonFS=927715(fs0) Cor-FS=142120(fs1) Cor-nonFS=338762(fs0) Ax=000631(fs0)
study=242917 n=10 | Sag-FS=179868(fs1) Sag-nonFS=801045(fs0) Cor-FS=246331(fs1) Cor-nonFS=822793(fs1) Ax=026910(fs1)
study=597540 n=10 | Sag-FS=892950(fs1) Sag-nonFS=659097(fs0) Cor-FS=622499(fs1) Cor-nonFS=082073(fs0) Ax=150428(fs1)
study=605778 n=10 | Sag-FS=412380(fs1) Sag-nonFS=939144(fs0) Cor-FS=546392(fs1) Cor-nonFS=243506(fs0) Ax=171303(fs0)
study=286535 n=10 | Sag-FS=351467(fs1) Sag-nonFS=270252(fs0) Cor-FS=100889(fs1) Cor-nonFS=179758(fs0) Ax=300688(fs1)
study=201782 n=10 | Sag-FS=703308(fs1) Sag-nonFS=816541(fs0) Cor-FS=036945(fs1) Cor-nonFS=901843(fs0) Ax=998425(fs0)
study=844346 n=10 | Sag-FS=498475(fs1) Sag-nonFS=060318(fs0) Cor-FS=389768(fs1) Cor-nonFS=940410(fs0) Ax=940422(fs1)
study=163428 n=10 | Sag-FS=401117(fs1) Sag-nonFS=377482(fs0) Cor-FS=713053(fs1) Cor-nonFS=146858(fs0) Ax=423445(fs0)
study=545893 n=10 | Sag-FS=732424(fs1) Sag-nonFS=314871(fs0) Cor-FS=053937(fs1) Cor-nonFS=393219(fs0) Ax=739196(fs1)
study=674643 n=10 | Sag-FS=825661(fs1) Sag-nonFS=753711(fs0) Cor-FS=469501(fs1) Cor-nonFS=161993(fs1) Ax=796309(fs0)
study=537207 n=10 | Sag-FS=868261(fs1) Sag-nonFS=002494(fs0) Cor-FS=299580(fs1) Cor-nonFS=694759(fs0) Ax=282651(fs1)
study=735815 n=10 | Sag-FS=932939(fs1) Sag-nonFS=427712(fs0) Cor-FS=787872(fs1) Cor-nonFS=227374(fs0) Ax=494568(fs1)
study=540792 n=10 | Sag-FS=988171(fs1) Sag-nonFS=398934(fs0) Cor-FS=478303(fs1) Cor-nonFS=544745(fs1) Ax=979768(fs0)
study=865485 n=10 | Sag-FS=648582(fs1) Sag-nonFS=233473(fs0) Cor-FS=603908(fs1) Cor-nonFS=359508(fs0) Ax=425214(fs0)
study=460530 n=10 | Sag-FS=139823(fs1) Sag-nonFS=157349(fs0) Cor-FS=974975(fs1) Cor-nonFS=255185(fs0) Ax=184611(fs1)
study=145948 n=10 | Sag-FS=550439(fs1) Sag-nonFS=483118(fs1) Cor-FS=574736(fs1) Cor-nonFS=843152(fs0) Ax=448721(fs1)
study=658177 n=10 | Sag-FS=488210(fs1) Sag-nonFS=222705(fs0) Cor-FS=102725(fs1) Cor-nonFS=812227(fs0) Ax=813750(fs0)
study=490190 n=10 | Sag-FS=053155(fs1) Sag-nonFS=522530(fs0) Cor-FS=152710(fs1) Cor-nonFS=978509(fs0) Ax=446173(fs1)
study=589563 n=10 | Sag-FS=330406(fs1) Sag-nonFS=635868(fs0) Cor-FS=364319(fs1) Cor-nonFS=130170(fs1) Ax=453245(fs0)
study=386752 n=10 | Sag-FS=190595(fs1) Sag-nonFS=199630(fs0) Cor-FS=447603(fs1) Cor-nonFS=116231(fs0) Ax=190906(fs1)
study=447010 n=10 | Sag-FS=045206(fs1) Sag-nonFS=293225(fs0) Cor-FS=096965(fs1) Cor-nonFS=824604(fs0) Ax=876722(fs1)
study=338455 n=10 | Sag-FS=772839(fs1) Sag-nonFS=259641(fs0) Cor-FS=985689(fs1) Cor-nonFS=664467(fs0) Ax=144575(fs0)
study=117756 n=10 | Sag-FS=551467(fs1) Sag-nonFS=574330(fs0) Cor-FS=988887(fs1) Cor-nonFS=317189(fs0) Ax=429007(fs0)
study=552840 n=10 | Sag-FS=759011(fs1) Sag-nonFS=155888(fs0) Cor-FS=750913(fs1) Cor-nonFS=455767(fs0) Ax=064285(fs0)
study=929056 n=10 | Sag-FS=450625(fs1) Sag-nonFS=235186(fs0) Cor-FS=549487(fs1) Cor-nonFS=760067(fs0) Ax=115386(fs0)
study=805522 n=10 | Sag-FS=518570(fs1) Sag-nonFS=814620(fs0) Cor-FS=920648(fs1) Cor-nonFS=409891(fs0) Ax=941607(fs1)
study=971577 n=10 | Sag-FS=843494(fs1) Sag-nonFS=812508(fs0) Cor-FS=401179(fs1) Cor-nonFS=130120(fs1) Ax=646169(fs1)
study=660873 n=10 | Sag-FS=966719(fs1) Sag-nonFS=676931(fs0) Cor-FS=673113(fs1) Cor-nonFS=174559(fs0) Ax=504736(fs1)
study=711653 n=10 | Sag-FS=247361(fs1) Sag-nonFS=613602(fs0) Cor-FS=897548(fs1) Cor-nonFS=917109(fs0) Ax=679635(fs1)
study=122220 n=10 | Sag-FS=475497(fs1) Sag-nonFS=741859(fs0) Cor-FS=979603(fs1) Cor-nonFS=193715(fs0) Ax=192415(fs1)
study=390972 n=10 | Sag-FS=247919(fs1) Sag-nonFS=499946(fs0) Cor-FS=379637(fs1) Cor-nonFS=389101(fs0) Ax=654288(fs0)
study=128908 n=10 | Sag-FS=158975(fs1) Sag-nonFS=413388(fs0) Cor-FS=813807(fs1) Cor-nonFS=423256(fs0) Ax=693971(fs0)
study=547497 n=10 | Sag-FS=685108(fs1) Sag-nonFS=223285(fs0) Cor-FS=115772(fs1) Cor-nonFS=371574(fs1) Ax=198135(fs1)
study=910585 n=10 | Sag-FS=874925(fs1) Sag-nonFS=963074(fs0) Cor-FS=899557(fs1) Cor-nonFS=387022(fs0) Ax=293057(fs0)
study=540506 n=10 | Sag-FS=218369(fs1) Sag-nonFS=325451(fs0) Cor-FS=842694(fs1) Cor-nonFS=491783(fs0) Ax=875222(fs0)
study=497308 n=10 | Sag-FS=356709(fs1) Sag-nonFS=835064(fs0) Cor-FS=428479(fs1) Cor-nonFS=928630(fs0) Ax=153712(fs1)
study=119916 n=10 | Sag-FS=679042(fs1) Sag-nonFS=267844(fs0) Cor-FS=292045(fs1) Cor-nonFS=331335(fs0) Ax=833365(fs0)
study=975364 n=10 | Sag-FS=080333(fs1) Sag-nonFS=729467(fs0) Cor-FS=685670(fs1) Cor-nonFS=937962(fs0) Ax=955208(fs1)
study=579391 n=10 | Sag-FS=008009(fs1) Sag-nonFS=856959(fs0) Cor-FS=382504(fs1) Cor-nonFS=523832(fs1) Ax=502365(fs0)
study=982381 n=10 | Sag-FS=135238(fs1) Sag-nonFS=142676(fs0) Cor-FS=899786(fs1) Cor-nonFS=474919(fs1) Ax=752350(fs1)
study=738314 n=11 | Sag-FS=165954(fs1) Sag-nonFS=293994(fs0) Cor-FS=878208(fs1) Cor-nonFS=857053(fs0) Ax=774430(fs1)
study=542129 n=11 | Sag-FS=914801(fs1) Sag-nonFS=080090(fs0) Cor-FS=702439(fs1) Cor-nonFS=425766(fs0) Ax=595776(fs1)
study=153419 n=11 | Sag-FS=031964(fs1) Sag-nonFS=278844(fs0) Cor-FS=299661(fs1) Cor-nonFS=720765(fs0) Ax=830781(fs0)
study=965719 n=11 | Sag-FS=532981(fs1) Sag-nonFS=158272(fs0) Cor-FS=888381(fs1) Cor-nonFS=429496(fs0) Ax=853341(fs0)
study=672982 n=11 | Sag-FS=254774(fs1) Sag-nonFS=974045(fs0) Cor-FS=181923(fs1) Cor-nonFS=264736(fs0) Ax=844855(fs0)
study=566620 n=11 | Sag-FS=352627(fs1) Sag-nonFS=519258(fs0) Cor-FS=828924(fs1) Cor-nonFS=119104(fs0) Ax=716268(fs1)
study=001922 n=11 | Sag-FS=362021(fs1) Sag-nonFS=423043(fs0) Cor-FS=600532(fs1) Cor-nonFS=504483(fs0) Ax=040433(fs1)
study=044465 n=11 | Sag-FS=471661(fs1) Sag-nonFS=117458(fs0) Cor-FS=666247(fs1) Cor-nonFS=095656(fs0) Ax=275862(fs0)
study=701872 n=11 | Sag-FS=495830(fs1) Sag-nonFS=945336(fs0) Cor-FS=047313(fs1) Cor-nonFS=633447(fs0) Ax=668325(fs1)
study=293545 n=11 | Sag-FS=544745(fs1) Sag-nonFS=661284(fs0) Cor-FS=221237(fs1) Cor-nonFS=907560(fs0) Ax=534026(fs1)
study=459619 n=11 | Sag-FS=291119(fs1) Sag-nonFS=779337(fs0) Cor-FS=199670(fs1) Cor-nonFS=922752(fs1) Ax=556059(fs0)
study=526112 n=11 | Sag-FS=987691(fs1) Sag-nonFS=965808(fs0) Cor-FS=388395(fs1) Cor-nonFS=764640(fs0) Ax=911117(fs0)
study=304265 n=11 | Sag-FS=869484(fs1) Sag-nonFS=159775(fs0) Cor-FS=489309(fs1) Cor-nonFS=091574(fs0) Ax=369583(fs0)
study=205950 n=11 | Sag-FS=096401(fs1) Sag-nonFS=743425(fs0) Cor-FS=692884(fs1) Cor-nonFS=397598(fs0) Ax=455615(fs1)
study=672103 n=11 | Sag-FS=763734(fs1) Sag-nonFS=942854(fs0) Cor-FS=400747(fs1) Cor-nonFS=151177(fs1) Ax=557658(fs1)
study=762771 n=11 | Sag-FS=287944(fs1) Sag-nonFS=415559(fs0) Cor-FS=636682(fs1) Cor-nonFS=601466(fs0) Ax=543596(fs0)
study=220525 n=11 | Sag-FS=966142(fs1) Sag-nonFS=729621(fs0) Cor-FS=165032(fs1) Cor-nonFS=659899(fs0) Ax=876370(fs0)
study=660942 n=11 | Sag-FS=054254(fs1) Sag-nonFS=288949(fs0) Cor-FS=639289(fs1) Cor-nonFS=001101(fs0) Ax=101535(fs0)
study=656162 n=11 | Sag-FS=011355(fs1) Sag-nonFS=579526(fs1) Cor-FS=647490(fs1) Cor-nonFS=938005(fs0) Ax=501577(fs1)
study=936291 n=11 | Sag-FS=456475(fs1) Sag-nonFS=431803(fs0) Cor-FS=595345(fs1) Cor-nonFS=946545(fs0) Ax=190337(fs1)
study=550187 n=11 | Sag-FS=679712(fs1) Sag-nonFS=451848(fs0) Cor-FS=657646(fs1) Cor-nonFS=156891(fs0) Ax=932495(fs1)
study=935632 n=12 | Sag-FS=899424(fs1) Sag-nonFS=837754(fs0) Cor-FS=729561(fs1) Cor-nonFS=233554(fs0) Ax=589246(fs0)
study=313299 n=12 | Sag-FS=464851(fs1) Sag-nonFS=429126(fs0) Cor-FS=010900(fs1) Cor-nonFS=909868(fs0) Ax=120952(fs0)
study=806038 n=12 | Sag-FS=818672(fs1) Sag-nonFS=258477(fs0) Cor-FS=488118(fs1) Cor-nonFS=336686(fs0) Ax=297556(fs1)
study=457090 n=12 | Sag-FS=629863(fs1) Sag-nonFS=905407(fs0) Cor-FS=062113(fs1) Cor-nonFS=415420(fs0) Ax=707006(fs0)
study=643386 n=12 | Sag-FS=690907(fs1) Sag-nonFS=384329(fs0) Cor-FS=975177(fs1) Cor-nonFS=342714(fs0) Ax=443000(fs0)
study=527983 n=13 | Sag-FS=930503(fs1) Sag-nonFS=226043(fs0) Cor-FS=670290(fs1) Cor-nonFS=710773(fs0) Ax=225736(fs0)
study=959510 n=13 | Sag-FS=328874(fs1) Sag-nonFS=578260(fs0) Cor-FS=675923(fs1) Cor-nonFS=393218(fs0) Ax=137614(fs0)
study=686941 n=14 | Sag-FS=059174(fs1) Sag-nonFS=109782(fs0) Cor-FS=571753(fs1) Cor-nonFS=798811(fs0) Ax=808390(fs1)
