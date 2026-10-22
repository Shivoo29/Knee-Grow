# RSNA Knee Abnormality Detection: what changed 16–20 Oct 2026, and what is moving scores above 0.958–0.960

How this was collected (20 Oct 2026, about 04:55–05:45 UTC). Everything was read-only. Nothing was submitted, posted, voted, forked or uploaded, and no downloaded code was run.

- **Leaderboard.** `kaggle competitions leaderboard --download`, file `publicleaderboard-2026-10-20T04:58:17.csv`, 6,250 teams. It is compared with the 16 Oct snapshot (15:59 UTC, 6,095 teams) from `Knee MRI top leaderboard tactics/leaderboard/`.
- **Efficiency board.** Output of Kaggle staff's efficiency notebook, run on 19 Oct at 21:14 UTC.
- **Notebooks.** `kaggle kernels list` (by run date and by score). Ten notebooks run on 16–20 Oct were pulled with `kaggle kernels pull -m` and read as text.
- **Datasets.** `kaggle datasets list --search … --sort-by updated`. I read `metadata`, `files` and a few small text files: READMEs, logs, `knee.py`, `judge_image.py`.
- **Forum.** `kaggle competitions topics list/show` and the same read-only Kaggle API call the CLI uses (`get_topic`) to get opening posts. That covered 96 recently active topics, of which 29 had posts dated 16 Oct or later, plus notebook comment threads. GitHub repositories pushed since 16 Oct were found through the unauthenticated GitHub search API. Web searches were run too.
- **Not reachable.** Kaggle's web `ListKernels` endpoint, used on 16 Oct for exact notebook scores, now returns a reCAPTCHA page, so exact notebook scores were **not** available. The CLI shows submission scores to 3 decimals only.
- **Local copies.** All under `research_notes/Knee MRI last 48 hour gains/kaggle_oct20/`:
  - `leaderboard/`, `efficiency_lb/`, `listings/`
  - `kernels/` (pulled notebooks)
  - `datasets_meta/`
  - `forum/` (`recent_comments_since_oct16.txt`, `opening_posts_oct16_20.txt`, `notebook_comments_since_oct16.txt`, topic JSONs)
  - `github_readmes/`
- Everything dated before 16 Oct is marked **(background)**.

---

## Q1. Current public leaderboard: ranks 1/5/10/20/50, the size of the 0.958/0.959/0.960 blocks, and whether the 0.958 plateau broke

### Takeaway
**On the leaderboard, yes.**
- The top moved to 0.969: #1 is RB & Matheus & Tucker & Tony.
- 128 teams are now at ≥ 0.960 (77 on 16 Oct) and 194 at ≥ 0.959 (119).

**As a ceiling for public forks, no.** The 0.958 block tripled to 1,338 teams (411 on 16 Oct), and **both the silver line (rank 312) and the bronze line (rank 625) now lie inside the 0.958 block**. Teams that broke out did so without publishing anything. Several jumped straight from 0.941–0.954 to 0.959–0.963.

### Cited Findings

**Snapshot.** 2026-10-20 04:58 UTC, 6,250 teams (6,095 on 16 Oct) — [public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard) (CLI download; local `kaggle_oct20/leaderboard/…2026-10-20T04:58:17.csv`).

**Rank anchors** (team, score, score on 16 Oct) — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

| Rank | Team | Score now | 16 Oct |
|---|---|---|---|
| 1 | RB & Matheus & Tucker & Tony (rashmibanthia, tomirol, tonylica, tuckerarrants; 339 subs) | **0.969** | 0.967 (#2) |
| 2 | azamatt (azamat1ch, matthewliew; 94 subs) | 0.968 | 0.967 (#3) |
| 3 | CentralyDS | 0.968 | 0.968 (#1) |
| 4 | Sergey Bryansky | 0.968 | 0.967 |
| 5 | Zhengru Li (5 members) | **0.967** | 0.964 (#16) |
| 6 | joinT fusion | 0.967 | 0.966 |
| 7 | Hyunwoong Kim (solo, 102 subs) | 0.967 | 0.961 (#50) |
| 8 | Team Kangaroo | 0.967 | 0.964 |
| 9 | BarryZhou (solo, 84 subs) | 0.967 | 0.961 (#39) |
| 10 | MEJAL | **0.967** | 0.966 (#5) |
| 20 | TomKroumov | **0.964** | 0.964 |
| 50 | dalab | **0.962** | 0.962 |
| 100 | Awshesh Nath | 0.960 | — |
| 150 | yuken | 0.959 | — |
| 200 | Ezra | 0.958 | — |

**Counts at or above each threshold, now vs 16 Oct** — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

| Threshold | Now | 16 Oct |
|---|---|---|
| ≥ 0.969 | 1 | 0 |
| ≥ 0.968 | 4 | 1 |
| ≥ 0.967 | 10 | 4 |
| ≥ 0.966 | 15 | 7 |
| ≥ 0.965 | 19 | 9 |
| ≥ 0.964 | 31 | 18 |
| ≥ 0.963 | 49 | 25 |
| ≥ 0.962 | 65 | 38 |
| ≥ 0.961 | 86 | 57 |
| ≥ 0.960 | 128 | 77 |
| ≥ 0.959 | 194 | 119 |
| ≥ 0.958 | 1,532 | 530 |
| ≥ 0.957 | 1,709 | 852 |
| ≥ 0.956 | 2,001 | 1,488 |

**Teams at each exact score (tie blocks)** — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

| Exact score | Teams now | Ranks now | Teams 16 Oct | Ranks 16 Oct |
|---|---|---|---|---|
| 0.960 | 42 | 87–128 | 20 | 58–77 |
| 0.959 | 66 | 129–194 | 42 | 78–119 |
| 0.958 | **1,338** | 195–1,532 | 411 | 120–530 |
| 0.957 | 177 | — | 322 | — |
| 0.956 | 292 | — | 636 | — |

**Medal lines** under Kaggle's ≥ 1,000-team rule (top 10 + 0.2 %, 5 %, 10 %) — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

| Medal | Rank line now | Score at line now | Score at line 16 Oct |
|---|---|---|---|
| Gold | 22 | **0.964** | 0.963 |
| Silver | 312 | **0.958** | 0.958 |
| Bronze | 625 | **0.958** | 0.957 |

**Where the 0.958 block came from** (teams present on both days; selected rows, smaller sources omitted) — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

| Score on 16 Oct | Teams now at 0.958 |
|---|---|
| 0.958 (stayed) | 379 |
| 0.957 | 228 |
| 0.956 | 339 |
| 0.954 | 69 |
| 0.950 | 33 |
| 0.943 | 45 |
| not on the board | 63 |

The block's last submissions fall on 16 Oct (99), 17 Oct (123), 18 Oct (227), 19 Oct (801) and 20 Oct (84).

**Where the 0.959 and 0.960 blocks came from** — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

| Score on 16 Oct | Now at 0.959 (66 teams) | Now at 0.960 (42 teams) |
|---|---|---|
| Same score (stayed) | 22 | 8 |
| 0.959 | — | 11 |
| 0.958 | 23 | 7 |
| 0.957 | 11 | 4 |
| 0.956 | 4 | 5 |
| 0.955 | — | 2 |
| 0.954 | 2 | 2 |
| 0.941–0.953 | 4 | — |
| 0.947 or 0.951 | — | 2 |
| New team | — | 1 |

**Largest movers 16 → 20 Oct** (score, submissions on 16 Oct → now) — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

| Team | Score | Submissions |
|---|---|---|
| Hyunwoong Kim | 0.961 → 0.967 | 84 → 102 |
| BarryZhou | 0.961 → 0.967 | 66 → 84 |
| YC & NB | 0.959 → 0.964 | — |
| Advanced CV (bloodaxe, kharintymofii, vladimirsydor) | 0.959 → 0.964 | — |
| hirokenn | 0.959 → 0.964 | 33 → 43 |
| Parag & Ian & Dieter & CroDoc (christofhenkel, crodoc, paragkale, vaillant) | 0.959 → 0.964 | — |
| IteraLab | 0.960 → 0.964 | — |
| **Leslie Yeh** | **0.954 → 0.963** | 15 → 31 |
| AICON_Shibata | 0.957 → 0.963 | — |
| Guilherme Detoni | 0.946 → 0.961 | — |
| Yoshiki Nakamura | 0.947 → 0.961 | — |
| Sugawarya | 0.947 → 0.960 | — |
| Ivan Vybornov | 0.941 → 0.959 | — |
| T.S | 0.943 → 0.959 | 7 → 15 |
| Khanh Nguyen | 0.944 → 0.959 | 2 → 10 |

**Where the public-component authors sit now** (16 Oct score in brackets) — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
- Every author of a public checkpoint or blend found is in the 0.958 block:
  - Heliosli: 0.958, #205 (0.958)
  - goodpjw2008: 0.958, #518
  - Danial Zakaria (nartaa): 0.958 (0.957)
  - Bend the Knee (mattiaangeli): 0.958 (0.956)
  - Pilkwang Kim: 0.958
  - Dread Development: 0.958
  - hengck23: 0.958 (0.956)
  - Chris Deotte: 0.958
  - varshitha kolupuri: 0.958
  - AlekseiProvorov: 0.958
  - Дворкин Евгений (Versia 7): 0.958
  - "My Knee Hurts!!" (lavinwins): 0.958
  - Laimer Botrolenk: 0.958
  - Igor V (rostigor): 0.958
- **The one exception is BliverRigPåDoge (bliverrigpdoge): 0.959, #177 (0.958).** That author blends privately trained knee-centred CNNs into the public stack.

**Efficiency board** (staff notebook run 19 Oct 21:14 UTC; one evaluated submission per team, not necessarily the team's best) — [Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb); local `kaggle_oct20/efficiency_lb/full_leaderboard.csv`
- Top five:

  | Efficiency rank | Team | Public score | Submitted |
  |---|---|---|---|
  | 1 | Pa3฿aJluHа | 0.961 | — |
  | 2 | **RB & Matheus & Tucker & Tony** | **0.965** | 15 Oct |
  | 3 | Optimo | 0.960 | 18 Oct |
  | 4 | tomoon33 | 0.961 | — |
  | 5 | Donato Riccio | 0.958 | — |

- Entries of other top teams:

  | Team | Efficiency rank | Public score | Submitted |
  |---|---|---|---|
  | Leslie Yeh | 68 | **0.962** | 18 Oct |
  | Claudio Verdú Ruiz | 111 | 0.964 | — |
  | azamatt | 162 | 0.966 | — |
  | Parag & Ian & Dieter & CroDoc | 185 | 0.962 | — |
  | Team Kangaroo | 283 | 0.967 | — |
  | CentralyDS | 3,674 | 0.968 | — |
  | MEJAL | 4,354 | 0.966 | — |

- 2,444 of 6,250 teams have a submission dated 16 Oct or later — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard).

**Notebook scores are capped by the author's team score.** A notebook's public score comes from its author's own submission, so it cannot exceed the author's team score. The Kaggle CLI and API show only 3 decimals; our own submissions list confirms it — [Kaggle CLI `competitions submissions`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions).

**Queue and quota constraints in the final days**
- On 17 Oct, submissions waited 4–6 hours in the queue (Andrey Chankin; Optimo "about 6 hours"; Yuriy Samorodov "4 hours") — [750318](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750318)
- Kaggle's weekly GPU quota resets on 24 Oct, after the deadline — [750790](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750790)
- Practical tips (SwapIY, 19 Oct) — [750318](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750318); [744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230):
  - At most 2 GPU batch sessions run per account.
  - A CLI push without `--timeout` reserves 12 hours and is refused when less quota is left.
  - Set `"machine_shape": "NvidiaTeslaT4"` to avoid landing on a P100.

### Inferences
- **The 0.958 tie block is the public-fork ceiling.** It holds 1,338 teams, every author of a public component is in it, and teams moved into it from 0.943–0.957 (some with a handful of submissions). Moving to 0.959+ has, in every case visible, come from components that are not public.
- **Within-block rank decides public medals.** Silver (rank 312) and bronze (rank 625) are both inside the 0.958 block, so on the public board a 0.958 team's medal depends on its unrounded score, which is invisible. The private board will reorder 1,338 near-identical submissions. Two consequences:
  - A fork of a public blend is a coin flip for a medal.
  - Any genuinely different, decent model that lifts the score to 0.959+ moves a team past roughly 1,340 others.
- **Gold now needs 0.964 or better.** In four days #1 moved +0.001 (0.968 → 0.969) and the top-10 line +0.003 (0.964 → 0.967). Hyunwoong Kim and BarryZhou gained +0.006 each, so own-model teams can still jump late.
- **Fast own models already score 0.965.** The #1 team's efficiency entry is a fast submission at 0.965, and Leslie Yeh's fast entry is at 0.962. The top teams' gains do not depend on 5-hour public blends. This fits the 16 Oct background that top teams train their own 2.5D CNNs on better labels. It is an inference, since no method was disclosed.

### Gaps
- No leaderboard snapshot for 18 Oct was available. The brief's "#1 ≈ 0.969, #10 = 0.966 on 18 Oct" could not be cross-checked. #10 is 0.967 now.
- The efficiency board does not include runtimes in its CSV, and it reports one submission per team, so it cannot show each team's best fast score.
- Unrounded scores within tie blocks are not exposed, so the size of the margin around the silver and bronze lines inside the 0.958 block is unknown.
- None of the big movers (Leslie Yeh, Hyunwoong Kim, BarryZhou, hirokenn, AICON_Shibata, Guilherme Detoni, Yoshiki Nakamura) has public knee notebooks or datasets (Kaggle CLI user listings). BarryZhou (songqizhou) has a "Biohub 4th place solution" from another competition. Their methods are unknown.

---

## Q2. New public notebooks or datasets (16–20 Oct) above 0.958, and what they change vs the standard public blend

### Takeaway
**No public notebook or dataset released or updated on 16–20 Oct is verifiably above 0.958.** Every author of one is at 0.958 or lower on the leaderboard, except bliverrigpdoge (0.959, private models).

What the new releases change:
1. **heliosli (17 Oct)** renamed and updated the "0.9576" notebook to "0.9586". It keeps the Blend mine in recipe, pushes the per-finding Raptor weights further, and adds goodpjw2008's reader at 1/10 weight. The 4-decimal title is a claim, not a displayed score.
2. **goodpjw2008 (17 Oct)** published three **measured negatives**:
   - S6 next to R6: 0.957.
   - Per-finding weights tuned on gold-58: 0.957 and 0.955.
3. **New checkpoints and labels** appeared: the R6 fine-tunes ("B-leg") on gold-calibrated severity targets, a Mycelis 4-fold ConvNeXt, OmerZalman's partial-coverage label sets, and bliverrigpdoge's training kit. Their authors' own teams are still at 0.958 (bliverrigpdoge: 0.959).
4. **nartaa (17 Oct)** released the full training code behind the CoAtNet 0.949/0.945 and S6-all-data weights.

### Cited Findings

**Public notebooks run or updated 16–20 Oct.** About two dozen in total ([CLI dateRun listing](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code); local `kaggle_oct20/listings/`). The final-week publishing lock means the high-profile ones are renamed updates of existing notebooks: heliosli's "0.9586" notebook has the same kernel id (138670458) as the 16 Oct "0.9576" notebook, whose best score was 0.957 on 16 Oct. goodpjw2008's "Does S6 help" is the renamed "CoAtNet + S6 + 2.5D MIL 0.954" notebook (id 138139982). kiyoshiohno's 0.957 "RSNA Knee Template" no longer appears in the listing.

Kaggle's score-descending order is now:
1. [RSNA_Versia_7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7)
2. [goodpjw2008 0.958](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)
3. [lavinwins RSNA Gold Medal](https://www.kaggle.com/code/lavinwins/rsna-gold-medal)
4. [heliosli "0.9586"](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)
5. [lavinwins Top 1% Gold](https://www.kaggle.com/code/lavinwins/top-1-gold)
6. [Blend mine in (0.958)](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters)
7. Then the 0.957 notebooks (prvsiyan, goodpjw2008 "Does S6 help", bliverrigpdoge, …).

All five authors above Blend mine in are at 0.958 on the leaderboard ([leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)).

**1. heliosli, "This is best public dataset new version =0.9586" (version of 17 Oct 04:21 UTC)** — [notebook](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586); local `kaggle_oct20/kernels/heliosli__this-is-best-public-dataset-new-version-0-9586/`
- **Stage 1:** nartaa's 384-px CoAtNet (mirror TTA).
- **Stage 1b:** nartaa's 224-px 0.8-crop CoAtNet, one view.
- **Stage 2:** R6 (bundle `fe055406fad6a12f`, 4 folds, own fluid and state stackers) with `FLIP_TTA = True`.
- **Stage 3:** percentile-rank blend, Raptor half = 0.65·rank(384) + 0.35·rank(224).
- **Change 1 vs Blend mine in (0.958): new per-finding Raptor weights** (R6 gets 1 − w). All other findings stay at 0.5.

  | Finding | `W_RAPTOR_C` "[V2]" | Blend mine in |
  |---|---|---|
  | Lateral OA | **0.15** | 0.20 |
  | PF OA | **0.65** | 0.60 |
  | Synovitis | **0.85** | 0.80 |
  | Baker's | **0.30** | 0.35 |
  | Medial OA | 0.5 | 0.5 |

- **Change 2: "Part B (V2)" adds goodpjw2008's 2.5D ConvNeXt MIL reader** at its per-finding weights of 0.003–0.035 (1/10 of v6). If the reader fails or runs late, the stage-3 submission is kept.
- The R6 files are read from `rsna-knee-r6-frozen-fe055406` with an assert that the bundle id is `fe055406…`. Per edteoh (16 Oct), the latest version of that dataset holds R7, not R6, so a fork must mount version 1 — [749858](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749858)
- The single-class probe switch `PROBE_CLASS` is still present, set to `None`.
- heliosli's team best is 0.958, rank 205, 11th inside the 0.958 block — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard). The "0.9586" title cannot be read from Kaggle, which shows 3 decimals.

**2. goodpjw2008, "Does S6 help next to R6? | LB 0.957" (v8, 17 Oct)** — [notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)
- Public LB results, all variants of the 0.958 recipe:

  | Variant | Public LB |
  |---|---|
  | v4/v5: the 0.958 recipe + heliosli S6 as a second compact model at 0.3 | **0.957** |
  | v6: heliosli's weights with Fracture 0.8 and Effusion 0.7 toward Raptor, ACL 0.3 toward R6, moved where gold-58 showed 0.03–0.08 gaps | **0.957** |
  | v7: every finding at its gold-58 optimum, clipped to 0.1–0.9 | **0.955** |
  | heliosli's weights unchanged (reference) | 0.958 |

- Conclusion quoted: "do not add S6 to R6, and do not tune per-finding weights on the 58 gold studies".
- The full per-finding gold-58 table is under Q4.

**3. evgendvorkin, RSNA_Versia_7 (v23, 19 Oct)** — [notebook](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7)
- It is a documented reproduction of goodpjw2008's 0.958 notebook (Raptor 384 + 224 mirror, R6 flip TTA, the reader at small weights). It attaches R7 as well.
- It states: "The score of this reproduction will be recorded after Kaggle evaluates its submission."
- The author's team is at 0.958.

**4. lavinwins, "RSNA Gold Medal" and "Top 1% Gold" (17 Oct)** — [Gold Medal](https://www.kaggle.com/code/lavinwins/rsna-gold-medal); [Top 1% Gold](https://www.kaggle.com/code/lavinwins/top-1-gold)
- Both are AI-assistant-written.
- "Gold Medal" v6 is a "combination layer on top of the 0.943 public pipeline" with "no new checkpoints": probit-space rank mixing, a flat 3-way blend, a fix for a subpopulation-rank bug. It is not a 0.958 recipe in its current form.
- "Top 1% Gold" says: "Rearranging the public models' predictions is exhausted — `v6a` moved the ordering and the score didn't budge… What's left is adding a model the public ensemble doesn't contain." It proposes a 16-channel, single sagittal-series 2D model.
- The team is at 0.958.

**5. seanmoch, "Tri Reader Submission" (19 Oct)** — [notebook](https://www.kaggle.com/code/seanmoch/rsna-knee-tri-reader-submission)
- A reproduction of a 0.954 tri-reader. The team is at 0.950.

**New or updated datasets, 16–20 Oct** ([Kaggle datasets search, sorted by update](https://www.kaggle.com/datasets?search=rsna+knee); local `kaggle_oct20/datasets_meta/`)

**"B-leg" R6 fine-tunes (18 Oct)**
- Datasets:
  - [kritikabenjwal/rsna-knee-bleg-s0-f01](https://www.kaggle.com/datasets/kritikabenjwal/rsna-knee-bleg-s0-f01) (518 downloads)
  - [sarthakgupta1801/rsna-knee-bleg-s0-f23](https://www.kaggle.com/datasets/sarthakgupta1801/rsna-knee-bleg-s0-f23) (346)
  - [xhaitanya/rsna-knee-bleg-s1-f01](https://www.kaggle.com/datasets/xhaitanya/rsna-knee-bleg-s1-f01) (292)
- **Recipe:**
  - ConvNeXt-small models warm-started from the public R6 fold models (V1, bundle `fe055406`).
  - Fine-tuned for 2 epochs on "gold-calibrated soft targets" with heliosli's `compact_train.py` v9.8 (fp16 patch).
  - Arm `E_prior`, LR × 0.5, batch 4 × 32 windows, crop 352.
  - Pixel cache of 3,383 studies, **including all 58 gold**, with hash folds (sha1 of the study ID mod 4).
- **Single-fold gold-58 macro AUC 0.911–0.927.** Weak findings: Lateral OA about 0.85, Synovitis 0.82–0.87, PF OA 0.86–0.89, Effusion 0.86–0.90.
- **Intended use:** "soup the fold models per seed… score them as a second model group next to the R6 recipe (`--models2`)".
- **The three uploaders form one team, "Ghutnoon mein dimaag": 0.958 (#1,279), 22 submissions, up from 0.957** — [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

**[kritikabenjwal/rsna-knee-labels-soft-v2](https://www.kaggle.com/datasets/kritikabenjwal/rsna-knee-labels-soft-v2) (17 Oct): the targets behind the B-leg models**
- Qwen2.5-7B-Instruct, run offline, labels each report into one ordinal state per finding. For example, Effusion: not_mentioned / none / trace_or_small / moderate / large.
- The states are calibrated on gold-58:
  - Each (finding, state) positive rate is shrunk toward a tier prior with 4 pseudo-observations.
  - Rates are made monotonic along the severity ladder.
  - `not_mentioned` is calibrated separately.
  - Hedged mentions are pulled 30 % toward the base rate.
- Gold macro AUC: 0.831 for the states vs 0.764 for binary report labels. These are in-sample figures.

**[neronrx/mycelis-rsna-knee-4fold-hive](https://www.kaggle.com/datasets/neronrx/mycelis-rsna-knee-4fold-hive) (18 Oct; CC0; 1,191 downloads; no description)**
- Contents: 4 fold checkpoints ("cnxt_s384_v22_hive"), goodpjw2008's reader code (`knee.py`, 6 slots, study transformer) with ConvNeXt-small at 384 px, initialised from R6's `E_lrprior` fold weights (342 of 381 layers), labels `labels_pseudo_v1`, 14 epochs.
- **Its own logs show gold-58 macro AUC falling from 0.670 (epoch 0) to about 0.58**, while the pseudo-label validation AUC rises to 0.92. Fold-0 per-class gold values include Medial Meniscus 0.395. Local `datasets_meta/neronrx__mycelis-rsna-knee-4fold-hive/log_fold0.txt`
- The author's team "NODUS - M.Y.C.E.L.I.S" is at 0.958 (0.956 on 16 Oct).

**OmerZalman's partial-coverage label sets (17 Oct)**
- Datasets:
  - [omerzalman/rsna-knee-11-label-evidence-fusion](https://www.kaggle.com/datasets/omerzalman/rsna-knee-11-label-evidence-fusion): "97.13 % accuracy at 60 % coverage", 705 downloads
  - [omerzalman/rsna-knee-98-accuracy-11-soft-labels](https://www.kaggle.com/datasets/omerzalman/rsna-knee-98-accuracy-11-soft-labels): 98.04 % at 55.96 %
  - [omerzalman/rsna-knee-99-accuracy-11-soft-labels](https://www.kaggle.com/datasets/omerzalman/rsna-knee-99-accuracy-11-soft-labels): 99.14 % at 36.36 %
- 11 findings, **no Synovitis**. Each provides soft labels, confidence, an acceptance mask and training weights. CC BY-NC 4.0.
- Accuracy is measured on gold-58, described as "a familiar development panel used during dataset development".
- The author claims "A few people tested it and it increased AUC by around 0.02-0.1 auc" (unverified) — [750334](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750334); [750791](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750791)
- The author's team went from 0.728 to 0.958 in 5 submissions.

**[bliverrigpdoge/rsna-knee-trainkit](https://www.kaggle.com/datasets/bliverrigpdoge/rsna-knee-trainkit) (19 Oct; 309 downloads)**
- "Training code, label tables and a pack index for private training runs."
- Label tables: `D1a_teacher.csv`, `consensus_labels.csv`, `labels_KCS_s50.csv`, `labels_N5.csv`, `labels_R1s50.csv`, `labels_S5.csv`, `labels_soft.csv`, `silver3.csv`, `targets_DM.csv`, `language.csv`.
- Backbones: timm ConvNeXt-tiny/small (in12k), EfficientNetV2-S, and `convnext_small_s6fall_enc.safetensors` converted from nartaa's S6-all-data encoder (CC BY-NC 4.0).
- `judge_image.py` is a "LOCKED judge": "model SELECTION runs on silver validation only; gold_check() is the rationed reality check".
- The author's team is at 0.959 (#177), up from 0.958. This is the only public-component author above 0.958.
- Background: the 16 Oct note credits this author's private knee-centred compartment ConvNeXt-small/EfficientNetV2-S models with +0.001 twice at 0.2 weight — [R7 + Raptor + knee-centred ConvNeXt](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext).

**[nartaa/rsna-knee-s6fall-weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights) (updated 17 Oct): `training_code/` added, Apache-2.0**
- Per the README and RUNS.md, the training code behind the CoAtNet 0.949/0.945 weights is:
  - `train_knee.py` + `ext_trainer.py`.
  - 2,399 OAI knees appended to training only. **OAI cells supervise only PF OA, Lateral OA and Synovitis**, using MOAKS cuts.
  - **"SWA: average of the top-3 epochs ranked by Gold-58 AUC"**.
  - Exact argv: coatnet_rmlp_2_rw_384, k 12, k_eval 24, bs 1 × accum 16, 16 epochs, bb_lr 4.01e-05, head_lr 0.00195, wd 0.0588, labels `labels_v1_blendjev`.
- S6-all-data: epoch 16, no SWA or EMA, nothing selected on Gold-58.
- `labels_v1_blendjev` = cell-wise mean of `labels_v1_blend` and riadmohamed42's JEV HYBRID table.
- Sources: [discussion 749359 comment, 17 Oct](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359); [dataset discussion 750180](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/discussion/750180)

**Other datasets, none with evidence above 0.958:**
- [ayodejiibrahimlateef/rsna-knee-r6-frozen-v1-public-20261018](https://www.kaggle.com/datasets/ayodejiibrahimlateef/rsna-knee-r6-frozen-v1-public-20261018): another file-for-file copy of R6 v1.
- cyruspinto "Engine 8–11" ConvNeXt, Swin and multi-plane weights (team at 0.958).
- srinath2495 "Dual-Teacher Consensus" labels (0.958).
- abdullahmathar "Argon4 SOTA Soft Labels" (0.958).
- riadmohamed42 DINO 5-fold (team 0.949 → 0.958).
- dkim54 blend code (0.954).
- jorgerome weights v2 (0.510).
- haungjiawei train v2 (0.891).
- Sources: [datasets search](https://www.kaggle.com/datasets?search=rsna+knee); [leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

**Public weight provenance disclosed in the forum (17–18 Oct)**
- [pilkwang/rsna-knee-weights](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights): one DINOv2-small slice-window model, 4 seeds × 5 folds, trained in August on competition data only — [750355](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750355)
- mattiaangeli's knee weights: competition data only, public pretrained backbones (RadImageNet for the RadImageNet arm; `convnext_small.dinov3_lvd1689m` for M448 under the DINOv3 licence), "around 60 variants" — [750356](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750356)

**GitHub, 16–20 Oct.** About 37 knee repositories pushed. None is from a top-50 team. Useful negatives:
- trevoreliot: Grad-CAM pseudo-boxes → YOLO "Tests whether the pseudo-box YOLO helps classification (as a score and as a cropper for raptor). It does not." — [repo](https://github.com/trevoreliot/kaggle-rsna-knee-abnormality-detection)
- betitanx: local diagnostics of 0.967–0.973 on already-exposed gold studies vs public 0.931–0.934 — [repo](https://github.com/betitanx/Kaggle-RSNA-Knee-Abnormality-Detection)
- ppx16: "Public leaderboard noise is ~0.008 while the effects we chase are 0.001–0.003". This is the repo's own estimate. Background estimates put the standard deviation near 0.003 — [repo](https://github.com/ppx16/rsna-knee-record)
- Search: [GitHub search API](https://api.github.com/search/repositories?q=rsna+knee+pushed:>=2026-10-16)

### Inferences
- **Re-weighting the public checkpoints is exhausted**, on four pieces of evidence:
  - heliosli's further weight moves and the reader add-on keep the author at 0.958.
  - goodpjw2008's three variants lost 0.001–0.003.
  - lavinwins found that probit-space and other re-blends "didn't budge".
  - S6 next to R6 loses 0.001.

  What still moves a 0.958 public blend is a **decorrelated, privately trained member**. bliverrigpdoge is the only public-component author at 0.959, and does it with private knee-centred models.
- **The B-leg models are unlikely to add diversity.** They are R6 warm-starts on new targets, so they are highly correlated with R6 by construction. Their own team stayed at 0.958. Mycelis is R6-initialised and looks broken on gold.
- **Low-cost options worth testing inside the block:**
  - heliosli's V2 weights plus the reader at 1/10.
  - goodpjw2008's 224 mirror view.

  These are free and may move a team up within the 0.958 block, which matters for the silver and bronze lines. They should not be expected to display 0.959.
- **External-data flags for our rules:**
  - R7 carries fastMRI and is banned by our team's rule. It also sits inside Versia 7 and appears in other public blends.
  - R6's fastMRI status is still unconfirmed (see Q3).
  - B-leg and Mycelis inherit R6's status.
  - All of R6, R7, S6 and B-leg rely on `anatomy.pt`, the SKM-TEA/KneeXNet localiser.
  - nartaa's CoAtNets are OAI-trained.
  - goodpjw2008's reader, which heliosli's V2 adds at 1/10, trains only on public report-label tables per its published code (see Q3).

### Gaps
- Exact notebook scores were not retrievable: the web ListKernels endpoint returns a reCAPTCHA page, and the CLI shows 3 decimals. The "0.9586" title and the order of the score sort cannot be verified at 4 decimals.
- Whether lavinwins' "Gold Medal" earned its ≥ 0.958 sort position with an earlier version is unknown; its current code is a 0.943-stack re-blend.
- No public result exists for B-leg or Mycelis models blended into the 0.958 recipe. Only their authors' team scores of 0.958 are known.
- No independent test of OmerZalman's label sets exists. The accuracy figures are in-sample on gold-58.

---

## Q3. New forum posts from top-20 teams or the host (16–20 Oct): what helped, and clarifications on external data, DICOM metadata and the test set

### Takeaway
- **No top-20 team member posted a technical hint on 16–20 Oct.** The only near-top technical comment is colum2131 (now #28, 0.964), on 16 Oct 04:28: image pseudo-labels blended with LLM labels at a fixed ratio.
- **The host and Kaggle gave no new ruling.** On 19 Oct Kaggle's María Cruz said they cannot interpret third-party data terms or pre-approve datasets or weights, and that each team is responsible.
- About ten new topics asked for yes/no rulings on OAI-trained weights, the SKM-TEA/KneeXNet localiser, R7 (fastMRI), RadImageNet and LLM-label datasets. All are unanswered and most are heavily downvoted.
- There was no new statement on test-set labelling, test composition or use of DICOM metadata.

### Cited Findings

**Host and Kaggle staff**
- **María Cruz, 19 Oct 22:46 UTC:**
  - "we cannot provide legal interpretations of third-party data-use agreements or licenses, or pre-approve specific datasets or model weights under the competition rules… the host is not designating individual external datasets as permitted or off limits… each team is responsible for ensuring that their submission and any external data or models they use comply with both the Competition Rules and the third-party provider's terms of use."
  - She is listed with Kaggle's Ryan Holbrook among the competition's citation authors.
  - Sources: [747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455); [overview citation as mirrored in a participant README](https://github.com/switzloco/creaky)

**Unanswered ruling requests, 16–20 Oct**
- Karan Bhatti, 17 Oct (votes −24 to −28):
  - nartaa's OAI-trained weights ("in 90 % public notebooks at 0.95+") — [750257](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750257)
  - heliosli's anatomy model, described on its page as "unchanged SKM-TEA/KneeXNet auxiliary weights", with "no new open-source license"; KneeXNet is "built from OAI images" — [750258](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750258)
  - R7 trained with fastMRI — [750259](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750259)
  - RadImageNet weights, whose images are shared only on request — [750260](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750260)
- Michał Choiński, 17 Oct: the rules describe compliance checks only for prize winners (2.8, 3.9.b) and "nothing for medal positions" — [750236](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750236)
- Other requests:
  - Abdul Wahab, 19 Oct: public LLM-label datasets and hosted LLM APIs under Rule 4.b — [750752](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750752)
  - Marvin O, 20 Oct: fastMRI and SKM-TEA weights — [750889](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750889)
  - manduinca, 20 Oct: OAI-fine-tuned public checkpoints — [750899](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750899)
  - Richard_61, 16 Oct: OAI geo-block, NOT-OD-25-083 — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
  - RealEclypso, 18 Oct — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- **nishchal jain, 18 Oct, asked heliosli directly: "Was R6 (bundle fe055406, run tag cnx_r6fm) trained with fastMRI or fastMRI+ data in any way, including an auxiliary task?"** No reply was seen by 20 Oct 05:00 UTC. The same post discloses a team using nartaa CoAtNet + heliosli R6 + `anatomy.pt` for inference only, four CC0 LLM label tables, and out-of-fold predictions of their own models as targets — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)
- edteoh, 16 Oct, summarising external data in the public blends:
  - nartaa CoAtNets = OAI.
  - R7 = fastMRI + fastMRI+ boxes (`fm_aux`).
  - R6 = no `fm_aux` in the manifest.
  - `anatomy.pt` = SKM-TEA.
  - goodpjw2008's "read the report then the knee" notebook uses frozen fastMRI+ ViTDet heads, "so add fastMRI+ if you use that one".
  - Source: [749858](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749858)
  - This refers to a different goodpjw2008 notebook, not to the 2.5D ConvNeXt reader used at 1/10 weight in the 0.958 blends. Checked on 20 Oct: the reader dataset's published `train.py` and `make_labels.py` contain no fastMRI, OAI or SKM-TEA references. Its labels come from four public tables (stevenleehans `llm_labels_v4_blend`, pilkwang `report_labels_v2`, lixin73 `labels_llm_gpt56sol`, laymond Qwen3-8B weak labels) — [goodpjw2008/rsna-knee-2-5d-convnext-reader](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader); local `datasets_meta/goodpjw2008__rsna-knee-2-5d-convnext-reader/`. Its inference code (`preprocess.py`) was not checked for an anatomy model.
- hengck23, 16 Oct: "time to move on. Don't get yourself trapped into external data details… they cannot make any changes now" — [743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416)

**Technical comments since 16 Oct (none from top-20 teams)**
- colum2131 (team colum2131, 0.962 → **0.964**, #28; 16 Oct 04:28): "The pseudo-labels are generated from the images, with the image model taking the same input as at submission time. I tried separate heads to predict the LLM labels and the pseudo-labels, but currently I just blend the two at a fixed ratio and use the blended labels as training targets." Follow-up questions on how labels were improved and how checkpoints were selected went unanswered — [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)
- Amanie El-Hajj (team at 0.960), 17 Oct, on the public blend: "the notebook doesn't seem to be this overfitted to gold". Dread Development, 17 Oct, on the R7/heliosli line: "their AI remake of my work with this additional data is overfit anyways… the private board should shake it out". Both are opinions — [748181](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748181); [750259](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750259)
- heliosli, 16 Oct: "I've done almost no manual annotation—the 0.958 score came entirely from my work on the modeling side… I genuinely haven't written a single line of code myself; it's all been done through AI agents" — [748295](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748295)
- M. Ehtasham, 18–19 Oct: claims no external data is needed, but posted as "my work" a description copied from public Raptor inference code. It describes a single CoAtNet at 0.924 public using `dreaddevelopment/raptor-knee-widedense` weights, 64 slices in 5 slots, a 140 mm crop and 336 px. The same text appears as the docstring of the 224-stage script in the seanmoch tri-reader notebook. Not a new method — [750529](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750529); [seanmoch notebook](https://www.kaggle.com/code/seanmoch/rsna-knee-tri-reader-submission)
- Optimo (#48, 0.963): only the queue complaint — [750318](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750318)
- Notebook comment threads on the 0.957–0.958 notebooks and the efficiency notebook had one comment since 16 Oct, a runtime question on goodpjw2008's 0.958 notebook — [goodpjw2008 0.958 comments](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958/comments)

**Test set, labelling, DICOM metadata: background only**
- **(background, 7–8 Aug)** Gold and test labels come from image review with the challenge rubric: two independent readers and a third adjudicating. "The same process was used for the testing set." Where image and report disagree, the image label is authoritative — [733491](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733491); [733826](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)
- **(background)** The overview thanks the Society of Skeletal Radiology and the International Skeletal Society "for recruiting its members to join the annotation team that labeled the dataset" — [overview as mirrored](https://github.com/switzloco/creaky); [competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
- **(background)** The data page says about 1,300 test studies, and prevalence "is not guaranteed to be the same across the training, public leaderboard, and final evaluation datasets". Bilateral studies had report text or DICOM metadata adjusted so the scored knee can be told apart — [data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data); [733826](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)
- **(background, 8 Oct; from the 16 Oct notes)** azamat1ch (#2 now): own 5-fold 2.5D ConvNeXt-T 0.959 alone, no external data. Labels: "AI, not by hand. And don't rely on reports alone." — [748295](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748295); [735304](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304)

### Inferences
- **External data stays at the team's own risk until the end.** No ruling is coming: Kaggle's 19 Oct reply closes the door on per-dataset answers, and there are 2 days left. Compliance checks bind prize winners, not medals (as read by participants), so medal-level submissions are unlikely to be audited.
- **Our team's rule fixes the R7 decision regardless of any ruling.** R7, Versia 7 and the four-model blends carry fastMRI. R6's fastMRI status was asked publicly and is still unanswered.
- **The "image-informed labels" story is still the only top-team hint**, and it is unchanged from before 16 Oct. colum2131's fixed-ratio blend of image pseudo-labels with LLM labels is the most concrete recipe from a team at 0.964. It came with a +0.002 move (0.962 → 0.964) over the four days, which does not prove the recipe caused it.

### Gaps
- None of these top teams posted in the window: CentralyDS, azamatt, Sergey Bryansky, Zhengru Li, joinT fusion, Hyunwoong Kim, Team Kangaroo, BarryZhou, MEJAL, Nischay and Sho, GM-Enhanced Kaggle Agent, Ilya Kolb, knee-deep, YOKKOISHO, Claudio Verdú Ruiz.
- There is no host clarification on DICOM metadata use or test-set composition (sites, planes per study, bilateral studies) in the window.
- Social media (X, LinkedIn, Rednote) gave no indexed posts about the competition for 16–20 Oct. Web searches returned only launch-era news — [RSNA news](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri); [Radiology Business](https://radiologybusiness.com/topics/artificial-intelligence/rsna-launches-new-knee-mri-ai-challenge). Rednote is not searchable with the tools available.

---

## Q4. Hardest findings (Lateral OA, PF OA, Synovitis, Lateral Meniscus) and how people improved them

### Takeaway
New per-finding data this window comes from public components only:
- goodpjw2008's gold-58 table for the Raptor half vs R6 with flip TTA.
- The B-leg fold logs.
- heliosli's V2 weights.

All three agree that Lateral OA, Synovitis, PF OA and Lateral Meniscus are the weakest (gold-58 AUC 0.78–0.92), and that Fracture is weak for R6.

The leaderboard **rejected** gold-58-guided weights (0.957 and 0.955), and heliosli's leaderboard-chosen weights point the other way from gold-58 on Synovitis and PF OA. No top team disclosed a per-finding fix in the window.

### Cited Findings

**Per-finding gold-58 AUC, Raptor half (nartaa 384 + 224) vs R6 with flip TTA** (goodpjw2008, 17 Oct; "the 58 radiologist-labelled training studies, which neither model trained on") — [goodpjw2008 Does S6 help](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)

| Finding | Gold-58 positives | Raptor half | R6 flip | heliosli w (0.958) | heliosli V2 w (17 Oct) |
|---|---|---|---|---|---|
| ACL | 24 | 0.969 | 0.996 | 0.5 | 0.5 |
| MCL | 9 | 0.955 | 0.964 | 0.5 | 0.5 |
| Medial Meniscus | 26 | 0.978 | 0.971 | 0.5 | 0.5 |
| **Lateral Meniscus** | 23 | **0.894** | **0.916** | 0.5 | 0.5 |
| Medial OA | 15 | 0.981 | 0.991 | 0.5 | 0.5 |
| **Lateral OA** | 11 | **0.776** | **0.865** | 0.2 | **0.15** |
| **PF OA** | 21 | **0.861** | **0.918** | 0.6 | **0.65** |
| Effusion | 35 | 0.983 | 0.945 | 0.5 | 0.5 |
| **Synovitis** | 27 | **0.812** | **0.859** | 0.8 | **0.85** |
| Baker's | 12 | 0.973 | 0.969 | 0.35 | **0.30** |
| Contusion | 19 | 0.941 | 0.960 | 0.5 | 0.5 |
| Fracture | 18 | 0.962 | **0.887** | 0.5 | 0.5 |

The "w" columns are the Raptor share. Macro means computed from the table: R6 flip 0.937, Raptor half 0.924. The V2 weights come from the [heliosli notebook](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586).

**Caveat on the Raptor numbers.** Its SWA epochs were "ranked by Gold-58 AUC", so its gold-58 numbers are optimistic. R6 never trained on gold — [nartaa RUNS.md, 17 Oct](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights); [discussion 750180](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/discussion/750180)

**The leaderboard contradicted gold-58:** v6 (Fracture/Effusion toward Raptor, ACL toward R6) scored 0.957, and v7 (all gold optima) 0.955, against 0.958 with heliosli's weights. goodpjw2008: "with 18 to 35 positives per finding the gaps are noise, and heliosli's weights, chosen on much more data, stand" — [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)

**heliosli's 17 Oct V2 weights** move further toward:
- R6 for Lateral OA (0.85 R6) and Baker's (0.70 R6);
- Raptor for Synovitis (0.85) and PF OA (0.65).

This happens even though gold-58 shows R6 ahead on Synovitis (0.859 vs 0.812) and PF OA (0.918 vs 0.861) — [heliosli notebook](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

**OAI supervision in the Raptor CoAtNet** covers only PF OA, Lateral OA and Synovitis (MOAKS cuts). Effusion, Contusion, Fracture and MCL "can never be enabled" — [nartaa RUNS.md](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights)

**R6 fine-tunes on severity labels (B-leg).** Single-fold gold-58:
- Strong: ACL 0.97–0.99, Medial OA 0.97–0.99, Medial Meniscus 0.97–0.98.
- Weak ("low report signal"): Lateral OA about 0.85, Synovitis 0.82–0.87, PF OA 0.86–0.89, Effusion 0.86–0.90.

Sources: [kritikabenjwal B-leg s0-f01](https://www.kaggle.com/datasets/kritikabenjwal/rsna-knee-bleg-s0-f01); [sarthakgupta1801 B-leg s0-f23](https://www.kaggle.com/datasets/sarthakgupta1801/rsna-knee-bleg-s0-f23); [xhaitanya B-leg s1-f01](https://www.kaggle.com/datasets/xhaitanya/rsna-knee-bleg-s1-f01)

**Severity-state calibration** (soft-v2): "ACL `signal_or_degeneration` is officially positive 8/21 while `high_grade_or_complete_tear` is 15/16". Calibrated states reach a gold macro of 0.831 vs 0.764 for binary report labels (in-sample) — [kritikabenjwal/rsna-knee-labels-soft-v2](https://www.kaggle.com/datasets/kritikabenjwal/rsna-knee-labels-soft-v2)

**OmerZalman's high-accuracy label sets omit Synovitis entirely** (11 findings) — [omerzalman/rsna-knee-98-accuracy-11-soft-labels](https://www.kaggle.com/datasets/omerzalman/rsna-knee-98-accuracy-11-soft-labels)

**Background from before 16 Oct, for comparison:**
- darcywjoseph's public-LB inversion probe: PF OA 0.890, Lateral OA 0.896, Synovitis 0.932, Lateral Meniscus 0.956, the other eight 0.963 — [SOLUTION.md](https://github.com/darcywjoseph/Knee-Abnormality-Detection/blob/HEAD/SOLUTION.md)
- colum2131's image pseudo-labels reached gold Synovitis 0.859 and Effusion 0.986 — [745949](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745949)
- A request on 19 Oct asking colum2131 for that Effusion/Synovitis model's architecture, the encoding of "not mentioned", and checkpoints went unanswered — [745949](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745949)

### Inferences
- **Lateral OA:** every source agrees that knee-anchored ConvNeXt (R6) beats the CoAtNet. Gold-58 favours R6 by 0.09, and the leaderboard-chosen weight gives R6 85 %. Lateral OA is also where report-based labels over-call (background). An own model trained on image-aware labels with compartment-anchored crops should be weighted heavily here.
- **Synovitis and PF OA:** the leaderboard favours the OAI-MOAKS-supervised CoAtNet even though gold-58 favours R6. Either:
  - gold-58 (21–27 positives) misleads; or
  - external MOAKS supervision (Hoffa-synovitis, PF full-thickness loss) transfers to the expert test labels.

  For a team without OAI data, this is the finding pair where an own model is least likely to match the public blend. That suggests keeping the public CoAtNet's share on these two findings if it is used at all, subject to our rules.
- **Do not tune per-finding weights on gold-58.** That is now measured twice: goodpjw2008 v6/v7 and darcywjoseph (background). Leaderboard-chosen weights (heliosli) beat gold-chosen ones by 0.001–0.003.
- **Fracture and Effusion are weak in R6 / R6-derived models** (R6 flip 0.887 Fracture, 0.945 Effusion; B-leg Effusion 0.86–0.90), while the CoAtNet is strong (0.962, 0.983). A decorrelated own model is most likely to add value where the two public families disagree. That is an inference, not a measurement.

### Gaps
- No per-finding public-LB numbers for any ≥ 0.960 team, and no top-team statement on how they fixed Lateral OA, PF OA, Synovitis or Lateral Meniscus in the window.
- No public measurement shows whether severity-calibrated targets (soft-v2) or partial-coverage label sets improve the hard findings on the leaderboard.
- The gold-58 table has 9–35 positives per finding. goodpjw2008 states the gaps are within noise. The per-finding comparisons above are directional only.
