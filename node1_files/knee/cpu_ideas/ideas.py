"""CPU tests of ideas 1-7 on round-2 ConvNeXt predictions. Two scorecards per finding:
OOF AUC vs report labels (4,349 studies, fold-aware fits) / gold-58 AUC vs expert labels. No report text is read."""
import re, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
auc = lambda y, p: roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan

def fam(oof_pat, gold_pat, folds=range(4)):
    zs = [np.load(oof_pat.format(f=f)) for f in folds]
    o = pd.DataFrame(lg(np.concatenate([z["pred"] for z in zs])), index=np.concatenate([z["ids"] for z in zs]))
    gs = []
    for f in folds:
        try: gs.append(np.load(gold_pat.format(f=f)))
        except OSError: pass
    return o, pd.DataFrame(lg(np.mean([g["pred"] for g in gs], 0)), index=gs[0]["ids"]), gs[0]["truth"]

base = pd.read_parquet("labels_sev_v1.parquet").set_index("StudyInstanceUID"); F = list(base.columns)
O, G, GT = fam("oof/raptor_oof_cnxt_r2_f{f}_swa_fold{f}.npz", "oof/raptor_gold_cnxt_r2_f{f}_swa.npz")
ids, gids = O.index.values, G.index.values
Y = (base.loc[ids].values > 0.5).astype(int)
fold = np.concatenate([np.full(n, f) for f, n in enumerate([len(np.load(f"oof/raptor_oof_cnxt_r2_f{f}_swa_fold{f}.npz")["ids"]) for f in range(4)])])
X, XG = O.values, G.values

# ---- study-level metadata from series headers + competition series table + masks + laterality ----
import os
if os.path.exists("series_headers.csv"):
    h = pd.read_csv("series_headers.csv", low_memory=False); h = h[h.split == "train_series"]
else:   # header scan not done yet: hospital from site_metadata, protocol from the series table only
    sm = pd.read_csv("site_metadata.csv")
    h = pd.read_csv("train_series.csv")[["StudyInstanceUID", "SeriesInstanceUID"]].merge(sm, on="StudyInstanceUID", how="left")
    for c in ["PatientSex", "SeriesDescription", "ProtocolName", "ContrastBolusAgent", "SliceThickness", "RepetitionTime", "EchoTime"]:
        h[c] = None
    h["n_images"] = 1
ser = pd.read_csv("train_series.csv")
h = h.merge(ser, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")
txt = (h.SeriesDescription.fillna("") + " " + h.ProtocolName.fillna("")).str.lower()
h["contrast"] = (h.ContrastBolusAgent.fillna("").str.strip() != "") | txt.str.contains(r"\bgad|\+ ?c\b|\bc\+|\bkm\b|contrast|gado|dotarem|post ?gd|\bpost\b(?! ?(erior|lat))")
num = lambda c: pd.to_numeric(h[c], errors="coerce")
h["thk"], h["tr"], h["te"], h["fs_t"] = num("SliceThickness"), num("RepetitionTime"), num("EchoTime"), num("MagneticFieldStrength")
g = h.groupby("StudyInstanceUID")
S = pd.DataFrame({
    "site": g.apply(lambda d: "|".join(d[["InstitutionName", "ManufacturerModelName", "StationName"]].fillna("na").astype(str).iloc[0])),
    "sex_f": g.PatientSex.first().eq("F").astype(float), "field": g.fs_t.max(), "n_series": g.size(),
    "contrast": g.contrast.any().astype(float), "n_img": g.n_images.sum(), "thk_min": g.thk.min(), "tr_max": g.tr.max(), "te_max": g.te.max()})
for pl in ["Sagittal", "Coronal", "Axial"]:
    S[f"n_{pl}"] = h[h.Anatomical_Plane == pl].groupby("StudyInstanceUID").size().reindex(S.index).fillna(0)
    S[f"fs_{pl}"] = h[(h.Anatomical_Plane == pl) & (h.Fat_Suppression == 1)].groupby("StudyInstanceUID").size().reindex(S.index).fillna(0)
m, mid = np.load("all_masks.npy"), np.load("all_ids.npy", allow_pickle=True)
bounds = [(0, 40), (40, 80), (80, 116), (116, 152), (152, 192)]
cov = pd.DataFrame({f"cov{s}": (m[:, a:b] > 0).mean(1) for s, (a, b) in enumerate(bounds)}, index=mid)
S = S.join(cov)
lat = pd.read_csv("study_laterality_orientation.csv").set_index("StudyInstanceUID")
flags = pd.read_csv("laterality_flags_192.csv").set_index("StudyInstanceUID")
S["left"] = lat.Laterality.reindex(S.index).eq("Left").astype(float)
S["normx_pos"] = (lat.Normal_X.reindex(S.index) > 0).astype(float)
S["side_src"] = flags.side_source.reindex(S.index).fillna("none")
SO, SG = S.reindex(ids), S.reindex(gids)
print(f"meta: {len(S)} studies, {S.site.nunique()} sites, contrast studies {int(S.contrast.sum())}, female {S.sex_f.mean():.2f}")

rows = []
def report(idea, j, po, pg):
    rows.append((idea, F[j], auc(Y[:, j], po), auc(GT[:, j], pg)))

def cv_fit(make, feats_o, feats_g, j):
    po, pg = np.zeros(len(ids)), []
    for f in range(4):
        tr, te = fold != f, fold == f
        mdl = make().fit(feats_o[tr], Y[tr, j])
        sc = (lambda M, A: M.decision_function(A)) if hasattr(mdl, "decision_function") else (lambda M, A: M.predict_proba(A)[:, 1])
        po[te] = sc(mdl, feats_o[te]); pg.append(sc(mdl, feats_g))
    return po, np.mean(pg, 0)

def site_stat(tr, vals, k):   # shrunk per-site mean of vals over training rows; returns lookup Series
    d = pd.DataFrame({"s": SO.site.values[tr], "v": vals}); gg = d.groupby("s").v.agg(["sum", "count"]); mu = vals.mean()
    return (gg["sum"] + k * mu) / (gg["count"] + k), mu

LR = lambda: LogisticRegression(C=1.0, max_iter=3000)
for j in range(12):
    report("0 baseline", j, X[:, j], XG[:, j])
    # 1. hospital prevalence prior (log-odds, shrinkage k)
    for k in (5, 20, 80):
        po, pg = np.zeros(len(ids)), []
        for f in range(4):
            tr, te = fold != f, fold == f
            tab, mu = site_stat(tr, Y[tr, j].astype(float), k)
            pri = lambda s: lg(pd.Series(s).map(tab).fillna(mu).values)
            A, B, C = np.c_[X[tr, j], pri(SO.site.values[tr])], np.c_[X[te, j], pri(SO.site.values[te])], np.c_[XG[:, j], pri(SG.site.values)]
            mdl = LR().fit(A, Y[tr, j]); po[te] = mdl.decision_function(B); pg.append(mdl.decision_function(C))
        report(f"1 hospital prior k{k}", j, po, np.mean(pg, 0))
    # 2. remove the hospital bias the model learned (unsupervised: subtract shrunk per-site mean logit)
    for beta in (0.5, 1.0):
        po, pg = X[:, j].copy(), []
        for f in range(4):
            tr, te = fold != f, fold == f
            tab, mu = site_stat(tr, X[tr, j], 20)
            po[te] = X[te, j] - beta * (pd.Series(SO.site.values[te]).map(tab).fillna(mu).values - mu)
            pg.append(XG[:, j] - beta * (pd.Series(SG.site.values).map(tab).fillna(mu).values - mu))
        report(f"2 debias b{beta}", j, po, np.mean(pg, 0))
    # 3. protocol fingerprint stacker (gradient boosting on own logit + header features)
    pcols = ["sex_f", "field", "n_series", "contrast", "n_img", "thk_min", "tr_max", "te_max", "n_Sagittal", "n_Coronal", "n_Axial",
             "fs_Sagittal", "fs_Coronal", "fs_Axial"]
    pcols = [c for c in pcols if SO[c].astype(float).nunique() > 1]   # skip features the header scan has not filled yet
    A, C = np.c_[X[:, j], SO[pcols].values.astype(float)], np.c_[XG[:, j], SG[pcols].values.astype(float)]
    po, pg = cv_fit(lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=40), A, C, j)
    report("3 protocol GBM", j, po, pg)
    po, pg = cv_fit(LR, np.nan_to_num((A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-6)), np.nan_to_num((C - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-6)), j)
    report("3 protocol LR", j, po, pg)
    # 4. contrast flag
    po, pg = cv_fit(LR, np.c_[X[:, j], SO.contrast.values], np.c_[XG[:, j], SG.contrast.values], j)
    report("4 +contrast", j, po, pg)
    # 5. coverage-aware trust: logit x coverage interactions
    cc = [f"cov{s}" for s in range(5)]
    A = np.c_[X[:, j], SO[cc].values, X[:, [j]] * SO[cc].values]; C = np.c_[XG[:, j], SG[cc].values, XG[:, [j]] * SG[cc].values]
    po, pg = cv_fit(LR, np.nan_to_num(A), np.nan_to_num(C), j)
    report("5 coverage trust", j, po, pg)
    # 7. side-aware: logit x left, plus paired-compartment logit x left (lets left knees borrow the other side's head)
    pair = {2: 3, 3: 2, 4: 5, 5: 4}.get(j)
    if pair is not None:
        L, LG_ = SO.left.values, SG.left.values
        A = np.c_[X[:, j], X[:, pair], L, X[:, j] * L, X[:, pair] * L]; C = np.c_[XG[:, j], XG[:, pair], LG_, XG[:, j] * LG_, XG[:, pair] * LG_]
        po, pg = cv_fit(LR, A, C, j); report("7 side-aware mix", j, po, pg)

# 6. per-finding family weights (4-fold families; fold-0 extras scored on fold-0 rows)
fams = {"sev": fam("oof/raptor_oof_cnxt_sev_f{f}_swa_fold{f}.npz", "oof/raptor_gold_cnxt_sev_f{f}_swa.npz"),
        "coat96": fam("oof/raptor_oof_oof4_f{f}_swa_fold{f}.npz", "oof/raptor_gold_oof4_f{f}_swa.npz")}
rk = lambda a: pd.Series(a).rank(pct=True).values
for name, (o, gpred, _) in fams.items():
    o = o.reindex(ids).values; gp = gpred.reindex(gids).values
    for j in range(12):
        for w in (0.25, 0.5):
            report(f"6 r2+{name} w{w}", j, (1 - w) * rk(X[:, j]) + w * rk(o[:, j]), (1 - w) * rk(XG[:, j]) + w * rk(gp[:, j]))
f0 = fold == 0
for name, op, gpat in [("pos", "oof/raptor_oof_cnxt_pos_f0_swa_fold0.npz", "oof/raptor_gold_cnxt_pos_f0_swa.npz"),
                       ("radimg", "oof/raptor_oof_rin_r2_f0_swa_fold0.npz", "oof/raptor_gold_rin_r2_f0_swa.npz"),
                       ("r2l", "oof/raptor_oof_cnxt_r2l_f0_swa_fold0.npz", "oof/raptor_gold_cnxt_r2l_f0_swa.npz"),
                       ("seed2", "oof/raptor_oof_cnxt_r2s2_f0_swa_fold0.npz", "oof/raptor_gold_cnxt_r2s2_f0_swa.npz")]:
    try: o, gpred, _ = fam(op, gpat, folds=[0])
    except OSError as e: print("skip", name, e); continue
    o = o.reindex(ids[f0]).values; gp = gpred.reindex(gids).values
    g0 = np.load("oof/raptor_gold_cnxt_r2_f0_swa.npz"); xg0 = lg(g0["pred"])
    for j in range(12):
        rows.append((f"6f0 r2 fold0 only", F[j], auc(Y[f0, j], X[f0, j]), auc(GT[:, j], xg0[:, j])))
        rows.append((f"6f0 r2+{name} w0.5", F[j], auc(Y[f0, j], rk(X[f0, j]) + rk(o[:, j])), auc(GT[:, j], rk(xg0[:, j]) + rk(gp[:, j]))))

R = pd.DataFrame(rows, columns=["idea", "finding", "oof", "gold"]).drop_duplicates(["idea", "finding"])
R.to_csv("ideas_results.csv", index=False)
b = R[R.idea == "0 baseline"].set_index("finding")
b0 = R[R.idea == "6f0 r2 fold0 only"].set_index("finding")
R["d_oof"] = R.apply(lambda r: r.oof - (b0 if r.idea.startswith("6f0") else b).loc[r.finding, "oof"], axis=1)
R["d_gold"] = R.apply(lambda r: r.gold - (b0 if r.idea.startswith("6f0") else b).loc[r.finding, "gold"], axis=1)
pd.set_option("display.width", 200)
print("\nMACRO (12 findings): idea, oof, gold, d_oof, d_gold")
print(R.groupby("idea")[["oof", "gold", "d_oof", "d_gold"]].mean().round(4).to_string())
weak = ["Lateral OA", "Synovitis", "PF OA", "Lateral Meniscus"]
print("\nWEAK FOUR, per finding (d_oof / d_gold):")
print(R[R.finding.isin(weak)].pivot(index="idea", columns="finding", values="d_oof").round(3).astype(str).add(" / ")
      .add(R[R.finding.isin(weak)].pivot(index="idea", columns="finding", values="d_gold").round(3).astype(str)).to_string())
print("\nWINS (both scorecards up, d_oof>=+0.003 and d_gold>=+0.010):")
print(R[(R.d_oof >= 0.003) & (R.d_gold >= 0.010)].round(4).to_string(index=False))
# left vs right diagnostic (idea 7)
L = SO.left.values == 1
side = [1, 2, 3, 4, 5]
print("\nLEFT vs RIGHT, OOF AUC vs reports, side-specific findings:")
for j in side:
    print(f"  {F[j]:18s} left {auc(Y[L, j], X[L, j]):.3f}  right {auc(Y[~L, j], X[~L, j]):.3f}")
for src, d in SO.groupby("side_src"):
    msk = SO.side_src.values == src
    print(f"  side source {src:18s} n={msk.sum():4d}  side-specific mean {np.nanmean([auc(Y[msk, j], X[msk, j]) for j in side]):.3f}")
print("\nCONTRAST: studies", int(SO.contrast.sum()), "| Synovitis report rate with/without",
      round(Y[SO.contrast.values == 1, 8].mean(), 3), round(Y[SO.contrast.values == 0, 8].mean(), 3))
