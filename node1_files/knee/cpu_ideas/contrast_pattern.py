"""Does model accuracy depend on which contrast each slot got? + stacker with per-study contrast-pattern features."""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sl = pd.read_csv("../ds/corpus192/slots.csv")
print("series_fluid values by slot:\n", sl.groupby("slot_idx").series_fluid.value_counts(dropna=False).unstack(fill_value=0).to_string())
ser = pd.read_csv("train_series.csv")[["SeriesInstanceUID", "Fluid_Sensitive"]]
sl = sl.merge(ser, on="SeriesInstanceUID", how="left")
piv = sl.pivot_table(index="StudyInstanceUID", columns="slot_idx", values="Fluid_Sensitive", aggfunc="first")  # NaN = empty slot
P = pd.DataFrame(index=piv.index)
P["sagFS_ok"] = (piv[0] == 1).astype(float); P["sagPlain_ok"] = (piv[1] == 0).astype(float)
P["corFS_ok"] = (piv[2] == 1).astype(float); P["corPlain_ok"] = (piv[3] == 0).astype(float)
P["ax_FS"] = (piv[4] == 1).astype(float); P["ax_missing"] = piv[4].isna().astype(float)
print("\npattern rates:", P.mean().round(3).to_dict())
base = pd.read_parquet("labels_sev_v1.parquet").set_index("StudyInstanceUID"); F = list(base.columns)
zs = [np.load(f"oof/raptor_oof_cnxt_r2_f{f}_swa_fold{f}.npz") for f in range(4)]
ids = np.concatenate([z["ids"] for z in zs]); X = lg(np.concatenate([z["pred"] for z in zs]))
fold = np.concatenate([np.full(len(z["ids"]), f) for f, z in enumerate(zs)])
Y = (base.loc[ids].values > 0.5).astype(int); PO = P.reindex(ids).fillna(0).values
gz = [np.load(f"oof/raptor_gold_cnxt_r2_f{f}_swa.npz") for f in range(4)]; gid = gz[0]["ids"]
XG = lg(np.mean([g["pred"] for g in gz], 0)); GT = gz[0]["truth"]; PG = P.reindex(gid).fillna(0).values
auc = lambda y, p: roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
print("\nOOF AUC (vs reports) by contrast pattern  [n]:")
for col in P.columns:
    c = P.columns.get_loc(col); on = PO[:, c] == 1
    row = " ".join(f"{F[j][:6]} {auc(Y[on, j], X[on, j]):.3f}/{auc(Y[~on, j], X[~on, j]):.3f}" for j in (3, 5, 6, 8))
    print(f"  {col:12s} yes/no [{on.sum():4d}/{(~on).sum():4d}]  {row}")
print("\nstacker [logit, pattern flags, logit x flags] per finding: OOF base->new | gold base->new")
tot = []
for j in range(12):
    A = np.c_[X[:, j], PO, X[:, [j]] * PO]; C = np.c_[XG[:, j], PG, XG[:, [j]] * PG]
    po, pg = np.zeros(len(ids)), []
    for f in range(4):
        tr, te = fold != f, fold == f
        m = LogisticRegression(C=0.3, max_iter=3000).fit(A[tr], Y[tr, j]); po[te] = m.decision_function(A[te]); pg.append(m.decision_function(C))
    r = (auc(Y[:, j], X[:, j]), auc(Y[:, j], po), auc(GT[:, j], XG[:, j]), auc(GT[:, j], np.mean(pg, 0))); tot.append(r)
    print(f"  {F[j]:17s} {r[0]:.3f}->{r[1]:.3f} | {r[2]:.3f}->{r[3]:.3f}")
t = np.array(tot); print(f"  MACRO             {t[:,0].mean():.4f}->{t[:,1].mean():.4f} | {t[:,2].mean():.4f}->{t[:,3].mean():.4f}")
