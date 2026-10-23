"""Lateral OA scanner prior: key = Manufacturer|ManufacturerModelName (present in test DICOMs). Fold-aware check + final table."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
h = pd.read_csv("series_headers.csv", low_memory=False); h = h[h.split == "train_series"]
key = (h.Manufacturer.fillna("na").astype(str).str.strip().str.upper() + "|" + h.ManufacturerModelName.fillna("na").astype(str).str.strip().str.upper())
scan = key.groupby(h.StudyInstanceUID).first()
base = pd.read_parquet("labels_sev_v1.parquet").set_index("StudyInstanceUID"); J = 5
zs = [np.load(f"oof/raptor_oof_cnxt_r2_f{f}_swa_fold{f}.npz") for f in range(4)]
ids = np.concatenate([z["ids"] for z in zs]); X = lg(np.concatenate([z["pred"] for z in zs]))[:, J]
fold = np.concatenate([np.full(len(z["ids"]), f) for f, z in enumerate(zs)])
Y = (base.loc[ids, "Lateral OA"].values > 0.5).astype(int); S = scan.reindex(ids).fillna("na|na").values
gz = [np.load(f"oof/raptor_gold_cnxt_r2_f{f}_swa.npz") for f in range(4)]; gid = gz[0]["ids"]
XG = lg(np.mean([g["pred"] for g in gz], 0))[:, J]; GT = gz[0]["truth"][:, J]; SG = scan.reindex(gid).fillna("na|na").values
print("scanners:", len(set(S)))
def table(mask, k):
    d = pd.DataFrame({"s": S[mask], "y": Y[mask]}); g = d.groupby("s").y.agg(["sum", "count"]); mu = Y[mask].mean()
    return ((g["sum"] + k * mu) / (g["count"] + k)).to_dict(), mu
for k in (5, 20, 80):
    po, pg = np.zeros(len(ids)), []
    for f in range(4):
        tr, te = fold != f, fold == f
        t, mu = table(tr, k); pri = lambda s: lg(np.array([t.get(x, mu) for x in s]))
        m = LogisticRegression(max_iter=2000).fit(np.c_[X[tr], pri(S[tr])], Y[tr])
        po[te] = m.decision_function(np.c_[X[te], pri(S[te])]); pg.append(m.decision_function(np.c_[XG, pri(SG)]))
    print(f"k{k}: LatOA OOF {roc_auc_score(Y, X):.3f} -> {roc_auc_score(Y, po):.3f} | gold {roc_auc_score(GT, XG):.3f} -> {roc_auc_score(GT, np.mean(pg, 0)):.3f}")
t, mu = table(np.ones(len(ids), bool), 20); pri = lambda s: lg(np.array([t.get(x, mu) for x in s]))
m = LogisticRegression(max_iter=2000).fit(np.c_[X, pri(S)], Y)
json.dump({"key": "Manufacturer|ManufacturerModelName upper", "k": 20, "prior": t, "default": mu,
           "coef_logit": float(m.coef_[0][0]), "coef_prior": float(m.coef_[0][1]), "intercept": float(m.intercept_[0])},
          open("latoa_scanner_prior.json", "w"), indent=1)
print("wrote latoa_scanner_prior.json | coef logit/prior", m.coef_[0].round(3))
