"""Idea 10 + contrast views + axial-FS swap: per-finding AUC on fold-0 OOF (report labels) and gold-58 (experts)."""
import os, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(D)
base = pd.read_parquet(os.path.join(R, "../pcseq/labels_sev_v1.parquet")).set_index("StudyInstanceUID"); F = list(base.columns)
g = np.load(os.path.join(R, "raptor_gold_cnxt_r2_f0_swa.npz")); GT = pd.DataFrame(g["truth"], index=g["ids"].astype(str), columns=F)
rd = lambda n: pd.read_parquet(os.path.join(D, n)).set_index("StudyInstanceUID")[F] if os.path.exists(os.path.join(D, n)) else None
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
views = [v for v in ["all", "cor", "axsag", "sag", "corax", "fs", "plain"] if rd(f"{v}_oof.parquet") is not None]
O = {v: rd(f"{v}_oof.parquet") for v in views}; G = {v: rd(f"{v}_gold.parquet") for v in views}
ids = O["all"].index; Y = (base.loc[ids] > 0.5).astype(int)
print("per-finding AUC  (report labels on 1088 OOF / experts on gold 58)")
print("finding".ljust(17), "  ".join(f"{v:>11s}" for v in views))
for c in F:
    print(c[:16].ljust(17), "  ".join(f"{roc_auc_score(Y[c], O[v].loc[ids, c]):.3f}/{roc_auc_score(GT[c], G[v].loc[GT.index, c]):.3f}" for v in views))
# view routing: per finding, LR over all view logits, 5-fold CV inside the 1088 OOF; gold = fit on all OOF
print("\nlearned view routing (CV): OOF all->routed | gold all->routed")
rng = np.random.RandomState(0); cvf = rng.randint(0, 5, len(ids)); tot = []
for c in F:
    A = np.c_[[lg(O[v].loc[ids, c].values) for v in views]].T; C = np.c_[[lg(G[v].loc[GT.index, c].values) for v in views]].T
    po = np.zeros(len(ids))
    for f in range(5):
        m = LogisticRegression(C=0.5, max_iter=3000).fit(A[cvf != f], Y[c].values[cvf != f]); po[cvf == f] = m.decision_function(A[cvf == f])
    pg = LogisticRegression(C=0.5, max_iter=3000).fit(A, Y[c].values).decision_function(C)
    r = (roc_auc_score(Y[c], A[:, 0]), roc_auc_score(Y[c], po), roc_auc_score(GT[c], C[:, 0]), roc_auc_score(GT[c], pg)); tot.append(r)
    print(f"  {c[:16]:17s} {r[0]:.3f}->{r[1]:.3f} | {r[2]:.3f}->{r[3]:.3f}")
t = np.array(tot); print(f"  MACRO             {t[:,0].mean():.4f}->{t[:,1].mean():.4f} | {t[:,2].mean():.4f}->{t[:,3].mean():.4f}")
ax = rd("axfix.parquet")
if ax is not None:
    sw = [u for u in ax.index if u in ids]
    print(f"\naxial-FS swap on {len(sw)} fold-0 studies whose axial slot was plain:")
    for c in ["PF OA", "Effusion", "Synovitis", "Lateral OA", "Medial OA"]:
        y = Y.loc[sw, c]
        print(f"  {c:11s} subset AUC {roc_auc_score(y, O['all'].loc[sw, c]):.3f} -> {roc_auc_score(y, ax.loc[sw, c]):.3f}   (pos {int(y.sum())})")
        full = O["all"][c].copy(); full.loc[sw] = ax.loc[sw, c]
        print(f"  {'':11s} all-1088 AUC {roc_auc_score(Y[c], O['all'][c]):.4f} -> {roc_auc_score(Y[c], full):.4f}")
