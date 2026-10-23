"""Score label CSVs on the 25 unread gold studies vs expert labels; baseline = current report labels."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
gz = np.load("../pcr2/raptor_gold_cnxt_r2_f0_swa.npz", allow_pickle=True)
F = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
T = pd.DataFrame(gz["truth"], index=gz["ids"].astype(str), columns=F)
test = open("test_ids.txt").read().split()
srcs = {"current": "../labels/labels_clef_sev.csv", **{p: p for p in sys.argv[1:]}}
print(f"{'source':28s} macro | " + " ".join(f"{c[:6]:>6s}" for c in F))
for n, p in srcs.items():
    d = pd.read_csv(p).set_index("StudyInstanceUID"); ok = [u for u in test if u in d.index]
    a = [roc_auc_score(T.loc[ok, c], d.loc[ok, c]) for c in F]
    print(f"{n[-28:]:28s} {np.mean(a):.3f} | " + " ".join(f"{x:6.3f}" for x in a) + f"  (n={len(ok)})")
