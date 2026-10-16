"""Per (study, finding): mask = 0 where the public LLM label tables are uncertain or disagree (forum: Prateek Grover)."""
import glob, numpy as np, pandas as pd
exec(open('/tmp/claude-1000/-home-codemaster29-Documents-still-learning-rsna-knee-abnormality-detection/8cbf309e-7b5f-49e4-b79c-0abc1b397a28/scratchpad/label_sources_gold.py').read().split("src={}")[0])          # L, P, load()
lab = pd.read_parquet(P + 'labels_avg.parquet'); lab['StudyInstanceUID'] = lab.StudyInstanceUID.astype(str)
ids = lab.StudyInstanceUID.tolist()
tabs = []
for f in sorted(glob.glob('labels/*/*.csv') + glob.glob(P + 'data/labels/*') + glob.glob(P + 'data/llm_qwen.parquet')):
    if 'gold74' in f: continue
    try: d = load(f)
    except Exception: continue
    cov = d.index.intersection(ids)
    if len(cov) >= 0.95 * len(ids): tabs.append(d.reindex(ids)[L])
print('tables covering >=95% of train studies:', len(tabs))
S = np.stack([t.values for t in tabs])                      # (T, N, 12), NaN where a table lacks a study
m = np.nanmean(S, 0); sd = np.nanstd(S, 0)
w = ~((m > 0.30) & (m < 0.70))   # only truly uncertain consensus; sd dropped (tables use different scales)
print('kept fraction per finding:', ' '.join(f'{c[:6]} {w[:, j].mean():.2f}' for j, c in enumerate(L)), '| overall %.2f' % w.mean())
out = pd.DataFrame(w.astype(np.float32), columns=['w_' + c for c in L]); out.insert(0, 'StudyInstanceUID', ids)
out.to_parquet(P + 'label_weights_consensus_v1.parquet', index=False); print('wrote label_weights_consensus_v1.parquet', out.shape)
