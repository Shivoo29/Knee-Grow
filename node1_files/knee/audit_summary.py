import json, glob, os

print(f'=== UNNATI MODEL AUDIT SUMMARY (Oct 14, 2026) ===')
files = sorted(glob.glob('/home/s05814802723/projects/knee/raptor_ft_*.json'), key=os.path.getmtime, reverse=True)
for p in files:
    try:
        with open(p) as f:
            d = json.load(f)
        tag = d.get('tag', os.path.basename(p))
        auc = d.get('best_gold_auc', 0)
        epochs = d.get('epochs_done', d.get('epochs', '?'))
        lat = d.get('aucs', {}).get('Lateral OA', 0)
        pf = d.get('aucs', {}).get('PF OA', 0)
        mcl = d.get('aucs', {}).get('MCL', 0)
        print(f'{tag:24s} | Gold: {auc:.4f} | LatOA: {lat:.4f} | PFOA: {pf:.4f} | MCL: {mcl:.4f} | Ep: {epochs}')
    except Exception as e:
        pass
