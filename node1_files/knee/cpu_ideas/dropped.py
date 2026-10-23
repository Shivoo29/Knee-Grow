import pandas as pd, numpy as np
sl = pd.read_csv("../ds/corpus192/slots.csv"); ser = pd.read_csv("train_series.csv")
h = pd.read_csv("series_headers.csv", low_memory=False)[["SeriesInstanceUID", "n_images", "SeriesDescription"]]
ser = ser.merge(h, on="SeriesInstanceUID", how="left")
used = set(sl.SeriesInstanceUID.dropna())
ser["used"] = ser.SeriesInstanceUID.isin(used)
ser["kind"] = ser.Anatomical_Plane + np.where(ser.Fat_Suppression == 1, "-FS", "-plain")
n_st = ser.StudyInstanceUID.nunique()
print(f"studies {n_st} | series {len(ser)} | used {ser.used.sum()} | dropped {(~ser.used).sum()} ({(~ser.used).mean():.0%})")
d = ser[~ser.used]
print("\nstudies with >=1 dropped series:", d.StudyInstanceUID.nunique(), f"({d.StudyInstanceUID.nunique()/n_st:.0%})")
print("\ndropped by kind (count | studies | median slices):")
for k, g in d.groupby("kind"):
    print(f"  {k:16s} {len(g):5d} | {g.StudyInstanceUID.nunique():5d} | {g.n_images.median():.0f}")
tiny = d.n_images <= 5
print(f"\ndropped series with <=5 images (scouts/localizers): {tiny.sum()} ({tiny.mean():.0%})")
# dropped real series whose kind has NO used series in that study? (= a plane/contrast the model never sees)
uk = ser[ser.used].groupby("StudyInstanceUID").kind.apply(set)
miss = d[~tiny & d.apply(lambda r: r.kind not in uk.get(r.StudyInstanceUID, set()), axis=1)]
print(f"dropped real series of a kind the model sees NOWHERE in that study: {len(miss)} series in {miss.StudyInstanceUID.nunique()} studies")
print(miss.kind.value_counts().to_string())
s = sl[sl.filled == True]
print(f"\nslices subsampled: {(s.n_available > s.slot_k).mean():.0%} of filled slots have more slices than the slot holds (median available {s.n_available.median():.0f} vs slot {s.slot_k.median():.0f})")
print("empty slots:", (sl.filled == False).sum(), "of", len(sl))
print("\nmost common dropped series names (top 12):"); print(d[~tiny].SeriesDescription.str.lower().str.strip().value_counts().head(12).to_string())
