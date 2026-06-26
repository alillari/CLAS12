"""
Characterize real CLAS12 events so the synthetic straight-track generator can
match them (particle count per event, hits per event, angular coverage).
Run inside the container or anywhere with pandas + the CSVs.
"""
import numpy as np
import pandas as pd

CSV_FILE_BST = "/workspace/PP_collision/data/THREE_HADRONS_bst_crosses.csv"
CSV_FILE_BMT = "/workspace/PP_collision/data/THREE_HADRONS_bmt_combined_points.csv"

KEEP = ["event", "x0", "y0", "z0", "trkID"]
df = pd.concat([
    pd.read_csv(CSV_FILE_BST)[KEEP],
    pd.read_csv(CSV_FILE_BMT)[KEEP],
], ignore_index=True)

# derive r, phi, eta
df["r"] = np.sqrt(df["x0"]**2 + df["y0"]**2)
df["phi"] = np.arctan2(df["y0"], df["x0"])
theta = np.arctan2(df["r"], df["z0"])
df["eta"] = -np.log(np.tan(theta / 2.0))

# hits per event
hits_per_event = df.groupby("event").size()
print("=== hits per event ===")
print(f"  n_events: {df['event'].nunique()}")
print(f"  mean: {hits_per_event.mean():.2f}, median: {hits_per_event.median():.0f}")
print(f"  min: {hits_per_event.min()}, max: {hits_per_event.max()}")
print(f"  percentiles 10/25/50/75/90: "
      f"{np.percentile(hits_per_event,[10,25,50,75,90]).round(1).tolist()}")

# distinct (non-noise) tracks per event — trkID != -1
real_trk = df[df["trkID"] != -1]
trks_per_event = real_trk.groupby("event")["trkID"].nunique()
print("\n=== distinct real tracks (trkID != -1) per event ===")
print(f"  mean: {trks_per_event.mean():.2f}, median: {trks_per_event.median():.0f}")
print(f"  min: {trks_per_event.min()}, max: {trks_per_event.max()}")
print(f"  value counts:\n{trks_per_event.value_counts().sort_index()}")

# how many noise hits (trkID == -1)
n_noise = (df["trkID"] == -1).sum()
print(f"\n=== noise hits (trkID == -1): {n_noise} / {len(df)} "
      f"({100*n_noise/len(df):.1f}%) ===")

# angular coverage
print("\n=== angular coverage (real tracks only) ===")
print(f"  phi: [{real_trk['phi'].min():.3f}, {real_trk['phi'].max():.3f}]")
print(f"  eta: [{real_trk['eta'].min():.3f}, {real_trk['eta'].max():.3f}]")

# hits per individual track (to see if tracks always hit all 6 layers)
hits_per_track = real_trk.groupby(["event", "trkID"]).size()
print("\n=== hits per individual track ===")
print(f"  mean: {hits_per_track.mean():.2f}")
print(f"  value counts:\n{hits_per_track.value_counts().sort_index()}")