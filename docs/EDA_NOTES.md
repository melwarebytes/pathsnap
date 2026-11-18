# EDA notes

Produced by `python -m src.eda` (about 50 s). Partitions: `train` (80 trips) and `student_test` (10 trips) only. No test partition is read.
Evidence files: `artifacts/results/eda_inventory.csv` (one row per trip), `artifacts/results/eda_summary.csv` (per partition: `train`, `train_clean` = train without 00000033/00000054, `student_test`), plots in `artifacts/plots/eda_*.png`, maps in `artifacts/maps/`.

## Partition used per analysis

| Analysis | Partition |
|---|---|
| Inventory, missing values, gaps, speed, sampling, network structure, GPS-to-route distance, origins, network vs track | train (all 80) and student_test, overlaid and coloured by partition |
| Candidate histogram, recall vs radius, candidate miss rate | cached candidate tables: train **without** 00000033 and 00000054 (78 trips), and student_test |
| Typical-trip plot and map | train 00000006 (1,450 points; closest to the train median of 1,529) |
| Bad-GT plot | train 00000033 |
| Second map | student_test 00000007 (1,354 points; closest to the student_test median) |

## Missing and invalid values (eda_inventory.csv, eda_summary.csv, eda_missing_invalid.png)

- NaN or out-of-range lon/lat in tracks or nodes: 0 in both partitions. Arc or route ids out of range: 0. Every trip has a `.route` file. No missing-value imputation is needed.
- Non-increasing timestamps: 0 (train and student_test).
- Isolated nodes (degree 0): 0.
- Sampling gaps > 10 s: 590 in train (7.4 per trip, 40 of 80 trips), 27 in student_test (2.7 per trip). Largest gaps: 153,064 s (trip 00000090) and 52,810 s (trip 00000078); student_test max is 210 s. The HMM must handle gaps (trips 78 and 90 are essentially several trips glued together).
- "Missing" in the sense of labels: 11.1 % of train points (`off50_frac_pooled`) have no label because the aligned arc is more than 50 m away, but only 1.4 % once the two bad-GT trips are removed. student_test: 0.14 %.

## Data-quality issues and decisions

1. **Bad ground truth, trips 00000033 and 00000054.** Median distance to the nearest route arc is 122 m (59 % of points over 50 m) and 139 m (87 %); every other train trip has a median under 12 m. For 00000033 (`eda_traj_bad_gt.png`) most of the points that are far from the route are dwell clusters away from the route (the track length is 47.8 km against a 6.4 km route, so the track is mostly jitter, not travel), so the route does not describe what the receiver recorded. Decision: excluded from training, CV and the candidate cache, reported separately (`EXCLUDED_TRIPS` in `src/data.py`).
2. **Off-network segment, trip 00000002**: 12 % of its points are over 50 m from the label (inventory `off50_frac`). Kept, with those points left unlabelled. Trips 66 (8 %), 43 (7 %) and 96 (4.4 %) are the next worst.
3. **Candidate coverage.** Cached candidate tables contain 183,152 of 183,260 clean train points (108 points have no arc within 200 m, so the HMM has to skip them). 0.85 % of candidate points come from the 200 m fallback in train, 0 % in student_test. Candidate miss rate (label arc not among the K=16 nearest within 50 m): 0.106 % train, 0.031 % student_test. Recall at 25 m: 98.1 % / 98.7 %, at 50 m: 99.89 % / 99.97 % (`eda_candidates.png`, `eda_summary.csv`). Median candidates per point: 6 (train), 8 (student_test); p95 = 16 (the cap).
4. **Network structure.** The networks are not connected: 79 of 80 train trips and 10 of 10 student_test trips have more than one weakly connected component (median 19 components per trip). The largest component holds a median 98.1 % (train) and 97.1 % (student_test) of nodes. Decision: no repair; shortest-path gap filling must tolerate unreachable node pairs.
5. **Route contiguity.** Mean over trips 99.93 % of consecutive route arcs are joined (`route_contig_frac`), min 98.1 % (trip 75). 9 of 78 clean train trips and 1 of 10 student_test trips contain at least one break. 70/80 train trips are fully contiguous (10 trips with a break: 2, 19, 33, 41, 42, 69, 75, 80, 86, 89). Treat the breaks as ground-truth gaps, not errors.
6. **Speed and geography.** Median trip speed is 12.7 m/s in train (range 5.7 to 41.4 m/s per-trip median) and 15.0 m/s in student_test (8.5 to 22.1). Trips are spread worldwide (`eda_origins.png`), each with its own network, which is why no absolute coordinates are used as features.
7. **Sampling.** 1 Hz for most steps (`eda_sampling_interval.png`); points per trip 361 to 19,313 (train), 262 to 3,958 (student_test).
8. **Label noise level.** Pooled median GPS-to-aligned-arc distance is 4.6 m (train median over trips), p90 11.8 m; student_test 4.7 m and 12.5 m (`eda_gps_to_route_dist.png`). Student_test is clean, which makes it a fair held-out check.
9. **Network size vs track.** Arcs per trip 649 to about 307k, median 19.5k (train) and 13.1k (student_test); points per trip and network size are plotted in `eda_network_vs_track.png`.

## Assumptions

- Per-trip equirectangular projection (from `src/data.py`) is used for all distances in metres.
- GPS-to-route distance = distance to the arc picked by the monotonic DP alignment (`align_labels`), measured on all points before the 50 m cutoff. For clean trips this equals the nearest-route-arc distance to float precision; the same holds for 00000033 (median 122.4 m both ways).
- "Gap" = step with dt > 10 s. Non-monotonic = dt <= 0. Speeds use only steps with 0 < dt <= 10 s.
- Maps subsample GPS points to about 1,200 per trip and keep each road once (reverse twins merged); both HTML files are under 2 MB (`artifacts/maps/eda_00000006.html` 1.9 MB, `eda_00000007.html` 1.7 MB).
