"""Build docs/PathSnap_Report.docx from the committed results. Run: python docs/make_report_docx.py
The draft holds the facts, tables and figures; the team writes the analysis in their own words."""
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RES, PLOTS, SHOTS = ROOT / "artifacts/results", ROOT / "artifacts/plots", ROOT / "docs/screenshots"

d = Document()
d.styles["Normal"].font.name = "Calibri"
d.styles["Normal"].font.size = Pt(10.5)


def h(text, level=1): d.add_heading(text, level)
def p(text): d.add_paragraph(text)
def bullets(*items):
    for i in items: d.add_paragraph(i, style="List Bullet")
def todo(text):
    r = d.add_paragraph().add_run("[TEAM TO WRITE] " + text)
    r.italic = True; r.font.color.rgb = RGBColor(0xC0, 0x30, 0x30)
def fig(path, caption, width=6.0):
    if Path(path).exists():
        d.add_picture(str(path), width=Inches(width))
        c = d.add_paragraph(caption); c.runs[0].italic = True
def table(df, fmt="{:.3f}"):
    t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row): cells[i].text = fmt.format(v) if isinstance(v, float) else str(v)
    d.add_paragraph()


ms = pd.read_csv(RES / "model_summary.csv").set_index("model")
rm = pd.read_csv(RES / "classifier_route_metrics.csv").set_index("model")
hpo = pd.read_csv(RES / "hpo_comparison.csv").set_index("model")
hs_all = pd.read_csv(RES / "hmm_summary.csv")
hs = hs_all[hs_all.split == "student_test"].set_index("mode")
he = hs_all[hs_all.split == "evaluator_test"].set_index("mode")
pms = pd.read_csv(RES / "point_metrics_by_split.csv", index_col=0)
eda = pd.read_csv(RES / "eda_summary.csv")

# ------------------------------------------------------------------ title
d.add_heading("PathSnap: Map Matching and Route Reconstruction", 0)
p("Technical report.  Repository: https://github.com/melwarebytes/pathsnap")
p("Team members: [names, SRNs].")

# ------------------------------------------------------------------ 1
h("1. Problem and approach")
bullets(
    "Input per trip: noisy GPS track (lon, lat, time) plus that trip's directed road graph (nodes, arcs). "
    "Output: the arc each GPS point was driven on and the connected route (ordered arcs).",
    "Formulated as binary classification over candidate arcs: for every GPS point, arcs within 50 m "
    "(max 16; fallback 4 nearest within 200 m) are candidates; the classifier scores each one.",
    "HMM/Viterbi then picks one candidate per point so that consecutive choices are connected on the road "
    "graph; shortest paths fill the gaps to give a continuous route.",
    "Primary metric: point-level Top-1 accuracy (share of GPS points whose predicted arc is correct). "
    "Also macro F1 and log loss over candidates; route precision/recall/F1 (length-weighted) and continuity.",
)

# ------------------------------------------------------------------ 2
h("2. Dataset understanding and EDA")
p("Partitions: train (80 trips; 78 used for training, trips 00000033 and 00000054 excluded) and "
  "student_test (10 trips, used only for final reporting). evaluator_test is never used for tuning. "
  "Code: src/data.py, src/eda.py. Tables: artifacts/results/eda_summary.csv, eda_inventory.csv.")
table(eda[["partition", "n_trips", "total_points", "points_median", "points_min", "points_max",
           "duration_s_median", "nodes_median", "arcs_median", "route_len_median"]], "{:.1f}")

h("2.1 Preprocessing", 2)
bullets(
    "Files are whitespace-separated; read with pandas. Coordinates projected to local metres "
    "(equirectangular around the trip's first point) so distances and bearings are in metres/degrees.",
    "Missing / invalid values: no NaN or invalid coordinates, no bad ids and no non-monotonic timestamps in any partition (figure below). Duplicate timestamps (dt <= 0) give "
    "speed 0; speed capped at 70 m/s to remove GPS jumps.",
    "Label alignment: each GPS point is aligned to the ground-truth route with a monotonic dynamic programme "
    "(points move forward along the route). Cost = distance to arc + heading term 2(1 - cos(heading - arc "
    "bearing)) when speed >= 1 m/s, so on U-turns the return leg gets the reverse-direction arc. Points "
    "farther than 50 m from the route are unlabelled. The heading term cut wrong-direction labels from "
    "5,380 to 5,138.",
    "Excluded trips: 00000033 (long dwell clusters far from the route: bad ground truth) and 00000054.",
    "Heavy-tailed features (time gaps up to 153,064 s, arcs up to 9 km) are log1p-scaled and standardised "
    "for LR and MLP only (src/preprocess.py); tree models use raw values.",
)
fig(PLOTS / "eda_missing_invalid.png", "Missing / invalid values per partition (train, student_test).")
fig(PLOTS / "eda_points_per_trip.png", "GPS points per trip (train vs student_test).", 5)
fig(PLOTS / "eda_sampling_interval.png", "Sampling interval distribution.", 5)
fig(PLOTS / "eda_speed.png", "Speed distribution.", 5)
fig(PLOTS / "eda_gps_to_route_dist.png", "Distance from GPS point to the ground-truth route.", 5)

h("2.2 Patterns observed", 2)
bullets(
    "Trips are worldwide, each with its own road network (figure: trip origins).",
    "Most GPS points lie within a few metres of the route (median 4.6 m, 90th percentile 11.8 m on train), but there is a long tail; "
    "a 50 m candidate radius misses the true arc for only 0.026% of student_test points.",
    "Every two-way road is two arcs with opposite direction ('reverse twins'). Distance alone cannot separate "
    "them, which is why heading features matter: the nearest-arc baseline gets only 55.1% (train) / 53.7% "
    "(student_test) Top-1; nearest arc + heading rule gets 83.1% / 84.2%.",
    "Time gaps over 10 s: 7.4 per trip in train vs 2.7 in student_test; the largest train gap is 153,060 s "
    "(42 h), student_test max 210 s. Median speed about 12.7 m/s (train) and 15.0 m/s (student_test).",
    "11.1% of train points are more than 50 m from the route (1.4% after excluding trips 33 and 54); 0.1% in student_test.",
)
fig(PLOTS / "eda_origins.png", "Trip origins (lon/lat) by partition.", 5.5)
fig(PLOTS / "eda_network_vs_track.png", "Road-network size vs track length.", 5)
fig(PLOTS / "eda_candidates.png", "Candidates per point and coverage of the true arc vs radius.")
fig(PLOTS / "eda_traj_typical.png", "GPS trajectory over road network: typical train trip.", 4.5)
fig(PLOTS / "eda_traj_bad_gt.png", "GPS trajectory with bad ground truth (trip 00000033, excluded).", 4.5)
fig(SHOTS / "eda_folium_00000007.png", "Folium EDA map, student_test trip 00000007: GPS observations "
    "over the trip's road network and the reference route. Interactive version: "
    "artifacts/maps/eda_00000007.html (generated by src/eda.py).")

h("2.3 Features (src/features.py)", 2)
feats = [
    ("dist, dist_rank, dist_gap, dist_ratio", "Distance from point to candidate arc and how it compares to the nearest candidate."),
    ("proj_frac, edge_proj", "Where along the arc the point projects; whether it falls beyond an end."),
    ("arc_len", "Arc length (long arcs attract more points)."),
    ("bearing_diff, cos_bearing, heading_ok", "Angle between vehicle heading and arc direction: separates reverse twins."),
    ("speed, dt_prev, dt_next", "Kinematics; heading is unreliable at low speed and across gaps."),
    ("n_cand, has_twin", "How crowded the neighbourhood is; whether the reverse arc is also a candidate."),
    ("out_deg, in_deg", "Graph degree of the arc's end nodes (junction vs plain road)."),
    ("prev_link, next_link", "Whether the arc connects to a candidate of the previous/next point."),
    ("fallback", "Candidate came from the 200 m fallback (no arc within 50 m)."),
]
table(pd.DataFrame(feats, columns=["Feature(s)", "Why"]))
p("Why these features. The EDA showed that almost every GPS point has several arcs within a few metres, "
  "and that the nearest arc is right only about half the time (53.7% on student_test). The reason is that "
  "every two-way road appears twice, once per direction, at exactly the same distance. Distance features "
  "therefore cannot decide alone; they are expressed relative to the nearest candidate (rank, gap, ratio) "
  "so that the model learns 'closer than the alternatives' rather than an absolute threshold that changes "
  "with GPS quality from city to city. The direction features (bearing difference, its cosine and "
  "heading_ok) are what separate a road from its reverse twin: adding only a heading rule to the "
  "nearest-arc baseline already lifts Top-1 to 84.2%. Heading is unreliable when the vehicle is slow or "
  "after a time gap, so speed and the gaps to the previous and next point let the model decide how much "
  "to trust the heading. Projection features catch points that fall past the end of an arc, which happens "
  "at junctions, and the degree and has_twin features tell the model whether it is in a complicated area. "
  "prev_link and next_link add a little sequence information (whether the arc connects to the neighbouring "
  "points' candidates) before the HMM. All features use only the GPS track and the road graph; the "
  "ground-truth route is never read at inference.")

# ------------------------------------------------------------------ 3
h("3. Model experimentation and comparison")
p("Validation: 5-fold GroupKFold by trip on the 78 clean train trips (no trip in both train and validation). "
  "student_test is scored once with a model refit on all train trips. Code: src/train.py, src/metrics.py.")
h("3.1 Overall model comparison", 2)
cfg = {"nb": "Gaussian, default", "lr": "balanced class weights, log1p + scaler, max_iter 1000",
       "rf": "300 trees, min_samples_leaf 20", "xgb": "400 trees, depth 6, lr 0.1, hist",
       "lgbm": "400 trees, 63 leaves, lr 0.05", "mlp": "(64, 32), early stopping, log1p + scaler"}
names = {"nb": "Naive Bayes", "lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
         "lgbm": "LightGBM", "mlp": "MLP"}
rows = []
for k in ["nb", "lr", "rf", "xgb", "lgbm", "mlp"]:
    a = ms.loc[k]
    rows.append([names[k], a.family, cfg[k], a.cv_macro_f1_mean, a.cv_top1_mean, a.st_top1,
                 rm.loc[k].continuity, a.st_macro_f1, a.st_log_loss, f"{a.infer_ms_per_trip:.1f} ms/trip"])
t = hpo.loc["tuned"]
rows.append(["XGBoost tuned (final)", "boosting (trees)", "Optuna TPE, 15 trials", t.cv_macro_f1, t.cv_top1,
             t.st_top1, hs.loc["classifier"].continuity, t.st_macro_f1, t.st_log_loss,
             f"{t.infer_ms_per_trip:.1f} ms/trip"])
table(pd.DataFrame(rows, columns=["Model", "Algorithm family", "Key configuration", "CV macro F1",
                                  "CV Top-1", "Student-test Top-1", "Route continuity",
                                  "Student-test macro F1", "Log loss", "Inference time"]))
p("Route continuity here is classifier-only (argmax per point, then shortest-path gap filling). With HMM "
  "decoding the final model reaches continuity 1.000 (section 5). Log loss applies to all six models since "
  "all output probabilities. Inference time = classifier scoring per trip (feature building excluded).")
p("Selection: best mean CV Top-1 with log loss as tie-break. RF leads XGBoost by 0.0005, far inside the fold "
  "standard deviation (about 0.026), so it is a tie. RF was rejected on practical grounds: 365 MB model "
  "file, 142 s training, 58 ms/trip inference, versus XGBoost 1.6 MB, 7 s, 10 ms/trip. XGBoost was tuned.")

h("3.2 Per-model results", 2)
fit_gap = {
    "nb": "Train-fold Top-1 0.834 vs validation 0.815: underfits. Correlated distance features break the "
          "independence assumption; log loss 0.74 shows badly calibrated probabilities.",
    "lr": "0.865 vs 0.853: small gap, mild underfitting. Linear model cannot represent interactions such "
          "as bearing x heading_ok x speed.",
    "rf": "0.895 vs 0.855: moderate gap; min_samples_leaf 20 limits overfitting.",
    "xgb": "0.916 vs 0.855: largest gap (mild overfitting), yet the best validation scores.",
    "lgbm": "0.906 vs 0.850: mild overfitting; 63 leaves is a fairly deep model.",
    "mlp": "0.888 vs 0.843: gap present; early stopping on a validation split (see loss curve).",
}
WHY = {
    "nb": "A fast probabilistic baseline that outputs probabilities directly, which the HMM needs. It shows "
          "how far a model gets when the features are treated as independent.",
    "lr": "A linear, well-calibrated baseline. If one weighted sum of distance and heading terms were enough, "
          "LR would match the tree models, so it tells us how much non-linearity the problem needs.",
    "rf": "Tree ensembles handle mixed-scale features (metres, degrees, seconds, counts) without scaling and "
          "learn interactions such as 'trust heading only when speed is high'. Bagging is robust to noisy labels.",
    "xgb": "Gradient boosting is usually the strongest model on tabular data of this size (about 1.5 million "
           "candidate rows); it learns interactions, handles heavy-tailed features and is fast to train and run.",
    "lgbm": "A second boosting implementation with leaf-wise growth and histogram splits, included to check "
            "whether the XGBoost result depends on the library.",
    "mlp": "A neural network learns smooth non-linear combinations of the features and has a different "
           "inductive bias from trees; it is the only model with an epoch-wise training history.",
}
PROS = {
    "nb": "Strength: trains in 0.2 s and predicts in 2 ms per trip. Limitation: the distance features are "
          "strongly correlated (dist, dist_gap, dist_ratio), so the independence assumption double-counts "
          "evidence; probabilities are over-confident (log loss 0.739, about nine times the boosting models) "
          "and Top-1 is the lowest (0.803 on student_test). It underfits.",
    "lr": "Strength: stable, fast (2.9 s training) and reasonably calibrated (log loss 0.183). Limitation: "
          "it cannot represent interactions such as bearing difference mattering only at speed, so it stays "
          "at 0.853 CV Top-1 with train-fold Top-1 of only 0.865: underfitting rather than overfitting.",
    "rf": "Strength: tied for the best CV Top-1 (0.855) and the best student_test log loss (0.078). "
          "Limitation: the saved model is 365 MB (too large for the repository), training takes 142 s and "
          "inference 57.6 ms per trip, about 6 times slower than XGBoost for no measurable accuracy gain.",
    "xgb": "Strength: same accuracy as RF (CV Top-1 0.855, student_test 0.861) with a 1.6 MB model, 6.9 s "
           "training and 10 ms per trip, and good calibration (log loss 0.080). Limitation: the largest gap "
           "between train-fold and validation Top-1 (0.916 vs 0.855), so it fits training trips more closely "
           "than the others; like every per-point model it still confuses reverse twins, about 80% of its "
           "errors. Chosen as the final model and tuned.",
    "lgbm": "Strength: close to XGBoost (CV Top-1 0.850). Limitation: slightly worse on every metric, slower "
            "inference (49 ms per trip) and the worst classifier-only route F1 (0.791), so it was not chosen.",
    "mlp": "Strength: fastest inference of the strong models (5.3 ms per trip) and good calibration (0.087). "
           "Limitation: needs log transforms and scaling, trains on a subsample, and is the weakest non-NB "
           "model (CV Top-1 0.843); the validation score flattens after a few epochs (loss curve), so more "
           "epochs would not help.",
}
for k in ["nb", "lr", "rf", "xgb", "lgbm", "mlp"]:
    a = ms.loc[k]
    h(f"{names[k]}", 3)
    bullets(
        f"Family / configuration: {a.family}; {cfg[k]}.",
        "Loss / objective: binary cross-entropy (log loss)" + (" via maximum likelihood of Gaussian "
        "class-conditional densities" if k == "nb" else "") + (" (Gini impurity for tree splits)" if k == "rf" else "") + ".",
        f"CV (5 folds): Top-1 {a.cv_top1_mean:.3f} ± {a.cv_top1_std:.3f}, macro F1 {a.cv_macro_f1_mean:.3f}, "
        f"log loss {a.cv_log_loss_mean:.3f}.",
        f"Student-test: Top-1 {a.st_top1:.3f}, macro F1 {a.st_macro_f1:.3f}, log loss {a.st_log_loss:.3f}; "
        f"route F1 (classifier only) {rm.loc[k].route_f1:.3f}, continuity {rm.loc[k].continuity:.3f}.",
        f"Over/underfitting: {fit_gap[k]} Student-test is within about 1 point of CV, so no sign of CV overfitting.",
        f"Training time {a.train_s:.1f} s; inference {a.infer_ms_per_trip:.1f} ms/trip.",
    )
    d.add_paragraph().add_run("Why we considered it: ").bold = True
    d.paragraphs[-1].add_run(WHY[k])
    d.add_paragraph().add_run("Strengths and limitations on this task: ").bold = True
    d.paragraphs[-1].add_run(PROS[k])
    fig(PLOTS / f"confusion_{k}.png", f"Confusion matrix, {names[k]} (student_test, candidate level).", 3.2)
    if k == "mlp":
        fig(PLOTS / "mlp_loss_curve.png", "MLP training loss and validation score per epoch.", 4.5)

h("3.3 Comparison plots", 2)
fig(PLOTS / "cv_macro_f1.png", "CV macro F1 across models (5 folds).")
fig(PLOTS / "cv_top1.png", "CV Top-1 across models.")
fig(PLOTS / "student_test_top1_f1.png", "Student-test Top-1 and macro F1.")
fig(PLOTS / "log_loss.png", "Log loss comparison.")
fig(PLOTS / "timing.png", "Training and inference time.")
p("Only the MLP produces an epoch-wise training history; no training curves are shown for the other models.")

# ------------------------------------------------------------------ 4
h("4. Bayesian hyperparameter optimisation (XGBoost)")
hs_ = pd.read_csv(ROOT / "artifacts/hpo/hpo_summary.csv").iloc[0]
bullets(
    "1. Searched: n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, "
    "reg_lambda, reg_alpha.",
    "2. Ranges: n_estimators 200-1500; learning_rate 0.01-0.3 (log); max_depth 3-10; min_child_weight 1-20 "
    "(log); subsample and colsample_bytree 0.5-1; reg_lambda and reg_alpha 1e-3-10 (log).",
    "3. Objective: maximise mean 5-fold grouped-CV Top-1. Optuna TPE sampler (seed 42), MedianPruner with 2 "
    "warm-up folds. student_test never used.",
    f"4. Trials: {int(hs_.trials)} ({int(hs_.complete)} complete, {int(hs_.pruned)} pruned), "
    f"{hs_.wall_s:.0f} s total.",
    f"5. Best: n_estimators {int(hs_.best_n_estimators)}, learning_rate {hs_.best_learning_rate:.3f}, "
    f"max_depth {int(hs_.best_max_depth)}, min_child_weight {hs_.best_min_child_weight:.2f}, subsample "
    f"{hs_.best_subsample:.2f}, colsample_bytree {hs_.best_colsample_bytree:.2f}, reg_lambda "
    f"{hs_.best_reg_lambda:.4f}, reg_alpha {hs_.best_reg_alpha:.2f}.",
    "6. Tuned beat untuned on every metric, but by less than the fold standard deviation (0.026): a small, "
    "consistent gain.",
)
hp = hpo.reset_index()[["model", "cv_macro_f1", "cv_top1", "st_top1", "st_macro_f1", "st_log_loss",
                        "infer_ms_per_trip"]]
hp.columns = ["XGBoost", "CV macro F1", "CV Top-1", "Student-test Top-1", "Student-test macro F1",
              "Log loss", "Inference ms/trip"]
table(hp, "{:.4f}")
fig(PLOTS / "hpo_history.png", "Optuna optimisation history.", 5.5)
fig(PLOTS / "hpo_param_importance.png", "Hyperparameter importance.", 5.5)
fig(PLOTS / "hpo_tuned_vs_untuned.png", "Tuned vs untuned XGBoost.", 5.5)

# ------------------------------------------------------------------ 5
h("5. HMM and Viterbi route reconstruction (src/hmm.py, src/predict.py)")
bullets(
    "1. Hidden states: the candidate arcs of each GPS point (up to 16).",
    "2. Emission: lam * log p(classifier) + (1 - lam) * (-0.5 (d / sigma)^2), sigma = 6.7 m, lam = 1 "
    "(lam = 0 gives the Gaussian 'no ML' HMM baseline).",
    "3. Transition: -|road distance - straight-line distance| / beta, beta = 5. Road distance from Dijkstra "
    "(scipy) on the trip graph; parallel duplicate arcs deduplicated to the shortest; U-turn onto the reverse "
    "twin penalised (3).",
    "4. Missing candidates / large gaps: fallback candidates within 200 m; beta scaled up with dt when "
    "dt > 10 s; the chain is broken and restarted when dt > 120 s.",
    "5. Disconnected segments: if no candidate pair is reachable, transition is -inf; Viterbi breaks the chain "
    "and restarts from emissions. Route reconstruction joins consecutive arcs with shortest paths.",
    "6. HMM effect: see table. Route F1 rises from 0.847 to 0.985 and continuity to 1.000.",
)
hsum = hs.reset_index()[["mode", "top1", "route_precision", "route_recall", "route_f1", "continuity",
                         "n_breaks", "ms_total"]]
hsum["mode"] = hsum["mode"].map({"classifier": "Classifier only (tuned XGBoost)",
                                 "hmm": "Classifier + HMM/Viterbi", "gauss_hmm": "Gaussian HMM, no ML (lam=0)"})
table(hsum)
p("Student-test, per-trip means (artifacts/results/hmm_summary.csv; per trip: hmm_comparison.csv).")
fig(PLOTS / "hmm_vs_classifier.png", "Classifier-only vs classifier + HMM.", 5.5)
fig(PLOTS / "hmm_per_trip_delta.png", "Per-trip change from HMM decoding.", 5.5)
fig(PLOTS / "hmm_beta.png", "Transition scale beta sweep (train CV).", 4.5)
fig(PLOTS / "hmm_lam.png", "Emission weight lam sweep (train CV).", 4.5)
bullets(
    "Where HMM did not help: the Gaussian HMM without any classifier is only about 0.7 points Top-1 behind, "
    "so most of the gain comes from the road-graph constraint. Remaining errors are mostly at reverse twins "
    "(about 80% of classifier errors).",
    "Known limitation: about 4,900 labelled moving points have a labelled arc pointing against the heading "
    "(trips 84, 21, 2, 82, 10): likely ground-truth direction issues; these cap achievable Top-1.",
)
p("Discussion. HMM decoding improved every summary metric on student_test: Top-1 from 0.883 to 0.949, "
  "route precision from 0.747 to 0.975, and continuity from 98.6% to 100% (breaks per trip from 23.8 to 0.1). "
  "The classifier decides each point alone, so on a two-way road it sometimes jumps to the reverse arc or "
  "to a parallel road for a few points; each jump adds wrong arcs to the route (low precision) and breaks "
  "continuity. The transition term makes those jumps expensive because the road distance needed for them "
  "is much longer than the distance the vehicle moved. Recall was already 0.997 without the HMM and did not "
  "change: the classifier rarely misses the true road, it adds extra wrong ones.")
p("Where it did not help. (1) The Gaussian HMM that ignores the classifier is only 0.7 Top-1 points behind, "
  "so most of the gain comes from the road-graph constraint and the classifier adds a smaller improvement. "
  "(2) On some trips the classifier + HMM is worse at point level than the Gaussian HMM (evaluator_test "
  "trip 00000015: 0.926 vs 0.960) because confident wrong classifier probabilities pull the path. The lam "
  "sweep on train CV agrees: lam = 0.5 gave Top-1 0.951 against 0.948 for lam = 1. (3) Decoding costs about "
  "0.7 s per trip against 0.04 s for the classifier alone, which is still acceptable for the demo.")
p("Improvements we would make: tune lam and beta jointly on out-of-fold classifier probabilities "
  "(lam = 0.5 looks better), clean the ground-truth direction errors before training, and let the "
  "transition model use elapsed time as well as distance.")
h("5.1 Generalisation to evaluator_test", 2)
p("The final model was run once on the 10 evaluator_test trips after all choices were fixed; nothing was "
  "tuned on them. Point level (pooled over points): training in-sample Top-1 "
  f"{pms.iloc[0].top1:.3f}, cross-validation {hpo.loc['tuned'].cv_top1:.3f}, student_test "
  f"{pms.iloc[1].top1:.3f}, evaluator_test {pms.iloc[2].top1:.3f} (macro F1 {pms.iloc[2].macro_f1:.3f}, "
  f"log loss {pms.iloc[2].log_loss:.3f}).")
ev = he.reset_index()[["mode", "top1", "route_precision", "route_recall", "route_f1", "continuity",
                       "n_breaks", "ms_total"]]
ev["mode"] = ev["mode"].map({"classifier": "Classifier only", "hmm": "Classifier + HMM/Viterbi",
                             "gauss_hmm": "Gaussian HMM, no ML"})
table(ev)
p("With HMM decoding the system reaches 0.957 Top-1, 0.988 route F1 and 100% continuity on evaluator_test "
  "(per-trip means), slightly above student_test, so the pipeline generalises. The classifier alone drops "
  "to 0.803 pooled Top-1, mostly because of one long, hard trip (00000015, classifier Top-1 0.589), which "
  "the HMM recovers to 0.926.")

# ------------------------------------------------------------------ 6
h("6. Geographical visualisation (Folium, src/maps.py)")
p("Layers: road network (grey), GPS observations (blue points), reference route (green), predicted route "
  "(red), optional classifier-only route (dashed); legend and layer control included. Interactive HTML maps "
  "committed in artifacts/maps/ (student_test 00000007, 00000009, 00000052, 00000077; train 00000078).")
fig(SHOTS / "api_folium_map_00000007.png", "Folium map for student_test trip 00000007 (served by the API).")

# ------------------------------------------------------------------ 7
h("7. API, UI and deployment")
bullets(
    "API (Laptop 1): FastAPI + uvicorn, api/server.py. Endpoints GET /health, GET /trips, POST /predict "
    "{trip_id}, GET /map/{trip_id}. Model loaded once from artifacts/models/final.joblib. Ground truth used "
    "only server-side for metrics.",
    "UI (Laptop 2): static ui/index.html with Leaflet, served by python -m http.server; calls the API over "
    "the network (?api=http://<LAPTOP1_IP>:8000).",
    "Setup: pip install -r requirements.txt; full steps in README.md and docs/DEPLOYMENT.md.",
)
fig(SHOTS / "ui_connected.png", "UI connected to the API, trip list loaded.")
fig(SHOTS / "ui_predict_00000007.png", "Prediction for trip 00000007: Top-1 0.995, route F1 0.993, continuity 100%.")
fig(SHOTS / "ui_predict_00000052.png", "Prediction for trip 00000052: Top-1 1.000, route F1 0.998.")
fig(SHOTS / "api_cmd_requests.png", "API terminal on the serving laptop logging requests (all 200).")
p("Screenshots from the deployment dry run: the UI calls the API through the laptop's network address "
  "(not localhost), and the API terminal logs every request with its response code and latency.")

# ------------------------------------------------------------------ 8
h("8. Academic integrity and AI disclosure")
p("AI assistance disclosure. The team used Claude (Anthropic), an AI coding assistant, throughout the "
  "project. It was used to: read the hackathon documents and propose the overall approach (candidate "
  "classification followed by HMM decoding) and the work split; write and debug large parts of the code "
  "(data loading, feature extraction, training and evaluation scripts, HMM, API, UI and the notebook); "
  "review team members' pull requests; run our scripts to produce the result tables and plots; and draft "
  "this report, including its written explanations. All results in this report come from running the code "
  "in the repository on the provided data, and every number can be regenerated with the commands in the "
  "appendix. Team members reviewed the code and results, made the final decisions, and can explain every "
  "part of the submission.")
todo("Check that this disclosure is accurate for your team and matches the event's AI-use policy. "
     "Fill in team names and SRNs at the top.")

h("Appendix: reproducing results")
for c in ["python -m src.features train", "python -m src.features student_test", "python -m src.eda",
          "python -m src.train", "python -m src.hpo --model xgb --trials 15",
          "python -m src.evaluate --split student_test", "Notebook: PathSnap.ipynb (runs everything end to end)"]:
    d.add_paragraph(c, style="List Number")

out = ROOT / "docs" / "PathSnap_Report.docx"
d.save(out)
print(out)
