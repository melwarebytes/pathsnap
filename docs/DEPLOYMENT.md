# Deployment and demo guide (two laptops)

This guide covers the demo setup: **Laptop 1** runs the model API, **Laptop 2** runs only the
browser UI. It also contains the dry-run procedure, the demo-day runbook, the verification
checklist and the screenshot list for report section 7.

The API contract is fixed in `docs/PLAN.md` section 6.6. Commands below assume the repository is
cloned as `codebase/` and that you run everything from inside it.

> The repository is **private**. Cloning requires an accepted GitHub invitation for an account that
> has read access. Do this before demo day, not on the day.

---

## 1. Architecture

Laptop 1 runs `uvicorn` and the whole pipeline: it builds candidates for the requested trip, scores
them with the trained classifier, decodes the arc sequence with the HMM/Viterbi decoder, reconstructs
the route on that trip's road network, and finally scores the result against the ground-truth
`.route` file. Laptop 2 runs only the static `ui/index.html` page in a browser; it never sees the
model or the ground truth, and it exchanges nothing with Laptop 1 except HTTP (JSON, plus folium HTML
for `/map`) over the hotspot. Ground truth is opened **only inside `score()`**, after `predict_trip`
has finished, and only
five scalar numbers (top-1, route F1, precision, recall, continuity) leave that function.

```
Laptop 1 (model + API)                         Laptop 2 (browser only)
+----------------------------------+           +-----------------------------+
| uvicorn api.server:app           |  hotspot  | ui/index.html               |
|  /trips   -> trip ids            |<--------->|  trip buttons               |
|  /predict -> candidates          |  HTTP     |  Leaflet map (GPS + route)  |
|              classifier          |  JSON     |  metrics card               |
|              Viterbi + route     |           |  API URL field              |
|              score() <- GT only  |           +-----------------------------+
|  /map/{id} -> folium HTML        |
|  /health                         |
+----------------------------------+
        |
        +-- data: student_test/ or evaluator_test/ (Laptop 1 only)
        +-- artifacts/models/final.joblib
```

Per request the server prints one line to its terminal:

```
[HH:MM:SS] <client_ip>  POST /predict  <trip_id>  200  <latency> ms
```

That terminal line is part of the demo evidence: it shows the request came from Laptop 2 and was
served live.

---

## 2. What you need

| Item | Notes |
|---|---|
| Repository | Private; clone as `codebase/`. Needs GitHub access. |
| Python | 3.10 or newer, 64-bit. |
| Python packages | `pip install -r requirements.txt` (numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, optuna, folium, fastapi, uvicorn, matplotlib, seaborn, joblib, networkx, pytest). |
| Data folder | One folder containing the splits; default is the parent of the repo folder. |
| `student_test/` | 10 trips, used for the dry run. |
| `evaluator_test/` | 10 trips, handed over on demo day, **Laptop 1 only**. |
| Model | `artifacts/models/final.joblib` (committed, under 50 MB). |
| Vendored Leaflet | `ui/vendor/leaflet.js`, `ui/vendor/leaflet.css` and `ui/vendor/images/` (offline fallback). |
| UI page | `ui/index.html` with an API URL field, Connect button, trip list, Leaflet map, metrics card, footer status and visible error text. |
| Laptop 2 | Any modern browser (Edge/Chrome), plus a copy of `ui/` (clone the repo or copy the folder; no venv needed). Python is only needed for the optional `http.server` route. |

Data layout (the folder that contains `codebase/`, or the folder named by `PATHSNAP_DATA`):

```
Hackathon/
  train/<id>/<id>.track              # optional on the demo laptop
  student_test/<id>/<id>.{track,nodes,arcs,route}
  evaluator_test/<id>/<id>.{track,nodes,arcs,route}    # demo day, Laptop 1 only
  codebase/
```

Each trip is a folder named by its zero-padded id (`00000007`) containing four whitespace-separated
text files with the same stem: `.track` (lon, lat, time), `.nodes` (lon, lat), `.arcs` (from, to),
`.route` (true arc sequence). If `.route` is missing for a trip, the route still displays but its
metrics are `null` in the API and shown as "n/a (no reference route)" in the UI.

**Expected demo count: exactly 10 evaluator trips.** Verify it; do not assume it.

**Readiness check:** the steps below assume the committed `api/server.py` and the real
`ui/index.html` (API URL field, trip list, map, metrics card). If opening the page shows a
placeholder, the UI work is not finished — stop and report that before debugging the network.

---

## 3. Setup on Laptop 1

### 3.1 Clone and install

Windows (PowerShell):

```powershell
git clone <REPO_URL> codebase
cd codebase
py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
git clone <REPO_URL> codebase
cd codebase
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, use `cmd` instead:
`.venv\Scripts\activate.bat`.

### 3.2 Verify the environment

```bash
python -c "import numpy, pandas, scipy, sklearn, lightgbm, joblib, fastapi, uvicorn, folium; print('imports ok')"
```

If `lightgbm` fails to import on Windows, reinstall it (`pip install --force-reinstall lightgbm`)
and re-run the check. The model artifact needs it at load time.

### 3.3 Point at the data (only if it is not in the default place)

The default `DATA_ROOT` is the folder that contains the repo (so `codebase/..` when cloned as
`codebase/`). If the data lives elsewhere — for example you cloned into `C:\dev\codebase` but the data
is on `D:` — set the variable in the same terminal that will start the API:

PowerShell:

```powershell
$env:PATHSNAP_DATA = "D:\pathsnap_data"
```

cmd:

```bat
set PATHSNAP_DATA=D:\pathsnap_data
```

bash:

```bash
export PATHSNAP_DATA=/path/to/data
```

These variables are read when the server module is imported, so **restart uvicorn after changing any
`PATHSNAP_*` variable**.

### 3.4 Optional: build the student_test candidate cache

Only needed if a script you run reads `cache/student_test_candidates.pkl` (for example
`python -m src.evaluate`). Serving a prediction does not use these dataset caches (and
`evaluator_test` must never have one), so skip this step unless a script needs it.

```bash
python -m src.features student_test
```

Never run this for `evaluator_test`: `build_dataset` refuses that split, and building any cache for
it would violate the leakage rules.

### 3.5 Verify trips and model

```bash
python -c "from src.data import list_trips; print(list_trips('student_test'))"
```

Expect 10 ids. Then confirm the model artifact loads and note its name:

```bash
python -c "import joblib; b = joblib.load('artifacts/models/final.joblib'); print(b['name'], b['cv_top1'])"
```

Expect the winning model name (for example `lgbm`) and a cross-validated top-1 around the reported
value. If this raises `FileNotFoundError`, the artifact is missing or you are not in `codebase/` —
fix it now. The server itself does not crash on a missing model: it prints a
`!! WARNING: model not loaded ...` line and continues with `bundle=None`, which is **not** a demo
state. Treat that warning during the dry run as a blocker.

---

## 4. Dry run with `student_test`

Use a separate terminal for the server; do not close it.

### 4.1 Start the API

Linux / macOS / Git Bash:

```bash
export PATHSNAP_SPLIT=student_test
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

PowerShell:

```powershell
$env:PATHSNAP_SPLIT = "student_test"
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

cmd:

```bat
set PATHSNAP_SPLIT=student_test
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

The startup output is one line with the split, the number of trips, the model name, its `cv_top1`,
the stub flag and the resolved data root, for example:

```
PathSnap API: split=student_test  trips=10  model=lgbm  cv_top1=0.xxxx  stub=False  data_root=...
```

Variables you may need:

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_DATA` | parent of the repo folder | Folder holding `student_test/`, `evaluator_test/`. |
| `PATHSNAP_SPLIT` | `evaluator_test` | Split to serve. Use `student_test` for the dry run. |
| `PATHSNAP_MODEL` | `artifacts/models/final.joblib` | Model bundle path. |
| `PATHSNAP_STUB` | `0` | Plumbing fallback only (serves the GPS track as the route). Never set it for a demo. |

### 4.2 Check the endpoints

```bash
curl http://localhost:8000/health
curl http://localhost:8000/trips
```

On Windows PowerShell `curl` is an alias for `Invoke-WebRequest`; use `curl.exe` to get raw JSON:

```powershell
curl.exe -s http://localhost:8000/health
curl.exe -s http://localhost:8000/trips
```

`/health` returns `{"ok": true, "model": ..., "split": ..., "trips": 10, "stub": false}`. `/trips`
returns `{"split": "student_test", "trips": [...]}` with 10 ids.

### 4.3 Loop over all 10 trips

`api/smoke.py` is the end-to-end check. It calls `/health` and `/trips`, posts every trip, validates
the contract keys, checks that no ground-truth keys leak, and confirms that an unknown id and
`../train/00000001` return 404:

```bash
python api/smoke.py --api http://localhost:8000
```

Use `--trip <ID>` to test a single trip and `--timeout 600` on a slow machine. Every trip must print
`PASS`, the two 404 checks must pass, and the `stub` column must say `no`. The script prints
`inference_ms` per trip; note the largest value, you will quote it in the report. The response also
carries `timing_ms` (an optional per-stage breakdown, may be empty) and `stub` (must be `false`); the
UI shows a red "stub pipeline" badge if it is ever true.

The same checks can be driven from the UI once it is connected (next step).

### 4.4 Drive the UI

Open the UI on Laptop 1 first (this isolates UI problems from networking problems):

```bash
python -m http.server 5500 --directory ui
```

Then browse to `http://localhost:5500/?api=http://localhost:8000`, press **Connect**, and click
through **all 10 trips**. Check for each: the map fits the trip, the GPS track and the predicted route
are both drawn, the metrics card is filled, a spinner appears while waiting, and no error banner
appears. Record the slowest latency you see in the metrics card.

Repeat once with the network cable/Wi-Fi of Laptop 1 disabled to confirm the offline behaviour in
section 7.

---

## 5. Demo day with `evaluator_test`

### 5.1 Place and flatten the data

`evaluator_test/` arrives on demo day. It goes on **Laptop 1 only**, under `DATA_ROOT`, and must not
be copied into the repository or committed.

The required path is exactly:

```
<DATA_ROOT>/evaluator_test/<ID>/<ID>.track
<DATA_ROOT>/evaluator_test/<ID>/<ID>.nodes
<DATA_ROOT>/evaluator_test/<ID>/<ID>.arcs
<DATA_ROOT>/evaluator_test/<ID>/<ID>.route      # if provided
```

Archives often contain an extra top-level folder. Flatten it so the id folders sit directly under
`evaluator_test/`:

```powershell
# inspect what you unzipped
Get-ChildItem "D:\pathsnap_data\evaluator_test" | Select-Object -First 12
# one id folder must look like this (True):
Test-Path "D:\pathsnap_data\evaluator_test\<ID>\<ID>.track"
```

```bash
# Linux / macOS / Git Bash
ls /path/to/data/evaluator_test | head
ls /path/to/data/evaluator_test/<ID>/<ID>.track
```

If you see `evaluator_test/evaluator_test/<ID>/...`, move the inner contents up one level and delete
the empty folder.

### 5.2 Confirm exactly 10 ids

```bash
python -c "from src.data import list_trips; t = list_trips('evaluator_test'); print(len(t), t)"
```

Must print `10` followed by the ids. Fewer means trips are missing or still nested; more means a
stray folder (an archive folder, a `__MACOSX`, an old split) is present. Fix the layout before
starting the server, because the trip list is read once at startup.

### 5.3 Start the API

```powershell
cd C:\pathsnap\codebase
.venv\Scripts\Activate.ps1
$env:PATHSNAP_SPLIT = "evaluator_test"
# $env:PATHSNAP_DATA = "D:\pathsnap_data"     # only if the data is not in the default place
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

```bash
# Linux / macOS equivalent
cd codebase && source .venv/bin/activate
export PATHSNAP_SPLIT=evaluator_test
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

On demo day:

- **Do not** rebuild caches, retrain, tune, or run anything from `src/features`/`src/train` on
  `evaluator_test`.
- **Do not** set `PATHSNAP_STUB`. It serves the GPS track as the route: plumbing only, never a demo.
  The startup line must read `stub=False` and `/predict` must return `"stub": false`.
- **Do not** edit code after the split arrives; the only change is `PATHSNAP_SPLIT`.
- Keep this terminal **visible** and readable from where the evaluator sits: enlarge the font
  (Ctrl + mouse wheel, or Terminal → Settings → Appearance), keep the window in front and undimmed.

### 5.4 Warm up

The first prediction is slower (imports, lazy initialisation, OS file cache). Run one yourself
before the evaluator arrives:

```powershell
curl.exe -s -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" `
  -d '{"trip_id":"<ID>"}'
```

```bash
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"trip_id":"<ID>"}'
```

Or use Swagger at `http://localhost:8000/docs` (collapsed JSON, easier to read). The terminal must
show one log line with status 200.

### 5.5 Connect Laptop 2

Follow section 6. Open the UI with `?api=http://<LAPTOP1_IP>:8000`, press **Connect**, and confirm the
10 trip buttons appear. Then let the evaluator choose trips.

---

## 6. Two-laptop networking (Windows)

1. **Same network.** Join both laptops to the same hotspot. University/campus Wi-Fi often isolates
   clients, so use a phone hotspot, or Laptop 1's Mobile Hotspot
   (Settings → Network & internet → Mobile hotspot → On) and connect Laptop 2 to that SSID.
2. **Network profile = Private** on Laptop 1: Settings → Network & internet → Wi-Fi → the network →
   Network profile type → Private. If the adapter has no profile picker (Mobile Hotspot), the
   firewall rule below covers `Public` as well.
3. **Find Laptop 1's IPv4 address:**

   ```powershell
   ipconfig
   ```

   Read the `IPv4 Address` of the active Wi-Fi / hotspot adapter (for a Windows Mobile Hotspot this
   is usually `192.168.137.1`; Android hotspots often use `192.168.43.x`, iPhones `172.20.10.x`).
   Write it down as `<LAPTOP1_IP>`. It changes when you switch networks — re-check on the day.
4. **Open the firewall port** in an **administrator** PowerShell on Laptop 1:

   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP `
     -LocalPort 8000 -Action Allow -Profile Private,Public
   ```

   Administrator `cmd` alternative:

   ```bat
   netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000
   ```

   If Windows shows "Allow Python to communicate on these networks", tick **Private** and allow.
   That prompt alone is not enough if the network was classified as Public when it appeared.
5. **Bind to all interfaces** when starting the server: `--host 0.0.0.0` (not `127.0.0.1`).
6. **Self-test from Laptop 1:**

   ```powershell
   curl.exe -s http://localhost:8000/trips
   curl.exe -s http://<LAPTOP1_IP>:8000/trips
   ```

   Both must return the 10 ids. If the first works and the second does not, the server is bound to
   localhost or the firewall blocks it.
7. **Test from Laptop 2** in a browser (not just the UI):

   ```
   http://<LAPTOP1_IP>:8000/trips
   http://<LAPTOP1_IP>:8000/health
   ```

   Raw JSON means the network path works. A timeout means firewall/profile/isolation; "connection
   refused" means the server is not running or is bound to `127.0.0.1`.
8. **Open the UI on Laptop 2** with the API address in the query string:

   ```
   http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
   ```

   served from Laptop 2, from the folder that contains `ui/` (for example `C:\pathsnap\codebase`):

   ```powershell
   cd C:\pathsnap\codebase
   python -m http.server 5500 --directory ui
   ```

   (only localhost is used here, so no extra firewall rule is needed). The alternative is to open
   the file directly. A query string still works if you type the full file URL in the address bar:

   ```
   file:///C:/pathsnap/codebase/ui/index.html?api=http://<LAPTOP1_IP>:8000
   ```

   Double-clicking the file opens it **without** the query string; in that case type the API URL into
   the field at the top and press **Connect**. The UI stores the last working URL in
   `localStorage`, but note that `file://` and `http://localhost:5500` are different origins, so the
   stored value does not carry over between them.

---

## 7. Offline behaviour

- Leaflet is loaded from the unpkg CDN first and falls back to `ui/vendor/` if that fails. The footer
  of the left panel shows which copy won: `leaflet: cdn`, `leaflet: vendor`, or `leaflet: failed`.
- Basemap tiles need internet. The UI tries Esri World Street Map first, then OSM France. The public
  `tile.openstreetmap.org` server and CARTO's keyless endpoint can return HTTP 403 or an "API key
  required" watermark, so neither is used as the primary layer.
- If a provider fails three times, its tile layer is removed (no broken-tile checkerboard) and the
  next provider is tried. If all providers fail, the page shows "base map unavailable — geometry
  still drawn"; the GPS and predicted polylines, markers, legend and metrics still render on a
  neutral background. That is acceptable — say so before the evaluator notices.
- Check before the demo:
  - `ui/vendor/leaflet.js`, `ui/vendor/leaflet.css` and `ui/vendor/images/` exist.
  - No external font or icon dependency (system fonts only).
  - The only external URLs are the Leaflet CDN (with the vendor fallback), the Esri/OSM-France tile
    servers and their attribution links.

  ```bash
  ls ui/vendor ui/vendor/images
  grep -o 'https://[^"]*' ui/index.html | sort -u
  ```

  ```powershell
  Get-ChildItem ui\vendor, ui\vendor\images
  Select-String -Path ui\index.html -Pattern "https://" -AllMatches
  ```

  Then, on Laptop 1, open `http://localhost:5500` with Wi-Fi disabled, reload, and click one trip.
  The footer must say `leaflet: vendor` (not `cdn`, not `failed`) and the route must draw on the
  blank background. (Laptop 2 cannot reach the API without the hotspot, so do this test on
  Laptop 1.)

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` / `ERR_CONNECTION_REFUSED` | Server not running, or bound to `127.0.0.1` | Start it with `--host 0.0.0.0`; check it is still running in the terminal. |
| Request hangs, then timeout | Firewall, Public network profile, or hotspot client isolation | Add the inbound TCP 8000 rule, set the profile to Private, or switch to a phone hotspot. |
| Browser console: CORS error / "Failed to fetch" | CORS middleware missing or the API URL is wrong | Confirm `CORSMiddleware` is enabled for `GET, POST`; load `/trips` directly in the browser to test the URL. |
| 404 on every trip | Wrong `PATHSNAP_SPLIT`, wrong `PATHSNAP_DATA`, or server started before the data was in place | Fix the path/split, then restart uvicorn (the trip list is read once at import). |
| 503 "the prediction pipeline is not implemented yet" | `src/predict.py` is still the stub, or the import failed | Check `src/predict.py` is B's real implementation, not the placeholder; restart. Never demo on a stub and do not "fix" it by setting `PATHSNAP_STUB`. |
| 503 on `/map/<ID>` | `src/maps.py` is not implemented yet, or exposes neither `trip_map` nor `make_map` | Restart once B's map helper is in place; `/predict` still works without it, so it is not a demo blocker. |
| 500 "prediction failed" | An exception inside `predict_trip` | Read the traceback printed in the Laptop 1 terminal; report the trip id to B. |
| 404 only for a hand-typed id | Id is not in the active split, or leading zeros were dropped | Use the ids from `GET /trips` verbatim; ids are strings. |
| Startup warns `model not loaded ... bundle=None`; `/health` shows `model: null` | Missing `artifacts/models/final.joblib`, or started outside `codebase/` | Run from `codebase/`; set `PATHSNAP_MODEL` to an absolute path. Do not demo with `bundle=None`. |
| `/predict` returns `"stub": true`, or the UI shows the red "stub pipeline" badge | `PATHSNAP_STUB` is set in that terminal | Unset it (`Remove-Item Env:PATHSNAP_STUB` / `unset PATHSNAP_STUB`) and restart uvicorn. |
| UI footer says `leaflet: failed` or "Map unavailable" | `ui/vendor/` missing, or the page was copied without it | Restore `ui/vendor/leaflet.js`, `leaflet.css`, `images/` beside `index.html`; serve with `--directory ui`. |
| More or fewer than 10 trips | Nested archive folder, stray folder, or missing trips | Flatten so `evaluator_test/<ID>/<ID>.track`; remove extras; restart the server. |
| First request takes many seconds | Imports, lazy model load, cold file cache | Warm up with one prediction before the evaluator arrives. |
| Metrics show `NaN`, or the JSON refuses to parse | Non-finite values serialised instead of `null` | The server converts NaN/Inf to `null`; expected display is "n/a (no reference route)". If all five are null, the scoring modules or `.route` files are missing — check the `score()` imports. |
| Windows firewall prompt appears | Python listening on a new port/network | Tick Private + allow; if dismissed, add the rule manually. |
| Works, then fails after switching networks | Laptop IP changed | Re-run `ipconfig`, update the UI URL / reconnect. |
| UI works from `http://localhost:5500` but not from `file://` | Different origin, stored API URL does not apply, or fetch blocked | Type `?api=...` in the URL or use the optional `http.server` route. |
| `[Errno 10048]` / port already in use | Old uvicorn still alive | Close it with Ctrl+C; or find the PID listening on 8000 with `netstat -ano`, then `taskkill /PID <PID> /F`. |

---

## 9. Verification checklist

Tick every box before the evaluator arrives. These map to the C definition of done in
`docs/handoff/TEAMMATE_C_api_ui.md` section 6.

- [ ] `uvicorn api.server:app --host 0.0.0.0 --port 8000` starts with the expected split and prints
      split, trip count, model name, `cv_top1`, `stub=False` and the data root.
- [ ] `GET /health` returns `ok: true` with the model name, split, trip count and `stub: false`.
- [ ] `GET /trips` returns exactly 10 ids for the active split, as strings with leading zeros.
- [ ] All 10 trips return HTTP 200 from `POST /predict` with every contract key present, and
      `python api/smoke.py --api http://localhost:8000` reports PASS and `stub=no` for all of them.
- [ ] `POST /predict` with an unknown id returns 404.
- [ ] `POST /predict` with `"../train/00000001"` returns 404.
- [ ] The response contains no reference arc ids, no `label_arc`, no `.route` content — only the five
      scalar metrics.
- [ ] The terminal shows exactly one log line per request, in the
      `[ts] client_ip METHOD path trip_id status latency_ms` format, including Laptop 2's IP.
- [ ] The UI works from Laptop 2 via `<LAPTOP1_IP>` (not only localhost).
- [ ] The API URL is editable in the UI without touching code, and Connect reports OK/error.
- [ ] Errors are visible in the UI (stop the server and try a trip).
- [ ] The UI still draws routes with the network off: the footer says `leaflet: vendor`, the "base map
      offline" note appears and the route still renders.
- [ ] `GET /map/<ID>` renders GPS + predicted route with **no** reference layer.
- [ ] `PATHSNAP_STUB` is not set in the demo terminal: startup shows `stub=False` and `/predict`
      returns `"stub": false`.

---

## 10. Report section 7: screenshots and paragraph

Capture these during the two-laptop dry run, on Laptop 2's screen where the UI is involved. Save them
under `artifacts/plots/screens/` (create the folder if it is not there) with these names.

| # | Suggested file | What it must show |
|---|---|---|
| 1 | `01_ui_trips_loaded.png` | UI on Laptop 2, API URL field showing `http://<LAPTOP1_IP>:8000`, Connect state OK, 10 trip buttons. |
| 2 | `02a_ui_route_a.png`, `02b_ui_route_b.png` | UI after predicting 2–3 different trips: blue GPS track, red predicted route, start/end markers, legend, metrics card (Top-1, route F1, precision, recall, continuity, inference ms), trip id. |
| 3 | `03_laptop1_request_log.png` | Laptop 1 terminal with several lines from Laptop 2's IP: `GET /trips`, then `POST /predict` for the same trips, status 200, latency in ms. Font large enough to read. |
| 4 | `04_predict_json.png` | One raw JSON response (`/docs` Try it out, or `curl.exe -X POST .../predict`), long coordinate arrays collapsed, metrics visible. |
| 5 | `05_folium_map.png` | `/map/<ID>` folium page: GPS + predicted route only, no reference route layer. |
| 6 | `06_ipconfig_firewall.png` | `ipconfig` output with the IPv4 address, plus the firewall rule (`Get-NetFirewallRule -DisplayName "PathSnap API 8000"`). |

Screenshots must come from the same code and setup that is demonstrated; do not mock them.

Paragraph template (fill the placeholders):

> The demo runs on two laptops joined by a <hotspot type> hotspot. Laptop 1 (<OS>, Python <version>)
> runs `uvicorn api.server:app` bound to `0.0.0.0:8000` and holds the road network, the trained
> classifier and the HMM/Viterbi decoder. For each request it builds candidates for the trip, predicts
> the per-point arcs, reconstructs the route and only then loads the trip's ground-truth `.route`
> inside `score()`; the response carries the GPS track, the predicted route and five scalar metrics,
> never the reference route. Laptop 2 opens `ui/index.html`, points it at
> `http://<LAPTOP1_IP>:8000`, and uses Leaflet (with a vendored copy as the offline fallback) to draw
> the returned geometry; it contains no model or data code. CORS is open for GET/POST so the static
> page can call the API, uvicorn binds all interfaces, and a Windows inbound TCP 8000 rule together
> with a Private network profile lets Laptop 2 through.
> Measured latency was <min>–<max> ms per trip on `student_test` (mean <mean> ms, slowest <ID>), and
> the request terminal on Laptop 1 showed one log line per call from Laptop 2's address.

---

## 11. Demo-day runbook

1. **T-60 min** — Both laptops charged and plugged in; phones/hotspot available; cables and a spare
   phone hotspot ready.
2. **Laptop 1: update and install** — `git pull`, activate the venv, `pip install -r requirements.txt`
   if the lockfile/requirements changed.
3. **Laptop 1: data** — confirm `evaluator_test/` is flattened under `DATA_ROOT`, exactly 10 ids:
   `python -c "from src.data import list_trips; print(list_trips('evaluator_test'))"`.
   Confirm `artifacts/models/final.joblib` loads and note `b['name']`.
4. **Both: network** — connect to the same hotspot; on Laptop 1 run `ipconfig`, note `<LAPTOP1_IP>`;
   confirm Private profile; confirm the firewall rule exists (create it again if the network name
   changed).
5. **Laptop 1: start the API** — set `PATHSNAP_SPLIT=evaluator_test` (and `PATHSNAP_DATA` if needed),
   run `python -m uvicorn api.server:app --host 0.0.0.0 --port 8000`, enlarge the font, keep the
   window visible. Confirm the startup line shows `trips=10`, the model name and `stub=False`, and
   that no `!! WARNING` line appears.
6. **Laptop 1: self-test and warm-up** — `curl.exe -s http://<LAPTOP1_IP>:8000/trips`, then one
   `POST /predict` from Swagger or curl; confirm one log line with status 200.
7. **Laptop 2: start the UI** — from the folder containing `ui/`, run
   `python -m http.server 5500 --directory ui` (or open the file URL), browse to
   `http://localhost:5500/?api=http://<LAPTOP1_IP>:8000`, press Connect, see 10 trips.
8. **Laptop 2: test from the client** — click two trips yourself, check the map and metrics, and point
   at the matching log lines on Laptop 1 to confirm the client IP is Laptop 2's.
9. **Demo** — hand control to the evaluator; let them pick any trips. After each click, point at the
   new terminal line (client IP, trip id, 200, latency). Keep hands off the keyboard otherwise.
10. **Close** — leave the server and terminal running until the evaluator is done; do not stop
    uvicorn, do not reboot, do not "clean up" caches.

### Fallbacks

- **Wi-Fi fails or clients are isolated:** switch both laptops to a phone hotspot, run `ipconfig`
  again, update the UI's API URL, reconnect. Do not change the server binding.
- **Firewall still blocks:** disable the Public/Private block for that one rule (or temporarily turn
  off Windows Firewall for Private/Public in the demo profile) and retry the Laptop 2 browser check.
- **Last resort:** run the UI on Laptop 1 against `http://localhost:8000` and explain that the
  two-laptop path was verified earlier (screenshot 1 and the request log are the evidence). This is a
  fallback, not the demo: the brief wants two laptops.
- **Restart commands:** Ctrl+C in the uvicorn terminal, then the same start command. For a stuck port:
  `netstat -ano | findstr :8000`, then `taskkill /PID <PID> /F`. For the UI server: Ctrl+C and rerun
  `python -m http.server 5500 --directory ui`.
- **A trip crashes:** note the id and the traceback from Laptop 1, move to another trip, and report it
  to B after the demo. Do not patch code in front of the evaluator.

### Viva questions

1. **Where is the ground truth loaded?**
   Only in `score()` in `api/server.py`, after `predict_trip` has returned. `predict_trip` calls
   `load_trip(..., with_route=False)`; `score()` then loads the `.route` file, aligns the labels and
   computes the metrics. Only the five scalars are serialised, so no reference arc id can reach the
   browser.

2. **Is anything precomputed?**
   The model and its hyperparameters are trained offline; no trip, route or metric is precomputed.
   Every `/predict` builds candidates, runs the classifier and the Viterbi decoder live, and the
   terminal line shows the per-request latency. The only in-memory cache is the last predicted route
   per trip, kept so `/map` does not recompute it; it never contains ground truth. Candidate caches
   are only for training/evaluation scripts, not for serving predictions.

3. **Why two laptops?**
   It is the required deployment and it demonstrates the client/server split honestly: the model,
   the data and the scoring stay on one machine, while the UI is a static page with no pipeline code.
   It also makes it easy to show that the metrics are produced server-side from ground truth the
   client never receives.

4. **Why are the CPU-bound endpoints plain `def` instead of `async def`?**
   Candidate generation, the classifier and Viterbi are CPU-bound and would block the event loop.
   FastAPI runs plain `def` endpoints in a worker thread, so `/trips` and `/health` stay responsive
   while a prediction is running.

5. **How are CORS, the firewall and the binding configured?**
   `CORSMiddleware` allows `GET` and `POST` from any origin, because the UI may be opened from
   `file://` (a `null` origin) or from `http://localhost:5500`. `uvicorn` binds `0.0.0.0:8000` so the
   service listens on the LAN address, and an inbound Windows firewall rule for TCP 8000 on the
   Private/Public profiles lets Laptop 2 connect over the hotspot.
