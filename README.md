# Verdant — Streamlit Edition

A standalone Streamlit copy of the redesigned IIoT Risk Observatory. This edition keeps the ivory, warm taupe and emerald palette, locally bundled luxury serif / sans-serif typography, soft neumorphic surfaces and a responsive dashboard layout.

**No Node.js, Vercel gateway, FastAPI service or remote USB agent is needed for this edition.** Extract the complete folder; app.py imports the included Python helpers.

## Quick start

Use Python **3.11 or 3.12** where possible. From the extracted project directory:

```bash
pip install -r requirements.txt
```

1. Place your real Edge-IIoTset ML CSV in this directory and name it **data.csv**.
2. Copy `.streamlit/secrets.toml.example` to **`.streamlit/secrets.toml`**.
3. Replace the placeholder with your own password of **at least 12 characters**:

```toml
app_password = "your-own-unique-strong-password"
```

4. Start the application:

```bash
streamlit run app.py
```

Open the URL Streamlit prints and sign in with your configured password. There is **no default password**. Security Monitor works without data.csv; analytical views wait for a real dataset rather than substituting synthetic data.

The APP_PASSWORD environment variable is an alternative to secrets.toml and takes precedence. Keep real credentials out of shared files and version control.

### Optional environment variables

| Name | Purpose | Default |
|---|---|---|
| `APP_PASSWORD` | Shared workspace password, 12+ characters | Read `app_password` from Streamlit secrets |
| `DATA_PATH` | Path to your real ML CSV | data.csv beside app.py |
| `STATE_DIR` | Writable persistent directory for audit/triage/USB baseline | state/ beside app.py |

Use a private, persistent location for STATE_DIR. Database files include IP addresses, device descriptions and investigation state; restrict access accordingly.

## Files

```text
verdant-streamlit/
├── app.py                       dashboard and five core analytical modules
├── pipeline.py                  original training / risk / SHAP logic
├── security_monitor.py          login rules, audit storage and triage
├── usb_inventory.py             opt-in server-host USB enumeration
├── requirements.txt             only the original six external libraries
├── .streamlit/
│   ├── config.toml              ivory / emerald theme
│   └── secrets.toml.example     copy and edit; not a working credential
├── assets/                      local fonts and their licence notices
├── tests/smoke_test.py           temporary-fixture software checks
├── README.md
└── data.csv                     YOU supply this separately
```

Do not copy just app.py without its helpers. The included .gitignore excludes your real secrets, CSV and SQLite state. The ZIP contains no dataset, credentials or generated security logs.

## Dashboard views

### Overview
- Known-class attack recall, High-risk record count, held-out-family detection and recent login alerts.
- Real risk distribution and the ten highest-risk known-test records.
- Global class-1 SHAP on up to 500 known-test records, computed automatically in a background worker.
- Click a table row—or use the optional Record ID selector—to inspect its local explanation.

### Threat explorer
- Accuracy, attack precision, attack recall, F1, confusion matrix and classification report.
- Risk histogram, search, risk-level filtering and record pagination.
- Persistent **Open / Investigating / Resolved** triage. Triage changes do not alter labels, predictions or training data.

### Zero-day lab
- Selected family excluded from all fitting, feature selection, scaling and calibration.
- Combined risk ≥40 compared with RF probability >0.50 and an RF ≥0.40 sensitivity check.
- Corresponding false-positive rates on known-test Normal rows.
- Separate held-out-family investigation queue.

### Scenario studio
- Original-unit sliders for the training-selected features.
- Known-test Normal/attack examples and training-median defaults.
- Risk and local SHAP update when a slider change is committed. No separate submit/next-page sequence.
- Examples outside training bounds are clipped for the sliders and explicitly labelled as clipped. Arbitrary feature combinations may not be valid network observations.

### Security Monitor
- Five-second refresh of login event summaries, scores and explanations.
- Explicit opt-in USB inventory on the **machine running Streamlit**.
- Baseline review/reset and clear disabled/unsupported states.
- Independent of dataset loading and model training.

### Data & methodology
- Full CSV, sampled and clean class counts; before/after cleaning counts.
- Removed identifiers, text fields and training-constant columns.
- Selected feature importance, experimental settings, file fingerprint and raw sample preview.

## USB monitoring: important boundary

**This Streamlit edition checks the server host, not the visitor's browser device.**

- Run Streamlit directly on your laptop to monitor that laptop's USB devices.
- Open **Security monitor → Host USB settings**, and explicitly enable monitoring.
- The first successful inventory establishes the baseline. Devices already connected then count as normal.
- Connect a USB device and wait for the next refresh; new inventory entries are displayed.
- Review the inventory, confirm the checkbox, and use **Trust current host inventory** to reset the baseline.
- Windows uses PowerShell device enumeration, macOS uses system_profiler, and Linux uses USB sysfs. OS permissions, charge-only cables, hubs and virtualisation affect what can be observed.
- An enumeration failure is shown as unavailable, not as zero suspicious devices.
- These are inventory changes—not proof of malware or a malicious device.

If this app runs on Streamlit Community Cloud or Render, “host USB” means that cloud machine. It does **not** inspect your laptop or phone. For remote laptop telemetry, use the separate Vercel + Python backend + local agent project supplied earlier; that agent API is not part of this standalone Streamlit edition.

A phone opening your Streamlit URL generates login events, but its USB devices are not exposed. The USB risk term is environmental context from the monitored host, not evidence that a particular login came from that device.

## Login behaviour

A transparent, independent 0–100 rule score is assigned to attempts:

```text
50 × min(failures in 10 minutes, 5) / 5
+ 20 × min(failures in 60 seconds, 3) / 3
+ 20 if this Streamlit session is unrecognised
+ 10 if enabled, supported host monitoring reports a new USB device
```

Low <40; Medium 40–70 inclusive; High >70. This is **not** an Edge-IIoTset prediction and the factor chart is **not SHAP**.

- Five failures from a source within ten minutes trigger a 120-second lockout after its last failure.
- A simple global failure gate also limits bursts; neither gate replaces provider-level bot protection.
- Shared networks may share a source limit. If Streamlit cannot identify a source, unknown sources share the fallback bucket.
- The password is verified server-side with salted PBKDF2 and constant-time hash comparison; it is not stored in the event log.
- Authenticated access lasts eight hours; signing out clears authenticated access.
- Recognition lasts within the current Streamlit session. A browser reload/reconnect may create a new session and require another sign-in. This edition does not recreate the Vercel version's signed persistent browser cookie.
- Events are stored in SQLite; events older than 30 days are removed during subsequent login attempts. There is no UI action that erases lockout history.
- USB baselines and triage persist in STATE_DIR. Define backup and deletion policies before multi-user use.

This is a shared-password educational application, not enterprise identity management. It does not include MFA, individual roles or account recovery. Preserve Streamlit's XSRF and CORS protections, use HTTPS for remote hosting, and keep the underlying host secured.

## Data and model contract

Only pandas, numpy, scikit-learn, shap, matplotlib and streamlit are direct external dependencies.

1. Read the real ML CSV with pandas and sample at most 40,000 rows by Attack_type, seed 42. The full-file read still requires sufficient RAM.
2. Exclude target columns, identifiers and text; keep numeric features; replace infinities with NaN; drop incomplete rows and duplicates.
3. Reserve SQL_injection by default when available; the sidebar allows another family.
4. Split the remaining known data 75/25. Remove training-constant columns, select up to twelve features using a preliminary Random Forest, and fit StandardScaler on training data only.
5. Fit a 100-tree Random Forest with n_jobs=-1 and seed 42. Fit Isolation Forest only on Normal training rows.
6. Normalize the negative Isolation Forest score using known-training 1st/99th percentiles, clip to [0,1], and calculate:

```text
risk = 100 × (0.6 × RF_attack_probability + 0.4 × normalized_anomaly)
```

7. Explain the RF with class-1 TreeExplainer outputs. Common list and 3-D array formats are supported. Global summaries use up to 500 randomly sampled known-test rows; local descriptions refer to training quartiles.

SHAP explains **Random Forest only**, not Isolation Forest or the combined risk. The score is a heuristic, not a calibrated compromise probability. The attack-only holdout yields detection rates, not accuracy or precision. Different thresholds are not a matched-FPR comparison, and fusion need not outperform RF.

Random row splitting can overstate generalisation across devices, captures and time. A held-out labelled family is a zero-day simulation, not proof of production resilience. No automated network blocking or industrial control actions are performed.

## Caching and responsiveness

- CSV preparation uses st.cache_data.
- Models use st.cache_resource, with two cached holdout configurations.
- Risk outputs and local explanations are cached for reuse.
- Global SHAP runs in a bounded background-worker cache; its panel refreshes every three seconds.
- Scenario controls and Security Monitor use native Streamlit fragments.
- Model fitting still blocks the initiating Streamlit run while a new configuration is trained. Large forests and full-size SHAP can be expensive; adequate RAM/CPU are necessary.
- Dataset cache keys include path, modification time and size. The app also computes a content fingerprint for persistent triage separation. To replace a CSV with artificially preserved metadata, clear Streamlit's cache or restart the service.
- Caches are in memory, not saved fitted models. Restarting the service retrains the selected configuration.

## Hosting this edition

This is a conventional long-running Streamlit application. It can run locally or on a host supporting its persistent process and WebSocket connection. It is **not a direct Vercel serverless deployment**; keep the previous web edition for Vercel.

For a generic persistent host:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Configure APP_PASSWORD through the host's secret settings; mount persistent private storage for DATA_PATH and STATE_DIR; place the real dataset there; and use HTTPS at the hosting layer. Do not put your CSV or secret file in a public repository. Ephemeral hosted filesystems may lose SQLite history after restarts/redeployments; do not assume persistence without an explicit disk.

For access from a phone on your trusted LAN, use your computer's LAN address and Streamlit port, and allow that port through your firewall. Do not expose development HTTP on an untrusted network. IP metadata behind proxies may group multiple visitors; it is not a dependable device identity.

## Verification

From the extracted project directory, after installing requirements:

```bash
python tests/smoke_test.py
```

This creates and removes a **temporary synthetic fixture solely for software checks**. It does not create your production data.csv or establish real-dataset performance. The checks cover setup, login, all analytical views, manual controls, holdout switching, record explanations, persistent triage and simulated USB-baseline handling.

The real CSV was not provided for this iteration. Actual Edge-IIoTset metrics, full-size runtime and your computer's physical USB enumeration remain unverified. No fixture results or credentials are included as application defaults.
