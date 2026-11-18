"""Model pipeline adapted from the uploaded app.py; no Streamlit runtime.
All learned preprocessing uses known training data only.
"""
import re
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
SEED = 42
MAX_ROWS = 40_000
TOP_FEATURES = 12
TARGETS = ["Attack_label", "Attack_type"]

def stratified_sample(frame, limit=MAX_ROWS):
    """Proportional random sample, retaining at least one row of each class.

    Integer allocation also works when a class has only one record. The small
    remainder is assigned by largest fractional part. Sampling is reproducible.
    """
    if len(frame) <= limit:
        return frame.sample(frac=1, random_state=SEED).copy()
    counts = frame["Attack_type"].value_counts().sort_index()
    if len(counts) > limit:
        raise ValueError("Too many attack classes to retain all in the sample.")
    remaining = limit - len(counts)
    weights = (counts - 1) / (len(frame) - len(counts))
    quotas = weights * remaining
    allocation = np.floor(quotas).astype(int) + 1
    extra = limit - int(allocation.sum())
    order = (quotas - np.floor(quotas)).sort_values(ascending=False).index
    allocation.loc[order[:extra]] += 1
    pieces = [group.sample(n=int(allocation.loc[name]), random_state=SEED)
              for name, group in frame.groupby("Attack_type", sort=True)]
    return pd.concat(pieces).sample(frac=1, random_state=SEED).copy()

def load_data(path, file_signature):
    # file_signature (mtime, size) invalidates the cache when data.csv changes.
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names after stripping whitespace.")
    missing = set(TARGETS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    original_shape = frame.shape
    frame["Attack_type"] = frame["Attack_type"].astype("string").str.strip()
    valid_type = frame["Attack_type"].notna() & frame["Attack_type"].ne("")
    missing_types = int((~valid_type).sum())
    frame = frame.loc[valid_type].copy()
    if frame.empty:
        raise ValueError("No rows with a usable Attack_type.")
    # Normal defines class 0; every other named attack defines class 1.
    frame.loc[frame["Attack_type"].str.casefold().eq("normal"), "Attack_type"] = "Normal"
    full_counts = frame["Attack_type"].value_counts().rename("Full CSV")
    sample = stratified_sample(frame)
    # Preserve the original zero-based CSV data-row index for traceability.
    sample.index.name = "record_id"
    return sample, original_shape, full_counts, missing_types

def is_identifier_or_text(name):
    """Remove addresses, payloads, identifiers and leakage-prone metadata."""
    name = name.lower()
    explicit = {
        "frame.time", "frame.number", "frame.len_cap", "ip.src_host",
        "ip.dst_host", "http.file_data", "http.request.full_uri",
        "http.request.uri", "tcp.options", "tcp.payload", "tcp.srcport",
        "tcp.dstport", "tcp.stream", "tcp.seq", "tcp.ack", "tcp.seq_raw",
        "tcp.ack_raw", "udp.srcport", "udp.dstport", "udp.stream",
        "icmp.transmit_timestamp", "icmp.originate_timestamp",
        "icmp.receive_timestamp", "icmp.seq", "icmp.seq_le",
        "dns.qry.name", "dns.qry.name.len", "mqtt.clientid", "mqtt.msg",
        "mqtt.topic", "http.request.method", "http.referer", "http.host",
        "http.user_agent", "http.request.version", "http.response.phrase",
        "id", "index", "row_id", "record_id", "flow_id", "flow.id",
        "label", "class", "attack", "category", "subcategory", "target",
    }
    return (
        name in explicit
        or name.startswith("unnamed:")
        or name.startswith("frame.time")
        or name.startswith("attack_")
        or "timestamp" in name
        or (name.startswith("arp.") and "address" in name)
        or re.search(r"(^|[._])(src|dst|source|destination)[._]?(ip|mac|host|port|addr|address)$", name) is not None
        or name.endswith((".payload", ".address", ".uri"))
    )

def clean_data(sample):
    candidates = sample.drop(columns=TARGETS)
    removed_ids = [c for c in candidates if is_identifier_or_text(c)]
    candidates = candidates.drop(columns=removed_ids)
    # No text encoding: keep only actual numeric CSV columns.
    numeric = candidates.select_dtypes(include=[np.number]).copy()
    removed_text = [c for c in candidates if c not in numeric.columns]
    if numeric.shape[1] == 0:
        raise ValueError("No numeric features remain after removing identifiers.")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    labels = pd.to_numeric(sample["Attack_label"], errors="coerce")
    valid_labels = labels.isin([0, 1])
    expected = sample["Attack_type"].ne("Normal").astype(int)
    mismatch = valid_labels & labels.ne(expected)
    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} sampled rows have inconsistent Attack_label "
            "and Attack_type. Expected Normal=0 and every attack=1. Check the CSV."
        )
    clean = numeric.copy()
    clean["Attack_type"] = sample["Attack_type"]
    clean["Attack_label"] = expected
    clean = clean.loc[valid_labels].dropna()
    after_na = len(clean)
    # Deduplicate features plus class metadata before any split, not row IDs.
    clean = clean.drop_duplicates()
    stats = {
        "Sampled rows": len(sample),
        "After dropping NaN / invalid labels": after_na,
        "After removing duplicates": len(clean),
        "Numeric candidate features": numeric.shape[1],
    }
    if clean.empty:
        raise ValueError("Cleaning removed every row. Check missing values and numeric columns.")
    return clean, stats, removed_ids, removed_text

def train_models(clean, held_out):
    # The held-out attack is never used for fitting, feature selection, scaling,
    # anomaly normalization, threshold tuning or the ordinary test metrics.
    known = clean.loc[clean["Attack_type"].ne(held_out)].copy()
    zero_day = clean.loc[clean["Attack_type"].eq(held_out)].copy()
    if zero_day.empty:
        raise ValueError("No records remain for the selected zero-day attack.")
    X = known.drop(columns=TARGETS)
    y = known["Attack_label"].astype(int)
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Need at least two Normal and two known-attack rows after cleaning.")
    if len(known) < 10:
        raise ValueError("Need at least ten known-class records for the train/test split.")
    # Stratify by attack type where possible, otherwise use the binary target.
    type_counts = known["Attack_type"].value_counts()
    n_test = int(np.ceil(len(known) * 0.25))
    can_stratify_types = (type_counts.min() >= 2
                          and len(type_counts) <= min(n_test, len(known) - n_test))
    stratify = known["Attack_type"] if can_stratify_types else y
    train_ids, test_ids = train_test_split(
        known.index.to_numpy(), test_size=0.25, random_state=SEED, stratify=stratify,
    )
    X_train, X_test = X.loc[train_ids], X.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]
    variable = X_train.nunique().gt(1)
    constants = variable.index[~variable].tolist()
    X_train = X_train.loc[:, variable]
    if X_train.empty:
        raise ValueError("No nonconstant training features remain.")
    if int(y_train.eq(0).sum()) < 2:
        raise ValueError("Not enough Normal training rows for IsolationForest.")
    # A preliminary RF selects features using TRAINING labels only.
    selector = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    selector.fit(X_train, y_train)
    importances = pd.Series(selector.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    features = importances.head(TOP_FEATURES).index.tolist()
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[features]), columns=features, index=train_ids,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(X_test[features]), columns=features, index=test_ids,
    )
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    rf.fit(train_scaled, y_train)
    isolation = IsolationForest(n_estimators=100, contamination="auto",
                                n_jobs=-1, random_state=SEED)
    isolation.fit(train_scaled.loc[y_train.eq(0)])
    # Bigger -score_samples means more anomalous. Calibrate on known TRAINING
    # rows only, using robust percentiles, then clip future scores to [0, 1].
    train_anomaly = -isolation.score_samples(train_scaled)
    anomaly_low, anomaly_high = np.percentile(train_anomaly, [1, 99])
    reference = X_train[features].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    explainer = shap.TreeExplainer(rf)
    return {
        "rf": rf, "isolation": isolation, "scaler": scaler,
        "features": features, "importances": importances.head(TOP_FEATURES),
        "constants": constants, "explainer": explainer,
        "anomaly_low": float(anomaly_low), "anomaly_high": float(anomaly_high),
        "reference": reference, "test_scaled": test_scaled, "y_test": y_test,
        "test_raw": known.loc[test_ids], "zero_raw": zero_day,
        "train_count": len(train_ids), "normal_train_count": int(y_train.eq(0).sum()),
        "test_count": len(test_ids), "held_out": held_out,
        "type_stratification": bool(can_stratify_types),
    }

def scale_rows(bundle, rows):
    return pd.DataFrame(bundle["scaler"].transform(rows[bundle["features"]]),
                        columns=bundle["features"], index=rows.index)

def assess_risk(bundle, rows):
    scaled = scale_rows(bundle, rows)
    class_one = int(np.flatnonzero(bundle["rf"].classes_ == 1)[0])
    probability = bundle["rf"].predict_proba(scaled)[:, class_one]
    anomaly = -bundle["isolation"].score_samples(scaled)
    span = bundle["anomaly_high"] - bundle["anomaly_low"]
    # An uninformative calibration range cannot support a graded anomaly score.
    normalized = (np.clip((anomaly - bundle["anomaly_low"]) / span, 0, 1)
                  if span > 1e-12 else np.zeros(len(rows)))
    risk = np.clip(100 * (0.6 * probability + 0.4 * normalized), 0, 100)
    level = np.where(risk < 40, "Low", np.where(risk <= 70, "Medium", "High"))
    result = pd.DataFrame({
        "RF attack probability": probability,
        "Normalized anomaly": normalized,
        "Risk score": risk, "Risk level": level,
    }, index=rows.index)
    return result

def class_one_shap(raw, n_rows, n_features, class_index=1):
    """Support older list-of-class arrays and newer (rows, features, classes).

    A 2-D result already represents a single output. An alternative
    (classes, rows, features) layout is handled defensively as well.
    """
    if isinstance(raw, (list, tuple)):
        values = np.asarray(raw[class_index])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            if values.shape[:2] == (n_rows, n_features):
                values = values[:, :, class_index]
            elif values.shape[1:] == (n_rows, n_features):
                values = values[class_index, :, :]
            else:
                raise ValueError(f"Unexpected 3-D SHAP shape: {values.shape}")
    if values.shape != (n_rows, n_features):
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")
    return values

def explain_rows(bundle, rows):
    scaled = scale_rows(bundle, rows)
    raw = bundle["explainer"].shap_values(scaled, check_additivity=False)
    class_index = int(np.flatnonzero(bundle["rf"].classes_ == 1)[0])
    return class_one_shap(raw, len(rows), len(bundle["features"]), class_index)
