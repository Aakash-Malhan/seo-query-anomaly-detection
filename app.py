import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import os, re
from collections import Counter

RANDOM_STATE = 42
REQUIRED_COLS = ["Top queries", "Clicks", "Impressions", "CTR", "Position"]

# IO helpers
def read_csv_any(file_obj_or_path):
    """Accepts gr.File (temp path) or a string path; returns DataFrame."""
    if file_obj_or_path is None:
        raise ValueError("No CSV provided.")
    if hasattr(file_obj_or_path, "name"):          
        path = file_obj_or_path.name
    else:
        path = str(file_obj_or_path)
    return pd.read_csv(path)

# Cleaning / Features
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize header names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    lower_map = {c.lower(): c for c in df.columns}
    fixed = {}
    # flexible alternatives for common GSC exports
    alts = {
        "top queries": ["query","queries","keyword","search term","top query"],
        "clicks": ["click","total_clicks"],
        "impressions": ["impr","impressions_total","total_impressions"],
        "ctr": ["click_through_rate","click-through rate","ctr (%)","ctr%"],
        "position": ["avg position","avg_position","rank","avg rank","avg_ranking"]
    }
    for col in REQUIRED_COLS:
        key = col.lower()
        if key in lower_map:
            fixed[col] = lower_map[key]
        else:
            m = None
            for alt in alts.get(key, []):
                if alt in lower_map:
                    m = lower_map[alt]; break
            if m is None:
                raise ValueError(f"Missing required column: {col}")
            fixed[col] = m

    df = df[[fixed[c] for c in REQUIRED_COLS]].copy()
    df.columns = REQUIRED_COLS

    # CTR can be like "35.8%" or 0.358
    def parse_ctr(x):
        if isinstance(x, str):
            x = x.strip().replace('%', '')
            x = float(x) if x else np.nan
            return x / 100.0
        try:
            x = float(x)
            return x/100.0 if x > 1 else x
        except:
            return np.nan

    df["CTR"] = df["CTR"].apply(parse_ctr)
    for c in ["Clicks", "Impressions", "Position"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # basic validity
    df = df.dropna(subset=["Clicks","Impressions","CTR","Position"])
    df = df[(df["Clicks"] >= 0) &
            (df["Impressions"] >= 0) &
            (df["CTR"].between(0, 1)) &
            (df["Position"] > 0)]
    return df.reset_index(drop=True)

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_clicks"] = np.log1p(out["Clicks"])
    out["log_impr"]   = np.log1p(out["Impressions"])

    # Expected CTR from Position via log–log regression
    eps = 1e-9
    x = np.log(out["Position"] + eps)
    y = np.log(out["CTR"] + eps)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = coef
    out["ctr_expected"] = np.clip(np.exp(intercept + slope*np.log(out["Position"] + eps)), 0, 1)
    out["ctr_gap"] = out["CTR"] - out["ctr_expected"]

    # z-scores for the detector
    feats = out[["log_clicks","log_impr","CTR","Position","ctr_gap"]]
    scaler = StandardScaler()
    out[["z_log_clicks","z_log_impr","z_CTR","z_Position","z_ctr_gap"]] = scaler.fit_transform(feats)
    return out

def _train_iforest(X, contamination, n_estimators):
    model = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=float(contamination),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X)
    return model

def _proxy_prec_k(df, scores, k=50):
    """Heuristic proxy label: high impressions (>=75th pct) & low CTR (<=25th pct)."""
    p75 = np.percentile(df["Impressions"], 75)
    p25 = np.percentile(df["CTR"], 25)
    proxy = (df["Impressions"] >= p75) & (df["CTR"] <= p25)
    order = np.argsort(scores)  # lower = more anomalous
    k = int(min(k, len(order)))
    return float(proxy.iloc[order[:k]].mean()), k

# Word freq
STOP = {
    "the","a","an","to","and","of","in","on","for","with","how","can",
    "is","are","from","by","code","using"
}

def word_freq_figure(series: pd.Series):
    """Tokenize queries: allow letters+digits, treat underscores as spaces, split on non-alphanumerics."""
    toks = []
    for q in series.astype(str):
        q = q.lower().replace("_", " ")
        parts = re.split(r"[^a-z0-9\-]+", q)  # keep hyphenated if needed
        for t in parts:
            if len(t) < 2:
                continue
            if t in STOP:
                continue
            toks.append(t)
    if not toks:
        fig = go.Figure()
        fig.update_layout(title="Top 20 Words (no tokens found — check query text)")
        return fig
    freq = Counter(toks).most_common(20)
    word_df = pd.DataFrame(freq, columns=["word","freq"])
    return px.bar(word_df, x="word", y="freq", title="Top 20 Words in Queries")

# Pipeline
def run_pipeline(file_or_default, use_default, contamination=0.01, n_estimators=200, topk=100):
    """file_or_default is gr.File or None. If use_default=True, read bundled Queries.csv."""
    try:
        if use_default:
            df = pd.read_csv("Queries.csv")  
        else:
            df = read_csv_any(file_or_default)
        df = _clean_df(df)
    except Exception as e:
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False),
                f"Error: {e}")

    eng = _engineer(df)
    X = eng[["z_log_clicks","z_log_impr","z_CTR","z_Position","z_ctr_gap"]]
    model = _train_iforest(X, contamination, n_estimators)
    scores = model.decision_function(X)
    preds  = model.predict(X)
    eng["anomaly_score"] = scores
    eng["prediction"]    = preds

    # Visuals
    heat = px.imshow(df[["Clicks","Impressions","CTR","Position"]].corr(), text_auto=True, title="Correlation Matrix")
    hist = px.histogram(eng, x="anomaly_score", nbins=50, title="Decision Score (lower = more anomalous)")
    words = word_freq_figure(df["Top queries"])

    # Anomalies table
    top = (eng[eng["prediction"]==-1]
           .sort_values("anomaly_score")
           .head(int(topk))
           [["Top queries","Clicks","Impressions","CTR","Position","ctr_expected","ctr_gap","anomaly_score"]]
           .copy())
    pretty = top.copy()
    for c in ["CTR","ctr_expected","ctr_gap"]:
        pretty[c] = (pretty[c]*100).round(2).astype(str) + "%"

    # Proxy evaluation
    p, used_k = _proxy_prec_k(eng, scores, k=int(topk))
    msg = f"Rows: {len(df)} | anomaly_rate: {(preds==-1).mean():.2%} | proxy Precision@{used_k}: {p:.2%}"

    # Downloadable CSV
    top.to_csv("anomalies.csv", index=False)
    file_out = gr.File("anomalies.csv")

    return heat, hist, words, pretty, file_out, msg

# UI
with gr.Blocks(title="SEO Query Anomaly Detection") as demo:
    gr.Markdown("# SEO Query Anomaly Detection\nUpload your Google Search Console CSV or use the bundled default to detect query-level anomalies.")

    with gr.Row():
        csv_file   = gr.File(label="Upload Queries.csv", file_types=[".csv"])
        use_default = gr.Checkbox(label="Use default bundled Queries.csv", value=True)

    with gr.Row():
        contamination = gr.Slider(0.001, 0.2, value=0.01, step=0.001, label="Contamination")
        n_estimators  = gr.Slider(100, 600, value=200, step=50, label="Trees")
        topk          = gr.Slider(10, 500, value=100, step=10, label="Max anomalies to display & eval (k)")

    run_btn = gr.Button("Run Pipeline", variant="primary")

    with gr.Tabs():
        with gr.Tab("Overview"):
            heat = gr.Plot()
            hist = gr.Plot()
            words = gr.Plot()
        with gr.Tab("Anomalies"):
            table = gr.Dataframe()
            dl = gr.File()
        with gr.Tab("Notes"):
            msg = gr.Textbox(lines=3)

    run_btn.click(
        fn=run_pipeline,
        inputs=[csv_file, use_default, contamination, n_estimators, topk],
        outputs=[heat, hist, words, table, dl, msg]
    )

if __name__ == "__main__":
    demo.launch()
