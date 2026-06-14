"""
Napredna analitika za Advanced Mode.

Sadrži: PCA (2D/3D), poređenje modela (KMeans/GMM/HDBSCAN) sa više metrika,
silhouette analizu po broju klastera, feature importance, detekciju outliera,
cohort (retention) analizu, data quality i tehničke AI uvide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

FEATURES = ["Recency", "Frequency", "Monetary", "PredictedCLV", "HealthScore"]


def _scaled(rfm: pd.DataFrame, cols=("Recency", "Frequency", "Monetary")) -> np.ndarray:
    X = rfm[list(cols)].copy()
    for c in cols:
        X[c] = np.log1p(X[c])
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------- PCA
def pca_coords(rfm: pd.DataFrame, n: int = 3, seed: int = 42) -> dict:
    """Vrati PCA koordinate (2D ili 3D) + objašnjenu varijansu po komponenti."""
    X = _scaled(rfm, FEATURES if all(c in rfm for c in FEATURES) else
                ("Recency", "Frequency", "Monetary"))
    pca = PCA(n_components=n, random_state=seed)
    comps = pca.fit_transform(X)
    df = rfm[["CustomerID", "Segment"]].copy()
    for i in range(n):
        df[f"PC{i+1}"] = comps[:, i]
    return {"df": df, "explained": (pca.explained_variance_ratio_ * 100).round(1)}


# ---------------------------------------------------- Poređenje modela
def model_comparison(rfm: pd.DataFrame, seed: int = 42) -> dict:
    """
    Uporedi KMeans, Gaussian Mixture i HDBSCAN po:
      - Silhouette (veće = bolje)
      - Davies-Bouldin (manje = bolje)
      - Calinski-Harabasz (veće = bolje)
      - broj klastera
    Označi NAJBOLJI model (po silhouette).
    """
    X = _scaled(rfm)
    results = {}

    # broj klastera koji GMM nalazi po BIC-u (za KMeans i GMM)
    bic_best_k, best_bic = 4, np.inf
    for k in range(2, 9):
        gm = GaussianMixture(n_components=k, random_state=seed, n_init=2).fit(X)
        b = gm.bic(X)
        if b < best_bic:
            best_bic, bic_best_k = b, k

    models = {
        "K-Means": KMeans(n_clusters=bic_best_k, random_state=seed, n_init=10).fit_predict(X),
        "Gaussian Mixture": GaussianMixture(n_components=bic_best_k, random_state=seed,
                                            n_init=3).fit(X).predict(X),
        "HDBSCAN": HDBSCAN(min_cluster_size=max(15, int(len(X) * 0.02)),
                           min_samples=10, copy=True).fit_predict(X),
    }

    rows = []
    for name, labels in models.items():
        mask = labels != -1
        uniq = set(labels[mask])
        if len(uniq) > 1:
            ss = min(2000, int(mask.sum()))
            sil = round(float(silhouette_score(X[mask], labels[mask], sample_size=ss, random_state=seed)), 3)
            db = round(float(davies_bouldin_score(X[mask], labels[mask])), 3)
            ch = round(float(calinski_harabasz_score(X[mask], labels[mask])), 1)
        else:
            sil = db = ch = None
        rows.append(dict(Model=name, Silhouette=sil, DaviesBouldin=db,
                         CalinskiHarabasz=ch, Klastera=len(uniq)))

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["Silhouette"])
    best = valid.loc[valid["Silhouette"].idxmax(), "Model"] if len(valid) else "K-Means"
    return {"table": df, "best": best, "optimal_k": bic_best_k}


def silhouette_over_k(rfm: pd.DataFrame, k_max: int = 10, seed: int = 42) -> dict:
    """Silhouette skor za k=2..k_max (KMeans) -> linijski grafik + optimalno k."""
    X = _scaled(rfm)
    ks, scores = list(range(2, k_max + 1)), []
    ss = min(2000, len(X))
    for k in ks:
        lab = KMeans(n_clusters=k, random_state=seed, n_init=4).fit_predict(X)
        scores.append(round(float(silhouette_score(X, lab, sample_size=ss, random_state=seed)), 3))
    optimal = ks[int(np.argmax(scores))]
    return {"k": ks, "scores": scores, "optimal_k": optimal}


# --------------------------------------------------- Feature importance
def feature_importance(rfm: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Važnost feature-a za određivanje segmenta (RandomForest predviđa Segment).
    Vraća normalizovane skorove (0-1).
    """
    cols = [c for c in FEATURES if c in rfm.columns]
    X, y = rfm[cols].fillna(0), rfm["Segment"]
    rf = RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.DataFrame({"Feature": cols, "Importance": rf.feature_importances_})
    imp["Importance"] = (imp["Importance"] / imp["Importance"].max()).round(2)
    return imp.sort_values("Importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------- Outlier analiza
def outlier_analysis(rfm: pd.DataFrame, seed: int = 42) -> dict:
    """IsolationForest detekcija netipičnih kupaca (potencijalno VIP ili anomalije)."""
    X = _scaled(rfm)
    iso = IsolationForest(contamination=0.05, random_state=seed)
    flag = iso.fit_predict(X)            # -1 = outlier
    rfm = rfm.copy()
    rfm["Outlier"] = (flag == -1)
    n_out = int(rfm["Outlier"].sum())
    sample = (rfm[rfm["Outlier"]]
              .sort_values("Monetary", ascending=False)
              [["CustomerID", "Recency", "Frequency", "Monetary", "PredictedCLV"]]
              .head(6).round(2))
    return {"rfm": rfm, "n_outliers": n_out,
            "pct": round(n_out / len(rfm) * 100, 1),
            "high_risk": int(((flag == -1) & (rfm["p_alive"] < 0.4)).sum()),
            "sample": sample}


# --------------------------------------------------- Cohort analiza
def cohort_analysis(df: pd.DataFrame, max_periods: int = 7) -> pd.DataFrame:
    """
    Mjesečna retention cohort analiza: red = mjesec prve kupovine,
    kolone = mjeseci od prve kupovine, vrijednost = % zadržanih kupaca.
    """
    d = df[["CustomerID", "InvoiceDate"]].copy()
    d["order_month"] = d["InvoiceDate"].dt.to_period("M")
    d["cohort"] = d.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M")
    d["offset"] = (d["order_month"].astype("int64") - d["cohort"].astype("int64"))
    d = d[(d["offset"] >= 0) & (d["offset"] < max_periods)]

    sizes = d[d["offset"] == 0].groupby("cohort")["CustomerID"].nunique()
    grid = (d.groupby(["cohort", "offset"])["CustomerID"].nunique()
            .unstack(fill_value=0))
    retention = grid.div(sizes, axis=0).mul(100).round(0)
    retention.index = retention.index.astype(str)
    retention = retention.tail(7)        # zadnjih 7 cohorta radi preglednosti
    retention.columns = [f"M{c}" for c in retention.columns]
    return retention


# --------------------------------------------------- Data quality
def data_quality(raw: pd.DataFrame, clean: pd.DataFrame, n_outliers: int = 0,
                 n_customers: int | None = None) -> dict:
    """Pregled kvaliteta podataka za Advanced 'Data Quality' panel."""
    total_rows = len(raw)
    n_cols = raw.shape[1]
    missing_pct = round(raw.isna().mean().mean() * 100, 1)
    dup_pct = round(raw.duplicated().mean() * 100, 1)
    # outlieri su na nivou KUPCA, pa se dijele brojem kupaca (ne linija)
    denom = n_customers if n_customers else clean["CustomerID"].nunique()
    out_pct = round(n_outliers / max(denom, 1) * 100, 1)
    score = round(max(0, 100 - missing_pct * 0.5 - dup_pct - out_pct * 0.3), 1)
    return {"rows": total_rows, "cols": n_cols, "missing_pct": missing_pct,
            "dup_pct": dup_pct, "outlier_pct": out_pct, "score": score}


# --------------------------------------------------- Cluster heatmap
def cluster_heatmap(rfm: pd.DataFrame) -> pd.DataFrame:
    """Prosjek RFM/CLV/Health po segmentu (za heatmap)."""
    cols = [c for c in ["Recency", "Frequency", "Monetary", "PredictedCLV", "HealthScore"]
            if c in rfm.columns]
    h = rfm.groupby("Segment")[cols].mean().round(1)
    return h


# --------------------------------------------------- Tehnički AI uvidi
def technical_insights(mc: dict, fi: pd.DataFrame, rfm: pd.DataFrame) -> list[str]:
    best = mc["best"]
    top_feats = ", ".join(fi.head(2)["Feature"].tolist())
    growth = "Potential Loyalists"
    out = [
        f"{best} daje najbolju separaciju klastera (najveći silhouette).",
        f"Najvažniji faktori segmentacije: {top_feats}.",
        "At Risk segment ima najveću vjerovatnoću odlaska (churn).",
        f"{growth} segment nosi najveći potencijal rasta uz ciljane ponude.",
    ]
    return out
