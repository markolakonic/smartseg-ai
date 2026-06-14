"""
Segmentacija kupaca - HIBRIDNI pristup.

1) Pravilima zasnovani (rule-based) RFM segmenti -> daju ČOVJEKU RAZUMLJIVA imena
   (Champions, Loyal Customers, At Risk, ...). Ovo je ono što menadžer vidi.

2) Nenadgledano (data-driven) klasterovanje -> otkriva prirodne grupe u podacima
   bez ručnog podešavanja:
     - GaussianMixture sa AUTOMATSKIM brojem klastera (po BIC kriterijumu)
     - HDBSCAN (gustinsko klasterovanje, sam bira broj klastera)
     - KMeans (baseline za poređenje)
   Kvalitet se mjeri silhouette skorom -> 'evaluacija + poređenje sa baseline'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score

# Standardna mapa: (R_level, FM_level) -> ime segmenta.
# R i FM se svode na 3 nivoa (low/mid/high) radi preglednosti.
SEGMENT_NAMES = [
    "Champions", "Loyal Customers", "Potential Loyalists",
    "New Customers", "At Risk", "Hibernating", "Ostali",
]

SEGMENT_COLORS = {
    "Champions": "#22d3ee",
    "Loyal Customers": "#a855f7",
    "Potential Loyalists": "#f59e0b",
    "At Risk": "#ef4444",
    "Hibernating": "#10b981",
    "New Customers": "#64748b",
    "Ostali": "#475569",
}


# ----------------------------------------------------------------------------
# 1) Rule-based RFM imenovani segmenti
# ----------------------------------------------------------------------------
def assign_rfm_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Mapiraj R/F/M skorove (1..5) na poslovno razumljiva imena segmenata."""
    rfm = rfm.copy()
    R, F, M = rfm["R_score"], rfm["F_score"], rfm["M_score"]
    FM = (F + M) / 2.0

    def classify(r, fm, freq):
        if r >= 4 and fm >= 4:
            return "Champions"
        if r >= 3 and fm >= 3:
            return "Loyal Customers"
        if r >= 4 and fm <= 2:
            # nedavni, ali malo kupuju -> ili novi ili potencijal
            return "New Customers" if freq <= 2 else "Potential Loyalists"
        if r >= 3 and fm >= 2:
            return "Potential Loyalists"
        if r <= 2 and fm >= 3:
            return "At Risk"
        if r <= 2 and fm <= 2:
            return "Hibernating"
        return "Ostali"

    rfm["Segment"] = [
        classify(r, fm, fr)
        for r, fm, fr in zip(R, FM, rfm["Frequency"])
    ]
    return rfm


# ----------------------------------------------------------------------------
# 2) Data-driven klasterovanje
# ----------------------------------------------------------------------------
def _feature_matrix(rfm: pd.DataFrame) -> np.ndarray:
    """Log-transform (smanjuje skewnost) + standardizacija RFM-a."""
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])
    X["Recency"] = np.log1p(X["Recency"])
    return StandardScaler().fit_transform(X)


def auto_gmm(X: np.ndarray, k_min: int = 2, k_max: int = 8, seed: int = 42):
    """GaussianMixture sa automatskim izborom broja komponenti po BIC-u."""
    best, best_bic, bic_curve = None, np.inf, {}
    for k in range(k_min, k_max + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=seed, n_init=3)
        gm.fit(X)
        bic = gm.bic(X)
        bic_curve[k] = float(bic)
        if bic < best_bic:
            best, best_bic = gm, bic
    labels = best.predict(X)
    return labels, best.n_components, bic_curve


def run_clustering(rfm: pd.DataFrame, method: str = "GMM (auto)",
                   seed: int = 42) -> dict:
    """
    Pokreni izabrani metod klasterovanja i vrati labele + metrike.
    method: 'GMM (auto)' | 'HDBSCAN' | 'KMeans'
    """
    X = _feature_matrix(rfm)
    info = {"method": method, "bic_curve": None}

    if method == "HDBSCAN":
        min_size = max(15, int(len(X) * 0.02))
        model = HDBSCAN(min_cluster_size=min_size, min_samples=10, copy=True)
        labels = model.fit_predict(X)
        info["n_clusters"] = int(len(set(labels)) - (1 if -1 in labels else 0))
        info["n_noise"] = int((labels == -1).sum())
    elif method == "KMeans":
        # baseline: broj klastera = broj nenadgledanih GMM klastera ili 5
        labels, k, _ = auto_gmm(X, seed=seed)
        km = KMeans(n_clusters=max(2, k), random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        info["n_clusters"] = int(k)
    else:  # GMM (auto)
        labels, k, bic = auto_gmm(X, seed=seed)
        info["n_clusters"] = int(k)
        info["bic_curve"] = bic

    # Silhouette (samo ako >1 klaster i bez previše šuma)
    mask = labels != -1
    info["silhouette"] = None
    if len(set(labels[mask])) > 1:
        try:
            info["silhouette"] = float(round(silhouette_score(X[mask], labels[mask]), 3))
        except Exception:
            info["silhouette"] = None

    rfm = rfm.copy()
    rfm["Cluster"] = labels
    return {"rfm": rfm, "info": info, "X": X}


def evaluate_baselines(rfm: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Uporedi GMM (auto) vs KMeans (baseline) vs HDBSCAN po silhouette skoru.
    Koristi se za sekciju 'evaluacija modela i poređenje sa baseline'.
    """
    X = _feature_matrix(rfm)
    rows = []

    # GMM auto
    labels, k, _ = auto_gmm(X, seed=seed)
    rows.append(("GMM (auto, BIC)", k, _safe_sil(X, labels)))

    # KMeans baseline (isti broj klastera)
    km = KMeans(n_clusters=max(2, k), random_state=seed, n_init=10).fit_predict(X)
    rows.append(("KMeans (baseline)", k, _safe_sil(X, km)))

    # HDBSCAN
    min_size = max(15, int(len(X) * 0.02))
    hb = HDBSCAN(min_cluster_size=min_size, min_samples=10, copy=True).fit_predict(X)
    nh = len(set(hb)) - (1 if -1 in hb else 0)
    rows.append(("HDBSCAN", nh, _safe_sil(X, hb)))

    return pd.DataFrame(rows, columns=["Model", "Broj klastera", "Silhouette"])


def _safe_sil(X, labels):
    mask = labels != -1
    if len(set(labels[mask])) <= 1:
        return None
    try:
        return round(float(silhouette_score(X[mask], labels[mask])), 3)
    except Exception:
        return None


def profile_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    """Prosječni RFM/CLV profil po data-driven klasteru (za interpretaciju)."""
    return (rfm[rfm["Cluster"] != -1]
            .groupby("Cluster")
            .agg(Velicina=("CustomerID", "count"),
                 Recency=("Recency", "mean"),
                 Frequency=("Frequency", "mean"),
                 Monetary=("Monetary", "mean"),
                 CLV=("PredictedCLV", "mean"))
            .round(1)
            .reset_index())
