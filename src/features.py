"""
Feature engineering: RFM, izvedene metrike i predikcija CLV-a.

RFM (Recency, Frequency, Monetary) je osnova segmentacije kupaca.
CLV (Customer Lifetime Value) se procjenjuje preko transparentne, objašnjive
formule koja diskontuje vrijednost vjerovatnoćom da je kupac još 'živ' (p_alive),
izvedenom iz njegovog ličnog ritma kupovine (inter-purchase time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MARGIN = 0.25          # bruto marža za pretvaranje prihoda u profit
DEFAULT_HORIZON_MONTHS = 12    # horizont predikcije CLV-a


def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Izračunaj RFM i izvedene karakteristike po kupcu.

    Recency  - broj dana od poslednje kupovine do snapshot datuma
    Frequency- broj jedinstvenih narudžbi (faktura)
    Monetary - ukupna potrošnja
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    g = df.groupby("CustomerID")
    rfm = g.agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        NumItems=("Quantity", "sum"),
        NumProducts=("StockCode", "nunique") if "StockCode" in df.columns else ("InvoiceNo", "nunique"),
        Country=("Country", lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"),
    ).reset_index()

    # Izvedene metrike
    rfm["Tenure"] = (snapshot_date - rfm["FirstPurchase"]).dt.days.clip(lower=1)
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"].clip(lower=1)
    # prosječno vrijeme između kupovina (dani); za 1 kupovinu -> tenure
    rfm["InterPurchaseDays"] = np.where(
        rfm["Frequency"] > 1,
        rfm["Tenure"] / (rfm["Frequency"] - 1),
        rfm["Tenure"],
    )
    return rfm


def add_clv(rfm: pd.DataFrame,
            margin: float = DEFAULT_MARGIN,
            horizon_months: int = DEFAULT_HORIZON_MONTHS) -> pd.DataFrame:
    """
    Procijeni 'p_alive' i predviđeni CLV za zadati horizont.

    p_alive: vjerovatnoća da je kupac još aktivan. Modeluje se eksponencijalnim
    opadanjem u odnosu na to koliko Recency premašuje tipičan razmak između
    kupovina tog kupca:  p_alive = exp(-Recency / (InterPurchaseDays + 7))
    (klipovano u [0.02, 0.99]).

    Predicted CLV (prihod) = AOV * očekivane_kupovine_u_horizontu * p_alive
    Predicted CLV (profit) = Predicted CLV (prihod) * margin
    """
    rfm = rfm.copy()
    median_ipt = float(np.median(rfm["InterPurchaseDays"])) or 30.0

    ipt = rfm["InterPurchaseDays"].replace(0, median_ipt)
    rfm["p_alive"] = np.exp(-rfm["Recency"] / (ipt + 7.0)).clip(0.02, 0.99)

    # godišnja frekvencija -> očekivane kupovine u horizontu
    annual_freq = 365.0 / ipt.clip(lower=1)
    expected_orders = annual_freq * (horizon_months / 12.0)
    expected_orders = expected_orders.clip(upper=rfm["Frequency"].max() * 2)

    rfm["PredictedCLV"] = (rfm["AvgOrderValue"] * expected_orders * rfm["p_alive"]).round(2)
    rfm["PredictedProfit"] = (rfm["PredictedCLV"] * margin).round(2)
    rfm["HistoricalValue"] = rfm["Monetary"].round(2)
    return rfm


def add_health_score(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Customer Health Score (0-100): kompozit zdravlja odnosa sa kupcem.
    Kombinuje Recency (manje = bolje), Frequency, Monetary i p_alive,
    svedene na percentile (rank), pa ponderisane.
    """
    rfm = rfm.copy()
    r = rfm["Recency"].rank(pct=True)            # veći percentil = lošije (stariji)
    f = rfm["Frequency"].rank(pct=True)
    m = rfm["Monetary"].rank(pct=True)
    p = rfm["p_alive"] if "p_alive" in rfm.columns else 0.5
    rfm["HealthScore"] = (100 * (0.30 * (1 - r) + 0.25 * f + 0.25 * m + 0.20 * p)).round(1)
    return rfm


def rfm_scores(rfm: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """
    Dodijeli skorove 1..bins po kvantilima.
    Recency je obrnut (manji recency = bolji => veći skor).
    """
    rfm = rfm.copy()

    def _score(series, reverse=False):
        try:
            ranks = pd.qcut(series.rank(method="first"), bins, labels=False) + 1
        except ValueError:  # premalo jedinstvenih vrijednosti
            ranks = pd.cut(series.rank(method="first"), bins, labels=False) + 1
        ranks = ranks.astype(int)
        return (bins + 1 - ranks) if reverse else ranks

    rfm["R_score"] = _score(rfm["Recency"], reverse=True)
    rfm["F_score"] = _score(rfm["Frequency"])
    rfm["M_score"] = _score(rfm["Monetary"])
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
    return rfm
