"""
ROI 'what-if' simulator.

Menadžer pomjera slidere (budžet, popusti, retention) i vidi predviđeni ROI,
broj zadržanih kupaca i profit. Model je jednostavan i transparentan:
oslanja se na predviđeni CLV po segmentu i pretpostavke o elastičnosti.
"""
from __future__ import annotations

import pandas as pd

# Pretpostavljene elastičnosti (koliko 1% poteza pomjera ishod).
# Namjerno konzervativne i objašnjive - menadžer ih može mijenjati.
BUDGET_LIFT = 0.6      # +1% budžeta -> +0.6% dosega/konverzije (opadajući prinos)
DISCOUNT_LIFT = 0.8    # +1% popusta -> +0.8% konverzije, ali jede maržu
RETENTION_LIFT = 1.0   # +1% retention poteza -> +1% zadržanih at-risk kupaca


def simulate_roi(rfm: pd.DataFrame,
                 budget_increase_pct: float = 0.0,
                 discount_increase_pct: float = 0.0,
                 retention_increase_pct: float = 0.0,
                 base_margin: float = 0.25) -> dict:
    """
    Vrati predviđene ishode na osnovu poteza (sve u procentima, npr. 20 = +20%).

    Logika (transparentna i objašnjiva):
      inkrementalni prihod = od budžeta + od popusta + od zadržanih at-risk kupaca
      inkrementalni profit = inkrementalni prihod * marža - troškovi poteza
      troškovi = dodatni marketing + maržа data kroz popuste + trošak retention-a
    """
    total_clv = float(rfm["PredictedCLV"].sum())
    n_customers = int(len(rfm))
    at_risk = rfm[rfm["Segment"].isin(["At Risk", "Hibernating"])]
    at_risk_value = float(at_risk["PredictedCLV"].sum())
    n_at_risk = int(len(at_risk))

    b, d, r = budget_increase_pct / 100.0, discount_increase_pct / 100.0, retention_increase_pct / 100.0

    # 1) Inkrementalni prihod (opadajući prinos je već u koeficijentima <1)
    inc_from_levers = total_clv * (b * BUDGET_LIFT + d * DISCOUNT_LIFT)
    retained_fraction = min(0.6, r * RETENTION_LIFT)
    retained_customers = int(round(n_at_risk * retained_fraction))
    retained_value = at_risk_value * retained_fraction
    incremental_revenue = inc_from_levers + retained_value

    # 2) Troškovi poteza
    cost_marketing = total_clv * 0.012 * b                 # dodatni ad spend
    cost_discount = inc_from_levers * d * 0.40             # marža data kroz popuste
    cost_retention = retained_value * 0.10                 # trošak retention kampanje
    total_cost = cost_marketing + cost_discount + cost_retention + 1.0

    # 3) Profit i ROI (ROI = neto profit / trošak poteza)
    incremental_margin = incremental_revenue * base_margin
    predicted_profit = incremental_margin - (cost_marketing + cost_discount + cost_retention)
    roi_pct = (predicted_profit / total_cost * 100.0) if total_cost else 0.0

    return {
        "predicted_roi_pct": round(roi_pct, 1),
        "retained_customers": retained_customers,
        "predicted_profit": round(predicted_profit, 2),
        "extra_revenue": round(incremental_revenue, 2),
        "total_cost": round(total_cost, 2),
        "n_customers": n_customers,
        "n_at_risk": n_at_risk,
    }
