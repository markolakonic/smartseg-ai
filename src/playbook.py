"""
Marketing playbook - preporučena strategija i kanal po segmentu.
Ovo je 'akcioni' dio: priča o kupcima -> konkretna akcija (kao na dashboardu).
"""
from __future__ import annotations

import pandas as pd

PLAYBOOK = {
    "Champions": dict(
        opis="Najvredniji kupci",
        strategija="Premium early access, ekskluzivne ponude, VIP tretman",
        kanal="Email, SMS",
        prioritet="Zadržavanje",
    ),
    "Loyal Customers": dict(
        opis="Lojalni i aktivni kupci",
        strategija="Loyalty rewards, bodovi, personalizacija, upsell",
        kanal="Email, App",
        prioritet="Rast vrijednosti",
    ),
    "Potential Loyalists": dict(
        opis="Kupci s potencijalom",
        strategija="Personalizovane ponude, cross-sell, podsticaji za ponovnu kupovinu",
        kanal="Email, Push",
        prioritet="Konverzija u lojalne",
    ),
    "New Customers": dict(
        opis="Novi kupci",
        strategija="Welcome ponude, onboarding, edukacija o asortimanu",
        kanal="Email, App",
        prioritet="Aktivacija",
    ),
    "At Risk": dict(
        opis="Rizik odlaska",
        strategija="Retention kampanja, win-back, specijalne ponude, ankete",
        kanal="Email, SMS",
        prioritet="Hitno zadržavanje",
    ),
    "Hibernating": dict(
        opis="Neaktivni kupci",
        strategija="Reaktivacijske kampanje, agresivniji popusti, 'nedostajete nam'",
        kanal="Email",
        prioritet="Reaktivacija",
    ),
    "Ostali": dict(
        opis="Ostali kupci",
        strategija="Standardne kampanje, A/B testiranje pristupa",
        kanal="Email",
        prioritet="Posmatranje",
    ),
}


def build_playbook(rfm: pd.DataFrame) -> pd.DataFrame:
    """Spoj playbook pravila sa stvarnim brojkama po segmentu."""
    agg = (rfm.groupby("Segment")
           .agg(Kupaca=("CustomerID", "count"),
                ProsjecanCLV=("PredictedCLV", "mean"),
                UkupnaVrijednost=("PredictedCLV", "sum"))
           .reset_index())

    rows = []
    for _, r in agg.iterrows():
        seg = r["Segment"]
        pb = PLAYBOOK.get(seg, PLAYBOOK["Ostali"])
        rows.append({
            "Segment": seg,
            "Opis": pb["opis"],
            "Kupaca": int(r["Kupaca"]),
            "Prosječan CLV": round(float(r["ProsjecanCLV"]), 2),
            "Preporučena strategija": pb["strategija"],
            "Glavni kanal": pb["kanal"],
            "Prioritet": pb["prioritet"],
        })
    order = {s: i for i, s in enumerate(PLAYBOOK.keys())}
    return pd.DataFrame(rows).sort_values(
        "Segment", key=lambda s: s.map(order)).reset_index(drop=True)
