"""
LLM interpretacija segmenata i 'executive summary'.

Ako je dostupan ANTHROPIC_API_KEY -> koristi pravi Claude model za bogatu,
prirodnu interpretaciju (npr. "loyal high-value buyers koji najbolje reaguju
na premium early access ponude").

Ako ključ NIJE dostupan -> koristi se transparentan fallback koji generiše
smislen tekst po pravilima, tako da app uvijek radi (bitno za reproduktivnost).
"""
from __future__ import annotations

import os
import pandas as pd

SEGMENT_BLURB = {
    "Champions": "najvredniji, nedavno aktivni kupci sa visokom učestalošću i potrošnjom",
    "Loyal Customers": "stabilni, lojalni kupci koji redovno kupuju",
    "Potential Loyalists": "kupci s dobrim potencijalom koji još nisu dostigli punu vrijednost",
    "New Customers": "novi kupci u ranoj fazi odnosa sa brendom",
    "At Risk": "vrijedni kupci koji su prestali da kupuju i prijete da odu",
    "Hibernating": "uspavani, dugo neaktivni kupci niske vrijednosti",
    "Ostali": "heterogena grupa koja ne pripada jasno nijednom profilu",
}


def _fallback_segment_insight(seg: str, n: int, total: int, clv: float,
                              share_rev: float) -> str:
    blurb = SEGMENT_BLURB.get(seg, "grupa kupaca")
    pct = (n / total * 100) if total else 0
    return (f"**{seg}** ({n} kupaca, {pct:.0f}%) — {blurb}. "
            f"Prosječan predviđeni CLV je €{clv:,.0f} i ovaj segment nosi "
            f"oko {share_rev:.0f}% ukupne predviđene vrijednosti.")


def _fallback_summary(rfm: pd.DataFrame) -> str:
    by_seg = (rfm.groupby("Segment")["PredictedCLV"]
              .sum().sort_values(ascending=False))
    total = by_seg.sum() or 1
    top = by_seg.index[0]
    top_share = by_seg.iloc[0] / total * 100
    at_risk_share = (by_seg.get("At Risk", 0) + by_seg.get("Hibernating", 0)) / total * 100
    return (
        f"Analiza pokazuje da segment **{top}** nosi najveću vrijednost "
        f"(oko {top_share:.0f}% predviđenog CLV-a) i treba mu prioritet u zadržavanju. "
        f"Istovremeno, oko {at_risk_share:.0f}% vrijednosti je vezano za rizične i "
        f"uspavane kupce — ciljanom retention/reaktivacijskom kampanjom tu se "
        f"krije najveći potencijal rasta prihoda."
    )


def segment_insights(rfm: pd.DataFrame) -> list[dict]:
    """Vrati listu kratkih uvida po segmentu (za 'AI Insights' panel)."""
    total = len(rfm)
    total_clv = float(rfm["PredictedCLV"].sum()) or 1
    out = []
    g = rfm.groupby("Segment").agg(
        n=("CustomerID", "count"),
        clv=("PredictedCLV", "mean"),
        clv_sum=("PredictedCLV", "sum"),
    )
    for seg, row in g.iterrows():
        out.append({
            "segment": seg,
            "text": _fallback_segment_insight(
                seg, int(row["n"]), total, float(row["clv"]),
                row["clv_sum"] / total_clv * 100),
        })
    return out


def build_full_report(rfm: pd.DataFrame, model_name: str = "", use_llm: bool = True) -> str:
    """Bogat MD izvještaj: KPI + profil svih segmenata + rizici + preporuke.
    Tekstualni uvod piše Claude ako postoji ključ, inače strukturiran fallback."""
    import datetime as _dt
    n = len(rfm)
    total_rev = rfm["Monetary"].sum()
    total_clv = rfm["PredictedCLV"].sum()
    seg = (rfm.groupby("Segment")
           .agg(kupaca=("CustomerID", "count"),
                udio=("CustomerID", lambda s: len(s) / max(n, 1) * 100),
                clv=("PredictedCLV", "mean"),
                clv_sum=("PredictedCLV", "sum"),
                recency=("Recency", "mean"),
                freq=("Frequency", "mean"))
           .sort_values("clv_sum", ascending=False))

    def _eur(x):
        if x >= 1e6:
            return f"€{x/1e6:.2f}M"
        if x >= 1e3:
            return f"€{x/1e3:.1f}k"
        return f"€{x:.0f}"

    L = []
    L.append("# 📊 SmartSeg AI — Izvještaj o segmentaciji kupaca")
    L.append(f"*Generisano: {_dt.datetime.now():%d.%m.%Y %H:%M}*\n")
    L.append("## 1. Pregled")
    L.append(f"- **Ukupno kupaca:** {n:,}".replace(",", "."))
    L.append(f"- **Ukupan prihod:** {_eur(total_rev)}")
    L.append(f"- **Predviđeni CLV baze:** {_eur(total_clv)}")
    L.append(f"- **Broj segmenata:** {rfm['Segment'].nunique()}")
    if model_name:
        L.append(f"- **Najbolji model (silhouette):** {model_name}")
    L.append("")

    L.append("## 2. Izvršni sažetak")
    L.append(executive_summary(rfm, use_llm=use_llm) + "\n")

    L.append("## 3. Profil segmenata\n")
    L.append("| Segment | Kupaca | Udio | Prosj. CLV | Ukupna vrijednost | Prosj. Recency |")
    L.append("|---|---|---|---|---|---|")
    for name, r in seg.iterrows():
        L.append(f"| {name} | {int(r['kupaca']):,} | {r['udio']:.0f}% | {_eur(r['clv'])} | "
                 f"{_eur(r['clv_sum'])} | {r['recency']:.0f} d. |".replace(",", "."))
    L.append("")

    risk = rfm[rfm["Segment"].isin(["At Risk", "Hibernating"])]
    if len(risk):
        L.append("## 4. Rizik od odlaska")
        L.append(f"- **{len(risk):,}** kupaca ({len(risk)/n*100:.0f}% baze) u segmentima "
                 f"At Risk i Hibernating".replace(",", "."))
        L.append(f"- Ugrožena vrijednost: **{_eur(risk['PredictedCLV'].sum())}**\n")

    L.append("## 5. Preporučene akcije")
    tips = {
        "Champions": "VIP program i early-access — zaštititi najvredniji segment",
        "Loyal Customers": "Loyalty bodovi i upsell — rast CLV-a",
        "Potential Loyalists": "Cross-sell i personalizovane ponude — gurnuti ka lojalnosti",
        "New Customers": "Onboarding i welcome ponude — izgraditi naviku",
        "At Risk": "HITNA win-back kampanja — vrijednost na odlasku",
        "Hibernating": "Reaktivacija agresivnim popustima ili prirodan odliv",
        "Ostali": "Pratiti ponašanje i premjestiti u jasniji segment",
    }
    for name in seg.index:
        if name in tips:
            L.append(f"- **{name}** ({int(seg.loc[name, 'kupaca']):,} kupaca): {tips[name]}"
                     .replace(",", "."))
    L.append("\n---\n*SmartSeg AI · interpretacija pokretana "
             + ("Claude LLM-om" if os.environ.get("ANTHROPIC_API_KEY") and use_llm
                else "rule-based modulom") + "*")
    return "\n".join(L)


def executive_summary(rfm: pd.DataFrame, use_llm: bool = True,
                      model: str = "claude-sonnet-4-6") -> str:
    """
    Generiši izvršni rezime. Pokušava Claude API ako postoji ključ,
    u suprotnom koristi fallback po pravilima.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not (use_llm and api_key):
        return _fallback_summary(rfm)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        stats = (rfm.groupby("Segment")
                 .agg(kupaca=("CustomerID", "count"),
                      prosjecan_clv=("PredictedCLV", "mean"),
                      ukupan_clv=("PredictedCLV", "sum"))
                 .round(0).to_string())

        prompt = (
            "Ti si analitičar marketinga. Na osnovu sljedeće segmentacije kupaca "
            "(RFM + CLV) napiši kratak izvršni rezime na crnogorskom/srpskom jeziku "
            "(3-4 rečenice), bez markdown naslova, fokusiran na konkretnu akciju i "
            "rast prihoda. VAŽNO: kad navodiš vrijednost segmenta, dosljedno koristi "
            "UKUPNU vrijednost (ukupan_clv) za sve segmente — ne miješaj prosječni i "
            "ukupni CLV u istom tekstu. Pazi na gramatiku (npr. '1.316 kupaca', ne "
            "'1.316 kupac'). Ne navodi simbol valute (biće dodat naknadno), samo brojeve "
            "u milionima/hiljadama:\n\n" + stats
        )
        msg = client.messages.create(
            model=model, max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    except Exception:
        # Bilo kakva greška (nema paketa, mreža, kvota) -> fallback
        return _fallback_summary(rfm)
