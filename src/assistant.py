"""
SmartSeg AI Assistant — odgovara na pitanja o segmentima, modelima i preporukama.
Radi bez interneta (rule-based), a koristi Claude ako postoji ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os
import pandas as pd


def _context(rfm: pd.DataFrame) -> str:
    g = rfm.groupby("Segment").agg(n=("CustomerID", "count"),
                                   clv=("PredictedCLV", "mean")).round(0)
    return g.to_string()


def _rule_based(q: str, rfm: pd.DataFrame) -> str:
    q = q.lower()
    seg_counts = rfm["Segment"].value_counts()

    if any(w in q for w in ["izvješt", "izvjest", "report", "rezime", "sažetak", "sazetak"]):
        top = seg_counts.index[0]
        risk = int(rfm["Segment"].isin(["At Risk", "Hibernating"]).sum())
        champ_val = rfm.loc[rfm["Segment"] == "Champions", "PredictedCLV"].sum()
        pct = champ_val / max(rfm["PredictedCLV"].sum(), 1) * 100
        return (f"**Mini izvještaj**\n\n"
                f"- Baza: **{len(rfm):,}** kupaca u **{rfm['Segment'].nunique()}** segmenata\n"
                f"- Najbrojniji segment: **{top}** ({int(seg_counts.iloc[0]):,})\n"
                f"- Champions nose **{pct:.0f}%** predviđene vrijednosti\n"
                f"- Pod rizikom (At Risk + Hibernating): **{risk:,}** kupaca\n\n"
                f"Puni AI izvještaj: dugme „AI Izvještaj (MD)“ u Download Centru — a uz "
                f"ANTHROPIC_API_KEY ovaj chat generiše kompletan izvještaj na zahtjev.")
    if any(w in q for w in ["churn", "odlaz", "rizik", "rizič", "rizic", "napušt", "napust"]):
        n = int((rfm["Segment"].isin(["At Risk", "Hibernating"])).sum())
        return (f"Najveći rizik od odlaska imaju **At Risk** i **Hibernating** segmenti "
                f"({n} kupaca). Preporuka: retention i reaktivacijske kampanje sa "
                f"personalizovanim ponudama i win-back popustima.")
    if "model" not in q and any(w in q for w in ["champ", "chamion", "champin", "šamp",
                                                 "sampion", "najbolj", "vrijedn", "vredn", "vip"]):
        n = int(seg_counts.get("Champions", 0))
        return (f"**Champions** ({n} kupaca) su najvredniji — nedavno aktivni, česti i "
                f"visoke potrošnje. Zadrži ih premium ponudama i early-access pristupom.")
    _seg_map = {"loyal": "Loyal Customers", "lojal": "Loyal Customers",
                "potencijal": "Potential Loyalists", "potential": "Potential Loyalists",
                "novi": "New Customers", "new": "New Customers",
                "hibern": "Hibernating", "uspav": "Hibernating", "neaktiv": "Hibernating",
                "ostali": "Ostali"}
    for kw, seg in _seg_map.items():
        if kw in q and seg in seg_counts.index:
            sub = rfm[rfm["Segment"] == seg]
            return (f"**{seg}** — {len(sub):,} kupaca ({len(sub)/len(rfm)*100:.0f}% baze), "
                    f"prosječan predviđeni CLV €{sub['PredictedCLV'].mean():,.0f}, prosječno "
                    f"{sub['Recency'].mean():.0f} dana od zadnje kupovine."
                    .replace(",", "."))
    if any(w in q for w in ["model", "hdbscan", "kmeans", "gauss", "klaster"]):
        return ("Koristimo tri modela: K-Means (baseline), Gaussian Mixture (broj klastera "
                "biran po BIC-u) i HDBSCAN (gustinsko, sam nalazi klastere + outliere). "
                "Kvalitet poredimo silhouette, Davies-Bouldin i Calinski-Harabasz metrikama.")
    if any(w in q for w in ["preporuk", "akcij", "savjet", "strateg"]):
        return ("Top preporuke: 1) VIP kampanja za Champions, 2) loyalty program za Loyal "
                "Customers, 3) hitna retention kampanja za At Risk segment.")
    if any(w in q for w in ["clv", "lifetime", "vrijednost kupca"]):
        return (f"Prosječan predviđeni CLV je €{rfm['PredictedCLV'].mean():,.0f}. CLV računamo "
                f"kao prosječnu vrijednost narudžbe × očekivane buduće kupovine × p_alive "
                f"(vjerovatnoća da je kupac još aktivan).")
    return ("Mogu da objasnim segmente kupaca, ML modele, rizik od churn-a, CLV i da dam "
            "marketinške preporuke. Pitaj npr.: „Ko su Champions?“ ili „Koji segment je rizičan?“")


def ask(question: str, rfm: pd.DataFrame, use_llm: bool = True,
        model: str = "claude-sonnet-4-6") -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    wants_report = any(w in question.lower() for w in
                       ["izvješt", "izvjest", "report", "rezime", "sažetak", "sazetak"])
    if not (use_llm and api_key):
        return _rule_based(question, rfm)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        if wants_report:
            prompt = (
                "Ti si analitičar marketinga. Napiši strukturiran izvještaj na "
                "crnogorskom/srpskom o segmentaciji kupaca (250-400 riječi). "
                "Koristi isključivo običan tekst i markdown (## za naslove, ** za podebljano, "
                "- za stavke). NE koristi HTML oznake (nikakve <b>, <div>, style). "
                "NE pominji PDF ni format. Struktura: 1) pregled baze, 2) ključni segmenti i "
                "vrijednost, 3) rizici (churn), 4) konkretne preporuke. Podaci po segmentima:\n\n"
                f"{_context(rfm)}\n\nDodatni zahtjev korisnika: {question}"
            )
            max_toks = 1200
        else:
            prompt = (
                "Ti si SmartSeg AI asistent za segmentaciju kupaca. Odgovori kratko (2-4 "
                "rečenice) na crnogorskom/srpskom, na osnovu ovih podataka o segmentima:\n\n"
                f"{_context(rfm)}\n\nPitanje korisnika: {question}"
            )
            max_toks = 350
        msg = client.messages.create(model=model, max_tokens=max_toks,
                                     messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    except Exception:
        return _rule_based(question, rfm)
