"""PDF izvještaj o segmentaciji — sklapa tekst (Claude/rule-based) + tabele u PDF.
Tekst piše LLM/insights modul; ovaj fajl ga samo formatira u uredan dokument."""
import io
import datetime as _dt

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.piecharts import Pie

# SmartSeg paleta
NAVY = colors.HexColor("#0a0f1d")
CARD = colors.HexColor("#101a2e")
PURPLE = colors.HexColor("#a855f7")
CYAN = colors.HexColor("#22d3ee")
INK = colors.HexColor("#1a1040")
GREY = colors.HexColor("#555555")

SEG_HEX = {
    "Champions": colors.HexColor("#22d3ee"), "Loyal Customers": colors.HexColor("#a855f7"),
    "Potential Loyalists": colors.HexColor("#f59e0b"), "New Customers": colors.HexColor("#7c8aa5"),
    "At Risk": colors.HexColor("#ef4444"), "Hibernating": colors.HexColor("#34d399"),
    "Ostali": colors.HexColor("#64748b"),
}


def _donut_drawing(rfm):
    """Donut segmenata kao reportlab Drawing (bez matplotlib)."""
    c = rfm["Segment"].value_counts()
    d = Drawing(440, 170)
    pie = Pie()
    pie.x, pie.y = 15, 15
    pie.width, pie.height = 140, 140
    pie.data = [int(v) for v in c.values]
    pie.innerRadiusFraction = 0.55
    pie.slices.strokeColor = colors.white
    pie.slices.strokeWidth = 1.5
    for i, name in enumerate(c.index):
        pie.slices[i].fillColor = SEG_HEX.get(name, colors.HexColor("#64748b"))
    d.add(pie)
    lx, ly = 190, 140
    for name, val in c.items():
        d.add(Rect(lx, ly, 9, 9, fillColor=SEG_HEX.get(name, colors.HexColor("#64748b")),
                   strokeColor=None))
        d.add(String(lx + 14, ly + 1, f"{_safe(name)} ({int(val)})", fontSize=8,
                     fillColor=colors.HexColor("#222222")))
        ly -= 17
    return d


def _safe(t):
    """Helvetica nema naše dijakritike -> transliteracija da nema crnih kvadrata."""
    m = {"č": "c", "ć": "c", "ž": "z", "š": "s", "đ": "dj",
         "Č": "C", "Ć": "C", "Ž": "Z", "Š": "S", "Đ": "Dj"}
    for a, b in m.items():
        t = t.replace(a, b)
    return t


def _eur(x):
    if x >= 1e6:
        return f"EUR {x/1e6:.2f}M"
    if x >= 1e3:
        return f"EUR {x/1e3:.1f}k"
    return f"EUR {x:.0f}"


def build_pdf(rfm: pd.DataFrame, summary_text: str, model_name: str = "",
              dataset_name: str = "") -> bytes:
    """Vrati PDF (bytes) sa: naslovna + KPI + sažetak + tabela segmenata + preporuke."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=16 * mm,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            title="SmartSeg AI - Izvjestaj")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=INK, fontSize=20,
                        spaceAfter=4, alignment=0)
    sub = ParagraphStyle("sub", parent=ss["Normal"], textColor=GREY, fontSize=9,
                         spaceAfter=14)
    sec = ParagraphStyle("sec", parent=ss["Heading2"], textColor=PURPLE, fontSize=13,
                        spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=10, leading=15,
                         textColor=colors.HexColor("#222222"), spaceAfter=4)
    foot = ParagraphStyle("foot", parent=ss["Normal"], fontSize=7.5, textColor=GREY,
                         alignment=1, spaceBefore=18)

    n = len(rfm)
    total_rev = rfm["Monetary"].sum()
    total_clv = rfm["PredictedCLV"].sum()
    story = []

    # naslov
    story.append(Paragraph("SmartSeg AI — Izvjestaj o segmentaciji kupaca", h1))
    meta = f"Generisano: {_dt.datetime.now():%d.%m.%Y %H:%M}"
    if dataset_name:
        meta += f"  |  Dataset: {dataset_name}"
    story.append(Paragraph(meta, sub))

    # KPI tabela
    kpi = [["Kupaca", "Prihod", "Predv. CLV", "Segmenata"],
           [f"{n:,}".replace(",", "."), _eur(total_rev), _eur(total_clv),
            str(rfm["Segment"].nunique())]]
    t = Table(kpi, colWidths=[42 * mm] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f4f1fb")),
        ("TEXTCOLOR", (0, 1), (-1, 1), INK),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, 1), 13),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 1, colors.white),
    ]))
    story.append(t)

    # sažetak
    story.append(Paragraph("Izvrsni sazetak", sec))
    for para in _safe(str(summary_text)).split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip().replace("**", ""), body))

    # tabela segmenata
    story.append(Paragraph("Profil segmenata", sec))
    seg = (rfm.groupby("Segment")
           .agg(kupaca=("CustomerID", "count"),
                clv=("PredictedCLV", "mean"),
                clv_sum=("PredictedCLV", "sum"),
                rec=("Recency", "mean"))
           .sort_values("clv_sum", ascending=False))
    rows = [["Segment", "Kupaca", "Udio", "Prosj. CLV", "Ukupna vrij.", "Recency"]]
    for name, r in seg.iterrows():
        rows.append([_safe(name), f"{int(r['kupaca']):,}".replace(",", "."),
                     f"{r['kupaca']/n*100:.0f}%", _eur(r["clv"]),
                     _eur(r["clv_sum"]), f"{r['rec']:.0f} d."])
    st = Table(rows, colWidths=[38 * mm, 20 * mm, 16 * mm, 28 * mm, 30 * mm, 22 * mm])
    st.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f5fc")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(st)

    # donut grafik segmenata
    try:
        story.append(Spacer(1, 8))
        _dr = _donut_drawing(rfm)
        _dr.hAlign = "CENTER"
        story.append(_dr)
    except Exception:
        pass

    # rizik
    risk = rfm[rfm["Segment"].isin(["At Risk", "Hibernating"])]
    if len(risk):
        story.append(Paragraph("Rizik od odlaska", sec))
        story.append(Paragraph(
            f"{len(risk):,}".replace(",", ".") +
            f" kupaca ({len(risk)/n*100:.0f}% baze) u segmentima At Risk i Hibernating. "
            f"Ugrozena vrijednost: {_eur(risk['PredictedCLV'].sum())}.", body))

    # preporuke
    story.append(Paragraph("Preporucene akcije", sec))
    tips = {
        "Champions": "VIP program i early-access - zastititi najvredniji segment.",
        "Loyal Customers": "Loyalty bodovi i upsell - rast CLV-a.",
        "Potential Loyalists": "Cross-sell i personalizovane ponude.",
        "New Customers": "Onboarding i welcome ponude.",
        "At Risk": "HITNA win-back kampanja - vrijednost na odlasku.",
        "Hibernating": "Reaktivacija popustima ili prirodan odliv.",
        "Ostali": "Pratiti ponasanje i premjestiti u jasniji segment.",
    }
    for name in seg.index:
        if name in tips:
            story.append(Paragraph(
                f"<b>{_safe(name)}</b> ({int(seg.loc[name, 'kupaca']):,}".replace(",", ".") +
                f" kupaca): {_safe(tips[name])}", body))

    if model_name:
        story.append(Paragraph(f"Najbolji model (silhouette): {model_name}", body))

    story.append(Paragraph(
        "Generisano pomocu SmartSeg AI &middot; segmentacija + CLV + RFM analiza", foot))

    doc.build(story)
    return buf.getvalue()
