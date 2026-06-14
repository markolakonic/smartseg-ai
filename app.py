"""
SmartSeg AI — AI Customer Intelligence Platform (Streamlit)
===========================================================
Standard Mode (Manager View)  +  Advanced Mode (Data Mining View)

Pokretanje:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import read_any, clean_transactions, dataset_summary
from src.features import build_rfm, add_clv, add_health_score, rfm_scores
from src.segmentation import assign_rfm_segments, run_clustering, SEGMENT_COLORS
from src.playbook import build_playbook
from src.roi import simulate_roi
from src.insights import segment_insights, executive_summary, build_full_report
from src.pdf_report import build_pdf
from src import analytics as A
from src.assistant import ask as assistant_ask

_FAV = Path(__file__).parent / "assets" / "favicon.png"
st.set_page_config(page_title="SmartSeg AI", page_icon=str(_FAV) if _FAV.exists() else "🌍",
                   layout="wide", initial_sidebar_state="expanded")

PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#cbd5e1", size=13), margin=dict(l=10, r=10, t=10, b=10))

LOGO_SVG = """
<svg width="40" height="40" viewBox="0 0 48 48" fill="none" style="flex:0 0 auto">
  <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#22d3ee"/><stop offset="100%" stop-color="#a855f7"/></linearGradient></defs>
  <circle cx="24" cy="24" r="20" fill="none" stroke="url(#lg)" stroke-width="2.4"/>
  <ellipse cx="24" cy="24" rx="9" ry="20" fill="none" stroke="url(#lg)" stroke-width="1.4" opacity="0.65"/>
  <line x1="4" y1="24" x2="44" y2="24" stroke="url(#lg)" stroke-width="1.4" opacity="0.65"/>
  <circle cx="24" cy="4" r="2.8" fill="#22d3ee"/><circle cx="44" cy="24" r="2.8" fill="#a855f7"/>
  <circle cx="24" cy="44" r="2.8" fill="#a855f7"/><circle cx="4" cy="24" r="2.8" fill="#22d3ee"/>
  <circle cx="24" cy="24" r="3.6" fill="url(#lg)"/>
</svg>"""


# ----------------------------------------------------------------- CSS
st.markdown("""
<style>
.stApp{background:radial-gradient(1200px 600px at 80% -10%,#1a1145 0%,#070b16 48%);}
section[data-testid="stSidebar"]{background:#0a0f1d;border-right:1px solid #1d2942;}
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1600px;}
[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(155deg,#101a2e,#0d1322);border:1px solid #233047!important;
  border-radius:16px;box-shadow:0 10px 26px rgba(0,0,0,.32);}
.kpi{transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;}
.kpi:hover{transform:translateY(-3px);border-color:rgba(168,85,247,.55)!important;
  box-shadow:0 10px 26px rgba(168,85,247,.14);}
[data-testid="stVerticalBlockBorderWrapper"]{transition:box-shadow .2s ease, border-color .2s ease;border-radius:16px;}
[data-testid="stVerticalBlockBorderWrapper"]:hover{box-shadow:0 8px 26px rgba(34,211,238,.07);}
.kpi{background:linear-gradient(155deg,#121a2e,#0e1424);border:1px solid #233047;
  border-radius:16px;padding:14px 16px 10px;height:100%;min-height:178px;
  display:flex;flex-direction:column;}
.kpi .lbl{font-size:.74rem;color:#8aa0c0;font-weight:600}
.kpi .ic{float:right;font-size:1.25rem}
.kpi .val{font-size:1.5rem;font-weight:800;color:#f1f5f9;margin:6px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi .up{color:#34d399;font-size:.76rem;font-weight:700}
.kpi .down{color:#f87171;font-size:.76rem;font-weight:700}
.kpi .s{color:#7c8aa5;font-size:.74rem;font-weight:500}
.kpi .kpi-sub{min-height:34px;line-height:1.15}
.alertc{border-radius:14px;padding:14px 16px;height:100%;display:flex;gap:12px;align-items:flex-start}
.alertc .em{font-size:1.4rem}
.alertc b{display:block;font-size:.92rem;margin-bottom:2px}
.alertc span{font-size:.8rem;opacity:.85}
.a-red{background:rgba(239,68,68,.10);border:1px solid rgba(239,68,68,.32);color:#fca5a5}
.a-amber{background:rgba(245,158,11,.10);border:1px solid rgba(245,158,11,.32);color:#fcd34d}
.a-cyan{background:rgba(34,211,238,.10);border:1px solid rgba(34,211,238,.32);color:#7ee9f7}
.sec{font-size:1rem;font-weight:700;color:#e8eefb;margin:2px 0 8px}
.sec .num{display:inline-flex;width:22px;height:22px;border-radius:7px;background:rgba(168,85,247,.20);
  border:1px solid rgba(168,85,247,.45);color:#c084fc;align-items:center;justify-content:center;
  font-size:.74rem;font-weight:800;margin-right:8px;vertical-align:text-bottom;}
.brand{font-size:1.4rem;font-weight:800;background:linear-gradient(90deg,#22d3ee,#a855f7);
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.brand-sub{color:#7c8aa5;font-size:.66rem;letter-spacing:.08em;margin:-4px 0 6px}
.modetitle{font-family:'Sora',sans-serif;font-weight:800;font-size:1.5rem;
  background:linear-gradient(90deg,#a855f7,#22d3ee);-webkit-background-clip:text;
  background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:.02em}
.modesub{color:#8aa0c0;font-weight:500;font-size:1rem}
.hbox{font-size:.72rem;color:#8aa0c0}
.hbox b{color:#e2e8f0;font-size:.84rem}
.badge{display:inline-block;background:rgba(52,211,153,.15);border:1px solid #34d399;
  color:#34d399;border-radius:8px;padding:2px 10px;font-size:.78rem;font-weight:700}
.badge.wait{background:rgba(245,158,11,.15);border-color:#f59e0b;color:#fbbf24}
.statusrow{display:flex;justify-content:space-between;font-size:.8rem;color:#c4d2e8;
  padding:5px 0;border-bottom:1px solid #182338}
.statusrow b{color:#34d399}
.pill{background:rgba(34,211,238,.12);border:1px solid #22d3ee;color:#7ee9f7;
  border-radius:8px;padding:3px 10px;font-size:.74rem;font-weight:700}
.stButton>button[kind="secondary"],.stDownloadButton>button{background:linear-gradient(90deg,#7c3aed,#4f46e5);
  color:#fff;border:0;border-radius:11px;font-weight:600;
  padding:9px 16px!important;font-size:.9rem!important;min-height:0!important;}
section[data-testid="stSidebar"] .stButton>button[kind="tertiary"]{background:transparent;color:#8aa0c0;
  border:1px solid transparent;border-radius:10px;text-align:left;justify-content:flex-start;
  font-weight:600;padding:8px 12px;box-shadow:none}
section[data-testid="stSidebar"] .stButton>button[kind="tertiary"]:hover{background:#111b2e;
  border-color:#233047;color:#cdd9ee}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{
  background:linear-gradient(90deg,rgba(124,58,237,.32),rgba(34,211,238,.06));color:#fff;
  border:1px solid rgba(124,58,237,.45);border-radius:10px;text-align:left;
  justify-content:flex-start;font-weight:700;padding:8px 12px;box-shadow:none}
div[data-testid="stMetricValue"]{font-size:1.3rem}
.recact{display:flex;gap:10px;align-items:center;background:#0e1626;border:1px solid #233047;
  border-radius:12px;padding:11px 14px;height:100%}
.recact .n{font-size:1.3rem}
.upgrade{background:linear-gradient(160deg,#2a1a5e,#160d33);border:1px solid #3a2d6b;
  border-radius:14px;padding:14px;margin-top:10px}
.upgrade b{color:#fff}.upgrade p{color:#b9a9e8;font-size:.78rem;margin:4px 0 10px}
/* tamni scrollbar uz temu */
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-track{background:#0a0f1d;}
::-webkit-scrollbar-thumb{background:#233047;border-radius:6px;border:2px solid #0a0f1d;}
::-webkit-scrollbar-thumb:hover{background:#7c3aed;}
html{scrollbar-color:#233047 #0a0f1d;scrollbar-width:thin;}
.empty{text-align:center;padding:44px 20px;color:#c4d2e8}
.empty .big{font-family:'Sora',sans-serif;font-size:1.4rem;font-weight:800;color:#e8eefb;margin-bottom:8px}
.icbox{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex:0 0 auto}
.navmenu{display:flex;flex-direction:column;gap:3px;margin:8px 0 2px}
.navitem{display:flex;align-items:center;gap:10px;padding:8px 11px;border-radius:10px;font-size:.85rem;color:#8aa0c0;font-weight:600}
.navitem.on{background:linear-gradient(90deg,rgba(124,58,237,.30),rgba(34,211,238,.08));color:#fff;box-shadow:inset 0 0 0 1px rgba(124,58,237,.4)}
.navitem .e{width:18px;text-align:center}
.adminbox{display:flex;align-items:center;gap:9px;justify-content:flex-end}
.adminbox .av{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#a855f7,#22d3ee);display:flex;align-items:center;justify-content:center;font-weight:800;color:#fff;font-size:.82rem}
.adminbox .nm{line-height:1.15;text-align:right}
.adminbox .nm b{font-size:.82rem;color:#e2e8f0;display:block}
.adminbox .nm small{font-size:.66rem;color:#8aa0c0}
.bell{width:34px;height:34px;border-radius:9px;background:#0e1626;border:1px solid #233047;display:flex;align-items:center;justify-content:center;color:#8aa0c0;font-size:1rem}
.chev{margin-left:auto;font-size:1.2rem;opacity:.5}
</style>
""", unsafe_allow_html=True)

# ---- UX/UI polish + animacije ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');
html, body, [class*="css"], .stApp, p, span, div, label, input, button{
  font-family:'Plus Jakarta Sans',system-ui,sans-serif!important;}
h1,h2,h3,.brand,.kpi .val,.sec,.modetitle{font-family:'Sora','Plus Jakarta Sans',sans-serif!important;letter-spacing:-.01em;}
#MainMenu, footer{visibility:hidden;height:0;}
[data-testid="stDecoration"], [data-testid="stStatusWidget"]{display:none!important;}
/* NE sakrivamo stToolbar — u njemu je dugme za otvaranje sidebara!
   Sakrivamo samo Streamlit-ov meni, deploy dugme i akcije (desna strana) */
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"]{display:none!important;}
header[data-testid="stHeader"]{background:transparent!important;box-shadow:none!important;}
[data-testid="stToolbar"]{background:transparent!important;}
/* dugme za otvaranje sidebara — stilizovano */
[data-testid="stExpandSidebarButton"]{
  background:#101a2e!important;border:1px solid #2b3a55!important;border-radius:10px!important;}
[data-testid="stExpandSidebarButton"]:hover{border-color:#a855f7!important;}
/* GLOBALNO: material ikonice (kad se font ne učita prikazuju se kao tekst poput
   'arrow_upward' i preklapaju natpise — npr. meni kolona u tabelama). Uklanjamo
   tekst svuda; bitna dugmad dolje imaju ::after zamjene. */
[data-testid="stIconMaterial"]{font-size:0!important;line-height:0!important;letter-spacing:0!important;}
/* X za zatvaranje dialoga (puna tabela i sl.) */
[data-testid="stDialog"] button [data-testid="stIconMaterial"]::after{
  content:"✕";font-size:15px;font-family:'Plus Jakarta Sans',sans-serif;color:#8aa0c0;}
/* zamijeni nepročitan material-icon tekst (keyboard_double...) čistom strelicom */
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"]{
  font-size:0!important;width:22px;display:inline-flex;justify-content:center;align-items:center;}
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]::after{
  content:"«";font-size:16px;font-family:'Plus Jakarta Sans',sans-serif;color:#8aa0c0;}
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"]::after,
[data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"]::after{
  content:"»";font-size:16px;font-family:'Plus Jakarta Sans',sans-serif;color:#7ee9f7;}
/* sakrij neučitan ikonski tekst u expander naslovu (AI Assistant) */
[data-testid="stExpander"] summary [data-testid="stIconMaterial"]{display:none!important;}
/* Top 3 akcije — klikabilne kartice */
.st-key-act1 button,.st-key-act2 button,.st-key-act3 button,
.st-key-rep_act1 button,.st-key-rep_act2 button,.st-key-rep_act3 button{
  text-align:left!important;justify-content:flex-start!important;align-items:flex-start!important;
  border-radius:12px!important;padding:13px 15px!important;min-height:96px!important;
  background:#0c1322!important;border:1px solid #2b3a55!important;color:#e2e8f0!important;
  font-weight:600!important;line-height:1.45!important;transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease!important;}
.st-key-act1 button:hover,.st-key-rep_act1 button:hover{border-color:#22d3ee!important;transform:translateY(-2px)!important;box-shadow:0 8px 20px rgba(34,211,238,.12)!important;}
.st-key-act2 button:hover,.st-key-rep_act2 button:hover{border-color:#a855f7!important;transform:translateY(-2px)!important;box-shadow:0 8px 20px rgba(168,85,247,.12)!important;}
.st-key-act3 button:hover,.st-key-rep_act3 button:hover{border-color:#ef4444!important;transform:translateY(-2px)!important;box-shadow:0 8px 20px rgba(239,68,68,.12)!important;}
/* plutajući AI chatbot (dole desno) */
.botavatar{width:46px;height:46px;flex:0 0 auto;border-radius:50%;
  background:#101a2e;border:2px solid #2b3a55;display:flex;align-items:center;justify-content:center;
  position:relative;box-shadow:0 0 0 0 rgba(168,85,247,.45);animation:botpulse 2.6s ease-in-out infinite;}
.botavatar::before{content:"";position:absolute;inset:-3px;border-radius:50%;
  background:conic-gradient(from 0deg,#22d3ee,#a855f7,#22d3ee);opacity:.0;transition:opacity .3s;
  z-index:-1;animation:botspin 4s linear infinite;}
.botavatar:hover::before{opacity:.55;}
@keyframes botpulse{0%,100%{box-shadow:0 0 0 0 rgba(168,85,247,.40);}
  50%{box-shadow:0 0 0 7px rgba(168,85,247,0);}}
@keyframes botspin{to{transform:rotate(360deg);}}
.st-key-chatfab{position:fixed;right:26px;bottom:26px;z-index:1000;width:auto!important;}
.st-key-chatfab button[data-testid="stPopoverButton"]{width:58px;height:58px;min-width:58px;border-radius:50%!important;font-size:0!important;background:#101a2e!important;border:2px solid #2b3a55!important;color:#fff!important;background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%20viewBox%3D%220%200%2048%2048%22%20fill%3D%22none%22%3E%3Cdefs%3E%3ClinearGradient%20id%3D%22fb%22%20x1%3D%220%22%20y1%3D%220%22%20x2%3D%221%22%20y2%3D%221%22%3E%3Cstop%20offset%3D%220%25%22%20stop-color%3D%22%2322d3ee%22/%3E%3Cstop%20offset%3D%22100%25%22%20stop-color%3D%22%23a855f7%22/%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle%20cx%3D%2224%22%20cy%3D%2224%22%20r%3D%2219%22%20stroke%3D%22url%28%23fb%29%22%20stroke-width%3D%222.8%22/%3E%3Cellipse%20cx%3D%2224%22%20cy%3D%2224%22%20rx%3D%228.5%22%20ry%3D%2219%22%20stroke%3D%22url%28%23fb%29%22%20stroke-width%3D%221.8%22%20opacity%3D%22.8%22/%3E%3Cline%20x1%3D%225%22%20y1%3D%2224%22%20x2%3D%2243%22%20y2%3D%2224%22%20stroke%3D%22url%28%23fb%29%22%20stroke-width%3D%221.8%22%20opacity%3D%22.8%22/%3E%3Ccircle%20cx%3D%2224%22%20cy%3D%225%22%20r%3D%223.2%22%20fill%3D%22%2322d3ee%22/%3E%3Ccircle%20cx%3D%2243%22%20cy%3D%2224%22%20r%3D%223.2%22%20fill%3D%22%23a855f7%22/%3E%3Ccircle%20cx%3D%2224%22%20cy%3D%2243%22%20r%3D%223.2%22%20fill%3D%22%23a855f7%22/%3E%3Ccircle%20cx%3D%225%22%20cy%3D%2224%22%20r%3D%223.2%22%20fill%3D%22%2322d3ee%22/%3E%3Ccircle%20cx%3D%2224%22%20cy%3D%2224%22%20r%3D%224.2%22%20fill%3D%22url%28%23fb%29%22/%3E%3C/svg%3E")!important;background-repeat:no-repeat!important;background-position:center!important;background-size:32px 32px!important;box-shadow:0 10px 28px rgba(0,0,0,.45)!important;transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease!important;justify-content:center!important;}
.st-key-chatfab button[data-testid="stPopoverButton"]:hover{
  transform:scale(1.08);border-color:#a855f7!important;box-shadow:0 14px 34px rgba(124,58,237,.4)!important;}
[data-testid="stPopoverBody"]{background:#101a2e!important;border:1px solid #2b3a55!important;
  border-radius:16px!important;min-width:330px;max-width:380px;}
.chatmsg{font-size:.82rem;line-height:1.55;padding:9px 12px;border-radius:13px;margin:6px 0;}
.chatmsg.ai{background:#0c1322;border:1px solid #1e2c44;color:#d4def0;margin-right:20px;}
.chatmsg.me{background:rgba(124,58,237,.20);border:1px solid rgba(168,85,247,.45);color:#f0f4fc;
  margin-left:20px;text-align:right;}
/* sakrij 'Press Enter to submit form' natpis ispod polja */
[data-testid="stPopoverBody"] [data-testid="InputInstructions"],
.st-key-chatfab [data-testid="InputInstructions"]{display:none!important;}
[data-testid="stPopoverBody"] input{background:#0c1322!important;border:1px solid #2b3a55!important;
  border-radius:11px!important;color:#e8eefb!important;font-size:.85rem!important;padding:9px 12px!important;}
[data-testid="stPopoverBody"] input::placeholder{color:#64748b!important;}
[data-testid="stPopoverBody"] input:focus{border-color:#a855f7!important;box-shadow:0 0 0 1px #a855f7!important;}
[data-testid="stPopoverBody"] button[kind="secondaryFormSubmit"]{
  background:linear-gradient(135deg,#7c3aed,#5b6cf5)!important;border:none!important;color:#fff!important;
  border-radius:11px!important;font-weight:800!important;}
[data-testid="stPopoverBody"] button[kind="secondaryFormSubmit"]:hover{filter:brightness(1.15);}
/* Export Report — gradijent + hover usklađen sa karticama */
.st-key-exportbtn button{background:linear-gradient(90deg,#7c3aed,#5b6cf5)!important;
  border:1px solid rgba(168,85,247,.5)!important;border-radius:11px!important;color:#fff!important;
  font-weight:700!important;box-shadow:0 4px 16px rgba(124,58,237,.25)!important;
  transition:transform .18s ease, box-shadow .18s ease!important;}
.st-key-exportbtn button:hover{transform:translateY(-1px)!important;
  box-shadow:0 8px 24px rgba(124,58,237,.4)!important;border-color:#a855f7!important;}
/* View All Insights link + View Outliers dugme */
.st-key-pb_open button,.st-key-roi_open button,.st-key-mc_open button{width:30px!important;min-width:30px!important;height:30px!important;
  border-radius:9px!important;background:rgba(34,211,238,.08)!important;border:1px solid rgba(34,211,238,.35)!important;
  color:#7ee9f7!important;font-size:.95rem!important;font-weight:800!important;padding:0!important;box-shadow:none!important;}
.st-key-pb_open button:hover,.st-key-roi_open button:hover,.st-key-mc_open button:hover{border-color:#a855f7!important;transform:translateY(-1px)!important;}
.st-key-viewall_ins button{background:transparent!important;border:none!important;color:#7ee9f7!important;
  font-size:.8rem!important;font-weight:700!important;padding:2px 0!important;justify-content:flex-end!important;
  width:100%!important;box-shadow:none!important;}
.st-key-viewall_ins button:hover{color:#a855f7!important;}
.st-key-view_out button{background:rgba(34,211,238,.08)!important;border:1px solid rgba(34,211,238,.4)!important;
  color:#7ee9f7!important;border-radius:9px!important;font-size:.8rem!important;font-weight:600!important;}
.st-key-view_out button:hover{border-color:#a855f7!important;}
/* klikabilne alert kartice (boje po stavci) */
.st-key-alert_red button,.st-key-alert_amber button,.st-key-alert_cyan button{
  text-align:left!important;justify-content:flex-start!important;align-items:flex-start!important;
  border-radius:14px!important;padding:15px 18px!important;min-height:88px!important;
  font-weight:600!important;line-height:1.4!important;transition:transform .16s ease!important;}
.st-key-alert_red button{background:rgba(239,68,68,.10)!important;border:1px solid rgba(239,68,68,.32)!important;color:#fca5a5!important;}
.st-key-alert_amber button{background:rgba(245,158,11,.10)!important;border:1px solid rgba(245,158,11,.32)!important;color:#fcd34d!important;}
.st-key-alert_cyan button{background:rgba(34,211,238,.10)!important;border:1px solid rgba(34,211,238,.32)!important;color:#7ee9f7!important;}
.st-key-alert_red button:hover,.st-key-alert_amber button:hover,.st-key-alert_cyan button:hover{
  transform:translateY(-2px)!important;filter:brightness(1.15)!important;}
.kpi, [data-testid="stVerticalBlockBorderWrapper"]{transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease;}
.kpi:hover{transform:translateY(-3px);border-color:#3a4a6b;box-shadow:0 18px 38px rgba(124,58,237,.20);}
[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-2px);border-color:#33456b!important;}
.alertc{transition:transform .16s ease;} .alertc:hover{transform:translateY(-2px);}
.recact{transition:transform .16s ease;} .recact:hover{transform:translateY(-2px);border-color:#3a4a6b;}
section[data-testid="stSidebar"] [role="radiogroup"]{gap:4px;}
section[data-testid="stSidebar"] [role="radiogroup"] label{border-radius:11px;padding:9px 12px;transition:.15s;border:1px solid transparent;}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#111b2e;border-color:#233047;}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,rgba(124,58,237,.32),rgba(34,211,238,.06));border-color:rgba(124,58,237,.45);color:#fff;font-weight:700;}
.stButton>button:hover, .stDownloadButton>button:hover{filter:brightness(1.08);transform:translateY(-1px);}
[data-testid="stMetric"]{background:#0e1626;border:1px solid #233047;border-radius:12px;padding:10px 14px;}
[data-testid="stMetricValue"]{color:#f1f5f9;font-weight:800;}
[data-testid="stMetricLabel"]{color:#8aa0c0;}
[data-testid="stDataFrame"]{border:1px solid #233047;border-radius:12px;overflow:hidden;}
[data-testid="stExpander"]{border:1px solid #233047!important;border-radius:14px;background:#0e1626;}
[data-testid="stExpander"] summary:hover{color:#22d3ee;}
::-webkit-scrollbar{width:8px;height:8px;}::-webkit-scrollbar-thumb{background:#1c2740;border-radius:8px;}::-webkit-scrollbar-thumb:hover{background:#2a3a59;}
.block-container{animation:fadein .4s ease;}
@keyframes fadein{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;}}
.brand,.modetitle{background-size:200% auto;animation:shine 5s linear infinite;}
@keyframes shine{to{background-position:200% center;}}
.kpi, .alertc, .recact{animation:rise .5s ease both;}
@keyframes rise{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:none;}}
.a-red{animation:pulse 2.4s ease-in-out infinite;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0);}50%{box-shadow:0 0 0 4px rgba(239,68,68,.12);}}
.kpi{position:relative;overflow:hidden;}
.kpi::after{content:"";position:absolute;top:0;left:-60%;width:45%;height:100%;
  background:linear-gradient(120deg,transparent,rgba(255,255,255,.06),transparent);
  transform:skewX(-20deg);animation:sweep 6s ease-in-out infinite;pointer-events:none;}
@keyframes sweep{0%,72%{left:-60%;}100%{left:150%;}}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"]{display:flex;justify-content:center;}
/* File uploader — sakrij neučitane material ikonice (prikazivale se kao 'upload' tekst) */
[data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"]{display:none!important;}
[data-testid="stFileUploaderDropzone"]{background:#0c1322!important;border:1.5px dashed #2b3a55!important;
  border-radius:12px!important;padding:18px 16px!important;transition:border-color .15s ease;
  display:flex!important;flex-direction:column!important;align-items:center!important;
  justify-content:center!important;text-align:center!important;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#a855f7!important;}
[data-testid="stFileUploaderDropzoneInstructions"]{display:flex!important;flex-direction:column!important;
  align-items:center!important;text-align:center!important;width:100%!important;}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small{color:#8aa0c0!important;text-align:center!important;}
[data-testid="stFileUploaderDropzone"] button{margin:0 auto 8px!important;}
/* AI Status — finija kartica */
.aistatus{background:linear-gradient(155deg,#101a2e,#0c1322);border:1px solid #1e2c44;
  border-radius:14px;padding:12px 14px;margin-top:4px}
.aistatus .dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:6px;
  vertical-align:middle}
/* Bell + Calendar popover — izgled kao kvadratna ikonica */
.st-key-bellpop button,.st-key-calpop button{width:38px!important;min-width:38px!important;height:38px!important;
  border-radius:10px!important;background:#0e1626!important;border:1px solid #233047!important;
  padding:0!important;font-size:1.05rem!important;box-shadow:none!important;
  display:flex!important;align-items:center!important;justify-content:center!important;}
.st-key-bellpop button>div,.st-key-calpop button>div,
.st-key-bellpop button p,.st-key-calpop button p,
.st-key-bellpop button span,.st-key-calpop button span{display:flex!important;
  align-items:center!important;justify-content:center!important;width:100%!important;height:100%!important;
  margin:0!important;padding:0!important;line-height:1!important;}
.st-key-bellpop button{color:#fbbf24!important;}
.st-key-calpop button{color:#7ee9f7!important;}
.st-key-bellpop button:hover,.st-key-calpop button:hover{border-color:#a855f7!important;transform:translateY(-1px)!important;}
/* sakrij strelicu popovera (prikazivala se kao 'expand_more' tekst) */
.st-key-bellpop button [data-testid="stIconMaterial"],
.st-key-calpop button [data-testid="stIconMaterial"],
.st-key-bellpop button svg,.st-key-calpop button svg{display:none!important;}
.st-key-bellpop p,.st-key-calpop p{margin:0!important}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] [role="radiogroup"],
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] > div{width:100%;}
section[data-testid="stSidebar"] [data-testid="stSegmentedControl"] button{flex:1;font-weight:700;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers
def fmt_eur(x): return "€" + f"{x:,.0f}".replace(",", ".")


def fmt_eur_kpi(x):
    """Kompaktan format za velike iznose na KPI karticama (€8.91M)."""
    if abs(x) >= 1_000_000:
        return f"€{x/1_000_000:.2f}M"
    if abs(x) >= 100_000:
        return f"€{x/1_000:.0f}k"
    return "€" + f"{x:,.0f}".replace(",", ".")


import re as _re
def md_bold(text):
    """Pretvara **tekst** (markdown) u <b>tekst</b> za prikaz u HTML-u."""
    return _re.sub(r"\*\*(.+?)\*\*", r"<b style='color:#e8eefb'>\1</b>", str(text))


def chat_md(text):
    """Čisti prikaz Claude/markdown odgovora u chat mjehuriću:
    ## naslovi, **bold**, - liste -> uredan HTML (bez sirovih oznaka)."""
    out = []
    for ln in str(text).split("\n"):
        s = ln.strip()
        if not s:
            out.append("<div style='height:5px'></div>"); continue
        if s.startswith("### "):
            s = f"<div style='font-weight:800;color:#e8eefb;margin:4px 0 2px'>{s[4:]}</div>"
        elif s.startswith("## "):
            s = f"<div style='font-weight:800;color:#c084fc;font-size:.9rem;margin:6px 0 2px'>{s[3:]}</div>"
        elif s.startswith("# "):
            s = f"<div style='font-weight:800;color:#c084fc;font-size:.95rem;margin:6px 0 3px'>{s[2:]}</div>"
        elif s.startswith(("- ", "* ")):
            s = f"<div style='margin:1px 0 1px 10px'>• {s[2:]}</div>"
        else:
            s = f"<div>{s}</div>"
        s = _re.sub(r"\*\*(.+?)\*\*", r"<b style='color:#e8eefb'>\1</b>", s)
        out.append(s)
    return "".join(out)


def spark(vals, color, w=150, h=30):
    vmin, vmax = min(vals), max(vals); rng = (vmax - vmin) or 1; n = len(vals)
    pts = " ".join(f"{i/(n-1)*w:.1f},{h-3-(v-vmin)/rng*(h-8):.1f}" for i, v in enumerate(vals))
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" style="width:100%;height:28px">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/></svg>')


def trend(base, n=14, seed=0):
    rng = np.random.default_rng(seed); v = base; out = []
    for _ in range(n):
        v = max(0.1, v * (1 + 0.04 + rng.normal(0, .05))); out.append(v)
    return out


def kpi(col, label, value, icon, color, seed, delta_pct=None, sub=""):
    """KPI kartica. delta_pct = stvarna promjena (% ili None); sub = činjenični opis."""
    if delta_pct is not None:
        up = delta_pct >= 0
        line = (f"<div class='{'up' if up else 'down'}'>{'▲' if up else '▼'} "
                f"{abs(delta_pct):.1f}% vs prošli mjesec</div>")
    else:
        line = f"<div class='s'>{sub}</div>"
    col.markdown(
        f"<div class='kpi'>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:2px'>"
        f"<div class='icbox' style='background:{color}22;color:{color}'>{icon}</div>"
        f"<div class='lbl'>{label}</div></div>"
        f"<div class='val'>{value}</div>"
        f"<div class='kpi-sub'>{line}</div>"
        f"<div style='margin-top:auto'>{spark(trend(100, seed=seed), color)}</div></div>",
        unsafe_allow_html=True)


def mom_delta(df):
    """Stvarna mjesečna promjena (zadnjih 30 dana vs prethodnih 30) za prihod i kupce."""
    last_end = df["InvoiceDate"].max()
    p1 = last_end - pd.Timedelta(days=30)
    p2 = last_end - pd.Timedelta(days=60)
    a = df[df["InvoiceDate"] > p1]
    b = df[(df["InvoiceDate"] > p2) & (df["InvoiceDate"] <= p1)]

    def pct(x, y):
        return None if y == 0 else round((x - y) / y * 100, 1)
    return (pct(a["TotalPrice"].sum(), b["TotalPrice"].sum()),
            pct(a["CustomerID"].nunique(), b["CustomerID"].nunique()))


# ----------------------------------------------------------------- pipeline (cached)
@st.cache_data(show_spinner="Analiziram podatke...")
def load_pipeline(file_bytes, filename, method):
    raw = read_any(file_bytes, filename)
    df = clean_transactions(raw)
    summary = dataset_summary(df)
    rfm = assign_rfm_segments(rfm_scores(add_health_score(add_clv(build_rfm(df)))))
    raw_stats = dict(rows=len(raw), cols=raw.shape[1],
                     missing_pct=round(raw.isna().mean().mean()*100, 1),
                     dup_pct=round(raw.duplicated().mean()*100, 1))
    return df, summary, rfm, raw_stats


@st.cache_data(show_spinner="Pokrećem napredne ML analize...")
def load_advanced(file_bytes, filename, method):
    df, summary, rfm, raw_stats = load_pipeline(file_bytes, filename, method)
    mc = A.model_comparison(rfm)
    oa = A.outlier_analysis(rfm)
    fi = A.feature_importance(rfm)
    out_pct = round(oa["n_outliers"] / max(len(rfm), 1) * 100, 1)
    dq = dict(raw_stats, outlier_pct=out_pct,
              score=round(max(0, 100 - raw_stats["missing_pct"] * 0.5
                              - raw_stats["dup_pct"] - out_pct * 0.3), 1))
    return dict(mc=mc, sk=A.silhouette_over_k(rfm), pca2=A.pca_coords(rfm, 2),
                pca3=A.pca_coords(rfm, 3), fi=fi, oa=oa,
                cohort=A.cohort_analysis(df), heatmap=A.cluster_heatmap(rfm),
                dq=dq, tech=A.technical_insights(mc, fi, rfm))


# ----------------------------------------------------------------- charts
def insight_icon(seg):
    if seg == "At Risk":
        return "<span style='color:#f87171;font-weight:800'>⚠</span>"
    if seg == "Hibernating":
        return "<span style='color:#fbbf24;font-weight:800'>●</span>"
    return "<span style='color:#34d399;font-weight:800'>✓</span>"


def interp(text, color="#22d3ee"):
    """Kutijica „Tumačenje“ ispod grafika — tekst se računa iz podataka."""
    if not text:
        return
    st.markdown(f"<div style='margin-top:4px;background:{color}0d;border-left:2px solid {color};"
                f"border-radius:0 9px 9px 0;padding:7px 10px;font-size:.73rem;color:#9fb2cc;"
                f"line-height:1.5'>📊 <b style='color:#c4d2e8'>Tumačenje:</b> {text}</div>",
                unsafe_allow_html=True)


def donut_fig(rfm, n, key, compact=False):
    c = rfm["Segment"].value_counts()
    tot = max(int(c.sum()), 1)
    labels = [f"{s}  {v/tot*100:.1f}% ({v:,})".replace(",", ".") for s, v in c.items()]
    pull = [0.04] + [0] * (len(c) - 1)            # najveći segment blago izvučen
    fig = go.Figure(go.Pie(labels=labels, values=c.values.tolist(), hole=.58, sort=False,
        pull=pull, rotation=-15,
        marker=dict(colors=[SEGMENT_COLORS.get(s, "#475569") for s in c.index],
                    line=dict(color="#0d1322", width=2.5)), textinfo="none",
        hovertemplate="%{label}<extra></extra>"))
    if compact:                                   # uska kolona: legenda ISPOD kruga
        legend = dict(font=dict(size=9.5), orientation="h", x=.5, xanchor="center",
                      y=-0.02, yanchor="top", itemsizing="constant")
        height = 420
    else:
        legend = dict(font=dict(size=11.5), orientation="v", x=1.02, y=.5,
                      yanchor="middle", itemsizing="constant", tracegroupgap=4)
        height = 310
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=13), margin=dict(l=0, r=0, t=4, b=4),
        height=height, showlegend=True, legend=legend,
        annotations=[dict(text=f"<b style='font-size:19px'>{n:,}</b><br>kupaca".replace(",", "."),
                          x=.5, y=.5, font_size=14, showarrow=False, font_color="#e2e8f0")])
    st.plotly_chart(fig, use_container_width=True, key=key)
    try:
        big_seg, big_n = c.index[0], int(c.iloc[0])
        champ_val = rfm.loc[rfm["Segment"] == "Champions", "PredictedCLV"].sum()
        champ_pct = champ_val / max(rfm["PredictedCLV"].sum(), 1) * 100
        risk_n = int((rfm["Segment"] == "At Risk").sum())
        txt = (f"Najbrojniji segment je <b>{big_seg}</b> ({big_n/tot*100:.0f}%). "
               + (f"<b>Champions</b> nose {champ_pct:.0f}% predviđene vrijednosti. " if champ_pct > 0 else "")
               + (f"<b>{risk_n}</b> kupaca je pod rizikom odlaska." if risk_n else ""))
        interp(txt)
    except Exception:
        pass


def exec_metric_rows(rfm):
    """Redovi metrika za Executive Summary — izračunato iz stvarnih podataka."""
    react_val = rfm.loc[rfm["Segment"].isin(["At Risk", "Hibernating"]), "PredictedCLV"].sum()
    champ_share = rfm.loc[rfm["Segment"] == "Champions", "PredictedCLV"].sum() / max(rfm["PredictedCLV"].sum(), 1) * 100
    avg_clv = rfm["PredictedCLV"].mean()
    hs = rfm["HealthScore"].mean()
    rows = [("Potencijal reaktivacije (At Risk + Hibernating)", fmt_eur(react_val), "#22d3ee"),
            ("Champions udio predviđene vrijednosti", f"{champ_share:.0f}%", "#a855f7"),
            ("Prosječan CLV po kupcu", fmt_eur(avg_clv), "#34d399"),
            ("Prosječan Health Score", f"{hs:.0f}/100", "#f59e0b")]
    return "".join(
        f"<div style='display:flex;justify-content:space-between;gap:8px;font-size:.78rem;"
        f"padding:6px 0;border-bottom:1px solid #182338'>"
        f"<span style='color:#c4d2e8'>✓ {l}</span><b style='color:{c};white-space:nowrap'>{v}</b></div>"
        for l, v, c in rows)


def roi_slider(label, lo, hi, default, key):
    """Slajder sa kutijicom trenutne vrijednosti (kao na mockupu)."""
    sc = st.columns([4, 1.15], vertical_alignment="center")
    val = sc[0].slider(label, lo, hi, default, key=key)
    sc[1].markdown(f"<div style='background:#0c1322;border:1px solid #2b3a55;border-radius:8px;"
                   f"padding:5px 8px;text-align:center;font-size:.8rem;color:#7ee9f7;font-weight:700;"
                   f"margin-top:14px'>{val}%</div>", unsafe_allow_html=True)
    return val


SEGMENT_ICONS = {"Champions": "👑", "Loyal Customers": "💜", "Potential Loyalists": "🌱",
                 "New Customers": "✨", "At Risk": "⚠️", "Hibernating": "😴", "Ostali": "👤"}
SEGMENT_TIPS = {
    "Champions": "VIP tretman, premium early access — čuvaj najvrednije",
    "Loyal Customers": "Loyalty program i bodovi — stabilan prihod, rast CLV-a",
    "Potential Loyalists": "Personalizovane ponude i cross-sell — gurni ka lojalnosti",
    "New Customers": "Welcome ponude i onboarding — izgradi naviku kupovine",
    "At Risk": "Win-back kampanja HITNO — vrijednost na odlasku",
    "Hibernating": "Reaktivacija agresivnim popustima ili pustiti",
    "Ostali": "Pratiti ponašanje i premjestiti u jasniji segment",
}


def segment_cards(rfm):
    """Profil kartica za svaki segment — brojke iz stvarnih podataka."""
    tot = max(len(rfm), 1)
    stats = (rfm.groupby("Segment").agg(
        n=("CustomerID", "count"), clv=("PredictedCLV", "mean"),
        rec=("Recency", "mean"), freq=("Frequency", "mean"))
        .reset_index().sort_values("clv", ascending=False))
    rows = [stats.iloc[i:i+3] for i in range(0, len(stats), 3)]
    for chunk in rows:
        cols = st.columns(3)
        for col, (_, s) in zip(cols, chunk.iterrows()):
            seg, color = s["Segment"], SEGMENT_COLORS.get(s["Segment"], "#64748b")
            icon, tip = SEGMENT_ICONS.get(seg, "👤"), SEGMENT_TIPS.get(seg, "")
            col.markdown(
                f"<div style='background:linear-gradient(155deg,#121a2e,#0e1424);border:1px solid #233047;"
                f"border-left:3px solid {color};border-radius:14px;padding:13px;height:100%'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:9px'>"
                f"<span style='font-size:.9rem;font-weight:800;color:{color}'>{icon} {seg}</span>"
                f"<span style='font-size:.66rem;background:{color}1f;border:1px solid {color};color:{color};"
                f"border-radius:7px;padding:2px 8px;white-space:nowrap'>{int(s['n']):,} · {s['n']/tot*100:.0f}%</span></div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;text-align:center;margin-bottom:9px'>"
                f"<div style='background:#0c1322;border-radius:9px;padding:6px 3px'><div style='font-size:.62rem;color:#8aa0c0'>Prosj. CLV</div>"
                f"<div style='font-size:.86rem;font-weight:800;color:#e8eefb'>{fmt_eur(s['clv'])}</div></div>"
                f"<div style='background:#0c1322;border-radius:9px;padding:6px 3px'><div style='font-size:.62rem;color:#8aa0c0'>Recency</div>"
                f"<div style='font-size:.86rem;font-weight:800;color:#e8eefb'>{s['rec']:.0f} d.</div></div>"
                f"<div style='background:#0c1322;border-radius:9px;padding:6px 3px'><div style='font-size:.62rem;color:#8aa0c0'>Kupovina</div>"
                f"<div style='font-size:.86rem;font-weight:800;color:#e8eefb'>{s['freq']:.1f}</div></div></div>"
                f"<div style='font-size:.72rem;color:#c4d2e8;background:{color}0d;border-radius:9px;padding:7px 9px'>"
                f"💡 {tip}</div></div>".replace(",", "."), unsafe_allow_html=True)
        st.write("")


def top3_actions(rfm, kp=""):
    """Top 3 akcije — klikabilne kartice sa brojkama iz stvarnih podataka."""
    acts = [("🥇", "VIP kampanja za Champions", "Champions",
             "Ekskluzivne ponude za najvrednije", "Marketing Playbook", "act1"),
            ("🥈", "Loyalty program za Loyal", "Loyal Customers",
             "Jačanje odnosa i rast CLV-a", "Customer Segments", "act2"),
            ("🥉", "Retention za At Risk", "At Risk",
             "Spriječi odlazak, vrati kupce", "ROI Simulator", "act3")]
    cols = st.columns(3)
    for col, (ic, title, seg, desc, target, key) in zip(cols, acts):
        sub = rfm[rfm["Segment"] == seg]
        n, val = len(sub), sub["PredictedCLV"].sum()
        stat = (f"{n:,} kupaca · {fmt_eur_kpi(val)} predviđene vrijednosti".replace(",", ".")
                if n else "trenutno nema kupaca u ovom segmentu")
        if col.button(f"{ic}  **{title}**\n\n{stat}\n\n{desc}  ›",
                      key=f"{kp}{key}", use_container_width=True):
            st.session_state.nav_target = target; st.rerun()


def playbook_table(rfm, head=None, skip=0):
    """Marketing Playbook tabela sa punim tekstom. head=N prikaže prvih N; skip preskoči prve."""
    pb = build_playbook(rfm)[["Segment", "Preporučena strategija", "Glavni kanal"]]
    if skip:
        pb = pb.iloc[skip:]
    if head:
        pb = pb.head(head)
    rows = ""
    for i, (_, r) in enumerate(pb.iterrows()):
        bg = "background:rgba(255,255,255,.022);" if i % 2 == 0 else ""
        chips = "".join(
            f"<span style='display:inline-block;background:#0c1322;border:1px solid #2b3a55;"
            f"color:#8aa0c0;border-radius:7px;padding:1px 7px;margin:1px 3px 1px 0;"
            f"font-size:.66rem;white-space:nowrap'>{k.strip()}</span>"
            for k in str(r["Glavni kanal"]).split(","))
        rows += (f"<tr style='{bg}'>"
                 f"<td style='color:{SEGMENT_COLORS.get(r['Segment'], '#c4d2e8')};font-weight:700;"
                 f"white-space:nowrap;padding:9px 10px 9px 6px;vertical-align:top'>{r['Segment']}</td>"
                 f"<td style='color:#c4d2e8;padding:9px 10px 9px 0;vertical-align:top;line-height:1.45'>"
                 f"{r['Preporučena strategija']}</td>"
                 f"<td style='padding:9px 6px 9px 0;vertical-align:top'>{chips}</td></tr>")
    st.markdown(
        "<table style='width:100%;border-collapse:collapse;font-size:.78rem'>"
        "<thead><tr>"
        "<th style='text-align:left;color:#8aa0c0;font-weight:600;padding:0 10px 7px 6px;border-bottom:1px solid #233047'>Segment</th>"
        "<th style='text-align:left;color:#8aa0c0;font-weight:600;padding:0 10px 7px 0;border-bottom:1px solid #233047'>Preporučena strategija</th>"
        "<th style='text-align:left;color:#8aa0c0;font-weight:600;padding:0 0 7px 0;border-bottom:1px solid #233047'>Kanal</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)


def scatter_fig(rfm, key):
    fig = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment",
                     color_discrete_map=SEGMENT_COLORS, log_x=True, log_y=True, opacity=.7,
                     hover_data=["Recency", "PredictedCLV"])
    fig.update_traces(marker=dict(size=6)); fig.update_layout(**PL, height=300,
        xaxis_title="Frequency (broj narudžbi)", yaxis_title="Monetary (€)")
    fig.update_xaxes(gridcolor="#1c2740", tickvals=[1, 2, 5, 10, 20, 50, 100, 200],
                     ticktext=["1", "2", "5", "10", "20", "50", "100", "200"],
                     minor=dict(showgrid=False), ticks="outside", ticklen=4)
    fig.update_yaxes(gridcolor="#1c2740", tickvals=[10, 100, 1000, 10000, 100000, 1000000],
                     ticktext=["10", "100", "1k", "10k", "100k", "1M"],
                     minor=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True, key=key)
    try:
        if rfm["Frequency"].nunique() <= 1:
            interp("Svi kupci u ovom skupu imaju po jednu kupovinu — raspored zato pokazuje "
                   "raspon potrošnje, a ne učestalost.")
        else:
            hi = rfm.loc[rfm["Frequency"] >= rfm["Frequency"].quantile(.75), "Monetary"].mean()
            lo = rfm.loc[rfm["Frequency"] <= rfm["Frequency"].quantile(.25), "Monetary"].mean()
            mult = hi / max(lo, 1)
            interp(f"Vrijednost je koncentrisana gore desno — najčešći kupci (top 25% po učestalosti) "
                   f"troše prosječno <b>{mult:.1f}×</b> više od najrjeđih. Svaka tačka je jedan kupac.")
    except Exception:
        pass


def header(mode_title, mode_sub, dataset_name, ready, notifs=None):
    st.markdown(f"<div style='text-align:center;margin-bottom:6px'>"
                f"<span class='modetitle'>{mode_title}</span> "
                f"<span class='modesub'>{mode_sub}</span></div>", unsafe_allow_html=True)
    h = st.columns([2, 1.25, 1.25, 1.1, 1.55], vertical_alignment="center")
    h[0].markdown("<div class='hbox'>SmartSeg AI<br><b>AI Customer Intelligence Platform</b></div>",
                  unsafe_allow_html=True)
    h[1].markdown(f"<div class='hbox'>Dataset<br><b>{dataset_name}</b></div>", unsafe_allow_html=True)
    h[2].markdown(f"<div class='hbox'>Vrijeme analize<br><b>{datetime.now():%d.%m.%Y %H:%M}</b></div>",
                  unsafe_allow_html=True)
    badge_cls = "badge" if ready else "badge wait"
    status = "✓ Analiza završena" if ready else "● Čeka podatke"
    h[3].markdown(f"<div class='hbox'>Status</div><div class='{badge_cls}'>{status}</div>",
                  unsafe_allow_html=True)
    # kalendar + zvonce (interaktivno) + admin
    ccol, bcol, icol = h[4].columns([1, 1, 2.1], vertical_alignment="center")
    with ccol.container(key="calpop"):
        with st.popover("📅"):
            _t = datetime.now()
            _dani = ["Ponedjeljak", "Utorak", "Srijeda", "Četvrtak", "Petak", "Subota", "Nedjelja"]
            _mj = ["januar", "februar", "mart", "april", "maj", "jun", "jul", "avgust",
                   "septembar", "oktobar", "novembar", "decembar"]
            st.markdown(f"<div style='text-align:center'>"
                        f"<div style='font-size:.78rem;color:#8aa0c0'>{_dani[_t.weekday()]}</div>"
                        f"<div style='font-size:2rem;font-weight:800;color:#7ee9f7;line-height:1.1'>{_t.day}</div>"
                        f"<div style='font-size:.85rem;color:#e8eefb'>{_mj[_t.month-1]} {_t.year}.</div>"
                        f"<div style='font-size:.75rem;color:#8aa0c0;margin-top:2px'>{_t:%H:%M}</div></div>",
                        unsafe_allow_html=True)
            st.date_input("Kalendar", value=date.today(), label_visibility="collapsed")
    with bcol.container(key="bellpop"):
        with st.popover("🔔"):
            st.markdown("<div style='font-weight:700;color:#e8eefb;margin-bottom:4px'>Obavještenja</div>",
                        unsafe_allow_html=True)
            items = notifs or [("⏳", "Čeka podatke — učitaj CSV/Excel da započneš.", "#fbbf24")]
            for ic, txt, col in items:
                st.markdown(f"<div style='display:flex;gap:8px;padding:7px 2px;border-bottom:1px solid "
                            f"#1a2436;font-size:.82rem'><span>{ic}</span>"
                            f"<span style='color:{col}'>{txt}</span></div>", unsafe_allow_html=True)
    icol.markdown("<div class='adminbox'><div class='av'>A</div>"
                  "<div class='nm'><b>Admin</b><small>Business User</small></div></div>",
                  unsafe_allow_html=True)
    st.write("")


# ================================================================= SIDEBAR (gornji dio)
file_bytes = filename = None
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:11px;margin-bottom:8px'>"
                + LOGO_SVG +
                "<div><div class='brand'>SmartSeg AI</div>"
                "<div class='brand-sub'>AI CUSTOMER INTELLIGENCE PLATFORM</div></div></div>",
                unsafe_allow_html=True)
    _m = st.segmented_control("Mode", ["📊 Standard", "🔬 Advanced"], default="📊 Standard",
                              label_visibility="collapsed", key="modesel")
    mode = "🔬 Advanced Mode" if (_m or "📊 Standard").startswith("🔬") else "📊 Standard Mode"
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    _menu_styles = {
        "container": {"padding": "2px 0", "background-color": "transparent"},
        "icon": {"color": "#7ee9f7", "font-size": "16px"},
        "nav-link": {
            "font-size": "13.5px", "font-weight": "600", "text-align": "left",
            "margin": "3px 0", "padding": "9px 13px", "border-radius": "11px",
            "color": "#8aa0c0", "--hover-color": "#111b2e",
        },
        "nav-link-selected": {
            "background": "linear-gradient(90deg,rgba(124,58,237,.40),rgba(34,211,238,.10))",
            "color": "#ffffff", "font-weight": "700", "border-left": "3px solid #a855f7",
        },
    }
    if mode.startswith("📊"):
        _pages = ["Dashboard", "Customer Segments", "AI Insights", "Marketing Playbook",
                  "ROI Simulator", "Reports", "Export Data", "Settings"]
        _icons = ["grid-1x2-fill", "people-fill", "stars", "clipboard2-check-fill",
                  "graph-up-arrow", "file-earmark-bar-graph-fill", "box-arrow-down", "gear-fill"]
        _default = _pages.index(st.session_state.get("page", "Dashboard")) if \
            st.session_state.get("page", "Dashboard") in _pages else 0
        _target = st.session_state.pop("nav_target", None)        # programatska navigacija (dugmad)
        _manual = _pages.index(_target) if _target in _pages else None
        page = option_menu(None, _pages, icons=_icons, default_index=_default,
                            manual_select=_manual, styles=_menu_styles, key="navmenu")
        st.session_state.page = page
    else:
        _apages = ["Advanced Mode", "AI Insights", "Reports", "Export Data", "Settings"]
        _aicons = ["cpu-fill", "stars", "file-earmark-bar-graph-fill", "box-arrow-down", "gear-fill"]
        page = option_menu(None, _apages, icons=_aicons, default_index=0,
                           styles=_menu_styles, key="navmenu_adv")
    st.markdown("---")
    up = st.file_uploader("📥 Upload Data (CSV / Excel)", type=["csv", "xlsx", "xls"], key="datafile")
    method = "GMM (auto)"

# odredi izvor podataka — isključivo upload korisnika
if up is not None:
    file_bytes, filename = up.getvalue(), up.name

mode_title = "STANDARD MODE" if mode.startswith("📊") else "ADVANCED MODE"
mode_sub = "(Manager View)" if mode.startswith("📊") else "(Data Mining View)"


# ================================================================= PRAZNO STANJE (početak)
if file_bytes is None:
    with st.sidebar:
        st.markdown("---")
        st.markdown("<div class='sec'>AI Status</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='aistatus'>"
            "<div class='statusrow'><span>Stanje</span>"
            "<b style='color:#fbbf24'><span class='dot' style='background:#fbbf24'></span>Čeka podatke</b></div>"
            "<div class='statusrow' style='border-bottom:none'><span>Dataset</span>"
            "<b style='color:#8aa0c0'>—</b></div></div>",
            unsafe_allow_html=True)
    header(mode_title, mode_sub, "—", ready=False)
    k = st.columns(6)
    for i, (lbl, ic, col) in enumerate([
            ("Total Customers", "👥", "#22d3ee"), ("Total Revenue", "💰", "#a855f7"),
            ("Average CLV", "💎", "#06b6d4"), ("VIP Customers", "👑", "#c084fc"),
            ("At Risk", "⚠️", "#ef4444"), ("Health Score", "❤️", "#34d399")]):
        kpi(k[i], lbl, "0", ic, col, i + 1, sub="—")
    st.write("")
    with st.container(border=True):
        st.markdown("<div class='empty'><div class='big'>Učitaj podatke da započneš analizu</div>"
                    "Prevuci svoj <b>CSV ili Excel</b> fajl u <b>Upload Data</b> u lijevom meniju.<br><br>"
                    "Sistem automatski prepoznaje kolone (kupac, datum, iznos), čisti podatke i "
                    "računa sve KPI-jeve, segmente i ML analize.</div>", unsafe_allow_html=True)
    st.stop()


# ================================================================= PIPELINE
try:
    df, summary, rfm, raw_stats = load_pipeline(file_bytes, filename, method)
except Exception as e:
    st.error(f"⚠️ Ne mogu obraditi ovaj fajl: {e}")
    st.info(
        "**Šta je potrebno u fajlu (CSV ili Excel):**\n\n"
        "- **Kupac** — kolona poput `CustomerID`, `Customer`, `UserID`, `ClientID`\n"
        "- **Datum** — `InvoiceDate`, `Date`, `OrderDate`, `Timestamp`\n"
        "- **Iznos** — ili `Amount`/`Total`/`Sales`/`Revenue`, ili `UnitPrice` + `Quantity`\n\n"
        "Nazivi kolona mogu varirati — sistem ih automatski prepoznaje."
    )
    st.stop()

n = summary["n_customers"]; total_rev = summary["total_revenue"]
avg_clv = rfm["PredictedCLV"].mean()
champs = rfm[rfm["Segment"] == "Champions"]; vip = len(champs)
at_risk = int((rfm["Segment"] == "At Risk").sum())
health = rfm["HealthScore"].mean()
rev_d, cust_d = mom_delta(df)

with st.sidebar:
    st.markdown("---")
    st.markdown("<div class='sec'>AI Status</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='aistatus'>"
        f"<div class='statusrow'><span>Stanje</span>"
        f"<b style='color:#34d399'><span class='dot' style='background:#34d399'></span>Analiza završena</b></div>"
        f"<div class='statusrow'><span>Model</span><b style='color:#c4d2e8'>{method}</b></div>"
        f"<div class='statusrow' style='border-bottom:none'><span>Dataset</span>"
        f"<b style='color:#c4d2e8'>{filename}</b></div></div>",
        unsafe_allow_html=True)
    try:                                       # prava linija: prihod po mjesecima iz fajla
        _mrev = df.set_index("InvoiceDate").resample("MS")["TotalPrice"].sum()
        if len(_mrev) < 3:                     # premalo mjeseci -> sedmični nivo
            _mrev = df.set_index("InvoiceDate").resample("W")["TotalPrice"].sum()
        st.markdown(spark(_mrev.values.tolist(), "#22d3ee"), unsafe_allow_html=True)
        st.markdown("<div style='font-size:.66rem;color:#7c8aa5;margin-top:-2px'>📈 Prihod po "
                    "mjesecima (iz učitanog fajla)</div>", unsafe_allow_html=True)
    except Exception:
        pass
    st.markdown(f"<div style='font-size:.72rem;color:#7c8aa5'>Last updated: "
                f"{datetime.now():%d.%m.%Y %H:%M}</div>", unsafe_allow_html=True)
    st.markdown("<div class='upgrade'><b>SmartSeg AI Pro</b><p>Napredni uvidi i prioritetna "
                "podrška.</p></div>", unsafe_allow_html=True)
    st.button("⚡ Upgrade Now", use_container_width=True)

_notifs = [
    ("✅", "Analiza završena — segmenti su spremni.", "#34d399"),
    ("⚠️", f"{at_risk} kupaca pod rizikom odlaska.", "#f87171"),
    ("👑", f"{vip} Champions kupaca — najvredniji segment.", "#fbbf24"),
    ("📊", f"Učitano {n:,} kupaca iz „{filename}“.".replace(",", "."), "#7ee9f7"),
]
header(mode_title, mode_sub, filename, ready=True, notifs=_notifs)
with st.columns([5, 1])[1]:
    st.download_button("⬇️ Export Report", rfm.to_csv(index=False).encode("utf-8"),
                       "smartseg_export.csv", "text/csv", use_container_width=True, key="exportbtn")


# ================================================================= STANDARD MODE
if mode.startswith("📊") or page != "Advanced Mode":
    def kpi_row():
        k = st.columns(6)
        kpi(k[0], "Total Customers", f"{n:,}".replace(",", "."), "👥", "#22d3ee", 1, delta_pct=cust_d, sub="ukupno kupaca")
        kpi(k[1], "Total Revenue", fmt_eur_kpi(total_rev), "💰", "#a855f7", 2, delta_pct=rev_d, sub="ukupan prihod")
        kpi(k[2], "Average CLV", fmt_eur_kpi(avg_clv), "💎", "#06b6d4", 3, sub="prosječna vrijednost kupca")
        kpi(k[3], "VIP Customers", f"{vip:,}".replace(",", "."), "👑", "#c084fc", 4, sub="Champions segment")
        kpi(k[4], "At Risk", f"{at_risk:,}".replace(",", "."), "⚠️", "#ef4444", 5, sub="kupci pod rizikom")
        kpi(k[5], "Health Score", f"{health:.0f}/100", "❤️", "#34d399", 6, sub="prosječno zdravlje baze")

    # ---------- DASHBOARD ----------
    if "Dashboard" in page:
        kpi_row()
        st.write("")
        a = st.columns(3)
        champ_share = champs["PredictedCLV"].sum() / max(rfm["PredictedCLV"].sum(), 1) * 100
        if a[0].button(f"⚠️  **At Risk kupci rastu**\n\n{at_risk} kupaca traži hitnu pažnju  ›",
                       key="alert_red", use_container_width=True):
            st.session_state.nav_target = "Customer Segments"; st.rerun()
        if a[1].button(f"👑  **Champions donose najviše**\n\nTop segment = {champ_share:.0f}% predviđene vrijednosti  ›",
                       key="alert_amber", use_container_width=True):
            st.session_state.nav_target = "AI Insights"; st.rerun()
        if a[2].button("📈  **Personalizacija diže ROI**\n\nCiljane kampanje do +35% ROI  ›",
                       key="alert_cyan", use_container_width=True):
            st.session_state.nav_target = "ROI Simulator"; st.rerun()
        st.write("")
        r = st.columns(3)
        with r[0].container(border=True):
            st.markdown("<div class='sec'>Customer Segmentation</div>", unsafe_allow_html=True)
            donut_fig(rfm, n, "std_donut")
        with r[1].container(border=True):
            st.markdown("<div class='sec'>Customer Map (RFM)</div>", unsafe_allow_html=True)
            scatter_fig(rfm, "std_scatter")
        with r[2].container(border=True):
            st.markdown("<div class='sec'>AI Insights</div>", unsafe_allow_html=True)
            for ins in sorted(segment_insights(rfm),
                              key=lambda d: -rfm[rfm.Segment == d["segment"]]["PredictedCLV"].sum())[:4]:
                st.markdown(f"<div style='display:flex;gap:8px;font-size:.82rem;color:#c4d2e8;padding:5px 0;"
                            f"border-bottom:1px solid #182338'>{insight_icon(ins['segment'])}"
                            f"<span>{md_bold(ins['text'])}</span></div>", unsafe_allow_html=True)
            if st.button("View All Insights →", key="viewall_ins"):
                st.session_state.nav_target = "AI Insights"; st.rerun()
        st.write("")
        r2 = st.columns(3)
        with r2[0].container(border=True):
            _h = st.columns([5, 1], vertical_alignment="center")
            _h[0].markdown("<div class='sec'>Marketing Playbook</div>", unsafe_allow_html=True)
            if _h[1].button("↗", key="pb_open", help="Otvori Marketing Playbook"):
                st.session_state.nav_target = "Marketing Playbook"; st.rerun()
            playbook_table(rfm, head=4)
            _npb = len(build_playbook(rfm))
            if _npb > 4:
                with st.expander(f"Prikaži još {_npb - 4} segmenata"):
                    playbook_table(rfm, skip=4)
        with r2[1].container(border=True):
            _h = st.columns([5, 1], vertical_alignment="center")
            _h[0].markdown("<div class='sec'>ROI Simulator</div>", unsafe_allow_html=True)
            if _h[1].button("↗", key="roi_open", help="Otvori ROI Simulator"):
                st.session_state.nav_target = "ROI Simulator"; st.rerun()
            b = roi_slider("Marketing budžet (%)", 0, 100, 20, key="d_b")
            d = roi_slider("Popusti (%)", 0, 50, 10, key="d_d")
            rt = roi_slider("Retention (%)", 0, 50, 15, key="d_rt")
            res = simulate_roi(rfm, b, d, rt)
            oc = st.columns(2)
            oc[0].metric("Predviđeni ROI", f"{res['predicted_roi_pct']:.0f}%")
            oc[1].metric("Predviđeni profit", fmt_eur(res["predicted_profit"]))
        with r2[2].container(border=True):
            st.markdown("<div class='sec'>Executive Summary</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:.86rem;color:#c4d2e8;line-height:1.55'>"
                        f"{md_bold(executive_summary(rfm, use_llm=True))}</div>"
                        f"<div style='height:8px'></div>{exec_metric_rows(rfm)}", unsafe_allow_html=True)
        st.write("")
        with st.container(border=True):
            st.markdown("<div class='sec'>Top 3 preporučene akcije</div>", unsafe_allow_html=True)
            top3_actions(rfm)

    # ---------- CUSTOMER SEGMENTS ----------
    elif "Customer Segments" in page:
        kpi_row()
        st.write("")
        r = st.columns(2)
        with r[0].container(border=True):
            st.markdown("<div class='sec'>Customer Segmentation</div>", unsafe_allow_html=True)
            donut_fig(rfm, n, "seg_donut")
        with r[1].container(border=True):
            st.markdown("<div class='sec'>Customer Map (RFM)</div>", unsafe_allow_html=True)
            scatter_fig(rfm, "seg_scatter")
        st.write("")
        st.markdown("<div class='sec'>Profil segmenata</div>", unsafe_allow_html=True)
        segment_cards(rfm)
        with st.container(border=True):
            st.markdown("<div class='sec'>Detaljna tabela</div>", unsafe_allow_html=True)
            prof = rfm.groupby("Segment").agg(
                Kupaca=("CustomerID", "count"),
                Recency=("Recency", "mean"), Frequency=("Frequency", "mean"),
                Monetary=("Monetary", "mean"), CLV=("PredictedCLV", "mean"),
                Health=("HealthScore", "mean")).round(0).reset_index().sort_values("CLV", ascending=False)
            st.dataframe(prof, use_container_width=True, hide_index=True)

    # ---------- AI INSIGHTS ----------
    elif "AI Insights" in page:
        c = st.columns(2)
        with c[0].container(border=True):
            st.markdown("<div class='sec'>AI Insights — uvidi po segmentima</div>", unsafe_allow_html=True)
            for ins in sorted(segment_insights(rfm),
                              key=lambda d: -rfm[rfm.Segment == d["segment"]]["PredictedCLV"].sum()):
                st.markdown(f"<div style='display:flex;gap:8px;font-size:.86rem;color:#c4d2e8;padding:7px 0;"
                            f"border-bottom:1px solid #182338'>{insight_icon(ins['segment'])}"
                            f"<span>{md_bold(ins['text'])}</span></div>", unsafe_allow_html=True)
        with c[1].container(border=True):
            st.markdown("<div class='sec'>Executive Summary</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:.88rem;color:#c4d2e8;line-height:1.6'>"
                        f"{md_bold(executive_summary(rfm, use_llm=True))}</div>"
                        f"<div style='height:8px'></div>{exec_metric_rows(rfm)}", unsafe_allow_html=True)

    # ---------- MARKETING PLAYBOOK ----------
    elif "Marketing Playbook" in page:
        with st.container(border=True):
            st.markdown("<div class='sec'>Marketing Playbook — strategija po segmentu</div>", unsafe_allow_html=True)
            st.dataframe(build_playbook(rfm), use_container_width=True, hide_index=True, height=430)

    # ---------- ROI SIMULATOR ----------
    elif "ROI Simulator" in page:
        c = st.columns(2)
        with c[0].container(border=True):
            st.markdown("<div class='sec'>Parametri kampanje</div>", unsafe_allow_html=True)
            b = roi_slider("Marketing budžet (%)", 0, 100, 20, key="r_b")
            d = roi_slider("Popusti (%)", 0, 50, 10, key="r_d")
            rt = roi_slider("Retention (%)", 0, 50, 15, key="r_rt")
            res = simulate_roi(rfm, b, d, rt)
        with c[1].container(border=True):
            st.markdown("<div class='sec'>Predviđeni rezultati</div>", unsafe_allow_html=True)
            m = st.columns(2)
            m[0].metric("Predviđeni ROI", f"{res['predicted_roi_pct']:.0f}%")
            m[1].metric("Predviđeni profit", fmt_eur(res["predicted_profit"]))
            m2 = st.columns(2)
            m2[0].metric("Zadržani kupci", f"{res.get('retained_customers', 0):,.0f}".replace(",", "."))
            m2[1].metric("Procijenjeni trošak", fmt_eur(res.get("total_cost", 0)))
            st.caption("Pomjeraj klizače da vidiš kako budžet, popusti i retention utiču na ROI i profit. "
                       "Računato na osnovu stvarne baze kupaca i njihove predviđene vrijednosti (CLV).")

    # ---------- REPORTS ----------
    elif "Reports" in page:
        kpi_row()
        st.write("")
        with st.container(border=True):
            st.markdown("<div class='sec'>Executive Summary</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:.9rem;color:#c4d2e8;line-height:1.6'>"
                        f"{md_bold(executive_summary(rfm, use_llm=True))}</div>"
                        f"<div style='height:8px'></div>{exec_metric_rows(rfm)}", unsafe_allow_html=True)
        st.write("")
        with st.container(border=True):
            st.markdown("<div class='sec'>Top 3 preporučene akcije</div>", unsafe_allow_html=True)
            top3_actions(rfm, kp="rep_")

    # ---------- EXPORT DATA ----------
    elif "Export Data" in page:
        _best = A.model_comparison(rfm)["best"]
        with st.container(border=True):
            st.markdown("<div class='sec'>Export Center</div>", unsafe_allow_html=True)
            st.write("Preuzmi rezultate analize:")
            e = st.columns(3)
            e[0].download_button("⬇️ Segmentacija (CSV)", rfm.to_csv(index=False).encode("utf-8"),
                                 "segmentacija.csv", "text/csv", use_container_width=True)
            e[1].download_button("⬇️ RFM podaci (CSV)",
                                 rfm[["CustomerID", "Recency", "Frequency", "Monetary", "PredictedCLV",
                                      "HealthScore", "Segment"]].to_csv(index=False).encode("utf-8"),
                                 "rfm.csv", "text/csv", use_container_width=True)
            report = build_full_report(rfm, model_name=_best, use_llm=True)
            e[2].download_button("⬇️ AI Izvještaj (MD)", report.encode("utf-8"),
                                 "smartseg_report.md", "text/markdown", use_container_width=True)
            st.write("")
            _pc = st.columns([1, 2, 1])
            with _pc[1]:
                if st.button("📄 Generiši PDF izvještaj", key="pdfbtn_adv", use_container_width=True):
                    with st.spinner("Pravim PDF izvještaj..."):
                        _sumtxt = executive_summary(rfm, use_llm=True)
                        _pdf = build_pdf(rfm, _sumtxt, model_name=_best, dataset_name=filename)
                        st.session_state["pdf_bytes"] = _pdf
                if st.session_state.get("pdf_bytes"):
                    st.download_button("⬇️ Preuzmi PDF", st.session_state["pdf_bytes"],
                                       "smartseg_izvjestaj.pdf", "application/pdf",
                                       use_container_width=True, key="pdfdl_adv")

    # ---------- SETTINGS ----------
    else:
        with st.container(border=True):
            st.markdown("<div class='sec'>Settings</div>", unsafe_allow_html=True)
            import os as _os
            llm = "Aktivan (Claude)" if _os.environ.get("ANTHROPIC_API_KEY") else "Rule-based (bez ključa)"
            st.markdown(
                f"<div class='statusrow'><span>Aktivni dataset</span><b style='color:#e2e8f0'>{filename}</b></div>"
                f"<div class='statusrow'><span>Broj kupaca</span><b style='color:#e2e8f0'>{n:,}</b></div>"
                f"<div class='statusrow'><span>Model</span><b style='color:#e2e8f0'>{method}</b></div>"
                f"<div class='statusrow'><span>AI asistent</span><b style='color:#e2e8f0'>{llm}</b></div>"
                f"<div class='statusrow'><span>Tema</span><b style='color:#e2e8f0'>Dark Neon</b></div>",
                unsafe_allow_html=True)
            if st.button("🔄 Resetuj (očisti podatke)"):
                st.session_state.pop("datafile", None)
                st.cache_data.clear()
                st.rerun()


# ================================================================= ADVANCED MODE
else:
    adv = load_advanced(file_bytes, filename, method)
    mc, sk, dq = adv["mc"], adv["sk"], adv["dq"]

    r = st.columns(3)
    with r[0].container(border=True):
        _h = st.columns([5, 1], vertical_alignment="center")
        _h[0].markdown("<div class='sec'><span class='num'>1</span>Model Comparison</div>", unsafe_allow_html=True)
        tbl = mc["table"].rename(columns={"DaviesBouldin": "Davies-Bouldin ↓",
                                          "CalinskiHarabasz": "Calinski-H. ↑", "Silhouette": "Silhouette ↑"})

        @st.dialog("Model Comparison — puna tabela", width="large")
        def _mc_dialog():
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            st.markdown(
                "<div style='font-size:.8rem;color:#9fb2cc;line-height:1.6'>"
                "<b style='color:#c4d2e8'>Kako čitati metrike:</b><br>"
                "• <b>Silhouette ↑</b> (−1 do 1) — koliko je svaki kupac bliži svom klasteru nego tuđem; veće = bolje.<br>"
                "• <b>Davies-Bouldin ↓</b> — prosječna sličnost klastera sa najsličnijim susjedom; manje = bolje.<br>"
                "• <b>Calinski-Harabasz ↑</b> — odnos razdvojenosti između i unutar klastera; veće = bolje.<br><br>"
                "<b style='color:#c4d2e8'>Algoritmi:</b> K-Means grupiše oko centara (baseline), "
                "Gaussian Mixture modeluje klastere kao raspodjele (broj komponenti biran automatski po BIC-u), "
                "HDBSCAN nalazi grupe po gustini i sam određuje njihov broj.</div>",
                unsafe_allow_html=True)

        if _h[1].button("↗", key="mc_open", help="Otvori punu tabelu sa objašnjenjima"):
            _mc_dialog()
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.markdown(f"<div class='badge'>🏆 Najbolji model: {mc['best']}</div> "
                    "<span style='font-size:.8rem;color:#8aa0c0'>(po silhouette skoru)</span>", unsafe_allow_html=True)
        try:
            interp(f"<b>{mc['best']}</b> daje najbolju separaciju klastera "
                   f"(silhouette {mc['table']['Silhouette'].max():.2f}); operativno koristimo "
                   f"k={mc['optimal_k']} radi poslovne interpretabilnosti.", "#a855f7")
        except Exception: pass
    with r[1].container(border=True):
        st.markdown("<div class='sec'><span class='num'>2</span>PCA 2D Visualization</div>", unsafe_allow_html=True)
        p2 = adv["pca2"]; f = px.scatter(p2["df"], x="PC1", y="PC2", color="Segment",
            color_discrete_map=SEGMENT_COLORS, opacity=.7)
        f.update_traces(marker=dict(size=5)); f.update_layout(**PL, height=300, showlegend=False,
            xaxis_title=f"PC1 ({p2['explained'][0]}%)", yaxis_title=f"PC2 ({p2['explained'][1]}%)")
        f.update_xaxes(gridcolor="#1c2740"); f.update_yaxes(gridcolor="#1c2740")
        st.plotly_chart(f, use_container_width=True, key="pca2")
        try:
            interp(f"Prve dvije komponente objašnjavaju <b>{p2['explained'][0]+p2['explained'][1]:.0f}%</b> "
                   f"varijanse — boje (segmenti) formiraju vidljivo odvojene grupe.", "#a855f7")
        except Exception: pass
    with r[2].container(border=True):
        st.markdown("<div class='sec'><span class='num'>3</span>PCA 3D Visualization</div>", unsafe_allow_html=True)
        p3 = adv["pca3"]; f = px.scatter_3d(p3["df"], x="PC1", y="PC2", z="PC3", color="Segment",
            color_discrete_map=SEGMENT_COLORS, opacity=.7)
        f.update_traces(marker=dict(size=3)); f.update_layout(**PL, height=300, showlegend=False,
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
        st.plotly_chart(f, use_container_width=True, key="pca3")
        try:
            interp(f"Tri komponente zajedno objašnjavaju <b>{sum(p3['explained'][:3]):.0f}%</b> "
                   f"varijanse — treća osa dodaje {p3['explained'][2]:.0f}% i otkriva dubinu "
                   f"razdvajanja segmenata.", "#a855f7")
        except Exception: pass

    r = st.columns(3)
    with r[0].container(border=True):
        st.markdown("<div class='sec'><span class='num'>4</span>Cluster Heatmap (prosjek po segmentu)</div>", unsafe_allow_html=True)
        hm = adv["heatmap"]; norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-9)
        f = go.Figure(go.Heatmap(z=norm.values, x=hm.columns.tolist(), y=hm.index.tolist(),
            text=hm.round(0).values, texttemplate="%{text}", colorscale="Turbo", showscale=False))
        f.update_layout(**PL, height=300); st.plotly_chart(f, use_container_width=True, key="hm")
        try:
            interp(f"<b>{hm['Monetary'].idxmax()}</b> ima najveću prosječnu potrošnju, a "
                   f"<b>{hm['Recency'].idxmax()}</b> najduže ne kupuje — toplije boje = veće vrijednosti.", "#a855f7")
        except Exception: pass
    with r[1].container(border=True):
        st.markdown("<div class='sec'><span class='num'>5</span>Feature Importance</div>", unsafe_allow_html=True)
        fi = adv["fi"]; f = px.bar(fi, x="Importance", y="Feature", orientation="h",
            color_discrete_sequence=["#7c3aed"])
        f.update_layout(**PL, height=300, yaxis=dict(autorange="reversed"))
        f.update_xaxes(gridcolor="#1c2740"); st.plotly_chart(f, use_container_width=True, key="fi")
        try:
            _f = fi.sort_values("Importance", ascending=False)
            interp(f"<b>{_f.iloc[0]['Feature']}</b> ({_f.iloc[0]['Importance']:.2f}) i "
                   f"<b>{_f.iloc[1]['Feature']}</b> ({_f.iloc[1]['Importance']:.2f}) najviše razdvajaju "
                   f"segmente — ponašanje potrošnje je ključni faktor.", "#a855f7")
        except Exception: pass
    with r[2].container(border=True):
        st.markdown("<div class='sec'><span class='num'>6</span>Silhouette Analysis</div>", unsafe_allow_html=True)
        f = go.Figure(go.Scatter(x=sk["k"], y=sk["scores"], mode="lines+markers",
            line=dict(color="#22d3ee", width=2), marker=dict(size=7)))
        f.add_scatter(x=[sk["optimal_k"]], y=[max(sk["scores"])], mode="markers",
            marker=dict(size=13, color="#34d399"), showlegend=False)
        f.update_layout(**PL, height=300, showlegend=False,
                        xaxis_title="Broj klastera (k)", yaxis_title="Silhouette")
        f.update_xaxes(gridcolor="#1c2740"); f.update_yaxes(gridcolor="#1c2740")
        st.plotly_chart(f, use_container_width=True, key="sil")
        st.markdown(f"<span class='badge'>Najbolji silhouette: k = {sk['optimal_k']}</span> "
                    f"<span style='font-size:.78rem;color:#8aa0c0'>· operativno k={mc['optimal_k']} (BIC + interpretabilnost)</span>",
                    unsafe_allow_html=True)
        try:
            interp(f"Razdvajanje je matematički najčistije za k={sk['optimal_k']} "
                   f"(skor {max(sk['scores']):.2f}); poslije pika kvalitet opada, pa veći k ne donosi korist.", "#a855f7")
        except Exception: pass

    r = st.columns(3)
    with r[0].container(border=True):
        oa = adv["oa"]
        st.markdown("<div class='sec'><span class='num'>7</span>Outlier Analysis</div>", unsafe_allow_html=True)
        st.markdown(f"Identifikovano **{oa['n_outliers']}** outliera "
                    f"(**{oa['pct']}%** kupaca), od toga {oa['high_risk']} visokog rizika.")
        st.dataframe(oa["sample"], use_container_width=True, hide_index=True, height=210)
        if st.button("👁 View Outliers — puna lista", key="view_out", use_container_width=True):
            st.session_state.show_out = not st.session_state.get("show_out", False)
        if st.session_state.get("show_out"):
            full_out = (oa["rfm"][oa["rfm"]["Outlier"]]
                        [["CustomerID", "Recency", "Frequency", "Monetary", "PredictedCLV", "Segment"]]
                        .sort_values("Monetary", ascending=False).head(25).round(2))
            st.dataframe(full_out, use_container_width=True, hide_index=True, height=240)
        try:
            interp(f"<b>{oa['pct']}%</b> kupaca odskače od svih grupa (IsolationForest) — najčešće "
                   f"ekstremno visoka potrošnja; {oa['high_risk']} ih je uz to rizično za odlazak.", "#a855f7")
        except Exception: pass
    with r[1].container(border=True):
        st.markdown("<div class='sec'><span class='num'>8</span>Cohort Analysis (retention %)</div>", unsafe_allow_html=True)
        co = adv["cohort"]
        f = go.Figure(go.Heatmap(z=co.values, x=co.columns.tolist(), y=co.index.tolist(),
            text=co.values, texttemplate="%{text:.0f}", colorscale="RdYlGn", showscale=False,
            zmin=0, zmax=100))
        f.update_layout(**PL, height=270, yaxis=dict(autorange="reversed"))
        st.plotly_chart(f, use_container_width=True, key="cohort")
        try:
            if co.shape[1] <= 1:
                interp("U ovom skupu nema ponovljenih kupovina kroz mjesece, pa je vidljiv samo mjesec "
                       "prve kupovine (M0) — retencija se ne može mjeriti na ovim podacima.", "#a855f7")
            else:
                interp(f"Prosječno <b>{co.iloc[:,1].mean():.0f}%</b> kupaca se vrati u prvom mjesecu "
                       f"nakon prve kupovine; zelenija polja = bolja retencija kohorte.", "#a855f7")
        except Exception: pass
    with r[2].container(border=True):
        st.markdown("<div class='sec'><span class='num'>9</span>RFM Distribution</div>", unsafe_allow_html=True)
        for colname, color in [("Recency", "#22d3ee"), ("Frequency", "#34d399"), ("Monetary", "#a855f7")]:
            st.markdown(f"<div style='font-size:.72rem;font-weight:700;color:{color};"
                        f"margin:2px 0 0'>{colname}</div>", unsafe_allow_html=True)
            f = px.histogram(rfm, x=colname, nbins=30, color_discrete_sequence=[color])
            f.update_layout(**PL, height=88, showlegend=False, yaxis_title="", xaxis_title="")
            f.update_xaxes(gridcolor="#1c2740"); f.update_yaxes(gridcolor="#1c2740", showticklabels=False)
            st.plotly_chart(f, use_container_width=True, key=f"hist_{colname}")
        try:
            interp(f"Medijan: recency <b>{rfm['Recency'].median():.0f} d.</b>, učestalost "
                   f"<b>{rfm['Frequency'].median():.0f}</b>, potrošnja <b>{fmt_eur(rfm['Monetary'].median())}</b> — "
                   f"dugi repovi udesno znače mali broj vrlo vrijednih kupaca.", "#a855f7")
        except Exception: pass

    r = st.columns(4)
    with r[0].container(border=True):
        st.markdown("<div class='sec'><span class='num'>10</span>Cluster Distribution</div>", unsafe_allow_html=True)
        donut_fig(rfm, n, "adv_donut", compact=True)
    with r[1].container(border=True):
        st.markdown("<div class='sec'><span class='num'>11</span>Data Quality</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='statusrow'><span>Ukupno redova</span><b style='color:#e2e8f0'>{dq['rows']:,}</b></div>"
            f"<div class='statusrow'><span>Ukupno kolona</span><b style='color:#e2e8f0'>{dq['cols']}</b></div>"
            f"<div class='statusrow'><span>Nedostajuće</span><b style='color:#e2e8f0'>{dq['missing_pct']}%</b></div>"
            f"<div class='statusrow'><span>Duplikati</span><b style='color:#e2e8f0'>{dq['dup_pct']}%</b></div>"
            f"<div class='statusrow'><span>Outlieri</span><b style='color:#e2e8f0'>{dq['outlier_pct']}%</b></div>"
            f"<div class='statusrow'><span>Data Quality Score</span><b>{dq['score']}%</b></div>",
            unsafe_allow_html=True)
        try:
            _lvl = "odlični" if dq["score"] >= 90 else ("dobri" if dq["score"] >= 75 else "umjereni")
            interp(f"Skor <b>{dq['score']}%</b> — podaci su {_lvl} za pouzdanu segmentaciju "
                   f"(malo nedostajućih vrijednosti i duplikata).", "#a855f7")
        except Exception: pass
    with r[2].container(border=True):
        st.markdown("<div class='sec'><span class='num'>12</span>AI Technical Insights</div>", unsafe_allow_html=True)
        for t in adv["tech"]:
            st.markdown(f"<div style='font-size:.8rem;color:#c4d2e8;padding:4px 0'>• {t}</div>",
                        unsafe_allow_html=True)
    with r[3].container(border=True):
        st.markdown("<div class='sec'><span class='num'>13</span>Download Center</div>", unsafe_allow_html=True)
        st.download_button("⬇️ Segmentacija (CSV)", rfm.to_csv(index=False).encode("utf-8"),
                           "segmentacija.csv", "text/csv", use_container_width=True)
        st.download_button("⬇️ RFM podaci (CSV)",
                           rfm[["CustomerID", "Recency", "Frequency", "Monetary", "PredictedCLV",
                                "HealthScore", "Segment"]].to_csv(index=False).encode("utf-8"),
                           "rfm.csv", "text/csv", use_container_width=True)
        report = build_full_report(rfm, model_name=mc["best"], use_llm=True)
        st.download_button("⬇️ AI Izvještaj (MD)", report.encode("utf-8"),
                           "smartseg_report.md", "text/markdown", use_container_width=True)
        st.write("")
        if st.button("📄 Generiši PDF izvještaj", key="pdfbtn", use_container_width=True):
            with st.spinner("Pravim PDF izvještaj..."):
                _sumtxt = executive_summary(rfm, use_llm=True)
                _pdf = build_pdf(rfm, _sumtxt, model_name=mc["best"], dataset_name=filename)
                st.session_state["pdf_bytes"] = _pdf
        if st.session_state.get("pdf_bytes"):
            st.download_button("⬇️ Preuzmi PDF", st.session_state["pdf_bytes"],
                               "smartseg_izvjestaj.pdf", "application/pdf",
                               use_container_width=True, key="pdfdl")


# ================================================================= AI ASSISTANT (plutajući)
with st.container(key="chatfab"):
    with st.popover("🤖"):
        _bot_logo = (
            "<span class='botavatar'>"
            "<svg width='30' height='30' viewBox='0 0 48 48' fill='none'>"
            "<defs><linearGradient id='botA' x1='0' y1='0' x2='1' y2='1'>"
            "<stop offset='0%' stop-color='#22d3ee'/><stop offset='100%' stop-color='#a855f7'/></linearGradient></defs>"
            "<circle cx='24' cy='24' r='19' stroke='url(#botA)' stroke-width='2.8'/>"
            "<ellipse cx='24' cy='24' rx='8.5' ry='19' stroke='url(#botA)' stroke-width='1.8' opacity='0.8'/>"
            "<line x1='5' y1='24' x2='43' y2='24' stroke='url(#botA)' stroke-width='1.8' opacity='0.8'/>"
            "<circle cx='24' cy='5' r='3.3' fill='#22d3ee'/><circle cx='43' cy='24' r='3.3' fill='#a855f7'/>"
            "<circle cx='24' cy='43' r='3.3' fill='#a855f7'/><circle cx='5' cy='24' r='3.3' fill='#22d3ee'/>"
            "<circle cx='24' cy='24' r='4.5' fill='url(#botA)'/></svg></span>")
        st.markdown("<div style='display:flex;align-items:center;gap:10px;margin-bottom:3px'>"
                    f"{_bot_logo}"
                    "<span style='font-weight:800;font-size:1.02rem;color:#e8eefb'>SmartSeg AI Assistant</span>"
                    "</div>"
                    "<div style='font-size:.73rem;color:#8aa0c0;margin:0 0 10px 48px'>"
                    "pitaj o segmentima, modelima i preporukama</div>", unsafe_allow_html=True)
        if "chat" not in st.session_state:
            st.session_state.chat = [("ai", "Zdravo! Pitaj me npr.: „Ko su Champions?“, "
                                            "„Koji segment je rizičan?“ ili „Koji model je najbolji?“")]
        for who, msg in st.session_state.chat[-8:]:
            klass = "ai" if who == "ai" else "me"
            st.markdown(f"<div class='chatmsg {klass}'>{chat_md(msg)}</div>", unsafe_allow_html=True)
        _last = st.session_state.chat[-1]
        if _last[0] == "ai" and len(_last[1]) > 450:      # dug odgovor = izvještaj
            st.download_button("⬇️ Sačuvaj kao izvještaj (.md)", _last[1].encode("utf-8"),
                               "smartseg_chat_izvjestaj.md", "text/markdown", key="chatdl")
        with st.form("chatform", clear_on_submit=True, border=False):
            _cq = st.columns([4, 1])
            _q = _cq[0].text_input("Pitanje", label_visibility="collapsed",
                                   placeholder="Postavi pitanje...")
            _send = _cq[1].form_submit_button("➤")
        if _send and _q.strip():
            st.session_state.chat.append(("user", _q.strip()))
            st.session_state.chat.append(("ai", assistant_ask(_q.strip(), rfm, use_llm=True)))
            st.rerun()
