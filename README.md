# 🌍 SmartSeg AI — AI Customer Intelligence Platform

Automatska segmentacija kupaca za e-commerce: RFM + CLV + Customer Health Score +
nenadgledano klasterovanje (GMM / HDBSCAN / KMeans), interaktivni dashboard,
LLM interpretacija, ROI "what-if" simulator, marketing playbook i napredna
ML analitika — u jednoj Streamlit aplikaciji sa DVA MODA.

Tema: "Segmentacija kupaca — Ko su zapravo naši korisnici?"

> **Preporučeni način korišćenja:** aplikacija se najbolje koristi preko
> **deploy-ovane verzije na Streamlit Community Cloud-u** , gdje
> radi pun set funkcija uključujući pravu Claude interpretaciju. Lokalno pokretanje
> je takođe podržano (vidi „Pokretanje").


## Dva moda

Standard Mode (za menadžere): KPI kartice, AI business alerts, donut segmenata,
customer map (scatter), AI insights, marketing playbook, ROI simulator, executive
summary i top 3 preporučene akcije.

Advanced Mode (za data mining): Model Comparison (KMeans / GMM / HDBSCAN sa
Silhouette, Davies-Bouldin, Calinski-Harabasz), PCA 2D i 3D, Cluster Heatmap,
Feature Importance, Silhouette analiza po k, Outlier detekcija, Cohort (retention)
analiza, RFM distribucije, Data Quality i AI Technical Insights + Download Center.

Plus AI Assistant (plutajući chat, dole desno) — objašnjava segmente, modele i daje
preporuke. Uz ANTHROPIC_API_KEY koristi pravi Claude; bez ključa radi rule-based.

## Izvoz rezultata
- Segmentacija (CSV) — cijela tabela kupaca sa segmentima i svim metrikama
- RFM podaci (CSV) — uži set (R/F/M, CLV, Health, Segment)
- AI Izvještaj (MD) — strukturiran tekstualni izvještaj (Claude/rule-based)
- PDF izvještaj — profesionalan dokument: KPI, sažetak, tabela segmenata, donut grafik, preporuke

## Pokretanje (lokalno, VS Code)

    python -m venv .venv
    # Windows:  .venv\Scripts\activate    | macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

Otvori http://localhost:8501 . U bočnom meniju biraš Standard ili Advanced mod.
Aplikacija počinje prazna (sve na nuli). Učitaj svoj **CSV ili Excel** fajl — sistem automatski prepoznaje kolone (kupac, datum, iznos), čisti podatke i izračunava sve KPI-jeve, segmente i ML analize iz pravih podataka.

Preporučeni demo dataset: **Online Retail II** (UCI) — daje pune segmente i cohort matricu.


## Deploy
- Streamlit Community Cloud
- Hugging Face Spaces (SDK: Streamlit)

## Metodologija
1. Čišćenje: storna (Invoice 'C'), negativne količine/cijene, redovi bez CustomerID.
2. RFM + izvedeno: Recency/Frequency/Monetary, AOV, tenure, inter-purchase time.
3. CLV: AOV * očekivane kupovine * p_alive,  p_alive = exp(-Recency/(IPT+7)).
4. Health Score: kompozit 0-100 iz R/F/M i p_alive (percentilno ponderisan).
5. Segmentacija: rule-based RFM nazivi + nenadgledani klasteri (GMM po BIC, HDBSCAN, KMeans).
6. Evaluacija: Silhouette, Davies-Bouldin, Calinski-Harabasz; najbolji model bira se po podacima.
7. Dodatno: PCA (2D/3D), feature importance (RandomForest), IsolationForest outlieri, cohort.
8. Akcija: ROI simulator + marketing playbook + AI rezime + AI asistent.
