"""
styles.py — CSS + sidebar chrome for the RACE Quiz UI.
"""
from pathlib import Path
import sys
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Step tracker ── */
.step-bar { display:flex; gap:8px; margin-bottom:24px; }
.step { flex:1; padding:10px 6px; border-radius:10px; text-align:center;
        font-size:0.78rem; font-weight:600; letter-spacing:.04em;
        background:#1e2130; color:#6b7280; border:2px solid #2a2f45; cursor:default; }
.step.active { background:linear-gradient(135deg,#6366f1,#8b5cf6);
               color:#fff; border-color:transparent;
               box-shadow:0 4px 14px rgba(99,102,241,.4); }
.step.done   { background:#14532d; color:#86efac; border-color:#166534; }

/* ── Cards ── */
.q-card { background:#1e2130; border:1px solid #2a2f45; border-radius:14px;
           padding:24px 28px; margin-bottom:16px; }
.q-card h2 { font-size:1.25rem; margin:6px 0 0; color:#e2e8f0; }
.q-card .label { font-size:0.72rem; text-transform:uppercase; letter-spacing:.1em;
                  color:#6366f1; font-weight:700; }

/* ── Option buttons ── */
.opt-btn { width:100%; text-align:left; padding:14px 18px; margin-bottom:8px;
           background:#1e2130; border:2px solid #2a2f45; border-radius:10px;
           color:#cbd5e1; font-size:0.95rem; cursor:pointer;
           transition:border-color .15s, background .15s; }
.opt-btn:hover { border-color:#6366f1; background:#252b42; }
.opt-btn.selected { border-color:#6366f1; background:#252b42; color:#fff; }
.opt-btn.correct  { border-color:#22c55e; background:#14532d22; color:#86efac; }
.opt-btn.wrong    { border-color:#ef4444; background:#7f1d1d22; color:#fca5a5; }

/* ── Result banners ── */
.banner { border-radius:10px; padding:16px 22px; margin:12px 0;
          font-weight:600; font-size:1rem; }
.banner.ok  { background:#14532d33; border:1.5px solid #22c55e; color:#86efac; }
.banner.err { background:#7f1d1d33; border:1.5px solid #ef4444; color:#fca5a5; }

/* ── Hint cards ── */
.hint-card { background:#1e2130; border:1.5px solid #374151; border-radius:12px;
             padding:18px 22px; margin-bottom:10px; }
.hint-card .hl { font-size:0.7rem; text-transform:uppercase; letter-spacing:.1em;
                 color:#f59e0b; font-weight:700; margin-bottom:6px; }
.hint-card p { color:#cbd5e1; margin:0; font-size:0.95rem; line-height:1.6; }
.hint-card.locked { opacity:.35; filter:grayscale(1); }

/* ── Metric tiles ── */
.metric-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }
.metric-tile { flex:1; min-width:130px; background:#1e2130;
               border:1px solid #2a2f45; border-radius:12px;
               padding:16px 18px; }
.metric-tile .mt { font-size:0.7rem; color:#6b7280; text-transform:uppercase;
                   letter-spacing:.08em; font-weight:600; }
.metric-tile .mv { font-size:1.6rem; font-weight:700; color:#e2e8f0;
                   line-height:1.2; margin-top:4px; }
.metric-tile .ms { font-size:0.78rem; color:#6366f1; margin-top:2px; }

/* ── Confusion matrix ── */
.cm-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:8px; }
.cm-cell { border-radius:8px; padding:14px; text-align:center; }
.cm-cell .cv { font-size:1.4rem; font-weight:700; }
.cm-cell .cl { font-size:0.7rem; color:#9ca3af; }
.cm-tp { background:#14532d33; color:#86efac; }
.cm-tn { background:#1e3a5f33; color:#93c5fd; }
.cm-fp { background:#7f1d1d22; color:#fca5a5; }
.cm-fn { background:#78350f22; color:#fcd34d; }

/* ── Section heading ── */
.sec { font-size:0.72rem; text-transform:uppercase; letter-spacing:.12em;
       color:#6366f1; font-weight:700; margin:24px 0 10px; }

/* ── Error / info boxes ── */
.err-box { background:#7f1d1d22; border:1.5px solid #ef4444; border-radius:10px;
           padding:14px 18px; color:#fca5a5; font-size:0.9rem; margin-top:8px; }
.info-box { background:#1e3a5f33; border:1.5px solid #3b82f6; border-radius:10px;
            padding:14px 18px; color:#93c5fd; font-size:0.9rem; margin-top:8px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.5rem !important; }
</style>
"""


# Loaded training metrics from the last model run (static for the session)
MODEL_A_METRICS = {
    "lr":  {"accuracy":0.5689,"precision":0.2907,"recall":0.5031,"f1":0.3685,
            "tn":23361,"fp":16179,"fn":6549,"tp":6631},
    "svm": {"accuracy":0.5746,"precision":0.2917,"recall":0.4913,"f1":0.3661,
            "tn":23820,"fp":15720,"fn":6705,"tp":6475},
    "nb":  {"accuracy":0.6579,"precision":0.3121,"recall":0.3058,"f1":0.3089,
            "tn":30654,"fp":8886,"fn":9149,"tp":4031},
    "rf":  {"accuracy":0.5864,"precision":0.3043,"recall":0.5086,"f1":0.3808,
            "tn":24211,"fp":15329,"fn":6476,"tp":6704},
    "ensemble": {"mcqAccuracy":0.3653,"binaryF1":0.3762,"binaryAccuracy":0.5788,
                 "tn":23814,"fp":15726,"fn":6482,"tp":6698,
                 "crossValF1Mean":0.5264,"crossValF1Std":0.0020,
                 "randomBaseline":0.25,"articlesUsed":87866,
                 "trainRows":149372,"testRows":52720,"features":12},
}
MODEL_B_METRICS = {
    "rf":  {"accuracy":0.88,"precision":0.85,"recall":0.82,"f1":0.83},
    "lr":  {"accuracy":0.81,"precision":0.76,"recall":0.78,"f1":0.77},
}


def injectCSS():
    st.markdown(CSS, unsafe_allow_html=True)


def renderStepBar(currentScreen: str):
    steps = [
        ("input",     "01 · Article Input"),
        ("quiz",      "02 · Quiz"),
        ("hints",     "03 · Hints"),
        ("dashboard", "04 · Dashboard"),
    ]
    order = [s[0] for s in steps]
    currentIdx = order.index(currentScreen) if currentScreen in order else 0

    parts = []
    for i, (key, label) in enumerate(steps):
        if i < currentIdx:
            cls = "done"
        elif i == currentIdx:
            cls = "active"
        else:
            cls = ""
        parts.append(f'<div class="step {cls}">{label}</div>')

    st.markdown(f'<div class="step-bar">{"".join(parts)}</div>', unsafe_allow_html=True)
