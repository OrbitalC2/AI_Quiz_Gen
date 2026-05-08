"""
app.py — Single-file Streamlit app for RACE Quiz Studio.

All four screens are rendered here based on st.session_state.currentScreen.
No multi-page navigation — avoids the confusion of Streamlit's sidebar page links.

Screens
-------
  input     → Article Input (Screen 1)
  quiz      → Quiz View     (Screen 2)
  hints     → Hint Panel    (Screen 3)
  dashboard → Dashboard     (Screen 4)
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.components.sample_data import SAMPLE_ARTICLES                    # noqa: E402
from ui.components.state import (                                         # noqa: E402
    checkAnswer,
    ensureRaceChunk,
    generateQuiz,
    getPipeline,
    goTo,
    initSessionState,
    loadRaceArticle,
    loadRandomRace,
    loadSampleArticle,
)
from ui.components.styles import (                                        # noqa: E402
    MODEL_A_METRICS,
    MODEL_B_METRICS,
    injectCSS,
    renderStepBar,
)

# Lazy T5 comparison (only imported on dashboard, never blocks startup)
def _getT5Generator():
    from src.inference import generateT5Question
    return generateT5Question

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RACE Quiz Studio",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)
initSessionState()

# Silently warm up the pipeline in a background thread so the UI loads
# instantly. getPipeline() is @st.cache_resource — once the thread populates
# the cache, all future calls (including the user's first quiz request) return
# the already-loaded singleton with no delay.
if not st.session_state.get("_pipelineWarmedUp"):
    import threading
    threading.Thread(target=getPipeline, daemon=True).start()
    st.session_state["_pipelineWarmedUp"] = True

injectCSS()

# ── Sidebar (minimal — just branding + jump links) ─────────────────────────
with st.sidebar:
    st.markdown("## 📚 RACE Quiz Studio")
    st.caption("AI Reading Comprehension System · FAST NUCES 2026")
    st.divider()
    if st.button("① Article Input",  use_container_width=True): goTo("input")
    if st.button("② Quiz View",      use_container_width=True): goTo("quiz")
    if st.button("③ Hints",          use_container_width=True): goTo("hints")
    if st.button("④ Dashboard",      use_container_width=True): goTo("dashboard")
    st.divider()
    if st.session_state.quizResult:
        quiz = st.session_state.quizResult
        st.caption("**Current question**")
        st.write(quiz["question"])
        st.caption(f"Correct: **{quiz['correctLabel']}) {quiz['correctAnswer']}**")

screen = st.session_state.currentScreen
renderStepBar(screen)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SCREEN 1 — Article Input                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if screen == "input":
    st.markdown('<div class="sec">Screen 01</div>', unsafe_allow_html=True)
    st.markdown("## 📄 Article Input")
    st.caption("Paste a passage, upload a file, or load a sample — then generate your quiz.")

    # ── Source selector ───────────────────────────────────────────────────
    srcTab, raceTab, uploadTab = st.tabs(["Built-in samples", "RACE dataset", "Upload .txt"])

    with srcTab:
        sampleName = st.selectbox("Choose a built-in passage", list(SAMPLE_ARTICLES.keys()))
        if st.button("Load this sample", use_container_width=True):
            loadSampleArticle(sampleName)
            st.rerun()

    with raceTab:
        ensureRaceChunk()
        raceChunk = st.session_state.raceChunk or {}
        if raceChunk:
            articleName = st.selectbox(
                "Choose a RACE article (lazy-loaded sample of 500)",
                list(raceChunk.keys()),
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load selected", use_container_width=True):
                    loadRaceArticle(articleName)
                    st.rerun()
            with col2:
                if st.button("Load random", use_container_width=True):
                    loadRandomRace()
                    st.rerun()
        else:
            st.info("RACE CSV not found at data/raw/train.csv — add it to enable RACE samples.")
            if st.button("Load random from CSV anyway"):
                loadRandomRace()
                st.rerun()

    with uploadTab:
        uploaded = st.file_uploader("Upload a plain-text passage (.txt)", type=["txt"])
        if uploaded:
            text = uploaded.read().decode("utf-8", errors="ignore").strip()
            if st.button("Use this file", use_container_width=True):
                st.session_state.articleText = text
                st.session_state.lastError   = None
                st.rerun()

    st.divider()

    # ── Text area + submit ────────────────────────────────────────────────
    with st.form("quizForm", clear_on_submit=False):
        articleText = st.text_area(
            "Reading passage",
            value=st.session_state.articleText,
            height=340,
            placeholder="Paste your passage here…",
            key="articleInput",
        )
        c1, c2, c3 = st.columns([4, 1, 1])
        with c1:
            submitted = st.form_submit_button(
                "🚀  Generate Quiz",
                type="primary",
                use_container_width=True,
            )
        with c2:
            cleared = st.form_submit_button("Clear", use_container_width=True)
        with c3:
            toDash = st.form_submit_button("Dashboard", use_container_width=True)

    if cleared:
        st.session_state.articleText = ""
        st.session_state.quizResult  = None
        st.session_state.lastError   = None
        st.rerun()

    if toDash:
        goTo("dashboard")
        st.rerun()

    if submitted:
        with st.spinner("Running Model A + Model B inference…"):
            ok = generateQuiz(articleText)
        if ok:
            st.markdown(
                '<div class="banner ok">✅ Quiz generated! Moving to Quiz View…</div>',
                unsafe_allow_html=True,
            )
            goTo("quiz")
            st.rerun()

    if st.session_state.lastError:
        st.markdown(
            f'<div class="err-box">⚠️ {st.session_state.lastError}</div>',
            unsafe_allow_html=True,
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SCREEN 2 — Quiz View                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
elif screen == "quiz":
    st.markdown('<div class="sec">Screen 02</div>', unsafe_allow_html=True)
    st.markdown("## 🧠 Question & Answer Quiz")

    quiz = st.session_state.quizResult
    if quiz is None:
        st.markdown(
            '<div class="info-box">No quiz loaded yet. Go to Article Input first.</div>',
            unsafe_allow_html=True,
        )
        if st.button("← Go to Article Input", type="primary"):
            goTo("input")
            st.rerun()
        st.stop()

    # Passage expander — always visible after quiz generation
    articleText = st.session_state.articleText
    if articleText:
        with st.expander("📄 Reading Passage", expanded=True):
            st.markdown(
                f'<div style="font-size:0.92rem;line-height:1.7;color:#d1d5db;white-space:pre-wrap;">{articleText}</div>',
                unsafe_allow_html=True,
            )

    # Question card
    st.markdown(
        f"""
        <div class="q-card">
            <div class="label">Generated Question</div>
            <h2>{quiz['question']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Option buttons (radio)
    options     = quiz["options"]
    prediction  = quiz["prediction"]
    answerCheck = st.session_state.answerCheck

    selectedLabel = st.radio(
        "Select your answer:",
        list(options.keys()),
        format_func=lambda k: f"{k})  {options[k]}",
        index=None,
        key="quizRadio",
    )

    if selectedLabel:
        st.session_state.selectedLabel = selectedLabel

    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        if st.button("✅  Check Answer", type="primary", use_container_width=True):
            if st.session_state.selectedLabel is None:
                st.warning("Select one of the four options first.")
            else:
                checkAnswer()
                st.rerun()
    with c2:
        if st.button("💡  Get Hints", use_container_width=True):
            goTo("hints")
            st.rerun()
    with c3:
        if st.button("📊  Dashboard", use_container_width=True):
            goTo("dashboard")
            st.rerun()

    # Result banner
    if answerCheck is not None:
        if answerCheck["isCorrect"]:
            st.markdown(
                '<div class="banner ok">🎉 Correct! Well done.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="banner err">❌ Incorrect. '
                f'The correct answer is <strong>{answerCheck["correctLabel"]}) '
                f'{answerCheck["correctAnswer"]}</strong>.</div>',
                unsafe_allow_html=True,
            )

    # Verifier scores — only shown AFTER the user has checked their answer
    if answerCheck is not None:
        st.divider()
        st.markdown('<div class="sec">Ensemble Verifier Scores (Model A)</div>', unsafe_allow_html=True)

        scoreRows = []
        for label, text in options.items():
            score = round(prediction["scores"][label], 4)
            isCorrect   = label == quiz["correctLabel"]
            isPredicted = label == prediction["label"]
            scoreRows.append({
                "Label": label,
                "Option Text": text,
                "Verifier Score": score,
                "Correct Answer": "✓" if isCorrect else "",
                "Verifier Pick":  "🎯" if isPredicted else "",
            })

        st.dataframe(
            pd.DataFrame(scoreRows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Verifier Score": st.column_config.ProgressColumn(
                    "Verifier Score", min_value=0, max_value=1, format="%.4f"
                )
            },
        )

    if st.button("← New Quiz", use_container_width=True):
        goTo("input")
        st.rerun()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SCREEN 3 — Hint Panel                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
elif screen == "hints":
    st.markdown('<div class="sec">Screen 03</div>', unsafe_allow_html=True)
    st.markdown("## 💡 Hint Panel")

    quiz = st.session_state.quizResult
    if quiz is None:
        st.markdown(
            '<div class="info-box">No quiz loaded. Generate a quiz first.</div>',
            unsafe_allow_html=True,
        )
        if st.button("← Article Input", type="primary"):
            goTo("input")
            st.rerun()
        st.stop()

    st.markdown(
        f"""
        <div class="q-card">
            <div class="label">Question</div>
            <h2>{quiz['question']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hints       = quiz.get("hints", [quiz.get("hint", ""), quiz.get("hint", ""), quiz.get("hint", "")])
    usedHints   = st.session_state.usedHints
    revealAnswer = st.session_state.revealAnswer

    HINT_META = [
        ("Hint 1 — General",      "🟡", "Broad topic context."),
        ("Hint 2 — Intermediate", "🟠", "Narrows to the relevant passage."),
        ("Hint 3 — Specific",     "🔴", "Near-explicit: fill in the blank."),
    ]

    for i, (title, dot, desc) in enumerate(HINT_META):
        hintText = hints[i] if i < len(hints) else "—"
        isUnlocked = usedHints > i

        if isUnlocked:
            st.markdown(
                f"""
                <div class="hint-card">
                    <div class="hl">{dot} {title}</div>
                    <p>{hintText}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="hint-card locked">
                    <div class="hl">{dot} {title}</div>
                    <p><em>🔒 Unlock by revealing the previous hint.</em></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Single progressive button — advances one step at a time
    st.write("")
    if usedHints < 3:
        nextTitle = HINT_META[usedHints][0]
        if st.button(f"Show {nextTitle}", type="primary", use_container_width=True):
            st.session_state.usedHints += 1
            st.rerun()

    st.divider()

    # Reveal Answer — only after all 3 hints used
    if usedHints >= 3:
        if not revealAnswer:
            if st.button("🔓  Reveal Answer", type="primary", use_container_width=True):
                st.session_state.revealAnswer = True
                st.rerun()
        else:
            st.markdown(
                f"""
                <div class="banner ok">
                    Answer: <strong>{quiz['correctLabel']}) {quiz['correctAnswer']}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        remaining = 3 - usedHints
        st.markdown(
            f'<div class="info-box">Use {remaining} more hint(s) to unlock the "Reveal Answer" button.</div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Quiz", use_container_width=True):
            goTo("quiz")
            st.rerun()
    with c2:
        if st.button("📊 Dashboard →", use_container_width=True):
            goTo("dashboard")
            st.rerun()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SCREEN 4 — Developer / Analytics Dashboard                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
elif screen == "dashboard":
    st.markdown('<div class="sec">Screen 04</div>', unsafe_allow_html=True)
    st.markdown("## 📊 Developer / Analytics Dashboard")

    quiz    = st.session_state.quizResult
    history = st.session_state.history

    # ── Live session metrics ───────────────────────────────────────────────
    st.markdown('<div class="sec">Current Inference</div>', unsafe_allow_html=True)
    if quiz:
        latency  = quiz.get("latencySec", quiz.get("uiLatencySec", "—"))
        verMatch = quiz["prediction"]["answer"].lower() == quiz["correctAnswer"].lower()
        m1, m2, m3, m4, m5 = st.columns(5)
        tiles = [
            (m1, "Latency",         f"{latency}s",             "pipeline inference"),
            (m2, "Verifier Pick",   quiz["prediction"]["label"], "ensemble soft-vote"),
            (m3, "Verifier Match",  "✓ Yes" if verMatch else "✗ No", "pick == correct"),
            (m4, "Hints Used",      str(st.session_state.usedHints), "of 3"),
            (m5, "Session Quizzes", str(len(history)),          "this session"),
        ]
        for col, title, value, sub in tiles:
            with col:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="mt">{title}</div>
                        <div class="mv">{value}</div>
                        <div class="ms">{sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        m1, m5 = st.columns([4, 1])
        with m1:
            st.markdown(
                '<div class="info-box">No quiz generated yet — live inference metrics will appear here after your first quiz.</div>',
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="mt">Session Quizzes</div>
                    <div class="mv">{len(history)}</div>
                    <div class="ms">this session</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("← Generate a Quiz", type="primary"):
            goTo("input")
            st.rerun()

    st.divider()

    # ── Model A training metrics ───────────────────────────────────────────
    st.markdown('<div class="sec">Model A — Verifier Ensemble (Training Metrics)</div>', unsafe_allow_html=True)
    ens = MODEL_A_METRICS["ensemble"]

    eCol1, eCol2, eCol3, eCol4, eCol5 = st.columns(5)
    ensembleTiles = [
        (eCol1, "MCQ Accuracy",    f"{ens['mcqAccuracy']:.1%}", f"Random baseline: {ens['randomBaseline']:.0%}"),
        (eCol2, "Binary Accuracy", f"{ens['binaryAccuracy']:.1%}", "natural 25/75 test split"),
        (eCol3, "Binary F1",       f"{ens['binaryF1']:.3f}",    "ensemble LR+SVM+RF"),
        (eCol4, "Cross-Val F1",    f"{ens['crossValF1Mean']:.3f} ± {ens['crossValF1Std']:.3f}", "5-fold, balanced data"),
        (eCol5, "Training Rows",   f"{ens['trainRows']:,}",     f"{ens['articlesUsed']:,} articles"),
    ]
    for col, title, value, sub in ensembleTiles:
        with col:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="mt">{title}</div>
                    <div class="mv">{value}</div>
                    <div class="ms">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Individual classifier table
    st.markdown('<div class="sec">Individual Classifiers</div>', unsafe_allow_html=True)
    classifierRows = []
    labels = {"lr":"Logistic Regression","svm":"Calibrated SVM (isotonic)","nb":"Naive Bayes (excluded from vote)","rf":"Random Forest"}
    inVote = {"lr":True,"svm":True,"nb":False,"rf":True}
    for key, name in labels.items():
        m = MODEL_A_METRICS[key]
        classifierRows.append({
            "Classifier": name,
            "In Ensemble Vote": "✓" if inVote[key] else "—",
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1":        m["f1"],
            "TP": m["tp"], "FP": m["fp"], "TN": m["tn"], "FN": m["fn"],
        })
    st.dataframe(pd.DataFrame(classifierRows), use_container_width=True, hide_index=True)

    # Confusion matrix for ensemble
    st.markdown('<div class="sec">Ensemble Confusion Matrix</div>', unsafe_allow_html=True)
    e = MODEL_A_METRICS["ensemble"]
    cmLeft, cmRight = st.columns([1, 2])
    with cmLeft:
        st.markdown(
            f"""
            <div class="cm-grid">
                <div class="cm-cell cm-tp"><div class="cv">{e['tp']:,}</div><div class="cl">TP</div></div>
                <div class="cm-cell cm-fp"><div class="cv">{e['fp']:,}</div><div class="cl">FP</div></div>
                <div class="cm-cell cm-fn"><div class="cv">{e['fn']:,}</div><div class="cl">FN</div></div>
                <div class="cm-cell cm-tn"><div class="cv">{e['tn']:,}</div><div class="cl">TN</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cmRight:
        st.caption(
            "Binary classification on natural 25%/75% test split (52,720 rows · 13,180 questions). "
            "MCQ accuracy measures whether the highest-scored option matches the correct answer. "
            f"Ensemble (LR+SVM+RF) MCQ accuracy: **{ens['mcqAccuracy']:.1%}** vs random baseline **{ens['randomBaseline']:.0%}**."
        )

    st.divider()

    # ── Model B metrics ────────────────────────────────────────────────────
    st.markdown('<div class="sec">Model B — Distractor Ranker (Training Metrics)</div>', unsafe_allow_html=True)
    bCols = st.columns(4)
    mbTiles = [
        ("Accuracy",  f"{MODEL_B_METRICS['rf']['accuracy']:.1%}", "RF ranker"),
        ("Precision", f"{MODEL_B_METRICS['rf']['precision']:.1%}", "RF ranker"),
        ("Recall",    f"{MODEL_B_METRICS['rf']['recall']:.1%}",    "RF ranker"),
        ("F1",        f"{MODEL_B_METRICS['rf']['f1']:.2f}",        "RF ranker"),
    ]
    for col, (t, v, s) in zip(bCols, mbTiles):
        with col:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="mt">{t}</div>
                    <div class="mv">{v}</div>
                    <div class="ms">{s}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── T5 Neural Baseline Comparison ─────────────────────────────────────
    st.markdown('<div class="sec">Question Generation — Template vs T5 Neural Baseline</div>', unsafe_allow_html=True)

    if quiz:
        colLeft, colRight = st.columns(2)
        with colLeft:
            st.markdown(
                f"""
                <div class="q-card">
                    <div class="label">🎯 Ranker QG (RF-ranked · Traditional — Production)</div>
                    <h2>{quiz['question']}</h2>
                    <p style="color:#6b7280;font-size:0.8rem;margin-top:8px;">
                        RF ranker · ROC-AUC 0.972 · top-1 match 66.9% · {quiz.get('latencySec','—')}s
                    </p>
                </div>
                """, unsafe_allow_html=True,
            )
        with colRight:
            st.session_state.setdefault("t5Result", None)
            t5Result = st.session_state.t5Result
            if t5Result is None:
                st.markdown(
                    '<div class="q-card"><div class="label">🤖 T5-small (Neural Baseline)</div>'
                    '<p style="color:#6b7280;">Click below to generate.</p></div>',
                    unsafe_allow_html=True,
                )
                if st.button("⚡ Generate T5 Question", type="primary", use_container_width=True):
                    with st.spinner("Loading T5 and generating… (~10-30s first run)"):
                        genFn = _getT5Generator()
                        st.session_state.t5Result = genFn(st.session_state.articleText, quiz["correctAnswer"])
                    st.rerun()
            elif t5Result.get("error"):
                st.markdown(f'<div class="err-box">T5 unavailable: {t5Result["error"]}</div>', unsafe_allow_html=True)
                if st.button("Retry T5", use_container_width=True):
                    st.session_state.t5Result = None
                    st.rerun()
            else:
                st.markdown(
                    f"""
                    <div class="q-card">
                        <div class="label">🤖 T5-small (Neural Baseline · Fine-tuned on RACE)</div>
                        <h2>{t5Result['question']}</h2>
                        <p style="color:#6b7280;font-size:0.8rem;margin-top:8px;">
                            20k RACE Wh-questions · 5 epochs · {t5Result.get('latencySec','—')}s
                        </p>
                    </div>
                    """, unsafe_allow_html=True,
                )
                if st.button("🔄 Regenerate T5", use_container_width=True):
                    st.session_state.t5Result = None
                    st.rerun()
        st.caption("Compare fluency and specificity: Ranker QG (RF-ranked rule-based) vs T5 (learned from data).")
    else:
        colLeft, colRight = st.columns(2)
        with colLeft:
            st.markdown(
                """
                <div class="q-card">
                    <div class="label">🎯 Ranker QG (RF-ranked · Traditional)</div>
                    <p style="color:#6b7280;margin-top:8px;">Generate a quiz on the Article Input screen to see a live question here.</p>
                </div>
                """, unsafe_allow_html=True,
            )
        with colRight:
            st.markdown(
                """
                <div class="q-card">
                    <div class="label">🤖 T5-small (Neural Baseline)</div>
                    <p style="color:#6b7280;margin-top:8px;">Available after generating a quiz.</p>
                </div>
                """, unsafe_allow_html=True,
            )
        if st.button("← Go to Article Input", type="primary"):
            goTo("input")
            st.rerun()

    st.divider()

    # ── Session log ────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Session Log</div>', unsafe_allow_html=True)
    if history:
        logDf = pd.DataFrame(history)
        st.dataframe(logDf, use_container_width=True, hide_index=True)

        csvData = logDf.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Export session log (CSV)",
            data=csvData,
            file_name="quiz_session_log.csv",
            mime="text/csv",
        )
    else:
        st.markdown(
            '<div class="info-box">No quiz sessions yet — generate a quiz to start logging.</div>',
            unsafe_allow_html=True,
        )

    if st.button("← Back to Quiz", use_container_width=True):
        goTo("quiz")
        st.rerun()
