"""
app.py
Streamlit Web Application: Resume-to-Role Predictor
Features:
- Dark cinematic UI with glassmorphism cards and neon accents
- Bigger, futuristic typography (Space Grotesk, Plus Jakarta Sans, JetBrains Mono)
- Glowing circular fit score gauge with SVG neon filters and ambient pulsing aura
- Staggered animations across cards, probability bars, and keyword chips
- Resume input area with instant sample-resume loader buttons
- Dual fit score gauge (Ridge Regression vs Logistic Regression Probability)
- Probability distribution bar chart
- Local keyword explainability (positive driving words chips & text highlighting)
- Model Insights tab with Confusion Matrix and NB vs LR benchmarks
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Resume-to-Role Predictor | AI Talent Match",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Cinematic Dark Theme with Bigger Typography & Staggered Animations
CINEMATIC_CSS = """
<style>
/* Import High-Impact Cinematic Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bg-color: #07090e;
    --card-bg: rgba(14, 20, 33, 0.72);
    --border-color: rgba(0, 242, 254, 0.2);
    --border-hover: rgba(0, 242, 254, 0.5);
    --neon-cyan: #00f2fe;
    --neon-blue: #4facfe;
    --neon-emerald: #00ff87;
    --neon-purple: #a855f7;
    --neon-coral: #ff5252;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

/* Background canvas with cinematic radial ambient illumination */
.stApp {
    background-color: var(--bg-color);
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 85% 20%, rgba(0, 255, 135, 0.06) 0%, transparent 40%),
        radial-gradient(circle at 50% 85%, rgba(168, 85, 247, 0.05) 0%, transparent 50%);
    background-attachment: fixed;
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
}

/* Base keyframe animations */
@keyframes cinematicFadeUp {
    0% {
        opacity: 0;
        transform: translateY(28px) scale(0.98);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes glowAuraPulse {
    0%, 100% {
        opacity: 0.3;
        transform: scale(0.96);
    }
    50% {
        opacity: 0.65;
        transform: scale(1.06);
    }
}

@keyframes chipPopIn {
    0% {
        opacity: 0;
        transform: scale(0.7) translateY(12px);
    }
    100% {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

@keyframes barGrow {
    0% {
        width: 0% !important;
    }
}

@keyframes dialFloat {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-5px);
    }
}

/* Staggered container wrapper classes */
.stagger-header {
    animation: cinematicFadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.stagger-card-1 {
    animation: cinematicFadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.15s both;
}

.stagger-card-2 {
    animation: cinematicFadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.3s both;
}

/* Glassmorphism Cards */
.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 28px;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.55), inset 0 1px 0 0 rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    margin-bottom: 24px;
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 16px 48px 0 rgba(0, 0, 0, 0.65), 0 0 25px rgba(0, 242, 254, 0.12);
}

/* BIGGER, BOLDER CINEMATIC TYPOGRAPHY */
.badge-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(90deg, rgba(0, 242, 254, 0.15), rgba(0, 255, 135, 0.15));
    border: 1px solid rgba(0, 242, 254, 0.45);
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.25);
    color: #00f2fe;
    padding: 6px 18px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 14px;
}

.glowing-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ffffff 0%, #00f2fe 50%, #00ff87 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -2px;
    line-height: 1.08;
    margin-bottom: 10px;
    filter: drop-shadow(0 0 35px rgba(0, 242, 254, 0.4));
}

.sub-headline {
    color: var(--text-secondary);
    font-size: 1.22rem;
    font-weight: 400;
    line-height: 1.6;
    letter-spacing: -0.2px;
    margin-bottom: 28px;
    max-width: 950px;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.6px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* HERO PREDICTED ROLE DISPLAY */
.role-hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.75rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00f2fe 0%, #00ff87 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1.2px;
    line-height: 1.1;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 25px rgba(0, 242, 254, 0.45));
}

.role-sub-badge {
    display: inline-block;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin-bottom: 6px;
}

/* SCORE METER WITH SUBTLE GLOW */
.gauge-wrapper {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 12px;
    animation: dialFloat 4s ease-in-out infinite;
}

.gauge-glow-aura {
    position: absolute;
    width: 170px;
    height: 170px;
    border-radius: 50%;
    filter: blur(28px);
    z-index: 0;
    animation: glowAuraPulse 3.5s ease-in-out infinite;
    pointer-events: none;
}

.gauge-svg {
    position: relative;
    z-index: 1;
}

.gauge-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-top: 10px;
    z-index: 1;
    text-align: center;
}

/* WORD CHIPS WITH STAGGERED REVEAL */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
}

.word-chip {
    display: inline-flex;
    align-items: center;
    background: rgba(0, 242, 254, 0.08);
    border: 1px solid rgba(0, 242, 254, 0.35);
    box-shadow: 0 4px 16px rgba(0, 242, 254, 0.08);
    color: #f1f5f9;
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: -0.1px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    animation: chipPopIn 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.word-chip:hover {
    background: rgba(0, 242, 254, 0.22);
    border-color: #00f2fe;
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 8px 24px rgba(0, 242, 254, 0.3);
    color: #ffffff;
}

.chip-weight {
    margin-left: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #00f2fe;
    font-weight: 700;
    background: rgba(0, 242, 254, 0.15);
    padding: 2px 7px;
    border-radius: 6px;
}

/* Staggered CSS Delays for Word Chips */
.word-chip:nth-child(1)  { animation-delay: 0.40s; }
.word-chip:nth-child(2)  { animation-delay: 0.45s; }
.word-chip:nth-child(3)  { animation-delay: 0.50s; }
.word-chip:nth-child(4)  { animation-delay: 0.55s; }
.word-chip:nth-child(5)  { animation-delay: 0.60s; }
.word-chip:nth-child(6)  { animation-delay: 0.65s; }
.word-chip:nth-child(7)  { animation-delay: 0.70s; }
.word-chip:nth-child(8)  { animation-delay: 0.75s; }
.word-chip:nth-child(9)  { animation-delay: 0.80s; }
.word-chip:nth-child(10) { animation-delay: 0.85s; }
.word-chip:nth-child(11) { animation-delay: 0.90s; }
.word-chip:nth-child(12) { animation-delay: 0.95s; }

/* PROBABILITY DISTRIBUTION BARS WITH STAGGERED FILL */
.prob-card {
    margin-bottom: 14px;
}

.prob-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 1.02rem;
    font-weight: 600;
    margin-bottom: 6px;
    letter-spacing: -0.2px;
}

.prob-bar-bg {
    width: 100%;
    height: 12px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 6px;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
}

.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.4);
    animation: barGrow 0.8s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.prob-bar-fill-top {
    background: linear-gradient(90deg, #00f2fe 0%, #00ff87 100%);
    box-shadow: 0 0 16px rgba(0, 255, 135, 0.6);
}

/* Staggered progress bar fills */
.prob-card:nth-child(1) .prob-bar-fill { animation-delay: 0.35s; }
.prob-card:nth-child(2) .prob-bar-fill { animation-delay: 0.45s; }
.prob-card:nth-child(3) .prob-bar-fill { animation-delay: 0.55s; }
.prob-card:nth-child(4) .prob-bar-fill { animation-delay: 0.65s; }
.prob-card:nth-child(5) .prob-bar-fill { animation-delay: 0.75s; }

/* HIGHLIGHTED RESUME TEXT */
.highlight-token {
    background: rgba(0, 242, 254, 0.28);
    border-bottom: 2px solid #00f2fe;
    color: #ffffff;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    box-shadow: 0 0 12px rgba(0, 242, 254, 0.3);
}

/* STREAMLIT CONTROLS ENHANCEMENT */
.stTextArea textarea {
    background-color: rgba(11, 17, 30, 0.85) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(0, 242, 254, 0.25) !important;
    border-radius: 14px !important;
    font-size: 1.02rem !important;
    line-height: 1.6 !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: #00f2fe !important;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.35) !important;
}

.stButton button {
    background: linear-gradient(135deg, #00f2fe 0%, #0072ff 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: -0.2px !important;
    padding: 0.7rem 2.0rem !important;
    box-shadow: 0 6px 24px rgba(0, 198, 255, 0.35) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 10px 32px rgba(0, 198, 255, 0.6) !important;
}

/* Sample buttons styling */
.sample-btn-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 8px;
}
</style>
"""

st.markdown(CINEMATIC_CSS, unsafe_allow_html=True)

# Sample resumes for quick testing
SAMPLE_RESUMES = {
    "AI / ML Engineer": (
        "Senior Data Scientist and Machine Learning Engineer with 5+ years of experience building and deploying "
        "NLP and deep learning systems. Proficient in Python, pandas, numpy, scikit-learn, PyTorch, and TensorFlow. "
        "Developed transformer-based text classification pipelines, clustering algorithms, and recommendation engines. "
        "Expertise in exploratory data analysis, feature engineering, regression modeling, hyperparameter tuning, "
        "model explainability with SHAP, and SQL for querying relational databases. Designed automated ETL pipelines "
        "and monitored predictive ML models in production environments."
    ),
    "Backend Developer": (
        "Senior Backend Developer with 6 years of experience architecting high-performance enterprise applications "
        "and distributed microservices. Core expertise in Java, Spring Boot, Hibernate, Python, and SQL Server. "
        "Designed and implemented RESTful APIs, optimized complex database queries in PostgreSQL and Oracle, "
        "and integrated Redis caching layers. Extensive experience in Docker, Kubernetes, CI/CD pipelines with Jenkins "
        "and GitLab, and messaging queues using Apache Kafka and RabbitMQ. Strong background in database design and OOP."
    ),
    "Web Developer / Designer": (
        "Creative Front-End Web Designer and Developer with 4 years of experience crafting modern, responsive web user interfaces. "
        "Proficient in HTML5, CSS3, JavaScript, jQuery, Bootstrap, React.js, and WordPress theme customization. "
        "Skilled in Adobe Photoshop, Figma, graphic design, responsive grid layouts, and cross-browser compatibility. "
        "Developed client websites, optimized UI/UX design workflows, and integrated REST APIs. Strong knowledge of web "
        "accessibility standards, animation effects, and mobile-first responsive design."
    ),
    "Business Analyst": (
        "Business Analyst with 5 years experience driving business intelligence, requirement gathering, and process re-engineering. "
        "Expert in translating complex business stakeholder requirements into functional specifications, user stories, and wireframes. "
        "Proficient in SQL, Excel advanced financial modeling, Tableau, Power BI, and ERP systems. "
        "Conducted GAP analysis, cost-benefit analysis, executive reporting, and KPI dashboard development. "
        "Facilitated sprint planning, client requirement workshops, and user acceptance testing."
    ),
    "QA / Other Tech": (
        "QA Automation Engineer and Software Tester with 5 years experience in manual and automated testing of web applications. "
        "Proficient in Selenium WebDriver, TestNG, Java, Python, and Postman API testing. Built robust test automation frameworks, "
        "executed regression testing suites, performance testing, and bug tracking using JIRA. Experience in network security protocols, "
        "vulnerability scanning, and blockchain smart contract testing. Certified ISTQB software quality engineer."
    )
}


@st.cache_resource
def load_predictor():
    """Initializes and caches predictor model artifacts."""
    from src.predict import get_predictor
    return get_predictor()


def render_circular_gauge(score: float, label: str = "Role Fit Score"):
    """
    Renders an elevated cinematic SVG circular gauge with:
    - Subtle neon glow filters (<feGaussianBlur>)
    - Soft ambient radial aura
    - Fluid stroke-dashoffset transition
    - Large Space Grotesk percentage numeral
    """
    # Radius = 52 -> Circumference = 2 * pi * 52 ≈ 326.72
    circumference = 326.72
    clamped_score = max(0.0, min(100.0, score))
    offset = circumference - (clamped_score / 100.0) * circumference

    if clamped_score >= 70:
        main_color = "#00ff87"
        grad_stop_1 = "#00f2fe"
        grad_stop_2 = "#00ff87"
        aura_rgba = "rgba(0, 255, 135, 0.28)"
    elif clamped_score >= 40:
        main_color = "#00f2fe"
        grad_stop_1 = "#4facfe"
        grad_stop_2 = "#00f2fe"
        aura_rgba = "rgba(0, 242, 254, 0.26)"
    else:
        main_color = "#ff5252"
        grad_stop_1 = "#ff7b7b"
        grad_stop_2 = "#ff5252"
        aura_rgba = "rgba(255, 82, 82, 0.25)"

    svg_html = f"""
    <div class="gauge-wrapper">
        <div class="gauge-glow-aura" style="background: radial-gradient(circle, {aura_rgba} 0%, transparent 70%);"></div>
        <svg class="gauge-svg" width="200" height="200" viewBox="0 0 140 140">
            <defs>
                <!-- Cinematic Neon Multi-Stage Glow Filter -->
                <filter id="neon-meter-glow" x="-30%" y="-30%" width="160%" height="160%">
                    <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur1" />
                    <feGaussianBlur in="SourceGraphic" stdDeviation="7" result="blur2" />
                    <feMerge>
                        <feMergeNode in="blur2" />
                        <feMergeNode in="blur1" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
                <linearGradient id="meter-gradient" x1="0%" y1="100%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="{grad_stop_1}" />
                    <stop offset="100%" stop-color="{grad_stop_2}" />
                </linearGradient>
            </defs>

            <!-- Background Track -->
            <circle cx="70" cy="70" r="52" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="10" />

            <!-- Ambient Glow Stroke -->
            <circle cx="70" cy="70" r="52" fill="none" stroke="{main_color}" stroke-width="12"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                    stroke-linecap="round" opacity="0.45" filter="url(#neon-meter-glow)"
                    transform="rotate(-90 70 70)"
                    style="transition: stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1);" />

            <!-- Sharp Foreground Neon Stroke -->
            <circle cx="70" cy="70" r="52" fill="none" stroke="url(#meter-gradient)" stroke-width="9"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                    stroke-linecap="round"
                    transform="rotate(-90 70 70)"
                    style="transition: stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1);" />

            <!-- Core Numerals -->
            <text x="70" y="68" text-anchor="middle" font-size="28" font-weight="900"
                  fill="#ffffff" font-family="'Space Grotesk', sans-serif"
                  style="filter: drop-shadow(0 0 10px {main_color});">{int(round(clamped_score))}%</text>

            <text x="70" y="87" text-anchor="middle" font-size="9" font-weight="700"
                  fill="#94a3b8" font-family="'Space Grotesk', sans-serif" letter-spacing="1.8">FIT INDEX</text>
        </svg>
        <div class="gauge-label">{label}</div>
    </div>
    """
    st.markdown(svg_html, unsafe_allow_html=True)


def highlight_resume_tokens(raw_text: str, driving_tokens: list[str]) -> str:
    """Highlights influential tokens inside resume text with neon cyan accents."""
    import re
    tokens_to_match = set([t['token'].lower() for t in driving_tokens] + [t['display'].lower() for t in driving_tokens])

    # Escape tokens for regex
    escaped = [re.escape(t) for t in tokens_to_match if len(t) > 2]
    if not escaped:
        return raw_text

    pattern = re.compile(r'(?i)\b(' + '|'.join(escaped) + r')\b')
    highlighted = pattern.sub(r'<span class="highlight-token">\1</span>', raw_text)
    return highlighted


def main():
    # Load predictor model
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}. Please run `python -m src.train` first.")
        st.stop()

    roles = predictor.roles

    # TOP HEADER with Staggered Fade Up
    st.markdown(
        """
        <div class="stagger-header">
            <div class="badge-tag">⚡ University NLP Mini-Project • Advanced AI Screening</div>
            <div class="glowing-title">Resume-to-Role Predictor</div>
            <div class="sub-headline">
                Cinematic NLP talent intelligence: real-time tech category classification with <b>TF-IDF</b>, 
                dual fit scoring via <b>Ridge Regression</b> & <b>Logistic Regression</b>, 
                and token-level coefficient explainability.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Main Tabs: Predictor vs Model Insights
    tab_predict, tab_insights = st.tabs(["🔮 Resume Predictor", "📊 Model Insights & Benchmarks"])

    with tab_predict:
        col_input, col_results = st.columns([1.08, 1.22], gap="large")

        with col_input:
            st.markdown('<div class="glass-card stagger-card-1">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📄 Input Candidate Resume</div>', unsafe_allow_html=True)

            # Sample buttons
            st.markdown('<div class="sample-btn-label">Quick Load Pre-built Profile:</div>', unsafe_allow_html=True)
            sample_cols = st.columns(3)

            if "resume_text" not in st.session_state:
                st.session_state.resume_text = SAMPLE_RESUMES["AI / ML Engineer"]

            if sample_cols[0].button("🤖 AI / ML", use_container_width=True):
                st.session_state.resume_text = SAMPLE_RESUMES["AI / ML Engineer"]
            if sample_cols[1].button("💼 Backend", use_container_width=True):
                st.session_state.resume_text = SAMPLE_RESUMES["Backend Developer"]
            if sample_cols[2].button("🎨 Web Dev", use_container_width=True):
                st.session_state.resume_text = SAMPLE_RESUMES["Web Developer / Designer"]

            sample_cols2 = st.columns(2)
            if sample_cols2[0].button("📊 Business Analyst", use_container_width=True):
                st.session_state.resume_text = SAMPLE_RESUMES["Business Analyst"]
            if sample_cols2[1].button("🛡️ QA / Testing", use_container_width=True):
                st.session_state.resume_text = SAMPLE_RESUMES["QA / Other Tech"]

            # Text Area
            input_text = st.text_area(
                "Resume Text Content",
                value=st.session_state.resume_text,
                height=270,
                placeholder="Paste complete resume or CV text here...",
                help="Requires at least 20 words for meaningful TF-IDF feature extraction."
            )

            # Target Role Selection & Action Row
            col_target, col_action = st.columns([1.2, 1.0])
            with col_target:
                target_role_choice = st.selectbox(
                    "Target Role to Evaluate:",
                    options=["Auto-Detect Best Match"] + roles,
                    index=0,
                    help="Target role against which the 0-100 fit score will be computed."
                )

            with col_action:
                st.write("")
                st.write("")
                analyze_btn = st.button("🚀 Analyze Resume", use_container_width=True)

            # Word counter
            word_count = len(input_text.strip().split()) if input_text.strip() else 0
            char_count = len(input_text)
            st.markdown(
                f"<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.82rem; color: #64748b; text-align: right; margin-top: 6px;'>"
                f"{word_count} words | {char_count} characters</div>",
                unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

        with col_results:
            st.markdown('<div class="glass-card stagger-card-2">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🎯 Prediction & Fit Assessment</div>', unsafe_allow_html=True)

            # Validate input
            is_valid, validation_msg = predictor.validate_input(input_text)

            if not is_valid:
                st.warning(f"⚠️ {validation_msg}")
                st.info("💡 Click one of the instant sample buttons on the left to immediately test a pre-loaded resume.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                target_role = None if target_role_choice == "Auto-Detect Best Match" else target_role_choice

                # Run inference
                with st.spinner("Processing NLP feature extraction & inference..."):
                    results = predictor.predict(input_text, target_role=target_role)

                pred_role = results['predicted_role']
                nb_role = results['nb_predicted_role']
                actual_target = results['target_role']
                ridge_score = results['ridge_fit_score']
                lr_prob_score = results['lr_prob_fit_score']
                probabilities = results['probabilities']
                driving_words = results['top_driving_words']

                # Hero Metrics row with Glowing Circular Gauge
                g_col1, g_col2 = st.columns([1, 1.25])

                with g_col1:
                    render_circular_gauge(ridge_score, f"Fit for {actual_target}")

                with g_col2:
                    st.markdown(
                        f"""
                        <div style="padding-top: 6px;">
                            <div class="role-sub-badge">Predicted Best-Fit Role</div>
                            <div class="role-hero-title">{pred_role}</div>
                            <div style="font-size: 0.95rem; color: #cbd5e1; margin-bottom: 14px;">
                                Model Certainty: <b style="color: #00f2fe;">{probabilities.get(pred_role, 0)}%</b>
                                &nbsp;•&nbsp; Naive Bayes: <span style="color: {'#00ff87' if pred_role == nb_role else '#ffb703'};"><b>{nb_role}</b></span>
                            </div>
                            <div style="font-size: 0.88rem; color: #cbd5e1; background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.18); padding: 10px 14px; border-radius: 12px; line-height: 1.5;">
                                <b style="color: #f8fafc; font-family: 'Space Grotesk', sans-serif;">Dual Fit Score Evaluation:</b><br>
                                • <span style="color: #94a3b8;">Ridge Regression:</span> <span style="color: #00ff87; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{ridge_score}/100</span><br>
                                • <span style="color: #94a3b8;">Logistic Reg Probability:</span> <span style="color: #00f2fe; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{lr_prob_score}/100</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 22px 0 16px 0;'>", unsafe_allow_html=True)

                # Probability Distribution Bar Chart with Staggered Animation
                st.markdown('<div style="font-family: \'Space Grotesk\', sans-serif; font-size: 1.25rem; font-weight: 800; margin-bottom: 12px; color: #f8fafc;">📊 Role Probability Spectrum</div>', unsafe_allow_html=True)
                for r in roles:
                    prob = probabilities[r]
                    is_top = (r == pred_role)
                    bar_class = "prob-bar-fill prob-bar-fill-top" if is_top else "prob-bar-fill"
                    color_txt = "#00f2fe" if is_top else "#e2e8f0"
                    st.markdown(
                        f"""
                        <div class="prob-card">
                            <div class="prob-label-row">
                                <span style="color: {color_txt}; font-weight: {'800' if is_top else '500'};">{r}</span>
                                <span style="color: {color_txt}; font-weight: {'800' if is_top else '500'}; font-family: 'JetBrains Mono', monospace;">{prob}%</span>
                            </div>
                            <div class="prob-bar-bg">
                                <div class="{bar_class}" style="width: {prob}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0 16px 0;'>", unsafe_allow_html=True)

                # Influential Word Chips with Staggered CSS Cascade
                st.markdown('<div style="font-family: \'Space Grotesk\', sans-serif; font-size: 1.25rem; font-weight: 800; color: #f8fafc; margin-bottom: 4px;">🔍 Influential Keywords Driving Decision</div>', unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 10px;'>Tokens with highest positive feature contribution (TF-IDF × Coefficient):</p>", unsafe_allow_html=True)

                if driving_words:
                    chip_html = '<div class="chip-container">'
                    for idx, w in enumerate(driving_words):
                        chip_html += f'<div class="word-chip">{w["display"]}<span class="chip-weight">+{w["contribution"]:.3f}</span></div>'
                    chip_html += '</div>'
                    st.markdown(chip_html, unsafe_allow_html=True)
                else:
                    st.info("No single dominant keyword detected; classification is informed by broad linguistic context.")

                # Resume Highlighting Expander
                with st.expander("👀 View Highlighted Resume Text", expanded=False):
                    highlighted_html = highlight_resume_tokens(input_text, driving_words)
                    st.markdown(
                        f"""
                        <div style="background: rgba(8, 12, 22, 0.95); border: 1px solid rgba(0, 242, 254, 0.15); padding: 18px; border-radius: 12px; font-size: 0.95rem; line-height: 1.7; color: #cbd5e1; max-height: 280px; overflow-y: auto;">
                            {highlighted_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown('</div>', unsafe_allow_html=True)

    with tab_insights:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔬 Model Performance & NLP Pipeline Insights</div>', unsafe_allow_html=True)
        meta = predictor.metadata

        # Metrics Overview Row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Training Samples", meta.get('train_size', 84))
        m_col2.metric("Testing Samples", meta.get('test_size', 22))
        m_col3.metric("LR Accuracy", f"{meta['lr_metrics']['Accuracy'] * 100:.1f}%")
        m_col4.metric("LR Macro-F1", f"{meta['lr_metrics']['Macro F1']:.4f}")

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)

        # Naive Bayes vs Logistic Regression Comparison
        col_table, col_reg = st.columns([1, 1], gap="medium")

        with col_table:
            st.markdown("#### ⚔️ Classification: Multinomial NB vs Logistic Regression")
            st.dataframe(
                meta['comparison_df'],
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "Evaluated using stratified 80/20 train/test split. "
                "Logistic Regression with balanced class weights significantly outperforms Naive Bayes on Macro-F1."
            )

        with col_reg:
            st.markdown("#### 📈 Fit Score Regression (0-100 scale on Test Set)")
            st.dataframe(
                meta['regression_comparison'],
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "Comparing Ridge Regression on TF-IDF against Logistic Regression class probabilities. "
                "Ridge Regression achieves lower MAE across target roles."
            )

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)

        # Confusion Matrices
        st.markdown("#### 🧩 Confusion Matrices")
        cm_col1, cm_col2 = st.columns(2)

        lr_cm_path = os.path.join(predictor.models_dir, "lr_confusion_matrix.png")
        nb_cm_path = os.path.join(predictor.models_dir, "nb_confusion_matrix.png")

        with cm_col1:
            if os.path.exists(lr_cm_path):
                st.image(lr_cm_path, caption="Logistic Regression Confusion Matrix", use_container_width=True)

        with cm_col2:
            if os.path.exists(nb_cm_path):
                st.image(nb_cm_path, caption="Multinomial Naive Bayes Confusion Matrix", use_container_width=True)

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)

        # Global Top 10 Coefficients per Role
        st.markdown("#### 🌐 Global Top 10 Positive Coefficients per Role (Logistic Regression)")
        st.caption("These features have the strongest positive log-odds weight for predicting each role.")

        coef_cols = st.columns(len(roles))
        for idx, role in enumerate(roles):
            with coef_cols[idx]:
                st.markdown(f"**{role}**")
                role_top_words = meta['top_coefficients'].get(role, [])
                for item in role_top_words[:8]:
                    st.markdown(
                        f"<div style='font-size: 0.84rem; background: rgba(255,255,255,0.03); padding: 4px 8px; border-radius: 6px; margin-bottom: 5px; border-left: 2px solid #00f2fe;'>"
                        f"<b>{item['display']}</b> <span style='color: #00f2fe; float: right; font-family: \"JetBrains Mono\", monospace;'>{item['weight']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
