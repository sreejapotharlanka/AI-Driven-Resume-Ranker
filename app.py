"""
Streamlit demo UI for the AI Resume Ranking System.
Run with: streamlit run app.py
"""

import os
import sys
import tempfile
import base64
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import jd_parser, data_loader, feature_extractor, embedder, scorer
from src import config

# Streamlit Page Config (Must be first)
st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="🎯",
    layout="wide"
)

# Helper function to convert local image to Base64 (Robust & offline-compatible)
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return ""

# Helper to generate the magnifying glass CSS-animated scanning layout
def show_scanning_animation(status_text):
    # Check if a GIF exists, otherwise fall back to the PNG image
    img_path = "searching_resume.gif" if os.path.exists("searching_resume.gif") else "searching_resume.png"
    mime = "image/gif" if img_path.endswith(".gif") else "image/png"
    img_b64 = get_image_base64(img_path)
    return f"""
        <div class="scanner-wrapper">
            <div class="scanner-container">
                <img src="data:{mime};base64,{img_b64}" class="scanner-image">
            </div>
            <div class="scanner-text">🔍 AI is scanning all resumes...</div>
            <div class="status-subtext">{status_text}</div>
        </div>
        <style>
            .scanner-wrapper {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 40px auto;
                padding: 20px;
                background: rgba(0, 0, 0, 0.01);
                border-radius: 20px;
                border: 1px solid rgba(0, 0, 0, 0.05);
                max-width: 450px;
            }}
            .scanner-container {{
                position: relative;
                width: 220px;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
                border: 1px solid rgba(0, 0, 0, 0.06);
            }}
            .scanner-image {{
                width: 100%;
                display: block;
            }}
            .scanner-text {{
                margin-top: 20px;
                font-family: 'Outfit', sans-serif;
                font-size: 18px;
                font-weight: 600;
                color: #059669;
                text-shadow: 0 0 10px rgba(5, 150, 105, 0.1);
            }}
            .status-subtext {{
                margin-top: 8px;
                font-family: 'Outfit', sans-serif;
                font-size: 14px;
                color: #64748b;
            }}
        </style>
    """

# Custom Premium Styling injection
def inject_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
            
            /* Custom typography and main bg */
            html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                font-family: 'Outfit', sans-serif !important;
                background-color: #f8fafc !important;
                color: #0f172a !important;
            }
            
            /* Sidebar background overrides */
            [data-testid="stSidebar"] {
                background-color: #f1f5f9 !important;
                border-right: 1px solid rgba(0, 0, 0, 0.06);
            }
            
            /* Sidebar text and label overrides */
            [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
                color: #334155 !important;
            }
            
            /* Glassmorphism card layouts */
            div.element-container:has(div.glass-card) {
                background: transparent;
            }
            .glass-card {
                background: #ffffff !important;
                border: 1px solid rgba(0, 0, 0, 0.06) !important;
                border-radius: 16px !important;
                padding: 22px !important;
                margin-bottom: 20px !important;
                box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.05);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .glass-card:hover {
                border-color: rgba(5, 150, 105, 0.3) !important;
                box-shadow: 0 10px 25px 0 rgba(5, 150, 105, 0.06);
                transform: translateY(-2px);
            }
            
            /* Custom Metric Card */
            .metric-card {
                background: rgba(0, 0, 0, 0.01) !important;
                border: 1px solid rgba(0, 0, 0, 0.03) !important;
                border-radius: 12px !important;
                padding: 15px !important;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            }
            .metric-card-val {
                font-size: 24px;
                font-weight: 700;
                color: #059669;
            }
            .metric-card-lbl {
                font-size: 11px;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-top: 4px;
            }
            
            /* Main Header styling */
            .main-title {
                background: linear-gradient(135deg, #059669 0%, #2563eb 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 42px;
                font-weight: 800;
                text-align: center;
                margin-bottom: 5px;
            }
            .main-subtitle {
                text-align: center;
                color: #64748b;
                font-size: 15px;
                font-weight: 400;
                margin-bottom: 30px;
            }
            
            /* Styled Buttons */
            div[data-testid="stFormSubmitButton"] button, button[kind="primary"] {
                background: linear-gradient(135deg, #059669 0%, #2563eb 100%) !important;
                color: #ffffff !important;
                font-weight: 700 !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 12px 24px !important;
                box-shadow: 0 4px 15px rgba(37, 99, 235, 0.15) !important;
                transition: all 0.2s ease !important;
            }
            div[data-testid="stFormSubmitButton"] button:hover, button[kind="primary"]:hover {
                transform: translateY(-1px) !important;
                box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3) !important;
            }
            
            /* Badges styling */
            .pill {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                margin-right: 5px;
                margin-bottom: 5px;
            }
            .pill-skill {
                background: rgba(37, 99, 235, 0.05);
                color: #2563eb;
                border: 1px solid rgba(37, 99, 235, 0.12);
            }
            .pill-match {
                background: rgba(5, 150, 105, 0.05);
                color: #059669;
                border: 1px solid rgba(5, 150, 105, 0.12);
            }
            .pill-alert {
                background: rgba(220, 38, 38, 0.05);
                color: #dc2626;
                border: 1px solid rgba(220, 38, 38, 0.12);
            }
            
            /* Clean input box overrides */
            div[data-baseweb="input"], div[data-baseweb="textarea"] {
                border-radius: 8px !important;
                border-color: rgba(0, 0, 0, 0.08) !important;
                background-color: #ffffff !important;
                color: #0f172a !important;
            }
            div[data-baseweb="input"]:focus-within, div[data-baseweb="textarea"]:focus-within {
                border-color: #2563eb !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Build a plain-English explanation of why this candidate ranked where they did
def _build_reasoning(row) -> str:
    title = row.get("current_title", "Candidate")
    yrs = row.get("years_of_experience", 0)
    n_skills = int(row.get("matched_skill_count", 0))
    response_rate = row.get("recruiter_response_rate", 0)
    
    reason = (
        f"{title} with {yrs:.1f} yrs; "
        f"{n_skills} matched skills; "
        f"response rate {response_rate:.2f}."
    )
    if "Preferred location" in str(row.get("triage_reason", "")):
         reason += " Preferred location match."
    return reason

# ── Sidebar Configuration ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Match Configuration")

    # 1. Job Description Inputs
    st.subheader("1. Job Description")
    jd_file = st.file_uploader("Upload Job Description (.docx, .txt)", type=["docx", "txt"])
    
    jd_text = ""
    if jd_file:
        ext = ".docx" if jd_file.name.endswith(".docx") else ".txt"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(jd_file.getvalue())
                tmp_path = tmp.name
            
            jd_text = data_loader.load_job_description(tmp_path)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            
            st.success(f"📄 Loaded: **{jd_file.name}**")
            with st.expander("🔍 View Extracted Text", expanded=False):
                st.text_area("Extracted Content", value=jd_text, height=200, disabled=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        # Load default text for text area if not already set in session state
        if "jd_pasted_text" not in st.session_state:
            default_jd = ""
            for p in ["data/job_description.docx", "sample_data/job_description.txt"]:
                if os.path.exists(p):
                    try:
                        default_jd = data_loader.load_job_description(p)
                        break
                    except Exception:
                        pass
            st.session_state["jd_pasted_text"] = default_jd

        jd_text = st.text_area(
            "Job Description Text",
            value=st.session_state["jd_pasted_text"],
            height=200,
            placeholder="Paste a Job Description here..."
        )
        st.session_state["jd_pasted_text"] = jd_text

    st.divider()

    # 2. Candidate Dataset Inputs
    st.subheader("2. Candidates")
    cand_file = st.file_uploader(
        "Upload Candidates File", 
        type=["jsonl", "json", "csv"],
        help="Upload candidate dataset. Leave empty to fallback to default sources."
    )
    
    local_exists = os.path.exists("data/candidates.jsonl")
    use_local = False
    if local_exists:
        use_local = st.checkbox("Use local candidates.jsonl (487MB)", value=True)

    # Determine candidate file source
    cand_path = None
    if cand_file:
        ext = os.path.splitext(cand_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(cand_file.getvalue())
            cand_path = tmp.name
    elif local_exists and use_local:
        cand_path = "data/candidates.jsonl"
    else:
        cand_path = "sample_data/sample_candidates.json"

    st.divider()

    # 3. Settings & Weights Adjuster
    st.subheader("3. Settings")
    top_n = st.slider("Shortlist size", 10, 100, 25)

    with st.expander("⚖️ Adjust Score Weights"):
        w_sem   = st.slider("🧠 Semantic Similarity", 0.0, 1.0, 0.35, 0.05)
        w_skill = st.slider("🔧 Skill Match",          0.0, 1.0, 0.30, 0.05)
        w_exp   = st.slider("📅 Experience Match",     0.0, 1.0, 0.15, 0.05)
        w_beh   = st.slider("⚡ Behavioral Signal",    0.0, 1.0, 0.10, 0.05)
        w_plat  = st.slider("📊 Platform Activity",    0.0, 1.0, 0.10, 0.05)
        total = w_sem + w_skill + w_exp + w_beh + w_plat
        if abs(total - 1.0) > 0.01:
            st.warning(f"Weights sum to {total:.2f} (ideally 1.0).")

    run = st.button("🚀 Rank Candidates", type="primary", use_container_width=True)

# ── Landing Page (Initial Load) ──────────────────────────────────────────────
if not run:
    inject_custom_css()
    st.markdown('<div class="main-title">🎯 AI-Driven Resume Ranker</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Rank and match candidate resumes semantically against job descriptions using recruiter-level heuristics.</div>', unsafe_allow_html=True)
    
    # Showcase standard metrics
    st.markdown("""
        <div style="display: flex; gap: 20px; margin-bottom: 30px;">
            <div class="glass-card metric-card" style="flex: 1;">
                <div class="metric-card-val">5 Layers</div>
                <div class="metric-card-lbl">Semantic · Skills · Exp · Behavior · Platform</div>
            </div>
            <div class="glass-card metric-card" style="flex: 1;">
                <div class="metric-card-val">2 GB Limit</div>
                <div class="metric-card-lbl">Maximum Dataset Upload Capacity</div>
            </div>
            <div class="glass-card metric-card" style="flex: 1;">
                <div class="metric-card-val">Zero Latency</div>
                <div class="metric-card-lbl">Instant Rule-Based Heuristic Fallbacks</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Getting Started")
    st.info("""
    1. **Provide a Job Description** by pasting text or uploading a document file in the sidebar.
    2. **Provide Candidate Profiles** by uploading a `.jsonl`, `.json`, or `.csv` file. If none is uploaded, the app automatically defaults to the sample dataset (50 candidates).
    3. Custom-tune the **Score Weights** or location/recruiter triage coefficients if required.
    4. Click **Rank Candidates** in the sidebar to visualize the ranking pipeline and custom scanning animation!
    """)
    st.stop()

# ── Validation ────────────────────────────────────────────────────────────────
if not jd_text:
    st.error("Please provide a Job Description text.")
    st.stop()

if not cand_path:
    st.error("Please provide a candidates dataset.")
    st.stop()

# Inject Score Weights
config.SCORE_WEIGHTS = {
    "semantic_similarity": w_sem,
    "skill_match":         w_skill,
    "experience_match":    w_exp,
    "behavioral_signal":   w_beh,
    "platform_activity":   w_plat,
}

# ── Execution Pipeline with Custom Scanning Animation ─────────────────────────
inject_custom_css()
st.markdown('<div class="main-title">🎯 Ranking Results</div>', unsafe_allow_html=True)

loader = st.empty()

# Step 1: Parse Job Description
loader.markdown(show_scanning_animation("Parsing Job Description requirements..."), unsafe_allow_html=True)
jd_spec = jd_parser.parse_job_description(jd_text, use_llm=False)

# Step 2: Load Candidates
loader.markdown(show_scanning_animation("Loading candidate profiles from dataset..."), unsafe_allow_html=True)
df = data_loader.load_candidates(cand_path)

# Step 3: Compute Semantic Similarity
loader.markdown(show_scanning_animation(f"Computing semantic similarity for {len(df)} candidates..."), unsafe_allow_html=True)
df["semantic_similarity"] = embedder.semantic_similarity(
    jd_text, df["full_text"].tolist(), use_transformers=False
)

# Step 4: Extract Features & Scoring
loader.markdown(show_scanning_animation("Analyzing skill match, experience fit, and platform activity..."), unsafe_allow_html=True)
df = feature_extractor.build_features(df, jd_spec)
ranked = scorer.compute_hybrid_score(df)

# Step 5: Build final outputs
loader.markdown(show_scanning_animation("Structuring output csv and ranking formats..."), unsafe_allow_html=True)
submission = scorer.build_submission(ranked, top_n=top_n)

# Clear animation loader
loader.empty()

# ── Results Presentation ──────────────────────────────────────────────────────
st.subheader("📋 Parsed Job Requirements")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
        <div class="glass-card metric-card">
            <div class="metric-card-val">{jd_spec.get('min_experience_years', 0)}+ yrs</div>
            <div class="metric-card-lbl">Min Experience</div>
        </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
        <div class="glass-card metric-card">
            <div class="metric-card-val">{jd_spec.get('seniority', 'mid').title()}</div>
            <div class="metric-card-lbl">Seniority Target</div>
        </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
        <div class="glass-card metric-card">
            <div class="metric-card-val">{len(jd_spec.get('required_skills', []))}</div>
            <div class="metric-card-lbl">Skills Detected</div>
        </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
        <div class="glass-card metric-card">
            <div class="metric-card-val">{(jd_spec.get('role_title') or 'AI Engineer')[:20]}</div>
            <div class="metric-card-lbl">Role Title</div>
        </div>
    """, unsafe_allow_html=True)

if jd_spec.get("required_skills"):
    skills_html = "".join(f'<span class="pill pill-skill">{s}</span>' for s in jd_spec["required_skills"][:15])
    st.markdown(f'<div style="margin-top: 10px; margin-bottom: 25px;"><strong>Keywords:</strong> {skills_html}</div>', unsafe_allow_html=True)

st.divider()

# Rendering Tabs for different visualization methods
tab_cards, tab_table, tab_submission = st.tabs(["✨ Modern Cards View", "📊 Detailed Table View", "📄 Submission Format"])

with tab_cards:
    st.subheader(f"🏆 Top {top_n} Match Candidates")
    
    # Display candidates using custom HTML cards
    for idx, row in ranked.head(top_n).iterrows():
        rank_num = idx + 1
        
        # Tech skill badge mapping
        candidate_skills = row.get("skill_names", [])
        matched_skills = [s for s in candidate_skills if s.lower() in [req.lower() for req in jd_spec.get("required_skills", [])]]
        skills_pills = "".join(f'<span class="pill pill-skill">{s}</span>' for s in matched_skills[:6])
        if len(matched_skills) > 6:
            skills_pills += f'<span class="pill pill-skill">+{len(matched_skills)-6} more</span>'
            
        # Target location boost pill
        loc_pill = ""
        location = str(row.get("location", "")).lower()
        is_target_location = any(loc in location for loc in config.TARGET_LOCATIONS)
        if is_target_location:
            loc_pill = '<span class="pill pill-match">📍 Local Preferred</span>'
            
        # Recruiter Triage warnings
        triage_status = ""
        triage_coeff = row.get("triage_coeff", 1.0)
        if triage_coeff == 0.0:
            triage_status = '<span class="pill pill-alert">⚠️ Disqualified</span>'
        elif triage_coeff < 1.0:
            triage_status = '<span class="pill pill-alert">⚠️ Penalized</span>'
            
        st.markdown(f"""<div class="glass-card">
<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
<div>
<div style="display: flex; align-items: center; gap: 10px;">
<span style="font-size: 20px; font-weight: 700; color: #0f172a;">#{rank_num} {row.get('name', 'Candidate')}</span>
<span style="font-size: 12px; color: #64748b; background: rgba(0,0,0,0.05); padding: 2px 6px; border-radius: 4px;">{row['candidate_id']}</span>
</div>
<div style="color: #2563eb; font-size: 14px; font-weight: 500; margin-top: 4px;">{row.get('current_title', 'Software Engineer')}</div>
</div>
<div style="text-align: right;">
<div style="font-size: 26px; font-weight: 800; color: #059669;">{row['hybrid_score']:.1f}</div>
<div style="font-size: 10px; color: #64748b; text-transform: uppercase;">Match Score</div>
</div>
</div>
<div style="margin-top: 10px; font-size: 13px; color: #475569; display: flex; gap: 15px; flex-wrap: wrap;">
<span>📍 <strong>Location:</strong> {row.get('location', 'N/A')}</span>
<span>💼 <strong>Exp:</strong> {row.get('years_of_experience', 0.0):.1f} yrs</span>
<span>📈 <strong>Responsiveness:</strong> {row.get('recruiter_response_rate', 0.0):.0%}</span>
</div>
<div style="margin-top: 12px; padding: 12px; background: rgba(0,0,0,0.015); border-radius: 8px; border: 1px solid rgba(0,0,0,0.04); font-size: 13.5px; color: #334155; line-height: 1.5;">
💡 {_build_reasoning(row)}
</div>
<div style="margin-top: 12px; display: flex; flex-wrap: wrap; align-items: center; gap: 4px;">
{skills_pills}
{loc_pill}
{triage_status}
</div>
</div>""", unsafe_allow_html=True)

with tab_table:
    st.subheader("📊 Candidate Ranking Grid")
    display_df = ranked.head(top_n)[[
        "candidate_id", "name", "current_title", "location",
        "years_of_experience", "matched_skill_count",
        "semantic_similarity", "skill_match", "experience_match",
        "behavioral_signal", "platform_score", "hybrid_score"
    ]].copy()
    display_df.insert(0, "rank", range(1, len(display_df) + 1))
    
    st.dataframe(
        display_df.round(3),
        use_container_width=True,
        hide_index=True,
    )

with tab_submission:
    st.subheader("📄 Submission Format Output")
    st.caption("Exact output format required by validate_submission.py.")
    st.dataframe(submission, use_container_width=True, hide_index=True)

# Styled Download Buttons
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "⬇️ Download Full Ranked CSV",
        ranked.head(top_n).to_csv(index=False).encode("utf-8"),
        "ranked_candidates_full.csv", "text/csv",
        use_container_width=True
    )
with col2:
    st.download_button(
        "⬇️ Download Submission CSV",
        submission.to_csv(index=False).encode("utf-8"),
        "ranked_candidates.csv", "text/csv",
        use_container_width=True
    )

st.success("✅ Candidates successfully ranked! Downloads are ready above.")
