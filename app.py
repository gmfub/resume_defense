import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from utils.analysis import calculate_similarity, find_missing_keywords
from utils.file_loader import smart_parser
from utils.styles import add_custom_css

load_dotenv()
my_secret_key = os.getenv("OPEN_API_KEY")
st.set_page_config(page_title="Resume Optimizer", layout="wide")
add_custom_css()

# Initialize AI Client (needed for image OCR)
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=my_secret_key)

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""

# --- LAYOUT ---
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown('<h1 class="hero-title">Resume Defense</h1>', unsafe_allow_html=True)

# INPUTS
grid_col1, grid_col2 = st.columns(2, gap="large")

with grid_col1:
    st.markdown("### 1. Upload Resume")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, or PNG",
        type=["pdf", "docx", "png", "jpg"],
        label_visibility="collapsed",
    )

with grid_col2:
    st.markdown("### 2. Job Context")
    tab1, tab2 = st.tabs(["Paste Text", "Upload File"])
    job_desc = ""
    with tab1:
        job_desc = st.text_area(
            "Job Description", height=150, label_visibility="collapsed"
        )
    with tab2:
        uploaded_jd = st.file_uploader(
            "Upload JD", type=["pdf", "docx"], label_visibility="collapsed"
        )
        if uploaded_jd:
            job_desc = smart_parser(uploaded_jd)

# ACTION
b1, b2, b3 = st.columns([1, 1, 1])
with b2:
    analyze_click = st.button("Start Optimization →")

# LOGIC FLOW
if analyze_click:
    if uploaded_file and job_desc:
        if not st.session_state.resume_text:
            with st.spinner("Reading file..."):
                # CALL TEAMMATE 1's FUNCTION
                st.session_state.resume_text = smart_parser(uploaded_file, client)

    else:
        st.error("Please upload files.")

# RESULTS
if st.session_state.resume_text and job_desc:
    st.markdown("---")
    doc_col, stat_col = st.columns([2, 1], gap="large")

    with doc_col:
        st.subheader("📝 Live Document")
        updated_text = st.text_area(
            "editor", value=st.session_state.resume_text, height=600
        )
        st.session_state.resume_text = updated_text

    with stat_col:
        st.subheader("Analysis")
        score = calculate_similarity(updated_text, job_desc)
        missing_words = find_missing_keywords(updated_text, job_desc)

        st.metric("Match Score", f"{score}%")
        st.progress(score / 100)

        if missing_words:
            st.markdown("**Missing:** " + " ".join([f"`{w}`" for w in missing_words]))

        if st.button("✨ Ask AI Coach"):
            pass
