import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from utils.analysis import calculate_similarity, find_missing_keywords
from utils.ats_analysis import ATSAnalyzer
from utils.file_loader import smart_parser
from utils.styles import add_custom_css

load_dotenv()

# Initialize Groq Client (via OpenAI SDK)
# This expects OPENAI_API_KEY (containing Groq Key) in .env
api_key = os.getenv("OPENAI_API_KEY")
client = None
if api_key:
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
else:
    client = None

st.set_page_config(page_title="Resume Optimizer", layout="wide")
add_custom_css()


if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_text" not in st.session_state:
    st.session_state.job_text = ""
if "job_data" not in st.session_state:
    st.session_state.job_data = None


# --- LAYOUT ---
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown('<h1 class="hero-title">Resume Defense</h1>', unsafe_allow_html=True)


# PART 2: INPUTS
grid_col1, grid_col2 = st.columns(2, gap="large")


def clear_cache():
    st.session_state.resume_text = ""


with grid_col1:
    st.markdown("### 1. Upload Resume")
    uploaded_file = st.file_uploader(
        "PDF or DOCX",
        type=["pdf", "docx"],
        key="resume",
        label_visibility="collapsed",
        on_change=clear_cache,
    )

with grid_col2:
    st.markdown("### 2. Job Context")
    tab1, tab2 = st.tabs(["Paste Text", "Upload Screenshot/File"])

    # Initialize from session state
    job_desc = st.session_state.job_text

    with tab1:
        job_text_input = st.text_area(
            "Job Description",
            height=150,
            label_visibility="collapsed",
            placeholder="Paste text here...",
            value=st.session_state.job_text if not st.session_state.job_data else "",
        )
        if job_text_input:
            job_desc = job_text_input
            st.session_state.job_text = job_desc
            if (
                st.session_state.job_data
                and st.session_state.job_data.get("raw_text") != job_desc
            ):
                st.session_state.job_data = None

    with tab2:
        uploaded_jd = st.file_uploader(
            "Upload JD",
            type=["pdf", "docx", "png", "jpg", "jpeg", "webp"],
            key="jd",
            label_visibility="collapsed",
        )
        if uploaded_jd:
            if uploaded_jd.size > 5 * 1024 * 1024:
                st.error("❌ File too large. Please upload an image under 5MB.")
                st.stop()

            if st.session_state.get("last_jd_name") != uploaded_jd.name:
                # 1. Image Preview
                if uploaded_jd.type.startswith("image"):
                    from PIL import Image

                    try:
                        image = Image.open(uploaded_jd)
                        w, h = image.size
                        if w < 400 or h < 400:
                            st.warning("⚠️ Image dimensions look small.")
                        st.image(image, caption="Uploaded Screenshot", width=300)
                    except Exception:
                        pass

                # 2. Process with Progress Steps
                progress_bar = st.progress(0, text="Starting upload...")

                try:
                    progress_bar.progress(30, text="Uploading image...")
                    progress_bar.progress(50, text="Extracting text with AI Vision...")

                    # Call Smart Parser (Pass OpenAI Client)
                    extracted_data = smart_parser(
                        uploaded_jd, client=client, is_jd=True
                    )

                    if isinstance(extracted_data, dict):
                        st.session_state.job_text = extracted_data.get("raw_text", "")
                        st.session_state.job_data = extracted_data
                    else:
                        st.session_state.job_text = extracted_data
                        st.session_state.job_data = None

                    if (
                        not st.session_state.job_text
                        or st.session_state.job_text.startswith("Error")
                    ):
                        raise Exception(st.session_state.job_text)
                    else:
                        progress_bar.progress(100, text="Text extracted successfully!")
                        st.success("Job Description loaded!")
                        st.session_state.last_jd_name = uploaded_jd.name
                        job_desc = st.session_state.job_text

                        if st.session_state.job_data:
                            with st.expander("Parsed Job Details", expanded=False):
                                st.json(st.session_state.job_data)

                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ Extraction Failed: {e}")

                    st.markdown("### Fallback: Paste Text Manually")
                    fallback_text = st.text_area(
                        "Manual JD Entry",
                        height=200,
                        placeholder="Paste the job description here since the image failed...",
                        key="fallback_jd_error",
                    )
                    if fallback_text:
                        job_desc = fallback_text
                        st.session_state.job_text = fallback_text
            else:
                if st.session_state.job_data:
                    with st.expander("Parsed Job Details", expanded=False):
                        st.json(st.session_state.job_data)


# ACTION
st.write("")
st.write("")
b1, b2, b3 = st.columns([3, 2, 3])
with b2:
    analyze_click = st.button("Start Optimization →", use_container_width=True)


# LOGIC FLOW
if analyze_click:
    if uploaded_file and job_desc:
        if not st.session_state.resume_text:
            with st.spinner("Reading file..."):
                # smart_parser call
                st.session_state.resume_text = smart_parser(
                    uploaded_file, client=client
                )

    else:
        st.error("Please upload files.")


# RESULTS (ATS Analysis)
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
        st.subheader("ATS Analysis")

        analyzer = ATSAnalyzer()
        report = analyzer.calculate_master_score(
            updated_text, job_desc, jd_data=st.session_state.job_data
        )

        st.metric(
            "ATS Probability",
            report.pass_probability,
            delta=f"Score: {report.master_score}/100",
        )
        st.progress(report.master_score / 100)

        t1, t2, t3 = st.tabs(["Keywords", "Format", "Risks"])
        with t1:
            st.caption(f"Score: {report.keyword_score}/100")
            if report.missing_critical_skills:
                st.error(
                    f"Missing Critical: {', '.join(report.missing_critical_skills)}"
                )

            missing_words = find_missing_keywords(updated_text, job_desc)
            if missing_words:
                st.markdown(
                    "**Consider adding:** "
                    + ", ".join([f"`{w}`" for w in missing_words[:5]])
                )

        with t2:
            st.caption(f"Score: {report.format_score}/100")
            if report.format_score == 100:
                st.success("Clean formatting!")
            else:
                for item in report.feedback:
                    if item.problem.startswith(
                        "Detected table"
                    ) or item.problem.startswith("Could not clearly"):
                        st.warning(f"⚠️ {item.problem}")

        with t3:
            st.caption(f"Safety: {report.manipulation_score}/100")
            if report.manipulation_score < 100:
                st.error("Manipulation detected!")

        st.markdown("### 🛠 Fixes Needed")
        for item in report.feedback:
            color = "red" if item.severity in ["critical", "high"] else "orange"
            st.markdown(f":{color}[**{item.problem}**]")
            st.caption(f"💡 Fix: {item.fix}")
            st.markdown("---")

        # AI COACH BUTTON - OPENAI VERSION
        if st.button("✨ Ask AI Coach"):
            with st.spinner("Thinking..."):
                try:
                    if not client:
                        st.error("❌ API Client missing. Check OPENAI_API_KEY in .env.")
                        st.stop()

                    feedback_text = "\n".join(
                        [f"- {i.problem}: {i.fix}" for i in report.feedback]
                    )

                    prompt = f"""
                    You are an expert career coach.
                    The candidate's resume has a {report.master_score}% ATS score ({report.pass_probability} pass rate).

                    Detected Issues:
                    {feedback_text}

                    Missing Keywords: {", ".join(missing_words[:5])}

                    Give 3 specific, actionable tips to improve this resume.
                    Focus on the most critical ATS matching issues first.
                    Keep it short and encouraging.
                    """

                    # Call Groq API
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                    )

                    # Show Answer
                    st.success("💡 **AI Advice:**")
                    st.info(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"AI Error: {str(e)}")
