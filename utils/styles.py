import streamlit as st


def add_custom_css():
    st.markdown(
        """
        <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Global Reset */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #000000;
            color: #EDEDED;
        }

        /* Grid Background */
        .stApp {
            background-color: #000000;
            background-image: linear-gradient(#111 1px, transparent 1px),
            linear-gradient(90deg, #111 1px, transparent 1px);
            background-size: 40px 40px;
        }

        /* HERO SECTION */
        .hero-title {
            text-align: center;
            font-size: 4rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            background: linear-gradient(180deg, #fff, #888);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .hero-subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #888;
            margin-bottom: 40px;
            font-weight: 400;
        }

        /* CARD STYLING */
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            background-color: #0A0A0A;
            border: 1px solid #222;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        /* INPUT STYLING */
        .stTextArea textarea {
            background-color: #000;
            border: 1px solid #333;
            color: #fff;
            border-radius: 8px;
        }

        /* BUTTON STYLING */
        div.stButton > button {
            width: 100%;
            background-color: #EDEDED;
            color: #000000;
            font-weight: 600;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            transition: all 0.2s ease;
            font-size: 16px;
        }
        div.stButton > button:hover {
            background-color: #FFFFFF;
            transform: scale(1.02);
            box-shadow: 0 0 15px rgba(255,255,255,0.2);
        }

        /* Hide Header */
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
