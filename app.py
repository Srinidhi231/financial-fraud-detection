import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from joblib import load
import base64
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# IMPORTANT: set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="FraudShield Pro - Advanced Fraud Detection",
  page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = 0

# Dark Theme CSS
st.markdown(
    """
<style>
  /* Dark Theme Base */
  .stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1629 100%);
    background-attachment: fixed;
  }
  
  /* Grid pattern overlay */
  .stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
      linear-gradient(rgba(25, 118, 210, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(25, 118, 210, 0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
  }
  
  /* Main content area */
  .main .block-container {
    background: transparent;
    padding-top: 1rem;
    z-index: 1;
    position: relative;
  }
  
  /* Text colors for dark theme */
  h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: #ffffff !important;
  }
  
  /* Dark theme for Streamlit components */
  .stMetric {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.6), rgba(59, 130, 246, 0.6)) !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    transition: transform 0.3s ease !important;
  }
  
  .stMetric:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
  }
  
  .stMetric label {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
  }
  
  .stMetric div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 24px !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
  }
  

  
  .stSelectbox label {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 16px !important;
  }
  
  .stSelectbox div[data-baseweb="select"] {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6) !important;
    border: 2px solid #60a5fa !important;
    border-radius: 8px !important;
    color: white !important;
  }

  .stSelectbox div[data-baseweb="select"] * {
    color: #ffffff !important;
}

 
  
  .stSelectbox div[data-baseweb="select"] input {
    color: #ffffff !important;
  }
  
  .stSelectbox div[data-baseweb="select"] [data-baseweb="popover"] {
    background: #1e3a8a !important;
    border: 1px solid #3b82f6 !important;
  }
  
  .stSelectbox div[data-baseweb="select"] [data-baseweb="option"] {
    background: #1e3a8a !important;
    color: #ffffff !important;
  }
  
  .stSelectbox div[data-baseweb="select"] [data-baseweb="option"]:hover {
    background: #3b82f6 !important;
  }
  
  /* Professional Dark Navigation Bar */
  .stTabs [data-baseweb="tab-list"] {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(10, 14, 39, 0.95) !important;
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(25, 118, 210, 0.3);
    padding: 0 2rem;
    margin: 0 -1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    display: flex;
    justify-content: flex-start;
    align-items: center;
    min-height: 70px;
    gap: 0.5rem;
    width: 100%;
  }
  
  .stTabs [data-baseweb="tab"] {
    font-size: 17px !important;
    font-weight: 600 !important;
    padding: 16px 28px !important;
    margin: 0 4px !important;
    color: #B0BEC5 !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
    min-height: 70px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
  }
  
  .stTabs [data-baseweb="tab"] > div {
    font-size: 17px !important;
    font-weight: 600 !important;
    color: #B0BEC5 !important;
  }
  
  .stTabs [data-baseweb="tab"]:hover {
    background: rgba(25, 118, 210, 0.1) !important;
    color: #64B5F6 !important;
  }
  
  .stTabs [data-baseweb="tab"]:hover > div {
    color: #64B5F6 !important;
  }
  
  .stTabs [aria-selected="true"] {
    font-weight: 700 !important;
    color: #64B5F6 !important;
    background: transparent !important;
    border-bottom: 3px solid #64B5F6 !important;
    padding-bottom: 13px !important;
  }
  
  .stTabs [aria-selected="true"] > div {
    font-weight: 700 !important;
    color: #64B5F6 !important;
  }
  
  /* Hero Section Styles */
  .hero-section {
    padding: 4rem 0;
    text-align: left;
    margin-bottom: 4rem;
  }
  
  .feature-tag {
    display: inline-block;
    padding: 8px 16px;
    border: 1px solid #64B5F6;
    border-radius: 20px;
    color: #64B5F6;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 2rem;
    background: rgba(25, 118, 210, 0.1);
  }
  
  .hero-title {
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #ffffff 0%, #64B5F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  
  .hero-description {
    font-size: 1.2rem;
    color: #B0BEC5;
    line-height: 1.6;
    margin-bottom: 2.5rem;
    max-width: 600px;
  }
  
  .cta-button-primary {
    display: inline-block;
    padding: 14px 32px;
    background: linear-gradient(135deg, #1976D2, #42A5F5);
    color: white !important;
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    text-decoration: none;
    margin-right: 1rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    border: none;
    cursor: pointer;
  }
  
  .cta-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(25, 118, 210, 0.4);
  }
  
  .cta-button-secondary {
    display: inline-block;
    padding: 14px 32px;
    background: rgba(255, 255, 255, 0.1);
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    text-decoration: none;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    cursor: pointer;
  }
  
  .cta-button-secondary:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.3);
  }
  
  /* Feature Cards */
  .features-section {
    margin: 5rem 0;
  }
  
  .section-title {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 1rem;
  }
  
  .section-title .highlight {
    color: #64B5F6;
  }
  
  .section-subtitle {
    text-align: center;
    color: #B0BEC5;
    font-size: 1.1rem;
    margin-bottom: 3rem;
  }
  
  .feature-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 2rem;
    height: 100%;
    transition: all 0.3s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }
  
  .feature-card:hover {
    transform: translateY(-5px);
    border-color: rgba(100, 181, 246, 0.5);
    box-shadow: 0 10px 30px rgba(25, 118, 210, 0.2);
  }
  
  /* Feature card button hover effects */
  .feature-card button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(25, 118, 210, 0.4);
    opacity: 0.9;
  }
  
  /* Hero section button hover effects */
  .hero-section button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(25, 118, 210, 0.4);
  }
  
  .feature-card.active {
    border-color: #4CAF50;
    box-shadow: 0 0 20px rgba(76, 175, 80, 0.3);
  }
  
  .feature-icon {
    width: 60px;
    height: 60px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin-bottom: 1.5rem;
    border: 2px solid;
  }
  
  .feature-icon.blue {
    background: rgba(25, 118, 210, 0.2);
    border-color: #64B5F6;
    color: #64B5F6;
  }
  
  .feature-icon.green {
    background: rgba(76, 175, 80, 0.2);
    border-color: #4CAF50;
    color: #4CAF50;
  }
  
  .feature-icon.orange {
    background: rgba(255, 152, 0, 0.2);
    border-color: #FF9800;
    color: #FF9800;
  }
  
  .feature-card-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: white;
  }
  
  .feature-card-desc {
    color: #B0BEC5;
    line-height: 1.6;
    margin-bottom: 1.5rem;
  }
  
  .feature-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  
  .feature-tag-item {
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    font-size: 12px;
    color: #B0BEC5;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  /* How It Works Section */
  .how-it-works-section {
    margin: 5rem 0;
  }
  
  .step-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    margin: 3rem 0;
    flex-wrap: wrap;
  }
  
  .step-item {
    text-align: center;
    position: relative;
    flex: 1;
    min-width: 200px;
    max-width: 250px;
  }
  
  .step-connector {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, #64B5F6, rgba(100, 181, 246, 0.3));
    position: absolute;
    top: 30px;
    right: -30px;
    z-index: 1;
  }
  
  .step-item:last-child .step-connector {
    display: none;
  }
  
  .step-icon-box {
    width: 80px;
    height: 80px;
    background: rgba(25, 118, 210, 0.2);
    border: 2px solid #64B5F6;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
    margin: 0 auto 1rem;
    position: relative;
    z-index: 2;
  }
  
  .step-number {
    position: absolute;
    top: -10px;
    right: -10px;
    width: 32px;
    height: 32px;
    background: #64B5F6;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    color: white;
    z-index: 3;
  }
  
  .step-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: white;
  }
  
  .step-description {
    color: #B0BEC5;
    font-size: 0.95rem;
    line-height: 1.5;
  }
  
  /* Dark theme for form elements */
  .stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #000000 !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
  }
  
  /* Text input labels */
  .stTextInput label {
    color: #ffffff !important;
  }
  
  .stFileUploader {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 2px dashed rgba(100, 181, 246, 0.8) !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    color: #000000 !important;
  }
  
  /* File uploader text visibility */
  .stFileUploader label {
    color: #000000 !important;
  }
  
  .stFileUploader small {
    color: #333333 !important;
  }
  
  .stButton > button {
    background: linear-gradient(135deg, #1976D2, #42A5F5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
  }
  
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(25, 118, 210, 0.4) !important;
  }
  
  /* Info boxes */
  .stInfo {
    background: rgba(25, 118, 210, 0.1) !important;
    border-left: 4px solid #64B5F6 !important;
    color: #B0BEC5 !important;
  }
  
  .stSuccess {
    background: rgba(76, 175, 80, 0.1) !important;
    border-left: 4px solid #4CAF50 !important;
    color: #B0BEC5 !important;
  }
  
  .stWarning {
    background: rgba(255, 152, 0, 0.1) !important;
    border-left: 4px solid #FF9800 !important;
    color: #B0BEC5 !important;
  }
  
  .stError {
    background: rgba(244, 67, 54, 0.1) !important;
    border-left: 4px solid #F44336 !important;
    color: #B0BEC5 !important;
  }
  
  /* Dataframes */
  .dataframe {
    background: rgba(255, 255, 255, 0.05) !important;
    color: white !important;
  }
  
  /* Footer */
  .footer {
    text-align: center;
    padding: 2rem;
    color: #757575;
    margin-top: 4rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  /* Hide Streamlit default elements */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
  
  /* Plotly chart dark theme */
  .js-plotly-plot {
    background: transparent !important;
  }
  /* Force stronger selectbox styles to avoid white-on-white issues */
  div[data-baseweb="select"] {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6) !important;
    color: #ffffff !important;
    border: 2px solid #60a5fa !important;
    border-radius: 8px !important;
  }
  div[data-baseweb="select"] * {
    color: #ffffff !important;
    background: transparent !important;
  }
  div[data-baseweb="select"] input {
    color: #ffffff !important;
  }
  div[role="listbox"] div[role="option"], div[data-baseweb="option"] {
    color: #ffffff !important;
    background: #1e3a8a !important;
  }
  div[role="option"]:hover {
    background: #3b82f6 !important;
    color: #ffffff !important;
  }
  /* =========================
   FIX WHITE-ON-WHITE ISSUES
   ========================= */

/* Text input fields (value + placeholder) - FIXED FOR VISIBILITY */
.stTextInput input,
.stTextInput textarea {
  color: #000000 !important;      /* black text for visibility */
  background-color: rgba(255, 255, 255, 0.95) !important;
  border: 1px solid rgba(148, 163, 184, 0.5) !important;
}

/* Placeholder text */
.stTextInput input::placeholder {
  color: #6b7280 !important;
}

/* File uploader container - FIXED FOR VISIBILITY */
.stFileUploader {
  background-color: rgba(255, 255, 255, 0.95) !important;
  color: #000000 !important;
}

/* File uploader label */
.stFileUploader label {
  color: #000000 !important;
  font-weight: 600;
}

/* File uploader instructions text */
.stFileUploader div,
.stFileUploader small,
.stFileUploader p,
.stFileUploader span {
  color: #333333 !important;
}

/* “Browse files” button text */
.stFileUploader button {
  color: #ffffff !important;
  background: linear-gradient(135deg, #1976D2, #42A5F5) !important;
  border: none !important;
}

/* BaseWeb input (used internally by Streamlit) - FIXED */
div[data-baseweb="input"] input {
  color: #000000 !important;
  background-color: rgba(255, 255, 255, 0.95) !important;
}

/* Disabled / readonly inputs */
input:disabled {
  color: #94a3b8 !important;
}
/* =========================
   STREAMLIT FILE UPLOADER FIX
   ========================= */

/* Main uploader box */
section[data-testid="stFileUploader"] > div {
  background-color: rgba(15, 23, 42, 0.9) !important;
  color: #e5e7eb !important;
}

/* Drag & drop text - FIXED FOR VISIBILITY */
section[data-testid="stFileUploader"] span,
section[data-testid="stFileUploader"] small,
section[data-testid="stFileUploader"] p,
section[data-testid="stFileUploader"] label {
  color: #000000 !important;
}

/* Cloud upload icon */
section[data-testid="stFileUploader"] svg {
  fill: #93c5fd !important;
  color: #93c5fd !important;
}

/* Browse files button */
section[data-testid="stFileUploader"] button {
  background: linear-gradient(135deg, #1976D2, #42A5F5) !important;
  color: #ffffff !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}

/* =========================
   TEXT INPUT (MODEL CONFIG)
   ========================= */

section[data-testid="stTextInput"] input {
  background-color: rgba(255, 255, 255, 0.95) !important;
  color: #000000 !important;
  border: 1px solid rgba(148, 163, 184, 0.5) !important;
}

/* Placeholder text */
section[data-testid="stTextInput"] input::placeholder {
  color: #6b7280 !important;
}

/* Labels */
section[data-testid="stTextInput"] label {
  color: #ffffff !important;
  font-weight: 600 !important;
}
/* =========================
   FINAL SELECTBOX VISIBILITY FIX (MODEL + SHAP) - FORCE BLACK TEXT
   ========================= */

/* Ensure all selectbox content (label, selected value, placeholder, arrow, options) is dark */
section[data-testid="stSelectbox"],
section[data-testid="stSelectbox"] * ,
section[data-testid="stSelectbox"] div[data-baseweb="select"],
section[data-testid="stSelectbox"] div[data-baseweb="select"] * ,
section[data-testid="stSelectbox"] div[data-baseweb="select"] [data-baseweb="select-value"],
section[data-testid="stSelectbox"] div[data-baseweb="select"] [data-baseweb="select-value"] * {
  color: #000000 !important;
  background: transparent !important;
}

/* Make the visible select box background neutral so black text is readable */
section[data-testid="stSelectbox"] > div,
section[data-testid="stSelectbox"] div[data-baseweb="select"] {
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  border-radius: 8px !important;
  padding: 6px 10px !important;
}

/* Dropdown popover and options — keep readable with black text */
div[data-baseweb="popover"] {
  background-color: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  color: #000000 !important;
  border-radius: 8px !important;
}

div[data-baseweb="option"],
div[role="option"] {
  background-color: #ffffff !important;
  color: #000000 !important;
  padding: 8px 12px !important;
}

div[data-baseweb="option"]:hover,
div[role="option"]:hover {
  background-color: rgba(0,0,0,0.04) !important;
  color: #000000 !important;
}

/* Ensure input inside select shows black text */
section[data-testid="stSelectbox"] input,
div[data-baseweb="select"] input {
  color: #000000 !important;
  background: transparent !important;
}

/* Extra fallback: very specific selector for Streamlit select internals */
.stSelectbox [data-baseweb="select"] [data-baseweb="select-value"],
.stSelectbox [data-baseweb="select"] [data-baseweb="select-value"] > div {
  color: #000000 !important;
  background: transparent !important;
}

/* =========================
     FINAL SELECTBOX VISIBILITY - FORCE BLACK TEXT EVERYWHERE
     ========================= */

  /* Broad catch-all: force black color for any selectbox element */
  section[data-testid="stSelectbox"] *,
  .stSelectbox *,
  div[data-baseweb="select"] *,
  div[data-baseweb="popover"] *,
  div[data-baseweb="option"],
  div[role="option"],
  div[role="listbox"] *,
  section[data-testid="stSelectbox"] input,
  section[data-testid="stSelectbox"] label,
  section[data-testid="stSelectbox"] span,
  section[data-testid="stSelectbox"] div,
  section[data-testid="stSelectbox"] button,
  section[data-testid="stSelectbox"] svg {
    color: #000000 !important;
    fill: #000000 !important;
    background: transparent !important;
  }

  /* Ensure the visible select box background is light so black text is readable */
  section[data-testid="stSelectbox"] > div,
  section[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
  }

  /* Make dropdown menu white so black option text is visible */
  div[data-baseweb="popover"] {
    background-color: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    color: #000000 !important;
    border-radius: 8px !important;
  }

  div[data-baseweb="option"],
  div[role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    padding: 8px 12px !important;
  }

  div[data-baseweb="option"]:hover,
  div[role="option"]:hover {
    background-color: rgba(0,0,0,0.04) !important;
    color: #000000 !important;
  }

  /* Extra very-specific fallbacks */
  .stSelectbox [data-baseweb="select"] [data-baseweb="select-value"],
  .stSelectbox [data-baseweb="select"] [data-baseweb="select-value"] > div,
  [data-testid="stSelectbox"] [data-baseweb="select"] [data-baseweb="select-value"] {
    color: #000000 !important;
    background: transparent !important;
  }

  /* Prevent Streamlit/Browser autofill or internal backgrounds from hiding text */
  section[data-testid="stSelectbox"] input:-webkit-autofill,
  section[data-testid="stSelectbox"] input:-internal-autofill-selected {
    -webkit-text-fill-color: #000000 !important;
    caret-color: #000000 !important;
    background-color: #ffffff !important;
  }
  #explain-label-only label {
  color: white !important;
}
/* === FIX GAP BETWEEN EXPLAIN LABEL TEXT AND DROPDOWN === */
#explain-label-only {
  margin-top: -18px !important;
}

#explain-label-only div[data-baseweb="select"] {
  margin-top: -28px !important;
}


  /* End selectbox force rules */
</style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts(model_path: str, mappings_path: str, features_path: str):
    model = load(model_path)
    with open(mappings_path, 'r', encoding='utf-8') as f:
        categorical_mappings = json.load(f)
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)
    return model, categorical_mappings, feature_columns


def preprocess(df: pd.DataFrame, categorical_mappings: dict, feature_columns: list) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.replace('-', '_') for c in out.columns]
    for col, mapping in categorical_mappings.items():
        if col in out.columns:
            out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            out[col] = -1
    missing = [c for c in feature_columns if c not in out.columns]
    for c in missing:
        out[c] = -999
    extra = [c for c in out.columns if c not in feature_columns]
    if extra:
        out = out.drop(columns=extra)
    out = out[feature_columns]
    out = out.fillna(-999)
    return out


def _download_href(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"


def show_tab_bar():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a, #3b82f6); padding: 1rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);">
    <h3 style="color: white; text-align: center; margin: 0; font-weight: 600;">Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(4)
    tab_names = ["Home", "Detect", "Visualize", "Explain"]
    
    for i, (col, name) in enumerate(zip(cols, tab_names)):
        with col:
            if st.button(name, key=f"tab_{i}", use_container_width=True):
                st.session_state['active_tab'] = i
                st.rerun()

def show_home():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="feature-tag">AI-Powered Fraud Detection</div>
        <h1 class="hero-title">Protect Your <span style="color: #64B5F6;">Transactions</span></h1>
        <p class="hero-description">
            Advanced machine learning algorithms detect fraudulent patterns in real-time, 
            protecting your financial data with unprecedented accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero buttons
    col1, col2 = st.columns(2)
    with col1:
      if st.button("Start Detection →", key="hero_detect", use_container_width=True):
        st.session_state['active_tab'] = 1
        st.rerun()
    with col2:
      if st.button("View Analytics", key="hero_analytics", use_container_width=True):
        st.session_state['active_tab'] = 2
        st.rerun()
    
    # Powerful Features Section
    st.markdown("""
    <div class="features-section">
        <h2 class="section-title">Powerful <span class="highlight">Features</span></h2>
        <p class="section-subtitle">
            Three integrated modules to detect, visualize, and understand fraud in your financial data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="position: absolute; top: 1rem; right: 1rem; font-size: 20px;">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 21l-4.35-4.35" stroke="#64B5F6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><circle cx="11" cy="11" r="6" stroke="#64B5F6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <div class="feature-icon blue">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 21l-4.35-4.35" stroke="#64B5F6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><circle cx="11" cy="11" r="6" stroke="#64B5F6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <h3 class="feature-card-title">Detect</h3>
            <p class="feature-card-desc">
                Upload transaction CSVs and let our ML model instantly identify fraudulent 
                transactions with high precision.
            </p>
            <div class="feature-tags">
                <span class="feature-tag-item">CSV Upload</span>
                <span class="feature-tag-item">Real-time Analysis</span>
                <span class="feature-tag-item">Fraud Flagging</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Detect →", key="goto_detect", use_container_width=True):
            st.session_state['active_tab'] = 1
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="position: absolute; top: 1rem; right: 1rem; font-size: 20px;">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 3v18h18" stroke="#4CAF50" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 13l4-6 4 8 4-10" stroke="#4CAF50" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <div class="feature-icon green">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 3v18h18" stroke="#4CAF50" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 13l4-6 4 8 4-10" stroke="#4CAF50" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <h3 class="feature-card-title">Visualize</h3>
            <p class="feature-card-desc">
                Explore interactive charts showing fraud distribution, probability patterns, 
                and risk level analysis.
            </p>
            <div class="feature-tags">
                <span class="feature-tag-item">Distribution Charts</span>
                <span class="feature-tag-item">Probability Graphs</span>
                <span class="feature-tag-item">Risk Analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Visualize →", key="goto_visualize", use_container_width=True):
            st.session_state['active_tab'] = 2
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div style="position: absolute; top: 1rem; right: 1rem; font-size: 20px;">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 18h6" stroke="#FF9800" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 3a7 7 0 00-4 12h8a7 7 0 00-4-12z" stroke="#FF9800" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <div class="feature-icon orange">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 18h6" stroke="#FF9800" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 3a7 7 0 00-4 12h8a7 7 0 00-4-12z" stroke="#FF9800" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <h3 class="feature-card-title">Explain</h3>
            <p class="feature-card-desc">
                Understand model decisions with SHAP explanations showing which features 
                contribute to fraud detection.
            </p>
            <div class="feature-tags">
                <span class="feature-tag-item">SHAP Values</span>
                <span class="feature-tag-item">Feature Impact</span>
                <span class="feature-tag-item">Model Insights</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Explain →", key="goto_explain", use_container_width=True):
            st.session_state['active_tab'] = 3
            st.rerun()
    
    # How It Works Section
    st.markdown("""
    <div class="how-it-works-section">
        <h2 class="section-title">How It <span class="highlight">Works</span></h2>
        <p class="section-subtitle">
            A simple four-step process to secure your financial transactions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-container">
        <div class="step-item">
            <div class="step-connector"></div>
            <div class="step-icon-box">
              <div class="step-number">1</div>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 3v12" stroke="#64B5F6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M8 7l4-4 4 4" stroke="#64B5F6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <h4 class="step-title">Upload Data</h4>
            <p class="step-description">Upload your transaction CSV file containing financial data.</p>
        </div>
        <div class="step-item">
            <div class="step-connector"></div>
            <div class="step-icon-box">
              <div class="step-number">2</div>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 8v8" stroke="#64B5F6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><path d="M8 12h8" stroke="#64B5F6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <h4 class="step-title">ML Processing</h4>
            <p class="step-description">Our trained model analyzes patterns to detect anomalies.</p>
        </div>
        <div class="step-item">
            <div class="step-connector"></div>
            <div class="step-icon-box">
              <div class="step-number">3</div>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2l6 3v5c0 5-3 9-6 11-3-2-6-6-6-11V5l6-3z" stroke="#64B5F6" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <h4 class="step-title">Fraud Detection</h4>
            <p class="step-description">Suspicious transactions are flagged with risk scores.</p>
        </div>
        <div class="step-item">
            <div class="step-icon-box">
              <div class="step-number">4</div>
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 17l6-6 4 4 8-8" stroke="#64B5F6" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </div>
            <h4 class="step-title">Insights</h4>
            <p class="step-description">Visualize results and understand feature contributions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Ensure navigation function is available (re-injection)
    st.markdown("""
    <script>
    // Re-inject navigation function to ensure it's always available
    if (typeof window.navigateToTab === '') {
        window.navigateToTab = function(tabIndex) {
            function findAndClickTab() {
                let tabs = document.querySelectorAll('div[data-testid="stTabs"] button[data-baseweb="tab"]');
                if (!tabs || tabs.length === 0) {
                    tabs = document.querySelectorAll('.stTabs button[data-baseweb="tab"]');
                }
                if (!tabs || tabs.length === 0) {
                    tabs = document.querySelectorAll('div[data-testid="stTabs"] [role="tab"]');
                }
                if (!tabs || tabs.length === 0) {
                    const tabContainer = document.querySelector('div[data-testid="stTabs"]');
                    if (tabContainer) {
                        tabs = tabContainer.querySelectorAll('button');
                    }
                }
                
                if (tabs && tabs.length > tabIndex && tabs[tabIndex]) {
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                    setTimeout(function() {
                        tabs[tabIndex].focus();
                        tabs[tabIndex].click();
                    }, 50);
                    return true;
                }
                return false;
            }
            
            if (!findAndClickTab()) {
                let attempts = 0;
                const maxAttempts = 30;
                const interval = setInterval(function() {
                    attempts++;
                    if (findAndClickTab() || attempts >= maxAttempts) {
                        clearInterval(interval);
                    }
                }, 200);
            }
        };
    }
    </script>
    """, unsafe_allow_html=True)


def show_detect():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
      <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; color: white;">
        Upload Transaction Data
      </h1>
        <p style="color: #B0BEC5; font-size: 1.1rem;">
            Upload your CSV file to begin fraud detection analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Tip: Ensure your CSV contains the required columns as per the training schema")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="file_uploader")
    
    with col2:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); 
              border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;">
            <h3 style="color: white; margin-bottom: 1rem;">Model Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Check available models and create selector
        available_models = []
        
        if os.path.exists('fraud_model.joblib'):
            available_models.append('fraud_model.joblib')
        
        if available_models:
          selected_model = st.selectbox(
            "Model",
            available_models,
            index=0,
            key="model_selector"
          )
          model_path = selected_model
        else:
          model_path = st.text_input("Model file", value="fraud_model.joblib", key="cfg_model")
        
        mappings_path = st.text_input("Categorical mappings", value="categorical_mappings.json", key="cfg_map")
        features_path = st.text_input("Feature columns", value="feature_columns.json", key="cfg_feat")
    
    if st.button("Detect Fraudulent Transactions", key="detect_fraud", use_container_width=True):
        if uploaded is None:
            st.warning("Please upload a CSV file before detecting fraud.")
            return
        
        with st.spinner("Loading model artifacts..."):
            try:
                model, categorical_mappings, feature_columns = load_artifacts(model_path, mappings_path, features_path)
            except Exception as e:
                st.error(f"Failed to load artifacts: {e}")
                return
        
        with st.spinner("Processing data and detecting fraud..."):
            try:
                raw = pd.read_csv(uploaded)
                X = preprocess(raw, categorical_mappings, feature_columns)
                probs = model.predict_proba(X)[:, 1]
                preds = (probs >= 0.5).astype(int)
                result = pd.DataFrame({"fraud_probability": probs, "isFraud_pred": preds})
                # If the uploaded CSV contains TransactionID and/or ground-truth isFraud,
                # include them so we can compute metrics later in Visualize.
                extra_cols = []
                if 'TransactionID' in raw.columns:
                  extra_cols.append('TransactionID')
                if 'isFraud' in raw.columns:
                  # include ground-truth label column if present
                  extra_cols.append('isFraud')
                if extra_cols:
                  result = pd.concat([raw[extra_cols].reset_index(drop=True), result], axis=1)
                
                st.success("✅ Fraud detection completed successfully!")
                
                total_transactions = len(result)
                fraud_cases = int(result[result['isFraud_pred'] == 1].shape[0])
                avg_fraud_probability = result['fraud_probability'].mean() if total_transactions > 0 else 0.0
                
                st.markdown("### Detection Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Transactions", f"{total_transactions:,}")
                c2.metric("Fraud Cases", f"{fraud_cases:,}")
                c3.metric("Avg. Risk Score", f"{avg_fraud_probability:.3f}")
                
                st.markdown("### All Detection Results")
                st.markdown(f"<p style='color: #B0BEC5;'>Showing all {len(result):,} transactions</p>", unsafe_allow_html=True)
                display_result = result.copy()
                display_result['isFraud_pred'] = display_result['isFraud_pred'].map({
                  0: 'Safe',
                  1: 'Fraud'
                  })
                # Make the serial/index start from 1 instead of 0 for display
                display_result.index = range(1, len(display_result) + 1)
                def _highlight_row(row):
                  # return a list of CSS styles for the row
                  if str(row.get('isFraud_pred', '')).strip().lower() == 'fraud':
                    return ['background-color: rgba(244, 67, 54, 0.12)'] * len(row)
                  else:
                    return ['background-color: rgba(76, 175, 80, 0.08)'] * len(row)

                styled_df = (
                  display_result.style
                  .apply(_highlight_row, axis=1)
                  .set_properties(subset=['isFraud_pred'], **{'text-align': 'right'})
                )

                st.dataframe(
                  styled_df,
                  use_container_width=True,
                  height=600,
                )

                
                st.markdown("### Download Results")
                href = _download_href(result, "fraudshield_predictions.csv")
                st.markdown(f'<a href="{href}" download="fraudshield_predictions.csv" style="color: #64B5F6; text-decoration: none; font-weight: 600;">Download CSV</a>', unsafe_allow_html=True)
                
                # persist
                st.session_state['X'] = X
                st.session_state['result'] = result
                st.session_state['model'] = model
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")


def show_visualize():
    result = st.session_state.get('result')
    if result is None:
        st.info("No results available. Run detection in the Detect tab first.")
        return
    
    st.markdown("""
    <div style="margin-bottom: 2rem;">
      <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; color: white;">
        Fraud Analytics Dashboard
      </h1>
        <p style="color: #B0BEC5; font-size: 1.1rem;">
            Interactive visualizations of fraud detection results
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fraud_counts = result['isFraud_pred'].value_counts().reset_index()
        fraud_counts.columns = ['Status', 'Count']
        fraud_counts['Status'] = fraud_counts['Status'].map({0: 'Legitimate', 1: 'Fraudulent'})
        fig_pie = px.pie(fraud_counts, values='Count', names='Status', title='Fraud Distribution', hole=0.4,
                        color_discrete_map={'Legitimate':'#4CAF50', 'Fraudulent':'#F44336'})
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font_color='white',
  
            
            legend=dict(
                font=dict(color='white', size=14)     # 👈 LEGEND TEXT FIX
              )
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_hist = px.histogram(result, x='fraud_probability', nbins=50, title='Fraud Probability Distribution',
                                color=result['isFraud_pred'].map({0: 'Legitimate', 1: 'Fraudulent'}),
                                color_discrete_map={'Legitimate':'#4CAF50', 'Fraudulent':'#F44336'})
        fig_hist.update_layout(
            xaxis_title="Fraud Probability",
            yaxis_title="Number of Transactions",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), 
            title_font_color='white',
            legend=dict(
        font=dict(color='white', size=14)     # 👈 THIS FIXES LEGEND
    )
        )

        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Ground-truth metrics removed per user request (precision/recall/F1 and confusion counts)
    
    st.markdown("---")
    st.markdown("### ⚖️ Risk Level Analysis")
    risk_levels = pd.cut(result['fraud_probability'], bins=[0, 0.2, 0.5, 0.8, 1.0], 
                         labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
    risk_counts = risk_levels.value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
    fig_risk = go.Figure(data=[go.Bar(x=risk_counts.index, y=risk_counts.values, 
                                     marker_color=['#4CAF50', '#FFC107', '#FF9800', '#F44336'])])
    fig_risk.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Number of Transactions",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # The detailed results table is shown on the Detect page; removed duplicate table here.


def show_explain():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; color: white;">
          Explain Flagged Transactions
        </h1>
           <p style="color: #B0BEC5; font-size: 1.05rem; line-height: 1.5; margin-bottom: 0.5rem;">
             Understand why transactions were flagged using SHAP explanations<br>
             Select flagged transaction to explain
           </p>
    </div>
    """, unsafe_allow_html=True)
    
    X = st.session_state.get('X')
    result = st.session_state.get('result')
    model = st.session_state.get('model')
    
    if X is None or result is None or model is None:
        st.info("Run detection in the Detect tab to enable explanations.")
        return
    
    fraud_idx = result[result['isFraud_pred'] == 1].index.tolist()
    if len(fraud_idx) == 0:
        st.info("No predicted fraud transactions to explain.")
        return
    
    # Select transaction — render selectbox inside a negative-margin container to pull it up
    st.markdown('<div id="explain-label-only" style="margin-top:-28px; margin-bottom:0px; padding-bottom:0px;">', unsafe_allow_html=True)
    labels = [result.loc[i, 'TransactionID'] if 'TransactionID' in result.columns else str(i) for i in fraud_idx]
    sel = st.selectbox("", labels, key='explain_select')
    selected_index = fraud_idx[labels.index(sel)]
    st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        import shap  # type: ignore
        shap_available = True
    except Exception as e:
        shap_available = False
        st.info("SHAP not installed. Install with: pip install shap to enable explanations.")
        return
    
    # Attempt transformed features if pipeline
    try:
        from sklearn.pipeline import Pipeline
        is_pipeline = isinstance(model, Pipeline)
    except Exception:
        is_pipeline = False
    
    try:
        if is_pipeline:
            pre = model[:-1]
            est = model[-1]
            try:
                X_trans = pre.transform(X)
            except Exception:
                X_trans = X.values
        else:
            est = model
            X_trans = X.values
    except Exception:
        X_trans = X.values
    
    if hasattr(X_trans, 'toarray'):
        X_trans = X_trans.toarray()
    
    try:
        feat_names = None;
        if is_pipeline and hasattr(pre, 'get_feature_names_out'):
            feat_names = list(pre.get_feature_names_out())
        elif hasattr(est, 'feature_names_in_'):
            feat_names = list(est.feature_names_in_)
        else:
            feat_names = list(X.columns)
    except Exception:
        feat_names = [f'f{i}' for i in range(X_trans.shape[1])]
    
    X_trans_df = pd.DataFrame(X_trans, index=X.index, columns=feat_names)
    
    try:
        expl = shap.Explainer(est, X_trans[: min(200, len(X_trans))])
    except Exception:
        try:
            expl = shap.KernelExplainer(lambda z: est.predict_proba(z), X_trans[: min(50, len(X_trans))])
        except Exception as e:
            st.error(f"Failed to initialize SHAP explainer: {e}")
            return
    
    try:
        row = X_trans_df.iloc[[selected_index]]
        out = expl(row.values)
        vals = out.values if hasattr(out, 'values') else out
        vals = np.asarray(vals)
        if vals.ndim == 2 and vals.shape[0] == 1:
            shap_vector = vals[0]
        elif vals.ndim == 1:
            shap_vector = vals
        else:
            shap_vector = vals.ravel()[: len(feat_names)]
        
        contrib = pd.Series(shap_vector, index=feat_names)
        top = contrib.abs().sort_values(ascending=False).head(12)
        plot_df = pd.DataFrame({'feature': top.index, 'shap_value': contrib.loc[top.index].values})
        plot_df['impact'] = np.where(plot_df['shap_value'] >= 0, '↑ increases fraud risk', '↓ decreases fraud risk')
        
        fig = px.bar(plot_df[::-1], x='shap_value', y='feature', color='impact', orientation='h',
                     color_discrete_map={'↑ increases fraud risk': '#F44336', '↓ decreases fraud risk': '#4CAF50'},
                     title='SHAP: Top contributing features')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                
                font=dict(color="white"),
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Provide a concise textual explanation listing the top contributing
        # features for the selected transaction instead of showing a table.
        st.markdown('### Top contributing features')
        top_feats = plot_df.copy()
        for i, r in top_feats.iterrows():
          feat = r['feature']
          shap_v = r['shap_value']
          impact = 'increases' if shap_v >= 0 else 'decreases'
          # omit raw feature value from the UI to keep explanations concise
          st.markdown(f"- **{feat}** — {impact} fraud risk (SHAP: {shap_v:.4f})")
    except Exception as e:
        st.error(f"Failed to compute SHAP values: {e}")


# Custom tab navigation
show_tab_bar()

# Render active tab content
if st.session_state.get('active_tab', 0) == 0:
    show_home()
elif st.session_state.get('active_tab', 0) == 1:
    show_detect()
elif st.session_state.get('active_tab', 0) == 2:
    show_visualize()
elif st.session_state.get('active_tab', 0) == 3:
    show_explain()

st.markdown('---')
st.markdown('<div class="footer">FraudShield Pro - Advanced Fraud Detection System | Powered by Machine Learning & AI</div>', unsafe_allow_html=True)