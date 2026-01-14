# =========================================================
# FINANCIAL STRESS INDEX & CREDIT PRICING SYSTEM
# WHY THIS APP EXISTS:
# Banks increase interest rates without explaining reasons.
# This app explains credit pricing in simple language
# using explainable AI and visual insights.
#
# BIGGEST BENEFIT:
# Clear explanation of loan pricing & improvement actions.
#
# LIMITATION:
# Decision-support only, not final bank approval.
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from fpdf import FPDF
import tempfile


# =========================================================
# SESSION STATE INITIALIZATION (ONE TIME)
# WHY:
# Persist AI explanation across Streamlit reruns
# =========================================================
if "ai_table" not in st.session_state:
    st.session_state.ai_table = pd.DataFrame(
        columns=["Bank / Risk Team View", "Customer View"]
    )

if "ai_generated" not in st.session_state:
    st.session_state.ai_generated = False

import re

def html_to_text_bullets(html):
    if not html:
        return ""

    text = html

    # Remove any <ul ...> or </ul>
    text = re.sub(r"<ul[^>]*>", "", text)
    text = text.replace("</ul>", "")

    # Replace list items with bullets
    text = re.sub(r"<li[^>]*>", "- ", text)
    text = text.replace("</li>", "\n")

    # Clean remaining HTML tags (safety)
    text = re.sub(r"<[^>]+>", "", text)

    return text.strip()



# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AI Credit Pricing System",
    layout="wide"
)

# =========================================================
# GLOBAL STYLING (BIG + BOLD DASHBOARD)
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 22px !important;
    font-weight: 700;
}
.main-title {
    background: linear-gradient(90deg, #8B0000, #006400);
    color: white;
    padding: 26px;
    border-radius: 14px;
    font-size: 44px;
    font-weight: 900;
    text-align: center;
}
.section-title {
    font-size: 32px;
    font-weight: 900;
    margin-top: 35px;
    border-left: 8px solid #00ff99;
    padding-left: 14px;
}
.sidebar-title {
    font-size: 26px;
    font-weight: 900;
}
.help-text {
    font-size: 18px;
    color: #cccccc;
}
.card {
    background-color: #111111;
    padding: 22px;
    border-radius: 12px;
    margin-top: 18px;
}
.kpi {
    font-size: 30px;
    font-weight: 900;
}
.kpi-label {
    font-size: 28px;
    font-weight: 900;
    color: #cccccc;
    letter-spacing: 0.5px;
}

.kpi-value {
    font-size: 46px;
    font-weight: 900;
    color: #ffffff;
    margin-top: 6px;
}

.kpi-card {
    background: linear-gradient(180deg, #0f0f0f, #141414);
    padding: 26px;
    border-radius: 16px;
    border: 2px solid #00ff99;
    text-align: left;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO TITLE
# =========================================================
st.markdown(
    '<div class="main-title">AI Financial Stress & Credit Pricing System</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div style="
    background: linear-gradient(180deg, #0f1a1a, #0b1111);
    border: 3px solid #00ff99;
    border-radius: 18px;
    padding: 32px;
    margin-top: 20px;
    margin-bottom: 30px;
">

<div style="
    font-size: 34px;
    font-weight: 900;
    margin-bottom: 20px;
">
Biggest Benefit
</div>

<div style="
    font-size: 26px;
    font-weight: 800;
    line-height: 1.6;
    margin-bottom: 28px;
">
Understand <b>why your interest rate is high or low</b> and
<b>what exactly you can improve</b> to reduce it.
</div>

<hr style="border:1px solid #00ff99; margin:24px 0;">

<div style="
    font-size: 34px;
    font-weight: 900;
    margin-bottom: 14px;
">
Limitation
</div>

<div style="
    font-size: 26px;
    font-weight: 800;
    line-height: 1.6;
">
This system <b>supports credit decisions</b>.
Final loan approval always depends on <b>bank policy</b>.
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================
MODEL_PATH = "models/risk_pricing_lgbm_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "annual_inc","emp_length","loan_amnt","term","dti",
    "revol_util","delinq_2yrs","inq_last_6mths",
    "FSI","dti_util_interaction","loan_income_ratio","stress_intensity"
]

# =========================================================
# SIDEBAR INPUTS (UPDATED: SLIDERS EXCEPT INCOME & LOAN)
# =========================================================
st.sidebar.markdown("<div class='sidebar-title'>Borrower Inputs</div>", unsafe_allow_html=True)

# --- Keep as NUMBER INPUT ---
annual_inc = st.sidebar.number_input(
    "Annual Income (₹)",
    min_value=0.0,
    value=60000.0,
    step=1000.0
)

# --- Keep as NUMBER INPUT ---
loan_amnt = st.sidebar.number_input(
    "Loan Amount (₹)",
    min_value=0.0,
    value=15000.0,
    step=500.0
)

# --- SLIDERS BELOW ---

emp_length = st.sidebar.slider(
    "Employment Length (Years)",
    min_value=0,
    max_value=40,
    value=5,
    step=1
)

term = st.sidebar.slider(
    "Loan Term (Months)",
    min_value=12,
    max_value=120,
    value=36,
    step=12
)

dti = st.sidebar.slider(
    "Debt-to-Income Ratio (%)",
    min_value=0.0,
    max_value=100.0,
    value=18.0,
    step=1.0
)

revol_util = st.sidebar.slider(
    "Credit Utilization (%)",
    min_value=0.0,
    max_value=100.0,
    value=45.0,
    step=1.0
)

delinq_2yrs = st.sidebar.slider(
    "Missed Payments (Last 2 Years)",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)

inq_last_6mths = st.sidebar.slider(
    "Credit Inquiries (6 Months)",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)


st.markdown("""
<div style="
    background: linear-gradient(180deg, #0f1a1a, #0b1111);
    border: 3px solid #00ff99;
    border-radius: 18px;
    padding: 28px;
    margin-top: 30px;
">

<div style="
    font-size: 36px;
    font-weight: 900;
    margin-bottom: 20px;
">
Why These Inputs Are Used
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.big-input-table table {
    width: 100%;
    border-collapse: collapse;
}
.big-input-table th {
    font-size: 26px;
    font-weight: 900;
    padding: 14px;
    text-align: left;
    border-bottom: 2px solid #00ff99;
}
.big-input-table td {
    font-size: 24px;
    font-weight: 700;
    padding: 14px;
    border-bottom: 1px solid #333333;
}
</style>
""", unsafe_allow_html=True)

input_explain_df = pd.DataFrame({
    "Input": [
        "Annual Income",
        "Employment Length",
        "Loan Amount",
        "Debt-to-Income Ratio",
        "Credit Utilization",
        "Missed Payments",
        "Credit Inquiries"
    ],
    "Reason": [
        "Higher income lowers default risk",
        "Stable job improves repayment confidence",
        "Higher exposure increases pricing risk",
        "Shows repayment burden",
        "High usage signals financial stress",
        "Past delays predict future risk",
        "Frequent checks indicate borrowing pressure"
    ]
})

st.markdown(
    '<div class="big-input-table">' +
    input_explain_df.to_html(index=False, escape=False) +
    '</div>',
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FINANCIAL STRESS INDEX
# =========================================================
fsi = (
    0.35*(revol_util/100) +
    0.25*(dti/50) +
    0.20*min(delinq_2yrs/5,1) +
    0.20*min(inq_last_6mths/5,1)
)
fsi = float(np.clip(fsi,0,1))

risk_level = "LOW" if fsi < 0.35 else "MEDIUM" if fsi < 0.65 else "HIGH"
decision = "APPROVED" if fsi < 0.65 else "MANUAL REVIEW"

# =========================================================
# FEATURE ENGINEERING
# =========================================================
input_df = pd.DataFrame([{
    "annual_inc": annual_inc,
    "emp_length": emp_length,
    "loan_amnt": loan_amnt,
    "term": term,
    "dti": dti,
    "revol_util": revol_util,
    "delinq_2yrs": delinq_2yrs,
    "inq_last_6mths": inq_last_6mths,
    "FSI": fsi,
    "dti_util_interaction": dti * revol_util,
    "loan_income_ratio": loan_amnt / (annual_inc + 1),
    "stress_intensity": fsi * dti
}])[FEATURES]

pred_rate = float(np.expm1(model.predict(input_df)[0]))

# =========================================================
# KPI DASHBOARD (HORIZONTAL)
# =========================================================
# =========================================================
# KPI DASHBOARD (BIG LABELS + BIG VALUES)
# =========================================================
st.markdown("<div class='section-title'>Credit Risk Summary</div>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Stress Score</div>
        <div class="kpi-value">{fsi:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Risk Level</div>
        <div class="kpi-value">{risk_level}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Interest Rate (%)</div>
        <div class="kpi-value">{pred_rate:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Decision</div>
        <div class="kpi-value">{decision}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# RAW BUSINESS FEATURES (FOR SHAP DISPLAY ONLY)
# =========================================================
RAW_FEATURES = [
    "annual_inc",
    "loan_amnt",
    "term",
    "dti",
    "revol_util",
    "delinq_2yrs",
    "inq_last_6mths",
    "FSI"
]

RAW_FEATURE_NAMES = {
    "annual_inc": "Annual Income",
    "loan_amnt": "Loan Amount",
    "term": "Loan Duration",
    "dti": "Debt-to-Income Ratio",
    "revol_util": "Credit Utilization",
    "delinq_2yrs": "Missed Payments",
    "inq_last_6mths": "Recent Credit Checks",
    "FSI": "Financial Stress Index"
}


# =========================================================
# SHAP EXPLANATION (RAW FEATURES ONLY – NATURAL VIEW)
# =========================================================
st.markdown("<div class='section-title'>Key Risk Drivers (Simple Explanation)</div>", unsafe_allow_html=True)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

raw_idx = [input_df.columns.get_loc(col) for col in RAW_FEATURES]

raw_shap_values = shap_values[0][raw_idx]
raw_feature_names = [RAW_FEATURE_NAMES[col] for col in RAW_FEATURES]

fig, ax = plt.subplots(figsize=(3, 2))  # half-size
shap.bar_plot(
    raw_shap_values,
    feature_names=raw_feature_names,
    show=False
)

st.pyplot(fig)


st.markdown("""
<div style="
    background: linear-gradient(180deg, #0f1a1a, #0b1111);
    border: 3px solid #00ff99;
    border-radius: 18px;
    padding: 28px;
    margin-top: 20px;
">

<div style="font-size: 34px; font-weight: 900; margin-bottom: 18px;">
How to read this chart
</div>

<div style="font-size: 26px; font-weight: 800; line-height: 1.6;">
• <span style="color:#ff4d6d;">Red bars (right side)</span> mean the factor is
<b>increasing the interest rate</b>.<br><br>

• <span style="color:#4da6ff;">Blue bars (left side)</span> mean the factor is
<b>reducing the interest rate</b>.
</div>

<hr style="border:1px solid #00ff99; margin:22px 0;">

<div style="font-size: 30px; font-weight: 900; margin-bottom: 10px;">
Important
</div>

<div style="font-size: 24px; font-weight: 800; line-height: 1.6;">
Some factors may look important even when their current value is low.<br><br>

This means the model has learned them as
<b>strong risk indicators overall</b>, not that they are high right now.
</div>

</div>
""", unsafe_allow_html=True)

def format_ai_bullets(text):
    lines = [l.strip().lstrip("*").strip() for l in text.split("\n") if l.strip()]
    html = "<ul style='padding-left:22px;'>"
    for l in lines:
        html += f"<li style='margin-bottom:10px;'>{l}</li>"
    html += "</ul>"
    return html

# =========================================================
# AI CREDIT ADVISOR (NO CONTRADICTION | FINAL)
# WHY:
# Explain model decision clearly for both bank & customer
# =========================================================

st.markdown("""
<div style="
    background: linear-gradient(180deg, #0f1a1a, #0b1111);
    border: 3px solid #00ff99;
    border-radius: 18px;
    padding: 32px;
    margin-top: 40px;
">

<div style="font-size:36px; font-weight:900; margin-bottom:22px;">
AI Credit Advisor
</div>
""", unsafe_allow_html=True)

ai_table = pd.DataFrame(columns=["Bank / Risk Team View", "Customer View"])
ai_error = False

if st.button("Generate AI Explanation"):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            ai_model = genai.GenerativeModel("gemini-2.5-flash-lite")

            prompt = f"""
You are explaining a CREDIT MODEL DECISION.

STRICT RULES:
- Do NOT contradict the model decision.
- Always align with approval status.
- Use ONLY the information given.
- Give AT LEAST 5 bullet points per section.

MODEL OUTPUT:
Stress Score: {fsi:.2f}
Risk Level: {risk_level}
Interest Rate: {pred_rate:.2f} %
Decision: {decision}

FORMAT STRICTLY LIKE THIS:

Bank Perspective:
- bullet
- bullet
- bullet
- bullet
- bullet

Customer Perspective:
- bullet
- bullet
- bullet
- bullet
- bullet

Language:
Very simple English.
Short bullets.
No theory.
No warnings if approved.
"""

            # -------- AI CALL --------
            response = ai_model.generate_content(prompt).text

            # -------- SPLIT & FORMAT --------
            raw_bank = response.split("Customer Perspective:")[0] \
                               .replace("Bank Perspective:", "") \
                               .strip()

            raw_customer = response.split("Customer Perspective:")[1].strip()

            bank = format_ai_bullets(raw_bank)
            customer = format_ai_bullets(raw_customer)

            st.session_state.ai_table.loc[0] = [bank, customer]
            st.session_state.ai_generated = True

            ai_error = False

        except Exception:
            ai_error = True

# -------- ERROR MESSAGE (ONLY WHEN LIMIT EXCEEDED) --------
if ai_error:
    st.markdown("""
    <div style="
        background:#2a1a1a;
        border:3px solid #ff4d4d;
        border-radius:14px;
        padding:22px;
        margin-top:20px;
        font-size:26px;
        font-weight:900;
        color:#ffcccc;
        text-align:center;
    ">
    Sorry, I’ve reached my AI usage limit.<br><br>
    Please try again after some time.
    </div>
    """, unsafe_allow_html=True)

# ----------------- TABLE STYLING -----------------
st.markdown("""
<style>
.ai-table table {
    width: 100%;
    border-collapse: collapse;
}
.ai-table th {
    font-size: 28px;
    font-weight: 900;
    padding: 16px;
    border-bottom: 2px solid #00ff99;
    text-align: left;
}
.ai-table td {
    font-size: 24px;
    font-weight: 800;
    padding: 16px;
    line-height: 1.8;
    border-bottom: 1px solid #333333;
}
</style>
""", unsafe_allow_html=True)

# ---------- SHOW AI TABLE IN APP ----------
if st.session_state.get("ai_generated", False) and len(st.session_state.ai_table) > 0:
    st.markdown(
        '<div class="ai-table">' +
        st.session_state.ai_table.to_html(index=False, escape=False) +
        '</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div style='font-size:24px; font-weight:800;'>Click above to generate AI explanation.</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PDF REPORT (FORMATTED FINTECH REPORT)
# WHY:
# Downloadable professional credit decision report
# =========================================================
if st.button("Download Full Credit Report (PDF)"):

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---------- TITLE ----------
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI CREDIT RISK & PRICING REPORT", ln=True)
    pdf.ln(6)

    # ---------- KPI SUMMARY ----------
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "1. Credit Risk Summary", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, f"""
Stress Score   : {fsi:.2f}
Risk Level     : {risk_level}
Interest Rate : {pred_rate:.2f} %
Decision       : {decision}
""")

    # ---------- DECISION REASON ----------
    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "2. Decision Reasoning", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, """
The credit decision is derived from the Financial Stress Index,
repayment stability, credit utilization, and recent credit behavior.
Lower stress and responsible credit usage support approval.
""")

    # ---------- INPUT SUMMARY ----------
    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "3. Borrower Input Summary", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, f"""
Annual Income      : {annual_inc}
Loan Amount        : {loan_amnt}
Loan Term          : {term} months
Debt-to-Income     : {dti} %
Credit Utilization : {revol_util} %
Missed Payments    : {delinq_2yrs}
Credit Inquiries   : {inq_last_6mths}
""")

    # ---------- AI CREDIT ADVISOR ----------
    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "4. AI Credit Advisor", ln=True)

    if st.session_state.get("ai_generated", False) and not st.session_state.ai_table.empty:

    # ---- BANK VIEW ----
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "Bank / Risk Team View", ln=True)
        pdf.set_font("Arial", size=11)

        bank_text = html_to_text_bullets(
            st.session_state.ai_table.iloc[0, 0]
        )

        pdf.multi_cell(0, 7, bank_text)

        # ---- CUSTOMER VIEW ----
        pdf.ln(2)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "Customer View", ln=True)
        pdf.set_font("Arial", size=11)

        customer_text = html_to_text_bullets(
            st.session_state.ai_table.iloc[0, 1]
        )

        pdf.multi_cell(0, 7, customer_text)

    else:
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(
            0,
            7,
            "AI explanation was not generated. Please generate AI explanation before downloading the report."
        )


    # ---------- FOOTER ----------
    pdf.ln(6)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0,
        6,
        "Developed by Anurag Kumar Singh\nData Scientist | AI/ML Engineer"
    )

    # ---------- SAVE ----------
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)

    with open(tmp.name, "rb") as f:
        st.download_button(
            "DOWNLOAD FORMATTED CREDIT REPORT (PDF)",
            f,
            file_name="AI_Credit_Risk_Report.pdf",
            mime="application/pdf"
        )
