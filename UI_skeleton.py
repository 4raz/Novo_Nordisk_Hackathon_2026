import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Force load the .env file FIRST (before importing agent.py)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Debug check: This will print your endpoint or 'None' to your terminal
print("DEBUG - Loading .env from:", env_path)
print("DEBUG - Endpoint value:", os.environ.get("AZURE_OPENAI_ENDPOINT"))

# 2. NOW import Streamlit and your modules
import streamlit as st
import pandas as pd
import time

from matcher import AlgorithmicMatcher
from agent import run_validation

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Novo Nordisk - AI Vendor Screening Engine",
    page_icon="🏥",
    layout="wide",
)

# --- CUSTOM CSS FOR ENTERPRISE LOOK ---
st.markdown(
    """
    <style>
    .main-header {font-size: 24px; font-weight: bold; color: #002060;}
    .card {background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef;}
    .metric-container {background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    </style>
""",
    unsafe_allow_html=True,
)


# --- RESOURCE CACHING ---
@st.cache_resource(show_spinner=False)
def load_matcher():
    """Caches the AlgorithmicMatcher so it doesn't reload the CSV and embeddings on every click."""
    csv_path = os.environ.get("VENDOR_MASTER_CSV", "novo_vendor_master.csv")
    if os.path.exists(csv_path):
        return AlgorithmicMatcher(master_db_path=csv_path, nrows=100_000)
    return None


matcher = load_matcher()

# --- SIDEBAR ---
# --- SIDEBAR ---
with st.sidebar:
    # Swapped SVG for a stable PNG render
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Novo_Nordisk_Logo.svg/512px-Novo_Nordisk_Logo.svg.png",
        use_container_width=True,
    )
    st.title("Governance Panel")
    st.markdown("---")

    if matcher is not None:
        st.metric("Vendor Master Records", f"{len(matcher.master_df):,} Active")
    else:
        st.metric("Vendor Master Records", "0 (CSV Not Found)")

    st.markdown("---")
    st.markdown("**Active Integrations**")
    st.markdown(
        "🏢 GLEIF (Corporate Hierarchy)<br>🛡️ OpenSanctions (UN Watchlist)",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Novo Nordisk GBS Data Hackathon 2026")

# --- MAIN TITLE ---
st.title("Intelligent Multi-Signal Vendor Screening Engine")
st.markdown(
    "Preventing vendor duplication and optimizing enterprise spend across global operations."
)

tab1, tab2 = st.tabs(["🔍 Single Vendor Onboarding", "📊 Vendor Master Governance"])

# --- TAB 1: SINGLE VENDOR SCREENING ---
with tab1:
    col_in, col_out = st.columns([1, 1.2], gap="large")

    with col_in:
        st.subheader("1. Enter Proposed Vendor Details")
        st.caption("Fill in the fields provided by the business unit request.")

        with st.form("vendor_onboarding_form"):
            v_name = st.text_input(
                "Proposed Vendor Name", placeholder="e.g., EdgeVerve Systems Limited"
            )
            v_country = st.selectbox(
                "Operating Country",
                [
                    "Argentina",
                    "Australia",
                    "Austria",
                    "Belgium",
                    "Brazil",
                    "Canada",
                    "China",
                    "Denmark",
                    "Finland",
                    "France",
                    "Germany",
                    "India",
                    "Ireland",
                    "Italy",
                    "Japan",
                    "Mexico",
                    "Netherlands",
                    "Norway",
                    "Poland",
                    "Singapore",
                    "South Africa",
                    "South Korea",
                    "Spain",
                    "Sweden",
                    "Switzerland",
                    "United Arab Emirates",
                    "United Kingdom",
                    "United States",
                ],
            )
            v_domain = st.text_input(
                "Website / Domain", placeholder="e.g., edgeverve.com"
            )

            submit_btn = st.form_submit_button(
                "Run Autonomous Screening", use_container_width=True
            )

    with col_out:
        st.subheader("2. Screening Pipeline & Evidence")

        if submit_btn and v_name:
            vendor_payload = {"name": v_name, "website": v_domain, "country": v_country}

            with st.status("Executing Pipeline...", expanded=True) as status:
                st.write("🧹 **Step 1: Normalizing Data**...")
                time.sleep(0.2)

                st.write("🔍 **Step 2 & 3: Semantic & Fuzzy Candidate Matching**...")
                match_result = None
                if matcher:
                    candidates = matcher.find_candidates(vendor_payload)
                    match_result = candidates[0] if candidates else None

                st.write("🌐 **Step 4: Agentic Validation (GLEIF & UN Sanctions)**...")
                outcome = run_validation(vendor_payload, match_result)

                st.write("📊 **Step 5: Synthesizing Recommendation**...")
                status.update(
                    label="Screening Complete!", state="complete", expanded=False
                )

            # --- RENDER ACTUAL RESULTS ---
            decision = outcome.get("decision", "MANUAL_REVIEW")
            justification = outcome.get(
                "business_justification", "No justification provided."
            )

            # Format UI Box based on decision
            if decision == "AUTO_REJECT":
                st.error(f"🚨 **DECISION BAND: {decision}**")
                border_color = "red"
            elif decision == "MANUAL_REVIEW":
                st.warning(f"⚠️ **DECISION BAND: {decision}**")
                border_color = "#ffc107"
            else:
                st.success(f"✅ **DECISION BAND: {decision}**")
                border_color = "green"

            # Display Justification
            st.info(f"**Agent Reasoner:** {justification}")

            # Show Safety Net Notes if any triggered
            if outcome.get("safety_net_notes"):
                for note in outcome["safety_net_notes"]:
                    st.error(f"🛡️ **Safety Net Trigger:** {note}")

            # Top Existing Candidate Match Box
            if match_result:
                st.markdown(
                    f"""
                <div class="card" style="border-left: 5px solid {border_color}; margin-bottom: 20px;">
                    <h4>Top Existing Candidate Match Found:</h4>
                    <p><b>Vendor ID:</b> {match_result.get('vendor_id', 'N/A')}<br>
                    <b>Existing Legal Entity:</b> {match_result.get('name', 'N/A')}<br>
                    <b>Existing Domain:</b> {match_result.get('website', 'N/A')} | <b>Country:</b> {match_result.get('country_code', 'N/A')}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="card" style="border-left: 5px solid {border_color}; margin-bottom: 20px;">
                    <h4>No Internal Match Found</h4>
                    <p>No vendor closely matching these details was found in the master database.</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Signal Breakdown Display
            # Signal Breakdown Display
            st.markdown("### Signal Breakdown")

            # Create a 2x2 grid instead of a 1x4 row to prevent truncation
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)

            # Populate metrics dynamically and force rounding to 1 decimal place
            fuzzy = match_result.get("fuzzy_score", 0.0) if match_result else 0.0
            semantic = match_result.get("semantic_score", 0.0) if match_result else 0.0
            comp_score = outcome.get("similarity_score", 0.0)

            sanctions_hit = outcome.get("sanctions_hit")
            sanctions_display = (
                "Flagged"
                if sanctions_hit
                else ("Clear" if sanctions_hit is False else "Unverified")
            )
            s_color = "inverse" if sanctions_hit else "normal"

            hierarchy = outcome.get("hierarchy_relationship_found")
            hierarchy_display = (
                "Found" if hierarchy else ("None" if hierarchy is False else "N/A")
            )

            # Row 1
            m1.metric(
                "Composite Score",
                f"{comp_score:.1f}%",
                delta=f"Fuzzy: {fuzzy:.1f}%",
                delta_color="off",
            )
            m2.metric("Semantic AI", f"{semantic:.1f}%", delta_color="off")

            # Row 2
            m3.metric(
                "UN Sanctions",
                sanctions_display,
                delta="Risk Check",
                delta_color=s_color,
            )
            m4.metric(
                "GLEIF Check",
                hierarchy_display,
                delta="Parent/Sub",
                delta_color="off",
            )

            if outcome.get("related_parent_entity"):
                st.caption(
                    f"**Discovered Parent Entity:** {outcome.get('related_parent_entity')}"
                )

            # Dynamic Actions
            st.markdown("---")
            act_col1, act_col2 = st.columns(2)
            if decision == "AUTO_APPROVE":
                act_col1.button(
                    "➕ Proceed to Onboard New Vendor",
                    type="primary",
                    use_container_width=True,
                )
            elif decision == "MANUAL_REVIEW":
                if match_result:
                    act_col1.button(
                        f"✅ Consolidate to {match_result.get('vendor_id')}",
                        type="primary",
                        use_container_width=True,
                    )
                act_col2.button("📩 Route to Data Steward", use_container_width=True)
            elif decision == "AUTO_REJECT":
                act_col1.button(
                    "🔒 Acknowledge & Block Request",
                    type="primary",
                    use_container_width=True,
                )

        elif submit_btn:
            st.error("Please fill in at least the Vendor Name to run screening.")

# --- TAB 2: GOVERNANCE DASHBOARD ---
with tab2:
    st.subheader("Master Vendor Database Overview")
    g1, g2, g3 = st.columns(3)

    active_count = len(matcher.master_df) if matcher else 184
    g1.metric("Total Active Vendors", f"{active_count:,}", "+12 this month")
    g2.metric("Duplicates Intercepted", "29", "100% manual effort saved")
    g3.metric("Spend Consolidated", "$4.2M USD", "Negotiation power preserved")

    st.markdown("---")
    # Display the actual loaded dataframe if available
    if matcher is not None:
        st.dataframe(matcher.master_df.head(50), use_container_width=True)
    else:
        st.warning("Master CSV not loaded. Showing empty dashboard.")
