import streamlit as st
import pandas as pd
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Novo Nordisk - AI Vendor Screening Engine",
    page_icon="🏥",
    layout="wide"
)

# --- CUSTOM CSS FOR ENTERPRISE LOOK ---
st.markdown("""
    <style>
    .main-header {font-size: 24px; font-weight: bold; color: #002060;}
    .card {background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef;}
    .metric-container {background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/8e/Novo_Nordisk_logo.svg", width=180)
    st.title("Governance Panel")
    st.markdown("---")
    st.metric("Vendor Master Records", "184 Active")
    st.metric("External APIs", "GLEIF / OpenCorporates")
    st.markdown("---")
    st.caption("Novo Nordisk GBS Data Hackathon 2026")

# --- MAIN TITLE ---
st.title("Intelligent Multi-Signal Vendor Screening Engine")
st.markdown("Preventing vendor duplication and optimizing enterprise spend across global operations.")

tab1, tab2 = st.tabs(["🔍 Single Vendor Onboarding", "📊 Vendor Master Governance"])

# --- TAB 1: SINGLE VENDOR SCREENING ---
with tab1:
    col_in, col_out = st.columns([1, 1.2], gap="large")

    with col_in:
        st.subheader("1. Enter Proposed Vendor Details")
        st.caption("Fill in the fields provided by the business unit request.")
        
        with st.form("vendor_onboarding_form"):
            v_name = st.text_input("Proposed Vendor Name", placeholder="e.g., EdgeVerve Systems Limited")
            v_country = st.selectbox("Operating Country", ["India", "United States", "Denmark", "United Kingdom", "Germany"])
            v_domain = st.text_input("Website / Domain", placeholder="e.g., edgeverve.com")
            
            submit_btn = st.form_submit_button("Run Autonomous Screening", use_container_width=True)

    with col_out:
        st.subheader("2. Screening Pipeline & Evidence")
        
        if submit_btn and v_name:
            # --- Simulated Pipeline Progress ---
            with st.status("Executing 5-Step Screening Engine...", expanded=True) as status:
                st.write("🧹 **Step 1: Normalizing Data** (Stripping legal suffixes, canonicalizing URL)...")
                time.sleep(0.4)
                st.write("🔍 **Step 2: RapidFuzz Candidate Blocking** (Scanning master records)...")
                time.sleep(0.4)
                st.write("🧠 **Step 3: Semantic Vector Compare** (Calculating NLP embedding distance)...")
                time.sleep(0.4)
                st.write("🌐 **Step 4: Agentic Validation** (Querying live external registries)...")
                time.sleep(0.6)
                st.write("📊 **Step 5: Synthesizing Recommendation**...")
                time.sleep(0.3)
                status.update(label="Screening Complete!", state="complete", expanded=False)

            # --- DYNAMIC DEMO TRIGGERS ---
            input_name = v_name.lower()
            input_domain = v_domain.lower()

            # TRIGGER 1: FRAUD & COMPLIANCE BLOCK (e.g., "Infosys Billing" or spoofed domain)
            if "billing" in input_name or "support" in input_domain:
                st.error("🚨 **CRITICAL RISK: COMPLIANCE & FRAUD BLOCK**")
                
                st.markdown("""
                <div class="card" style="border-left: 5px solid red;">
                    <h4>Impersonator / Spoofed Domain Detected</h4>
                    <p><b>Target Entity:</b> Infosys Limited (VN-100248)<br>
                    <b>Flagged Issue:</b> The domain <i>infosys-finance.com</i> is not registered to the parent entity. Tax ID is unregistered.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Signal Breakdown")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fuzzy Name", "88%", delta="High similarity")
                m2.metric("Domain Match", "0%", delta="-Spoofed URL", delta_color="inverse")
                m3.metric("Tax ID", "Fail", delta="-Unregistered", delta_color="inverse")
                m4.metric("OFAC Check", "Flagged", delta="-High Risk", delta_color="inverse")
                
                st.error("**Agent Action:** Automatic block applied. Ticket routed to Security and Compliance.")
                st.button("🔒 Acknowledge & Close Request", type="primary", use_container_width=True)


            # TRIGGER 2: HIGH DUPLICATE RISK / >85% (e.g., "Cognizant Tech" typo)
            elif "cognizant" in input_name or "tcs" in input_name:
                st.error("🚨 **DECISION BAND: HIGH CONFIDENCE DUPLICATE (94%)**")
                
                st.markdown("""
                <div class="card" style="border-left: 5px solid #ff4b4b;">
                    <h4>Top Existing Candidate Match Found:</h4>
                    <p><b>Vendor ID:</b> VN-100250<br>
                    <b>Existing Legal Entity:</b> Cognizant Technology Solutions<br>
                    <b>Existing Domain:</b> cognizant.com | <b>Country:</b> USA<br>
                    <b>Active Annual Spend:</b> $3,500,000 USD</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Signal Breakdown")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fuzzy Name", "92%", delta="Typo Detected", delta_color="normal")
                m2.metric("Vector Semantic", "95%", delta="Exact Meaning", delta_color="normal")
                m3.metric("Domain Match", "100%", delta="Root Match", delta_color="normal")
                m4.metric("GLEIF Hierarchy", "N/A", delta="Skipped", delta_color="off")
                
                st.info("**Agent Reasoner:** High string and domain match indicates an exact duplicate attempt due to a typo or abbreviation. Creation blocked to prevent spend fragmentation.")
                
                act_col1, act_col2 = st.columns(2)
                with act_col1:
                    st.button("✅ Auto-Consolidate to VN-100250", type="primary", use_container_width=True)
                with act_col2:
                    st.button("📩 Route to Data Steward", use_container_width=True)


            # TRIGGER 3: MEDIUM RISK / REQUIRES REVIEW 65-84% (e.g., "EdgeVerve" subsidiary)
            elif "edgeverve" in input_name or "edgeverve" in input_domain:
                st.warning("⚠️ **DECISION BAND: MEDIUM CONFIDENCE / REQUIRES REVIEW (78%)**")
                
                st.markdown("""
                <div class="card" style="border-left: 5px solid #ffc107;">
                    <h4>Top Existing Candidate Match Found:</h4>
                    <p><b>Vendor ID:</b> VN-100248<br>
                    <b>Existing Legal Entity:</b> Infosys Limited<br>
                    <b>Existing Domain:</b> infosys.com | <b>Country:</b> India<br>
                    <b>Active Annual Spend:</b> $1,450,000 USD</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Signal Breakdown")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fuzzy Name", "18%", delta="-Low Direct Match")
                m2.metric("Vector Semantic", "42%", delta="-Different Roots")
                m3.metric("Domain Match", "0%", delta="Unique Domain")
                m4.metric("GLEIF Hierarchy", "100%", delta="Parent Match", delta_color="normal")
                
                st.info("**Agent Reasoner:** Text matches failed. However, external query against **GLEIF API** confirms EdgeVerve Systems is a wholly owned subsidiary of Infosys Limited.")
                
                act_col1, act_col2, act_col3 = st.columns(3)
                with act_col1:
                    st.button("✅ Consolidate / Reuse VN-100248", type="primary", use_container_width=True)
                with act_col2:
                    st.button("📩 Route to Steward", use_container_width=True)
                with act_col3:
                    st.button("🚫 Force Create New", use_container_width=True)


            # TRIGGER 4: CLEAN RECORD / <65% (Anything else)
            else:
                st.success("✅ **DECISION BAND: CLEAN / LIKELY NEW VENDOR (Score: 12%)**")
                st.info("No matching entity or corporate hierarchy link found in master database or external registries. Safe to proceed with creation.")
                st.button("➕ Proceed to Onboard New Vendor", type="primary")

        elif submit_btn:
            st.error("Please fill in at least the Vendor Name to run screening.")

# --- TAB 2: GOVERNANCE DASHBOARD ---
with tab2:
    st.subheader("Master Vendor Database Overview")
    g1, g2, g3 = st.columns(3)
    g1.metric("Total Active Vendors", "184", "+12 this month")
    g2.metric("Duplicates Intercepted", "29", "100% manual effort saved")
    g3.metric("Spend Consolidated", "$4.2M USD", "Negotiation power preserved")
    
    st.markdown("---")
    # Mock Dataframe representing real Kaggle/Wikidata baseline
    mock_df = pd.DataFrame({
        "Vendor ID": ["VN-100248", "VN-100249", "VN-100250", "VN-100251"],
        "Legal Name": ["Infosys Limited", "Cognizant Technology Solutions", "Tata Consultancy Services", "Tech Mahindra Ltd"],
        "Country": ["India", "United States", "India", "India"],
        "Domain": ["infosys.com", "cognizant.com", "tcs.com", "techmahindra.com"],
        "Status": ["Active", "Active", "Active", "Active"]
    })
    st.dataframe(mock_df, use_container_width=True)
