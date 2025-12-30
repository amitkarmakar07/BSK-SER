import streamlit as st
import pandas as pd
import sys
import os
import re

# Set base directory as parent of frontend directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Add the parent directory to the path so we can import from backend
sys.path.append(BASE_DIR)

from backend.inference.district import get_top_services_for_district_from_csv
from backend.inference.content import find_similar_services_from_csv
from backend.inference.demo import recommend_services_2  # Demographic recommendations function

# Load under-18 services
@st.cache_data
def load_under_18_services():
    """Load under-18 eligible services from CSV."""
    try:
        csv_path = os.path.join(DATA_DIR, "under18_top_services.csv")
        df = pd.read_csv(csv_path, encoding='latin-1')
        # Handle CSVs with or without service_id column
        if 'service_id' in df.columns:
            df = df[['service_id', 'service_name']].drop_duplicates()
        else:
            # CSV only has service_name, create a simple DataFrame
            df = df[['service_name']].drop_duplicates()
        return df
    except Exception as e:
        st.warning(f"Could not load under18_top_services.csv: {e}")
        return pd.DataFrame(columns=['service_name'])

def normalize_service_name(name):
    """Normalize service names for comparison."""
    if not isinstance(name, str):
        return ""
    normalized = name.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.replace('-', ' ').replace('_', ' ')
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def filter_recommendations_for_under_18(recommendations, under_18_services_df):
    """Filter recommendations to only include services eligible for under-18 users."""
    if under_18_services_df.empty:
        return recommendations
    
    # Get list of eligible service names (normalized)
    eligible_service_names = set(under_18_services_df['service_name'].apply(normalize_service_name))
    
    # Filter recommendations - only list type for Streamlit
    if isinstance(recommendations, list):
        filtered = []
        for rec in recommendations:
            if isinstance(rec, str):
                rec_normalized = normalize_service_name(rec)
                if rec_normalized in eligible_service_names:
                    filtered.append(rec)
        return filtered
    
    return recommendations


# Load CSV files with absolute paths
grouped_df = pd.read_csv(os.path.join(DATA_DIR, "grouped_df.csv"), encoding="latin-1")
service_df = pd.read_csv(os.path.join(DATA_DIR, "services.csv"), encoding="latin-1")
final_df = pd.read_csv(os.path.join(DATA_DIR, "final_df.csv"), encoding="latin-1")

# Load cluster_service_map from pickle
import pickle
with open(os.path.join(DATA_DIR, "cluster_service_map.pkl"), "rb") as f:
    cluster_service_map = pickle.load(f)

# Build service_id_to_name mapping
df_service_names = pd.read_csv(os.path.join(DATA_DIR, "service_id_with_name.csv"), encoding="latin-1")
service_id_to_name = dict(zip(df_service_names['service_id'], df_service_names['service_name']))

# Load CSV files instead of using database
@st.cache_data
def load_citizen_master():
    file_path = os.path.join(DATA_DIR, "ml_citizen_master.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, encoding="latin-1")
    else:
        # Return empty DataFrame if file doesn't exist
        return pd.DataFrame()

@st.cache_data
def load_provision_data():
    file_path = os.path.join(DATA_DIR, "ml_provision.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, encoding="latin-1")
    else:
        # Return empty DataFrame if file doesn't exist
        return pd.DataFrame()

@st.cache_data
def get_citizen_details(citizen_id):
    citizen_master = load_citizen_master()
    
    # Handle case when citizen_master is empty
    if citizen_master.empty:
        return pd.DataFrame()
    
    df = citizen_master[citizen_master['citizen_id'] == citizen_id]
    return df

@st.cache_data
def get_services_used(citizen_id):
    provision_data = load_provision_data()
    
    # Handle case when provision_data is empty
    if provision_data.empty:
        return pd.DataFrame(columns=['customer_id', 'customer_name', 'service_id', 'service_name', 'prov_date', 'docket_no'])
    
    df = provision_data[provision_data['customer_id'] == citizen_id]
    # Rename columns to match expected format
    df = df.rename(columns={
        'customer_id': 'customer_id',
        'customer_name': 'customer_name', 
        'service_id': 'service_id',
        'service_name': 'service_name',
        'prov_date': 'prov_date',
        'docket_no': 'docket_no'
    })
    if not df.empty:
        df['service_id'] = df['service_id'].astype(int)
        return df.sort_values('prov_date', ascending=False)
    return df

@st.cache_data
def preprocess_data(citizen_data, provision_data):
    merged_df = pd.merge(citizen_data, provision_data, left_on='citizen_id', right_on='customer_id', how='inner')
    clean_merged_df = merged_df.dropna(subset=merged_df.columns)
    clean_df = clean_merged_df[['citizen_id', 'district_id', 'sub_div_id', 'gp_id', 'gender', 'dob',
       'age', 'caste', 'religion','service_id']]
    
    # Assuming your DataFrame is called clean_df
    # Step 1: Create a one-hot encoded DataFrame for service_id
    service_ohe = pd.get_dummies(clean_df['service_id'], prefix='service')

    # Step 2: Concatenate one-hot columns with original DataFrame
    clean_ohe = pd.concat([clean_df[['citizen_id']], service_ohe], axis=1)

    # Step 3: Group by citizen_id and sum to aggregate service flags
    service_agg = clean_ohe.groupby('citizen_id').sum().reset_index()

    # Step 4: Get unique citizen attributes (since they are identical per citizen_id)
    citizen_info = clean_df.drop_duplicates(subset='citizen_id').drop(columns=['service_id'])

    # Step 5: Merge one-hot service matrix with citizen attributes
    final_df = pd.merge(citizen_info, service_agg, on='citizen_id')

    # Find the citizen with the maximum number of unique services used
    service_columns = [col for col in final_df.columns if col.startswith('service_')]
    final_df['unique_services_used'] = final_df[service_columns].gt(0).sum(axis=1)
    max_services = final_df['unique_services_used'].max()
    top_citizens = final_df.loc[final_df['unique_services_used'] == max_services, 'citizen_id']
    print("Citizen(s) with maximum unique services used:", top_citizens.tolist())
    print("Number of unique services used:", max_services)

    # Add a column for the total number of services used (sum of all service columns)
    service_columns = [col for col in final_df.columns if col.startswith('service_')]
    final_df['total_services_used'] = final_df[service_columns].sum(axis=1)
    final_df[['citizen_id', 'total_services_used']].head()

    # Count 1 for any nonzero value in each service column (i.e., unique services used)
    service_columns = [col for col in final_df.columns if col.startswith('service_')]
    final_df['unique_services_used'] = final_df[service_columns].gt(0).sum(axis=1)

    df=final_df.copy()

    bins = [0, 18, 35, 60, 200]
    labels = ['child', 'youth', 'adult', 'senior']
    
    # Handle missing or invalid age values
    def assign_age_group(age):
        if pd.isna(age) or age is None or age <= 0:
            return 'adult'  # Default to adult for missing/invalid ages
        return pd.cut([age], bins=bins, labels=labels, right=False)[0]
    
    df['age_group'] = df['age'].apply(assign_age_group)
    
    # Assign 'minority' to all religions except 'Hindu', handle None/NaN values
    def assign_religion_group(religion):
        if pd.isna(religion) or religion is None or religion == '':
            return 'Minority'  # Default to Minority for missing values
        return 'Hindu' if religion == 'Hindu' else 'Minority'
    
    df['religion_group'] = df['religion'].apply(assign_religion_group)
    
    df.drop(columns=['age','dob','sub_div_id','gp_id','religion'], inplace=True)

    return df

DISTRICT_CSV_PATH = os.path.join(DATA_DIR, "district_top_services.csv")


# ==========================================
# --- UI & APP LOGIC START ---
# ==========================================

st.set_page_config(
    page_title="Bangla Sahayata Kendra Service Recommendation",
    page_icon="🏛️",
    layout="wide",
)

st.title("Bangla Sahayata Kendra Service Recommendation")

# --- Helper Function for Phone Search ---
def get_citizen_ids_by_phone(phone):
    citizen_master = load_citizen_master()
    if citizen_master.empty:
        st.error("⚠️ Citizen master data not available.")
        return pd.DataFrame()
    
    phone_columns = ['citizen_phone', 'phone', 'mobile']
    phone_col = None
    for col in phone_columns:
        if col in citizen_master.columns:
            phone_col = col
            break
    
    if phone_col is None:
        return pd.DataFrame()
    
    try:
        phone_int = int(phone)
        df = citizen_master[citizen_master[phone_col] == phone_int]
    except ValueError:
        df = citizen_master[citizen_master[phone_col].astype(str) == phone]
    
    if df.empty:
        st.warning(f"No registered citizen found for: {phone}")
    
    return df

# --- MAIN APP LAYOUT ---

st.header("Search & Identification")

# Updated Tab Names
tab1, tab2 = st.tabs(["📱 Phone Number Search (Pre-Existed Customer)", "📝 Manual Entry (New customer)"])

# --- TAB 1: PHONE SEARCH ---
with tab1:
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        phone_input = st.text_input("Enter Registered Phone Number", placeholder="e.g., 9800361474", key="phone_input_field")
        st.caption("Try: 9800361474, 8293058992, 9845120211")
    with col_p2:
        st.write("##") # Spacer
        search_phone_btn = st.button("Search Profile", key="btn_phone_search")

    # Session State Management for Search
    if search_phone_btn:
        st.session_state['phone_search_active'] = True
        st.session_state['current_phone'] = phone_input
    
    # Reset if input changes
    if 'current_phone' in st.session_state and phone_input != st.session_state['current_phone']:
            st.session_state['phone_search_active'] = False

    if st.session_state.get('phone_search_active') and st.session_state.get('current_phone'):
        phone = st.session_state['current_phone']
        citizens_df = get_citizen_ids_by_phone(phone)
        
        if not citizens_df.empty:
            for idx, citizen_row in citizens_df.iterrows():
                citizen_id = citizen_row['citizen_id']
                
                # --- Citizen Profile Card ---
                st.markdown("---")
                st.subheader(f"👤 Profile: Citizen `{citizen_id}`")
                
                info_cols = st.columns(4)
                
                # Safe data retrieval
                name_val = citizen_row.get('citizen_name', '-')
                masked_name = '####' if isinstance(name_val, str) and name_val.strip() else '--'
                
                age_val = citizen_row.get('age', '-')
                age_display = str(age_val) if (not pd.isna(age_val) and age_val != 0) else '--'

                with info_cols[0]: st.info(f"**Name:** {masked_name}")
                with info_cols[1]: st.info(f"**Age:** {age_display}")
                with info_cols[2]: st.info(f"**Gender:** {citizen_row.get('gender','-')}")
                with info_cols[3]: st.info(f"**Caste:** {citizen_row.get('caste','-')}")

                # --- History Section ---
                st.write("##")
                st.markdown("#### 📜 Service History")
                services_df = get_services_used(citizen_id)
                
                used_service_ids = []
                if not services_df.empty:
                    # Count unique services
                    service_counts = services_df.groupby(['service_id', 'service_name']).size().reset_index(name='count')
                    # Filter birth/death
                    service_counts = service_counts[~service_counts['service_name'].str.lower().str.contains('birth|death', na=False)]
                    service_counts = service_counts.sort_values(by='count', ascending=False)
                    
                    used_service_ids = service_counts['service_id'].unique().tolist()
                    
                    st.dataframe(
                        service_counts.rename(columns={'service_name': 'Service Name', 'count': 'Times Used'})[['Service Name', 'Times Used']],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No previous service history found.")

                # --- Recommendation Selector ---
                st.write("##")
                st.markdown("#### 🎯 Get Recommendations")
                
                service_master_df = pd.read_csv(os.path.join(DATA_DIR, "services.csv"), encoding="utf-8")
                service_master_df = service_master_df[~service_master_df['service_name'].str.lower().str.contains('birth|death', na=False)]
                
                service_options = [f"{row['service_id']} - {row['service_name']}" for _, row in service_master_df.iterrows()]
                
                sel_col1, sel_col2 = st.columns([3, 1])
                with sel_col1:
                    selected_service_str = st.selectbox(f"Current Service - What are they here for?", ["None"] + service_options, key=f"svc_sel_{citizen_id}")
                
                selected_service_id = None
                if selected_service_str != "None":
                        selected_service_id = int(selected_service_str.split(" - ")[0])

                with sel_col2:
                        st.write("##")
                        generate_recs = st.button("Generate Recommendations", key=f"btn_rec_{citizen_id}")

                # --- GENERATE RECOMMENDATIONS LOGIC (Phone Mode) ---
                if generate_recs:
                    with st.spinner("Analyzing profile and generating insights..."):
                        district_id = int(citizen_row["district_id"])
                        user_caste = citizen_row.get('caste', None)
                        
                        # Prepare IDs for logic
                        item_service_ids = list(used_service_ids)
                        if selected_service_id and selected_service_id not in item_service_ids:
                            item_service_ids.append(selected_service_id)

                        # Recs per service logic
                        max_total_recs = 5
                        recs_per_service = {}
                        if item_service_ids:
                            n_services = len(item_service_ids)
                            if selected_service_id:
                                recs_per_service[selected_service_id] = min(3, max_total_recs)
                                remaining = max_total_recs - recs_per_service[selected_service_id]
                                others = [s for s in item_service_ids if s != selected_service_id]
                                if others:
                                    base = remaining // len(others)
                                    extra = remaining % len(others)
                                    for i, s in enumerate(others): recs_per_service[s] = base + (1 if i < extra else 0)
                            else:
                                base = max_total_recs // n_services
                                extra = max_total_recs % n_services
                                for i, s in enumerate(item_service_ids): recs_per_service[s] = base + (1 if i < extra else 0)

                        # --- RENDER RESULTS SECTION (Phone) ---
                        st.markdown("---")
                        st.subheader("🚀 Personalized Recommendations")
                        
                        res_c1, res_c2, res_c3 = st.columns(3)
                        
                        # Block Function
                        def block_service(service, caste=None):
                            if not isinstance(service, str): return False
                            s = service.lower()
                            if "birth" in s or "death" in s: return False
                            if caste and caste.lower() == "general" and "caste" in s: return False
                            return True

                        # 1. District
                        with res_c1:
                            st.info("🏢 **District Trends**")
                            top_dist = get_top_services_for_district_from_csv(DISTRICT_CSV_PATH, district_id, top_n=5)
                            top_dist = [s for s in top_dist if block_service(s, user_caste)][:5]
                            if top_dist:
                                for s in top_dist: st.markdown(f"- {s}")
                            else:
                                st.write("No data available.")

                        # 2. Demographic
                        with res_c2:
                            st.success("👥 **Demographic Match**")
                            try:
                                searched_name = None
                                if selected_service_id:
                                    try:
                                            searched_name = service_master_df[service_master_df['service_id'] == selected_service_id]['service_name'].iloc[0]
                                    except: pass
                                
                                demo_recs = recommend_services_2(
                                    citizen_id=citizen_id,
                                    df=final_df,
                                    grouped_df=grouped_df,
                                    cluster_service_map=cluster_service_map,
                                    service_id_to_name=service_id_to_name,
                                    service_df=service_df,
                                    top_n=5,
                                    citizen_master=get_citizen_details(citizen_id),
                                    searched_service_name=searched_name
                                )
                                demo_recs = [s for s in demo_recs if isinstance(s, str) and block_service(s, user_caste)]
                                if demo_recs:
                                        for s in demo_recs: st.markdown(f"- {s}")
                                else:
                                    st.write("No data available.")
                            except Exception as e:
                                st.write("Grouping error.")

                        # 3. Content
                        with res_c3:
                            st.warning("🔄 **Similar Services**")
                            if item_service_ids:
                                data_file = os.path.join(DATA_DIR, "service_with_domains.csv")
                                sim_file = os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
                                found_any = False
                                for sid in item_service_ids:
                                    n = recs_per_service.get(sid, 0)
                                    if n > 0:
                                        try:
                                            sims = find_similar_services_from_csv(data_file, sim_file, int(sid), n)
                                            sims = [s for s in sims if isinstance(s,str) and block_service(s, user_caste)]
                                            if sims:
                                                found_any = True
                                                # Try get name
                                                s_name = str(sid)
                                                try: s_name = service_master_df[service_master_df['service_id']==sid]['service_name'].iloc[0]
                                                except: pass
                                                st.markdown(f"**Because you used: {s_name}**")
                                                for s in sims: st.markdown(f"- {s}")
                                        except: pass
                                if not found_any: st.write("No similar services found.")
                            else:
                                st.write("Select a service or use history for this.")


# --- TAB 2: MANUAL ENTRY ---
with tab2:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with c3:
        caste = st.selectbox("Caste", ["General", "SC", "ST", "OBC-A", "OBC-B"])
        
    c4, c5 = st.columns(2)
    with c4:
            # District Selection with DEDUPLICATION FIX
        district_df = pd.read_csv(DISTRICT_CSV_PATH, encoding="utf-8")
        district_names = sorted(district_df['district_name'].dropna().unique().tolist())
        selected_district_name = st.selectbox("District", district_names)
        district_id = int(district_df[district_df['district_name'] == selected_district_name]['district_id'].iloc[0])
    with c5:
        religions = ["Hindu", "Muslim", "Christian", "Sikh", "Other"]
        selected_religion = st.selectbox("Religion", religions)
    
    # Service Intent
    service_master_df = pd.read_csv(os.path.join(DATA_DIR, "services.csv"), encoding="utf-8")
    service_master_df = service_master_df[~service_master_df['service_name'].str.lower().str.contains('birth|death', na=False)]
    service_options = [f"{row['service_id']} - {row['service_name']}" for _, row in service_master_df.iterrows()]
    
    selected_service_str_man = st.selectbox("Current Service - What are they here for?", ["None"] + service_options)
    
    st.write("##")
    manual_submit = st.button("Generate Recommendations", key="btn_manual")

    if manual_submit:
        with st.spinner("Processing demographics..."):
            # Groups
            age_group = 'child' if age < 18 else ('youth' if age < 60 else 'elderly')
            religion_group = 'Hindu' if selected_religion == 'Hindu' else 'Minority'
            user_caste = caste
            
            selected_service_id = None
            if selected_service_str_man != "None":
                    selected_service_id = int(selected_service_str_man.split(" - ")[0])

            # --- RENDER RESULTS SECTION (Manual) ---
            st.markdown("---")
            st.subheader("🚀 Personalized Recommendations")
            
            res_m1, res_m2, res_m3 = st.columns(3)
            
            # Block Function (Redefined for scope safe)
            def block_service(service, caste=None):
                if not isinstance(service, str): return False
                s = service.lower()
                if "birth" in s or "death" in s: return False
                if caste and caste.lower() == "general" and "caste" in s: return False
                return True

            # 1. District
            with res_m1:
                st.info("🏢 **District Trends**")
                top_dist = get_top_services_for_district_from_csv(DISTRICT_CSV_PATH, district_id, top_n=5)
                top_dist = [s for s in top_dist if block_service(s, user_caste)][:5]
                if top_dist:
                    for s in top_dist: st.markdown(f"- {s}")
                else:
                    st.write("No data available.")
            
            # 2. Demographic
            with res_m2:
                st.success("👥 **Demographic Match**")
                try:
                    manual_data = pd.DataFrame([{
                        'citizen_id': 'manual_entry',
                        'gender': gender, 'caste': caste, 'age': age,
                        'religion': selected_religion, 'age_group': age_group,
                        'religion_group': religion_group, 'district_id': district_id
                    }])

                    searched_name = None
                    if selected_service_id:
                            searched_name = service_master_df[service_master_df['service_id'] == selected_service_id]['service_name'].iloc[0]

                    demo_recs = recommend_services_2(
                        citizen_id='manual_entry',
                        df=final_df,
                        grouped_df=grouped_df,
                        cluster_service_map=cluster_service_map,
                        service_id_to_name=service_id_to_name,
                        service_df=service_df,
                        top_n=5,
                        citizen_master=manual_data,
                        searched_service_name=searched_name
                    )
                    demo_recs = [s for s in demo_recs if isinstance(s, str) and block_service(s, user_caste)]
                    if demo_recs:
                            for s in demo_recs: st.markdown(f"- {s}")
                    else:
                        st.write("No data available.")
                except Exception as e:
                    st.write(f"Error: {e}")

            # 3. Content
            with res_m3:
                st.warning("🔄 **Similar Services**")
                if selected_service_id:
                    data_file = os.path.join(DATA_DIR, "service_with_domains.csv")
                    sim_file = os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
                    try:
                        sims = find_similar_services_from_csv(data_file, sim_file, int(selected_service_id), 5)
                        sims = [s for s in sims if isinstance(s,str) and block_service(s, user_caste)]
                        if sims:
                            st.markdown(f"**Similar to selected:**")
                            for s in sims: st.markdown(f"- {s}")
                        else:
                            st.write("No similar services found.")
                    except:
                        st.write("Error fetching similarities.")
                else:
                    st.write("Select a 'Target Service' to see similar recommendations.")
