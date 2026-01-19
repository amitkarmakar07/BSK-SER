import streamlit as st
import pandas as pd
import sys
import os
import re

# ==========================================
# --- PAGE CONFIG - MUST BE FIRST ---
# ==========================================
st.set_page_config(page_title="Service Recommendation for BSK Users", page_icon="🧑‍💼", layout="wide")

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
        if 'service_id' in df.columns:
            df = df[['service_id', 'service_name']].drop_duplicates()
        else:
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
    eligible_service_names = set(under_18_services_df['service_name'].apply(normalize_service_name))
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

# Load service eligibility data
services_eligibility_df = pd.read_csv(os.path.join(DATA_DIR, "services_updated22.csv"), encoding="latin-1")

# Load cluster_service_map from pickle
import pickle
with open(os.path.join(DATA_DIR, "cluster_service_map.pkl"), "rb") as f:
    cluster_service_map = pickle.load(f)

# Build service_id_to_name mapping
df_service_names = pd.read_csv(os.path.join(DATA_DIR, "service_id_with_name.csv"), encoding="latin-1")
service_id_to_name = dict(zip(df_service_names['service_id'], df_service_names['service_name']))

# Eligibility checker function
def check_service_eligibility(service_name, user_age, user_gender, user_caste, user_religion):
    """Check if a service is eligible for the user based on criteria in services_updated22.csv"""
    # Find service in eligibility dataframe
    service_row = services_eligibility_df[services_eligibility_df['service_name'] == service_name]
    
    if service_row.empty:
        return True  # If not found in eligibility list, allow it
    
    service_row = service_row.iloc[0]
    
    # Check age eligibility FIRST
    min_age = service_row.get('min_age', 0)
    max_age = service_row.get('max_age', 120)
    if not pd.isna(min_age) and not pd.isna(max_age):
        if user_age < min_age or user_age > max_age:
            return False

    # Check for_all column
    if service_row.get('for_all', 0) == 1:
        return True  # Service is for everyone (if age is valid)
    
    # Check caste eligibility
    if user_caste == 'SC' and service_row.get('is_sc', 0) == 0:
        return False
    if user_caste == 'ST' and service_row.get('is_st', 0) == 0:
        return False
    if user_caste == 'OBC-A' and service_row.get('is_obc_a', 0) == 0:
        return False
    if user_caste == 'OBC-B' and service_row.get('is_obc_b', 0) == 0:
        return False
    if user_caste == 'General':
        # For general caste, all caste flags should be 0
        if any([service_row.get('is_sc', 0) == 1, 
                service_row.get('is_st', 0) == 1,
                service_row.get('is_obc_a', 0) == 1,
                service_row.get('is_obc_b', 0) == 1]):
            return False
    
    # Check gender eligibility
    if user_gender == 'Female' and service_row.get('is_female', 0) == 0:
        return False
    if user_gender == 'Male' and service_row.get('is_female', 0) == 1:
        return False  # Service is for females only
    
    # Check religion eligibility
    is_user_minority = user_religion not in ['Hindu']
    if not is_user_minority and service_row.get('is_minority', 0) == 1:
        return False  # Hindu user but service is for minorities only
    if is_user_minority and service_row.get('is_minority', 0) == 0:
        return False  # Minority user but service is for Hindus only
    
    return True  # All checks passed

# Load CSV files instead of using database
@st.cache_data
def load_citizen_master():
    file_path = os.path.join(DATA_DIR, "ml_citizen_master.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, encoding="latin-1")
    else:
        return pd.DataFrame()

@st.cache_data
def load_provision_data():
    file_path = os.path.join(DATA_DIR, "ml_provision.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, encoding="latin-1")
    else:
        return pd.DataFrame()

@st.cache_data
def get_citizen_details(citizen_id):
    citizen_master = load_citizen_master()
    if citizen_master.empty:
        return pd.DataFrame()
    df = citizen_master[citizen_master['citizen_id'] == citizen_id]
    return df

@st.cache_data
def get_services_used(citizen_id):
    provision_data = load_provision_data()
    if provision_data.empty:
        return pd.DataFrame(columns=['customer_id', 'customer_name', 'service_id', 'service_name', 'prov_date', 'docket_no'])
    
    df = provision_data[provision_data['customer_id'] == citizen_id]
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

def get_citizen_by_phone(phone):
    """Get citizen details by phone number. Returns (citizen_row, citizen_id) or (None, None)"""
    citizen_master = load_citizen_master()
    if citizen_master.empty:
        return None, None
    
    phone_columns = ['citizen_phone', 'phone', 'mobile']
    phone_col = None
    for col in phone_columns:
        if col in citizen_master.columns:
            phone_col = col
            break
    
    if phone_col is None:
        return None, None
    
    try:
        phone_int = int(phone)
        df = citizen_master[citizen_master[phone_col] == phone_int]
    except ValueError:
        df = citizen_master[citizen_master[phone_col].astype(str) == phone]
    
    if df.empty:
        return None, None
    
    # Return first matching citizen
    citizen_row = df.iloc[0]
    citizen_id = citizen_row['citizen_id']
    return citizen_row, citizen_id

DISTRICT_CSV_PATH = os.path.join(DATA_DIR, "district_top_services.csv")
BLOCK_CSV_PATH = os.path.join(DATA_DIR, "block_wise_top_services.csv")

@st.cache_data
def load_block_data():
    """Load block-wise top services from CSV."""
    try:
        df = pd.read_csv(BLOCK_CSV_PATH, encoding='utf-8')
        return df
    except Exception as e:
        st.warning(f"Could not load block_wise_top_services.csv: {e}")
        return pd.DataFrame()

def get_top_services_for_block(block_id, top_n=5):
    """Get top N services for a specific block."""
    block_df = load_block_data()
    if block_df.empty:
        return []
    block_services = block_df[block_df['block_id'] == block_id].sort_values('rank_in_block')
    top_services = block_services.head(top_n)['service_name'].tolist()
    return top_services

@st.cache_data
def get_block_id_for_citizen(citizen_id):
    """Get block_id for a citizen based on their latest service provision."""
    try:
        provision_data = load_provision_data()
        if provision_data.empty:
            return None
        
        citizen_provisions = provision_data[provision_data['customer_id'] == citizen_id]
        if citizen_provisions.empty:
            return None
        
        if 'prov_date' in citizen_provisions.columns:
            latest_provision = citizen_provisions.sort_values('prov_date', ascending=False).iloc[0]
        else:
            latest_provision = citizen_provisions.iloc[0]
        
        bsk_id = latest_provision.get('bsk_id', None)
        if pd.isna(bsk_id):
            return None
        
        bsk_master_path = os.path.join(DATA_DIR, "ml_bsk_master.csv")
        if not os.path.exists(bsk_master_path):
            return None
        
        bsk_master = pd.read_csv(bsk_master_path, encoding='latin-1')
        bsk_record = bsk_master[bsk_master['bsk_id'] == bsk_id]
        
        if bsk_record.empty:
            return None
        
        block_mun_id = bsk_record.iloc[0].get('block_mun_id', None)
        return block_mun_id if not pd.isna(block_mun_id) else None
        
    except Exception as e:
        print(f"Error getting block_id for citizen {citizen_id}: {e}")
        return None

def block_service(service, caste=None):
    """Filter function to block birth/death and caste services for General caste."""
    if not isinstance(service, str):
        return False
    s = service.lower()
    if "birth" in s or "death" in s:
        return False
    if caste is not None and caste.lower() == "general" and "caste" in s:
        return False
    return True

# ==========================================
# --- PAGE TITLE ---
# ==========================================
st.title("Bangla Sahayata Kendra")

# ==========================================
# --- UNIFIED SINGLE FORM ---
# ==========================================
st.subheader("📋 Enter Citizen Details")

# Load data for dropdowns
district_df = pd.read_csv(DISTRICT_CSV_PATH, encoding="utf-8")
district_names = district_df['district_name'].tolist()
block_df = load_block_data()
service_master_df = pd.read_csv(os.path.join(DATA_DIR, "services.csv"), encoding="utf-8")
service_master_df = service_master_df[~service_master_df['service_name'].str.lower().str.contains('birth|death', na=False)]

# --- Row 1: Phone Number ---
phone = st.text_input("📱 Mobile Number (Enter to check existing Citizen)", placeholder="e.g., 9800361474")
st.caption("💡 If phone exists in our records, we'll show service history. Otherwise, enter details manually below.")

# --- Row 2: Age, Gender, Caste ---
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
with col3:
    caste = st.selectbox("Caste", ["General", "SC", "ST", "OBC-A", "OBC-B"])

# --- Row 3: District, Block, Religion ---
col4, col5, col6 = st.columns(3)
with col4:
    selected_district_name = st.selectbox("District", district_names)
    district_id = int(district_df[district_df['district_name'] == selected_district_name]['district_id'].iloc[0])
with col5:
    if not block_df.empty:
        unique_blocks = block_df[['block_id', 'block_name']].drop_duplicates().sort_values('block_name')
        block_options = [f"{row['block_id']} - {row['block_name']}" for _, row in unique_blocks.iterrows()]
        selected_block = st.selectbox("Block/Municipality", ["None"] + block_options)
        selected_block_id = int(selected_block.split(" - ")[0]) if selected_block != "None" else None
    else:
        selected_block_id = None
        st.info("Block selection not available")
with col6:
    religions = ["Hindu", "Muslim", "Christian", "Sikh", "Other"]
    selected_religion = st.selectbox("Religion", religions)

# --- Row 4: Current Service Selection ---
service_options = [f"{row['service_id']} - {row['service_name']}" for _, row in service_master_df.iterrows()]
selected_service = st.selectbox("🎯 Service Citizen came to apply for:", options=service_options)
selected_service_id = int(selected_service.split(" - ")[0]) if selected_service else None

# --- Generate Recommendations Button ---
st.markdown("---")
if st.button("🚀 Generate Recommendations", type="primary"):
    
    # --- Check if citizen exists by phone ---
    citizen_exists = False
    citizen_row = None
    citizen_id = None
    services_df = pd.DataFrame()
    used_service_ids = []
    citizen_block_id = None
    
    if phone and phone.strip():
        citizen_row, citizen_id = get_citizen_by_phone(phone.strip())
        if citizen_id is not None:
            citizen_exists = True
            services_df = get_services_used(citizen_id)
            citizen_block_id = get_block_id_for_citizen(citizen_id)
            
            # Get used service IDs from history
            if not services_df.empty:
                service_counts = services_df.groupby(['service_id', 'service_name']).size().reset_index(name='count')
                service_counts = service_counts[~service_counts['service_name'].str.lower().str.contains('birth|death', na=False)]
                used_service_ids = service_counts['service_id'].unique().tolist()
    
    # --- Display User Status ---
    if citizen_exists:
        st.success(f"✅ **Existing Citizen Found!** Citizen ID: `{citizen_id}`")
        
        # Show citizen info
        info_cols = st.columns(5)
        name_val = citizen_row.get('citizen_name', '-')
        masked_name = '####' if isinstance(name_val, str) and name_val.strip() else '--'
        db_age = citizen_row.get('age', age)
        db_gender = citizen_row.get('gender', gender)
        db_caste = citizen_row.get('caste', caste)
        db_religion = citizen_row.get('religion', selected_religion)
        
        info_cols[0].info(f"**Name:** {masked_name}")
        info_cols[1].info(f"**Age:** {db_age if not pd.isna(db_age) and db_age != 0 else '--'}")
        info_cols[2].info(f"**Gender:** {db_gender}")
        info_cols[3].info(f"**Caste:** {db_caste}")
        info_cols[4].info(f"**Religion:** {db_religion}")
        
        # Use database values for recommendations
        user_age = db_age if not pd.isna(db_age) and db_age != 0 else age
        user_gender = db_gender if db_gender else gender
        user_caste = db_caste if db_caste else caste
        user_religion = db_religion if db_religion else selected_religion
        
        # Show service history
        if not services_df.empty:
            st.markdown("### 📜 Service History")
            service_counts = services_df.groupby(['service_id', 'service_name']).size().reset_index(name='count')
            service_counts = service_counts[~service_counts['service_name'].str.lower().str.contains('birth|death', na=False)]
            service_counts = service_counts.sort_values(by='count', ascending=False).reset_index(drop=True)
            st.dataframe(service_counts.rename(columns={'service_id': 'Service ID', 'service_name': 'Service Name', 'count': 'Times Used'}), use_container_width=True, hide_index=True)
        else:
            st.info("No previous service history found for this user.")
    else:
        if phone and phone.strip():
            st.warning(f"📝 **New User** - Phone number `{phone}` not found in our records. Using entered details.")
        else:
            st.info("📝 **New User Entry** - No phone number provided.")
        
        # Use form values
        user_age = age
        user_gender = gender
        user_caste = caste
        user_religion = selected_religion
    
    # --- Calculate age group and religion group ---
    if user_age < 18:
        age_group = 'child'
    elif user_age < 60:
        age_group = 'youth'
    else:
        age_group = 'elderly'
    religion_group = "Hindu" if user_religion == "Hindu" else "Minority"
    
    # --- Prepare item service IDs for content-based recommendations ---
    item_service_ids = list(used_service_ids)
    if selected_service_id and selected_service_id not in item_service_ids:
        item_service_ids.append(selected_service_id)
    
    # --- Calculate recommendations per service ---
    max_total_recs = 5
    n_services = len(item_service_ids)
    recs_per_service = {}
    if n_services > 0:
        if selected_service_id and selected_service_id in item_service_ids:
            recs_per_service[selected_service_id] = min(3, max_total_recs)
            remaining_recs = max_total_recs - recs_per_service[selected_service_id]
            other_services = [sid for sid in item_service_ids if sid != selected_service_id]
            n_other = len(other_services)
            if n_other > 0:
                base = remaining_recs // n_other
                extra = remaining_recs % n_other
                for i, sid in enumerate(other_services):
                    recs_per_service[sid] = base + (1 if i < extra else 0)
        else:
            base = max_total_recs // n_services
            extra = max_total_recs % n_services
            for i, sid in enumerate(item_service_ids):
                recs_per_service[sid] = base + (1 if i < extra else 0)
    
    # Use block from citizen if exists, otherwise from form
    final_block_id = citizen_block_id if citizen_block_id else selected_block_id
    
    # --- COLLECT ALL RECOMMENDATIONS FROM 4 ENGINES ---
    all_recommendations = {}  # {service_name: [sources]}
    
    # --- RECOMMENDATIONS DISPLAY ---
    st.markdown("---")
    st.subheader("🚀 Personalized Recommendations")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # --- 1. District Recommendations ---
    district_recs = []
    with col1:
        st.markdown("🏢 **District Recommendations**")
        st.caption("Popular services in user's district")
        top_services = get_top_services_for_district_from_csv(DISTRICT_CSV_PATH, district_id, top_n=5)
        top_services = [s for s in top_services if block_service(s, user_caste)][:5]
        district_recs = top_services
        if top_services:
            st.markdown("<ul>" + "".join([f"<li>{s}</li>" for s in top_services]) + "</ul>", unsafe_allow_html=True)
            for svc in top_services:
                if svc not in all_recommendations:
                    all_recommendations[svc] = []
                all_recommendations[svc].append("District")
        else:
            st.info("No district recommendations found.")
    
    # --- 2. Block Recommendations ---
    block_recs = []
    with col2:
        st.markdown("🏘️ **Block Recommendations**")
        st.caption("Popular services in user's block")
        if final_block_id:
            block_services = get_top_services_for_block(final_block_id, top_n=5)
            block_services = [s for s in block_services if block_service(s, user_caste)][:5]
            block_recs = block_services
            if block_services:
                st.markdown("<ul>" + "".join([f"<li>{s}</li>" for s in block_services]) + "</ul>", unsafe_allow_html=True)
                for svc in block_services:
                    if svc not in all_recommendations:
                        all_recommendations[svc] = []
                    all_recommendations[svc].append("Block")
            else:
                st.info("No block recommendations found.")
        else:
            st.info("Select a block to see recommendations.")
    
    # --- 3. Demographic Recommendations ---
    demo_recs = []
    with col3:
        st.markdown("👥 **Demographic Recommendations**")
        st.caption("Based on age, gender, caste, religion")
        try:
            if citizen_exists:
                citizen_master_data = get_citizen_details(citizen_id)
            else:
                citizen_master_data = pd.DataFrame([{
                    'citizen_id': 'manual_entry',
                    'gender': user_gender,
                    'caste': user_caste,
                    'age': user_age,
                    'religion': user_religion,
                    'age_group': age_group,
                    'religion_group': religion_group,
                    'district_id': district_id
                }])
            
            searched_service_name = None
            if selected_service_id and not service_master_df.empty:
                service_row = service_master_df[service_master_df['service_id'] == selected_service_id]
                if not service_row.empty:
                    searched_service_name = service_row['service_name'].iloc[0]
            
            demo_recommendations = recommend_services_2(
                citizen_id=citizen_id if citizen_exists else 'manual_entry',
                df=final_df,
                grouped_df=grouped_df,
                cluster_service_map=cluster_service_map,
                service_id_to_name=service_id_to_name,
                service_df=service_df,
                top_n=5,
                citizen_master=citizen_master_data,
                searched_service_name=searched_service_name
            )
            
            filtered_demo = [s for s in demo_recommendations if isinstance(s, str) and block_service(s, user_caste)]
            demo_recs = filtered_demo
            if filtered_demo:
                st.markdown("<ul>" + "".join([f"<li>{s}</li>" for s in filtered_demo]) + "</ul>", unsafe_allow_html=True)
                for svc in filtered_demo:
                    if svc not in all_recommendations:
                        all_recommendations[svc] = []
                    all_recommendations[svc].append("Demographic")
            else:
                st.info("No demographic recommendations found.")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # --- 4. Content-based Recommendations ---
    content_recs = []
    with col4:
        st.markdown("🔄 **Content-based Recommendations**")
        st.caption("Similar services based on history")
        data_file = os.path.join(DATA_DIR, "service_with_domains.csv")
        similarity_file = os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
        
        if n_services > 0:
            found_any = False
            for sid in item_service_ids:
                try:
                    sid_int = int(sid)
                    num_similar = recs_per_service.get(sid, 0)
                    if num_similar <= 0:
                        continue
                    
                    similar_services = find_similar_services_from_csv(data_file, similarity_file, sid_int, min(num_similar, 5))
                    filtered_similar = [s for s in similar_services if isinstance(s, str) and block_service(s, user_caste)]
                    
                    if filtered_similar:
                        found_any = True
                        content_recs.extend(filtered_similar)
                        # Get service name
                        if not services_df.empty and sid in services_df['service_id'].values:
                            service_name = services_df[services_df['service_id'] == sid]['service_name'].iloc[0]
                        else:
                            service_name = service_master_df[service_master_df['service_id'] == sid]['service_name'].iloc[0] if sid in service_master_df['service_id'].values else f"Service {sid}"
                        
                        st.markdown(f"**{service_name}**")
                        st.markdown("<ul>" + "".join([f"<li>{s}</li>" for s in filtered_similar]) + "</ul>", unsafe_allow_html=True)
                        
                        for svc in filtered_similar:
                            if svc not in all_recommendations:
                                all_recommendations[svc] = []
                            all_recommendations[svc].append("Content-based")
                except Exception as e:
                    pass
            
            if not found_any:
                st.info("No similar services found.")
        else:
            st.info("No services to compare.")
    
    # --- CONSOLIDATED RECOMMENDATIONS WITH ELIGIBILITY FILTERING ---
    st.markdown("---")
    st.markdown("## 🎯 **Consolidated Eligible Recommendations**")
    st.caption("Services recommended by all engines, filtered by your eligibility")
    
    if all_recommendations:
        # Filter by eligibility
        eligible_services = []
        for service_name, sources in all_recommendations.items():
            if check_service_eligibility(service_name, user_age, user_gender, user_caste, user_religion):
                eligible_services.append(service_name)
        
        if eligible_services:
            # Create dataframe for display with only service names
            consolidated_df = pd.DataFrame({'Service Name': eligible_services})
            
            st.dataframe(
                consolidated_df,
                use_container_width=True,
                hide_index=True
            )
            
            st.success("✅ **Eligible services found**")
        else:
            st.warning("⚠️ No services match your eligibility criteria from the recommendations.")
    else:
        st.info("No recommendations available.")
