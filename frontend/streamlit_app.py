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

st.set_page_config(page_title="Service Recommendation for BSK Users", page_icon="🧑‍💼", layout="wide")
st.title("🧑‍💼 Service Recommendation for BSK Users")


mode = st.radio("Select Input Mode:", ["Phone Number", "Manual Entry"])

def get_citizen_ids_by_phone(phone):
    citizen_master = load_citizen_master()
    
    # Check if citizen master data is available
    if citizen_master.empty:
        st.error("⚠️ Citizen master data not available. ml_citizen_master.csv is missing from deployment.")
        st.info("Phone number search requires the citizen master file. Please use Manual Entry mode instead.")
        return pd.DataFrame()
    
    # Check if citizen_phone column exists, if not try phone or mobile
    phone_columns = ['citizen_phone', 'phone', 'mobile']
    phone_col = None
    for col in phone_columns:
        if col in citizen_master.columns:
            phone_col = col
            break
    
    if phone_col is None:
        st.error("No phone number column found in citizen master data")
        return pd.DataFrame()
    
    # Try to convert phone to integer for matching
    try:
        phone_int = int(phone)
        df = citizen_master[citizen_master[phone_col] == phone_int]
    except ValueError:
        # If conversion fails, try string matching
        df = citizen_master[citizen_master[phone_col].astype(str) == phone]
    
    if df.empty:
        st.warning(f"No citizens found for phone number: {phone}")
        st.info("Try entering the phone number without any special characters (e.g., 7602690034)")
    
    return df

if mode == "Phone Number":
    phone = st.text_input("Enter Phone Number:", placeholder="e.g., 76026XXXXX")
    st.caption("💡 Try sample phone numbers: 9800361474, 8293058992, 9845120211")
    if phone:
        citizens_df = get_citizen_ids_by_phone(phone)
        if not citizens_df.empty:
            for idx, citizen_row in citizens_df.iterrows():
                citizen_id = citizen_row['citizen_id']
                st.markdown(f"---\n### Recommendations for Citizen ID: `{citizen_id}`")
                # --- Citizen Details Section ---
                st.subheader("👤 Citizen Information")
                info_cols = st.columns(5)
                # Mask name: show first letter + ### if name is present
                name_val = citizen_row.get('citizen_name', '-')
                if isinstance(name_val, str) and name_val.strip():
                    masked_name =  '####'
                else:
                    masked_name = '--'
                info_cols[0].markdown(f"**Name:**<br>{masked_name}", unsafe_allow_html=True)
                info_cols[1].markdown(f"**Gender:**<br>{citizen_row.get('gender','-')}", unsafe_allow_html=True)
                # Show -- if age is missing or 0
                age_val = citizen_row.get('age', '-')
                if pd.isna(age_val) or age_val == 0:
                    age_display = '--'
                else:
                    age_display = str(age_val)
                info_cols[2].markdown(f"**Age:**<br>{age_display}", unsafe_allow_html=True)
                info_cols[3].markdown(f"**Caste:**<br>{citizen_row.get('caste','-')}", unsafe_allow_html=True)
                info_cols[4].markdown(f"**Religion:**<br>{citizen_row.get('religion','-')}", unsafe_allow_html=True)

                # --- Services Used Section ---
                services_df = get_services_used(citizen_id)
                st.subheader("📝 Services Availed")
                if not services_df.empty:
                    # Show a table of unique services with count of times used, block any with birth or death
                    service_counts = services_df.groupby(['service_id', 'service_name']).size().reset_index(name='count')
                    # Block any service containing "birth" or "death" (case-insensitive) in name
                    service_counts = service_counts[~service_counts['service_name'].str.lower().str.contains('birth|death', na=False)]
                    service_counts = service_counts.sort_values(by='count', ascending=False).reset_index(drop=True)
                    st.markdown(f"**Total Unique Services Used:** {len(service_counts)}")
                    
                    # Display service counts without pyarrow dependency
                    try:
                        import pyarrow
                        st.table(service_counts.rename(columns={'service_id': 'Service ID', 'service_name': 'Service Name', 'count': 'Times Used'}))
                    except ImportError:
                        # Display as markdown table if pyarrow not available
                        st.markdown("**Service Usage Details:**")
                        display_df = service_counts.rename(columns={'service_id': 'Service ID', 'service_name': 'Service Name', 'count': 'Times Used'})
                        
                        # Create markdown table
                        markdown_table = "| Service ID | Service Name | Times Used |\\n"
                        markdown_table += "|------------|--------------|------------|\\n"
                        
                        for _, row in display_df.head(20).iterrows():  # Show top 20 services
                            service_name = str(row['Service Name']).replace('|', '\\|')  # Escape pipes
                            markdown_table += f"| {row['Service ID']} | {service_name} | {row['Times Used']} |\\n"
                        
                        if len(display_df) > 20:
                            markdown_table += f"| ... | ... | ... |\\n"
                            markdown_table += f"| Total: {len(display_df)} services | | |\\n"
                        
                        st.markdown(markdown_table)
                    except Exception as e:
                        st.error(f"Error displaying table: {e}")
                        # Fallback to simple text display
                        st.markdown("**Service Usage (Text Format):**")
                        for _, row in service_counts.head(10).iterrows():
                            st.text(f"Service {row['service_id']}: {row['service_name']} (Used {row['count']} times)")
                        if len(service_counts) > 10:
                            st.text(f"... and {len(service_counts) - 10} more services")
                else:
                    st.info("No services found for this citizen.")

                # --- Service selection for item-based recommendations ---
                service_master_df = pd.read_csv(os.path.join(DATA_DIR, "services.csv"), encoding="utf-8")
                # Block any service containing "birth" or "death" (case-insensitive) in name from dropdown
                service_master_df = service_master_df[~service_master_df['service_name'].str.lower().str.contains('birth|death', na=False)]
                service_options = [
                    f"{row['service_id']} - {row['service_name']}" for _, row in service_master_df.iterrows()
                ]
                selected_service = st.selectbox(
                    f"Select the service the user came to apply for (Citizen ID: {citizen_id}):",
                    options=service_options,
                    key=f"service_select_{citizen_id}"
                )
                selected_service_id = int(selected_service.split(" - ")[0]) if selected_service else None

                if st.button(f"Recommend for {citizen_id}"):
                    # Check if user is under 18
                    user_age = citizen_row.get('age', None)
                    is_under_18 = user_age is not None and user_age < 18
                    
                    district_id = int(citizen_row["district_id"])
                    used_service_ids = services_df['service_id'].dropna().unique() if not services_df.empty else []
                    item_service_ids = list(used_service_ids) + ([selected_service_id] if selected_service_id and selected_service_id not in used_service_ids else [])
                    # --- Improved balanced logic for item-based recommendations ---
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
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("🏢 **District Recommendations**")
                        st.caption("Recommends the most popular services within a user's district based on historical service usage trends.")
                        top_services = get_top_services_for_district_from_csv(DISTRICT_CSV_PATH, district_id, top_n=5)
                        def block_service(service, caste=None):
                            if not isinstance(service, str):
                                return False
                            s = service.lower()
                            if "birth" in s or "death" in s:
                                return False
                            if caste is not None and caste.lower() == "general" and "caste" in s:
                                return False
                            return True
                        user_caste = citizen_row.get('caste', None)
                        top_services = [service for service in top_services if block_service(service, user_caste)][:5]
                        if top_services:
                            st.markdown("**Top Services in District:**")
                            st.markdown("<ul>" + "".join([f"<li>{service}</li>" for service in top_services]) + "</ul>", unsafe_allow_html=True)
                        else:
                            st.info("No district recommendations found.")
                    with col2:
                        st.markdown("👥 **Demographic Recommendations**")
                        st.caption("Suggests services using demographic clustering based on attributes like age, gender, caste, and religion for users.")
                        try:
                            # Get citizen master data for this citizen
                            citizen_master_data = get_citizen_details(citizen_id)
                            # Get searched service name from selected service
                            searched_service_name = None
                            if n_services > 0 and item_service_ids:
                                # Get the first service name as searched service
                                first_sid = item_service_ids[0]
                                if not services_df.empty and first_sid in services_df['service_id'].values:
                                    searched_service_name = services_df[services_df['service_id'] == first_sid]['service_name'].iloc[0]
                                elif not service_master_df.empty and first_sid in service_master_df['service_id'].values:
                                    searched_service_name = service_master_df[service_master_df['service_id'] == first_sid]['service_name'].iloc[0]
                            
                            demo_recommendations = recommend_services_2(
                                citizen_id=citizen_id,
                                df=final_df,
                                grouped_df=grouped_df,
                                cluster_service_map=cluster_service_map,
                                service_id_to_name=service_id_to_name,
                                service_df=service_df,
                                top_n=5,
                                citizen_master=citizen_master_data,
                                searched_service_name=searched_service_name
                            )
                            
                            # Block any service containing "birth" or "death" (case-insensitive)
                            filtered_demo_recommendations = [service for service in demo_recommendations if isinstance(service, str) and block_service(service, user_caste)]
                            if filtered_demo_recommendations:
                                st.markdown("**Top Demographic Recommendations:**")
                                st.markdown("<ul>" + "".join([f"<li>{service}</li>" for service in filtered_demo_recommendations]) + "</ul>", unsafe_allow_html=True)
                            else:
                                st.info("No demographic recommendations found.")
                        except Exception as e:
                            st.error(f"Error in demographic recommendations: {e}")
                    with col3:
                        st.markdown("🔄 **Content-based Recommendations**")
                        st.caption("Recommends semantically similar services to those already used by the citizen using embeddings and cosine similarity on enriched service descriptions.")
                        data_file = os.path.join(DATA_DIR, "service_with_domains.csv")
                        similarity_file = os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
                        if n_services > 0:
                            for sid in item_service_ids:
                                try:
                                    sid_int = int(sid)
                                    num_similar_services = recs_per_service.get(sid, 0)
                                    if num_similar_services <= 0:
                                        continue
                                    similar_services = find_similar_services_from_csv(
                                        data_file, similarity_file, sid_int, num_similar_services
                                    )
                                    # Block any service containing "birth" or "death" (case-insensitive) in content-based recs
                                    import re
                                    def clean_service_name(name):
                                        if not isinstance(name, str):
                                            return name
                                        # Replace all non-alphanumeric (except space and dash) with space
                                        cleaned = re.sub(r'[^\\w\\s\\-]', ' ', name)
                                        # Replace multiple spaces with single space
                                        cleaned = re.sub(r'\\s+', ' ', cleaned)
                                        # Remove leading/trailing spaces and dashes
                                        cleaned = cleaned.strip(' -')
                                        # Remove trailing/leading asterisks and spaces
                                        cleaned = cleaned.strip(' *')
                                        return cleaned
                                    filtered_similar_services = [sim_name for sim_name in similar_services if isinstance(sim_name, str) and block_service(sim_name, user_caste)]
                                    
                                    # Find the service name for display and clean it
                                    if not services_df.empty and sid in services_df['service_id'].values:
                                        service_name = services_df[services_df['service_id'] == sid]['service_name'].iloc[0]
                                    else:
                                        service_name = service_master_df[service_master_df['service_id'] == sid]['service_name'].iloc[0]
                                    service_name = service_name
                                    
                                    if filtered_similar_services:
                                        st.markdown(f"**{service_name}**")
                                        st.markdown("<ul>" + "".join([f"<li>{sim_name}</li>" for sim_name in filtered_similar_services]) + "</ul>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.write(f"Error for service_id {sid}: {e}")
                        else:
                            st.info("No item-based recommendations available (no services used).")
        else:
            st.error("No citizens found for this phone number.")

elif mode == "Manual Entry":
    st.subheader("Enter Demographic and Location Details")
    # --- District selection by name ---
    district_df = pd.read_csv(DISTRICT_CSV_PATH, encoding="utf-8")
    district_names = district_df['district_name'].tolist()
    selected_district_name = st.selectbox("District", district_names)
    district_id = int(district_df[district_df['district_name'] == selected_district_name]['district_id'].iloc[0])

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    caste = st.selectbox("Caste", ["General", "SC", "ST", "OBC-A", "OBC-B"])
    # --- Age input and group inference ---
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    
    # Calculate age group without pandas
    if age < 18:
        age_group = 'child'
    elif age < 60:
        age_group = 'youth'
    else:
        age_group = 'elderly'
    # --- Religion selection and grouping ---
    religions = ["Hindu", "Muslim", "Christian", "Sikh", "Other"]
    selected_religion = st.selectbox("Religion", religions)
    religion_group = "Hindu" if selected_religion == "Hindu" else "Minority"

    # --- Service selection ---
    service_master_df = pd.read_csv(os.path.join(DATA_DIR, "services.csv"), encoding="utf-8")
    # Block any service containing "birth" or "death" (case-insensitive) in name from dropdown
    service_master_df = service_master_df[~service_master_df['service_name'].str.lower().str.contains('birth|death', na=False)]
    service_options = [
        f"{row['service_id']} - {row['service_name']}" for _, row in service_master_df.iterrows()
    ]
    selected_service = st.selectbox(
        "Select a service for item-based recommendations:",
        options=service_options
    )
    selected_service_id = int(selected_service.split(" - ")[0]) if selected_service else None

    if st.button("Recommend"):
        # --- Item-based recommendation logic: only the selected service, up to 5 ---
        item_service_ids = [selected_service_id] if selected_service_id else []
        max_total_recs = 5
        n_services = len(item_service_ids)
        recs_per_service = {}
        if n_services > 0:
            recs_per_service[selected_service_id] = min(3, max_total_recs)
            remaining_recs = max_total_recs - recs_per_service[selected_service_id]
            # No other services in manual mode, so all go to selected
            recs_per_service[selected_service_id] += remaining_recs

        col1, col2, col3 = st.columns(3)
        # --- District Recommendations ---
        with col1:
            st.markdown("🏢 **District Recommendations**")
            st.caption("Recommends the most popular services within a user's district based on historical service usage trends.")
            top_services = get_top_services_for_district_from_csv(DISTRICT_CSV_PATH, district_id, top_n=5)
            def block_service(service, caste=None):
                if not isinstance(service, str):
                    return False
                s = service.lower()
                if "birth" in s or "death" in s:
                    return False
                if caste is not None and caste.lower() == "general" and "caste" in s:
                    return False
                return True
            user_caste = caste
            top_services = [service for service in top_services if block_service(service, user_caste)][:5]
            if top_services:
                st.markdown("**Top Services in District:**")
                st.markdown("<ul>" + "".join([f"<li>{service}</li>" for service in top_services]) + "</ul>", unsafe_allow_html=True)
            else:
                st.info("No district recommendations found.")

        # --- Demographic Recommendations ---
        with col2:
            st.markdown("👥 **Demographic Recommendations**")
            st.caption("Suggests services using demographic clustering based on attributes like age, gender, caste, and religion for users.")
            
            try:
                # Create a DataFrame with the manual entry data
                manual_citizen_data = pd.DataFrame([{
                    'citizen_id': 'manual_entry',
                    'gender': gender,
                    'caste': caste,
                    'age': age,
                    'religion': selected_religion,
                    'age_group': age_group,
                    'religion_group': religion_group,
                    'district_id': district_id
                }])
                
                # Get searched service name
                searched_service_name = None
                if selected_service_id and not service_master_df.empty:
                    service_row = service_master_df[service_master_df['service_id'] == selected_service_id]
                    if not service_row.empty:
                        searched_service_name = service_row['service_name'].iloc[0]
                
                demo_recommendations = recommend_services_2(
                    citizen_id='manual_entry',
                    df=final_df,
                    grouped_df=grouped_df,
                    cluster_service_map=cluster_service_map,
                    service_id_to_name=service_id_to_name,
                    service_df=service_df,
                    top_n=5,
                    citizen_master=manual_citizen_data,
                    searched_service_name=searched_service_name
                )
                filtered_demo_recommendations = [service for service in demo_recommendations if isinstance(service, str) and block_service(service, caste)]
                if filtered_demo_recommendations:
                    st.markdown("**Top Demographic Recommendations:**")
                    st.markdown("<ul>" + "".join([f"<li>{service}</li>" for service in filtered_demo_recommendations]) + "</ul>", unsafe_allow_html=True)
                else:
                    st.info("No demographic recommendations found.")
            except Exception as e:
                st.error(f"Error in demographic recommendations: {e}")

        # --- Item-based Recommendations ---
        with col3:
            st.markdown("🔄 **Content-based Recommendations**")
            st.caption("Recommends semantically similar services to those already used by the citizen using embeddings and cosine similarity on enriched service descriptions.")
            data_file = os.path.join(DATA_DIR, "service_with_domains.csv")
            similarity_file = os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
            if n_services > 0:
                for sid in item_service_ids:
                    try:
                        sid_int = int(sid)
                        num_similar_services = recs_per_service.get(sid, 0)
                        if num_similar_services <= 0:
                            continue
                        # Limit content-based recommendations to 5
                        num_similar_services = min(num_similar_services, 5)
                        similar_services = find_similar_services_from_csv(
                            data_file, similarity_file, sid_int, num_similar_services
                        )
                        service_name = service_master_df[service_master_df['service_id'] == sid]['service_name'].iloc[0]
                        filtered_similar_services = [sim_name for sim_name in similar_services if isinstance(sim_name, str) and block_service(sim_name, caste)]
                        st.markdown(f"**{service_name}**")
                        st.markdown("<ul>" + "".join([f"<li>{sim_name}</li>" for sim_name in filtered_similar_services]) + "</ul>", unsafe_allow_html=True)
                    except Exception as e:
                        st.write(f"Error for service_id {sid}: {e}")
            else:
                st.info("No item-based recommendations available (no service selected).")
