"""
FastAPI Backend for Bangla Sahayata Kendra Service Recommendation System
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
import pandas as pd
import pickle
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import sys

# Add parent directory to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.append(BASE_DIR)

from backend.inference.district import get_top_services_for_district_from_csv
from backend.inference.content import find_similar_services_from_csv
from backend.inference.demo import recommend_services_2

# ============= Load Data on Startup =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all required data on startup with memory optimization"""
    global grouped_df, service_df, final_df, cluster_service_map, service_id_to_name
    global citizen_master, provision_data, district_df, service_master_df
    
    try:
        print("📊 Loading data with aggressive memory optimization...")
        
        # Load essential CSVs
        grouped_df = pd.read_csv(
            os.path.join(DATA_DIR, "grouped_df.csv"), 
            encoding="latin-1",
            dtype={'service_id': 'int32'}
        )
        
        service_df = pd.read_csv(
            os.path.join(DATA_DIR, "services.csv"), 
            encoding="latin-1",
            dtype={'service_id': 'int32'}
        )
        
        # Load final_df - SAMPLE 30% to reduce memory by 70%
        final_df_path = os.path.join(DATA_DIR, "final_df.csv")
        if os.path.exists(final_df_path):
            # Read in chunks and sample
            chunk_size = 100000
            sampled_chunks = []
            for chunk in pd.read_csv(
                final_df_path,
                encoding="latin-1",
                usecols=['citizen_id', 'gender', 'caste', 'age', 'religion', 'district_id', 'age_group', 'religion_group'],
                dtype={
                    'citizen_id': 'str',
                    'gender': 'category',
                    'caste': 'category',
                    'religion': 'category',
                    'age_group': 'category',
                    'religion_group': 'category'
                },
                chunksize=chunk_size
            ):
                # Fill NAs and convert integer columns
                chunk['age'] = pd.to_numeric(chunk['age'], errors='coerce').fillna(0).astype('int16')
                chunk['district_id'] = pd.to_numeric(chunk['district_id'], errors='coerce').fillna(0).astype('int16')
                # Sample 30% of each chunk
                sampled_chunks.append(chunk.sample(frac=0.3, random_state=42))
            
            final_df = pd.concat(sampled_chunks, ignore_index=True)
            print(f"✅ Loaded final_df (30% sample) - Memory: {final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        else:
            final_df = pd.DataFrame()
            print("⚠️ final_df.csv not found")
        
        # Load cluster map
        with open(os.path.join(DATA_DIR, "cluster_service_map.pkl"), "rb") as f:
            cluster_service_map = pickle.load(f)
        
        # Load service ID mapping
        df_service_names = pd.read_csv(
            os.path.join(DATA_DIR, "service_id_with_name.csv"),
            encoding="latin-1",
            dtype={'service_id': 'int32'}
        )
        service_id_to_name = dict(zip(df_service_names['service_id'], df_service_names['service_name']))
        
        # Load citizen master - SAMPLE 40% to save memory
        citizen_master_path = os.path.join(DATA_DIR, "ml_citizen_master.csv")
        if os.path.exists(citizen_master_path):
            # Read all first, then sample
            citizen_master_full = pd.read_csv(
                citizen_master_path,
                encoding="latin-1",
                usecols=['citizen_id', 'citizen_name', 'citizen_phone', 'gender', 'age', 'caste', 'religion', 'district_id'],
                dtype={
                    'citizen_id': 'str',
                    'citizen_name': 'str',
                    'citizen_phone': 'int64',
                    'gender': 'category',
                    'caste': 'category',
                    'religion': 'category'
                }
            )
            # Handle NA values in numeric columns
            citizen_master_full['age'] = pd.to_numeric(citizen_master_full['age'], errors='coerce').fillna(0).astype('int16')
            citizen_master_full['district_id'] = pd.to_numeric(citizen_master_full['district_id'], errors='coerce').fillna(0).astype('int16')
            # Keep 40% sample but ensure unique phone numbers are distributed
            citizen_master = citizen_master_full.sample(frac=0.4, random_state=42)
            del citizen_master_full  # Free memory
            print(f"✅ Loaded ml_citizen_master (40% sample) - Memory: {citizen_master.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        else:
            citizen_master = pd.DataFrame()
            print("⚠️ ml_citizen_master.csv not found")
        
        # Load provision data - SAMPLE 35% to save memory
        provision_path = os.path.join(DATA_DIR, "ml_provision.csv")
        if os.path.exists(provision_path):
            # Read in chunks and sample
            chunk_size = 50000
            sampled_prov = []
            for chunk in pd.read_csv(
                provision_path,
                encoding="latin-1",
                usecols=['customer_id', 'service_id', 'service_name', 'prov_date'],
                dtype={
                    'customer_id': 'str',
                    'service_id': 'int32',
                    'service_name': 'str',
                    'prov_date': 'str'
                },
                chunksize=chunk_size
            ):
                sampled_prov.append(chunk.sample(frac=0.35, random_state=42))
            
            provision_data = pd.concat(sampled_prov, ignore_index=True)
            print(f"✅ Loaded ml_provision (35% sample) - Memory: {provision_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        else:
            provision_data = pd.DataFrame()
            print("⚠️ ml_provision.csv not found")
        
        # Load district and service master
        district_df = pd.read_csv(
            os.path.join(DATA_DIR, "district_top_services.csv"), 
            encoding="utf-8",
            dtype={'district_id': 'int16'}
        )
        service_master_df = pd.read_csv(
            os.path.join(DATA_DIR, "services.csv"), 
            encoding="utf-8",
            dtype={'service_id': 'int32'}
        )
        
        # Filter out birth/death services
        service_master_df = service_master_df[~service_master_df['service_name'].str.lower().str.contains('birth|death', na=False)]
        
        # Calculate total memory
        total_memory = sum([
            grouped_df.memory_usage(deep=True).sum(),
            service_df.memory_usage(deep=True).sum(),
            final_df.memory_usage(deep=True).sum() if not final_df.empty else 0,
            citizen_master.memory_usage(deep=True).sum() if not citizen_master.empty else 0,
            provision_data.memory_usage(deep=True).sum() if not provision_data.empty else 0,
            district_df.memory_usage(deep=True).sum(),
            service_master_df.memory_usage(deep=True).sum()
        ]) / 1024**2
        
        print(f"✅ Data loaded - Total Memory: {total_memory:.1f} MB")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise
    
    yield
    print("🔄 Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Bangla Sahayata Kendra API",
    description="Service Recommendation System for BSK Users",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(BASE_DIR, "api", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============= Pydantic Models =============
class CitizenInfo(BaseModel):
    citizen_id: str
    name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    caste: Optional[str] = None
    religion: Optional[str] = None

class ServiceUsage(BaseModel):
    service_id: int
    service_name: str
    count: int

class ManualEntryRequest(BaseModel):
    district_id: int
    gender: str
    caste: str
    age: int
    religion: str
    selected_service_id: int

class RecommendationRequest(BaseModel):
    citizen_id: str
    selected_service_id: Optional[int] = None

class RecommendationResponse(BaseModel):
    district_recommendations: List[str]
    demographic_recommendations: List[str]
    content_recommendations: Dict[str, List[str]]

# ============= Helper Functions =============
def block_service(service: str, caste: Optional[str] = None) -> bool:
    """Filter out birth/death services and caste services for general caste"""
    if not isinstance(service, str):
        return False
    s = service.lower()
    if "birth" in s or "death" in s:
        return False
    if caste and caste.lower() == "general" and "caste" in s:
        return False
    return True

def get_citizen_by_phone(phone: str) -> pd.DataFrame:
    """Get citizen details by phone number"""
    if citizen_master.empty:
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
    
    return df

def get_services_used(citizen_id: str) -> pd.DataFrame:
    """Get services used by citizen"""
    if provision_data.empty:
        return pd.DataFrame()
    
    df = provision_data[provision_data['customer_id'] == citizen_id]
    if not df.empty:
        df['service_id'] = df['service_id'].astype(int)
        return df.sort_values('prov_date', ascending=False)
    return df

# ============= API Endpoints =============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML"""
    html_path = os.path.join(BASE_DIR, "api", "static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>Bangla Sahayata Kendra API</h1><p>Frontend not found. Please access /docs for API documentation.</p>")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Bangla Sahayata Kendra API is running"}

@app.get("/api/districts")
async def get_districts():
    """Get list of all districts"""
    districts = district_df[['district_id', 'district_name']].to_dict('records')
    return {"districts": districts}

@app.get("/api/services")
async def get_services():
    """Get list of all services"""
    services = service_master_df[['service_id', 'service_name']].to_dict('records')
    return {"services": services}

@app.get("/api/citizen/phone/{phone}")
async def get_citizen_by_phone_api(phone: str):
    """Get citizen details by phone number"""
    if citizen_master.empty:
        raise HTTPException(status_code=503, detail="Citizen database not available in this deployment. Please use manual entry mode.")
    
    citizens_df = get_citizen_by_phone(phone)
    
    if citizens_df.empty:
        raise HTTPException(status_code=404, detail=f"No citizen found with phone number: {phone}")
    
    citizens = []
    for _, row in citizens_df.iterrows():
        # Mask name
        name_val = row.get('citizen_name', '-')
        masked_name = '####' if isinstance(name_val, str) and name_val.strip() else '--'
        
        age_val = row.get('age', None)
        age_display = None if pd.isna(age_val) or age_val == 0 else int(age_val)
        
        citizens.append({
            "citizen_id": row['citizen_id'],
            "name": masked_name,
            "gender": row.get('gender', '-'),
            "age": age_display,
            "caste": row.get('caste', '-'),
            "religion": row.get('religion', '-'),
            "district_id": int(row.get('district_id', 0))
        })
    
    return {"citizens": citizens}

@app.get("/api/citizen/{citizen_id}/services")
async def get_citizen_services(citizen_id: str):
    """Get services used by a citizen"""
    services_df = get_services_used(citizen_id)
    
    if services_df.empty:
        return {"services": [], "total_count": 0}
    
    # Group and count services
    service_counts = services_df.groupby(['service_id', 'service_name']).size().reset_index(name='count')
    service_counts = service_counts[~service_counts['service_name'].str.lower().str.contains('birth|death', na=False)]
    service_counts = service_counts.sort_values(by='count', ascending=False)
    
    services_list = service_counts.to_dict('records')
    
    return {
        "services": services_list,
        "total_count": len(services_list)
    }

@app.post("/api/recommend/phone")
async def recommend_by_phone(request: RecommendationRequest):
    """Get recommendations for a citizen by citizen ID"""
    citizen_id = request.citizen_id
    
    # Get citizen details
    if not citizen_master.empty:
        citizen_row = citizen_master[citizen_master['citizen_id'] == citizen_id]
        if citizen_row.empty:
            raise HTTPException(status_code=404, detail=f"Citizen {citizen_id} not found")
        citizen_row = citizen_row.iloc[0]
    else:
        raise HTTPException(status_code=404, detail="Citizen master data not available")
    
    # Get services used
    services_df = get_services_used(citizen_id)
    used_service_ids = services_df['service_id'].dropna().unique().tolist() if not services_df.empty else []
    
    # Add selected service
    selected_service_id = request.selected_service_id
    item_service_ids = list(used_service_ids) + ([selected_service_id] if selected_service_id and selected_service_id not in used_service_ids else [])
    
    # Calculate recommendations per service
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
    
    district_id = int(citizen_row["district_id"])
    user_caste = citizen_row.get('caste', None)
    user_age = citizen_row.get('age', None)
    
    # District Recommendations
    district_recs = get_top_services_for_district_from_csv(
        os.path.join(DATA_DIR, "district_top_services.csv"),
        district_id,
        top_n=5
    )
    district_recs = [s for s in district_recs if block_service(s, user_caste)][:5]
    
    # Demographic Recommendations
    try:
        citizen_master_data = citizen_master[citizen_master['citizen_id'] == citizen_id]
        searched_service_name = None
        if n_services > 0 and item_service_ids:
            first_sid = item_service_ids[0]
            if not services_df.empty and first_sid in services_df['service_id'].values:
                searched_service_name = services_df[services_df['service_id'] == first_sid]['service_name'].iloc[0]
            elif first_sid in service_id_to_name:
                searched_service_name = service_id_to_name[first_sid]
        
        demo_recs = recommend_services_2(
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
        demo_recs = [s for s in demo_recs if isinstance(s, str) and block_service(s, user_caste)]
    except Exception as e:
        demo_recs = []
    
    # Content-based Recommendations
    content_recs = {}
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
                
                # Get service name
                if sid in service_id_to_name:
                    service_name = service_id_to_name[sid]
                else:
                    service_name = f"Service {sid}"
                
                filtered_similar = [s for s in similar_services if isinstance(s, str) and block_service(s, user_caste)]
                if filtered_similar:
                    content_recs[service_name] = filtered_similar
            except Exception as e:
                continue
    
    return RecommendationResponse(
        district_recommendations=district_recs,
        demographic_recommendations=demo_recs,
        content_recommendations=content_recs
    )

@app.post("/api/recommend/manual")
async def recommend_by_manual_entry(request: ManualEntryRequest):
    """Get recommendations for manual entry"""
    district_id = request.district_id
    gender = request.gender
    caste = request.caste
    age = request.age
    religion = request.religion
    selected_service_id = request.selected_service_id
    
    # Calculate age group
    if age < 18:
        age_group = 'child'
    elif age < 60:
        age_group = 'youth'
    else:
        age_group = 'elderly'
    
    religion_group = "Hindu" if religion == "Hindu" else "Minority"
    
    # District Recommendations
    district_recs = get_top_services_for_district_from_csv(
        os.path.join(DATA_DIR, "district_top_services.csv"),
        district_id,
        top_n=5
    )
    district_recs = [s for s in district_recs if block_service(s, caste)][:5]
    
    # Demographic Recommendations
    try:
        manual_citizen_data = pd.DataFrame([{
            'citizen_id': 'manual_entry',
            'gender': gender,
            'caste': caste,
            'age': age,
            'religion': religion,
            'age_group': age_group,
            'religion_group': religion_group,
            'district_id': district_id
        }])
        
        searched_service_name = None
        if selected_service_id and selected_service_id in service_id_to_name:
            searched_service_name = service_id_to_name[selected_service_id]
        
        demo_recs = recommend_services_2(
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
        demo_recs = [s for s in demo_recs if isinstance(s, str) and block_service(s, caste)]
    except Exception as e:
        demo_recs = []
    
    # Content-based Recommendations
    content_recs = {}
    if selected_service_id:
        try:
            data_file = os.path.join(DATA_DIR, "service_with_domains.csv")
            similarity_file = os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
            
            similar_services = find_similar_services_from_csv(
                data_file, similarity_file, selected_service_id, 5
            )
            
            service_name = service_id_to_name.get(selected_service_id, f"Service {selected_service_id}")
            filtered_similar = [s for s in similar_services if isinstance(s, str) and block_service(s, caste)]
            
            if filtered_similar:
                content_recs[service_name] = filtered_similar
        except Exception as e:
            pass
    
    return RecommendationResponse(
        district_recommendations=district_recs,
        demographic_recommendations=demo_recs,
        content_recommendations=content_recs
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
