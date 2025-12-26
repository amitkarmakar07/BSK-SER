# main.py - Entry point for the application
# TODO: Implement main application entry point

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import pickle
import os
import logging
import re
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit

from backend.inference.district import get_top_services_for_district_from_csv
from backend.inference.content import find_similar_services_from_csv
from backend.inference.demo import recommend_services_2
from backend.helpers.district_helper import generate_district_csv_files
from backend.helpers.demo_helper import generate_demo_csv_files
from backend.helpers.content_helper import main as generate_content_csv_files

# Import database conversion functions
from backend.config.database import convert_database_to_csv, batch_convert_with_validation
from backend.utils.database_checker import (
    check_database_availability, 
    can_use_database_operations, 
    should_skip_database_operations,
    get_operational_mode,
    db_checker
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Get max recommendations from environment variable (default: 5)
MAX_RECOMMENDATIONS = int(os.getenv('MAX_RECOMMENDATIONS', 5))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check database availability at startup
logger.info("Checking database table availability...")
db_availability = check_database_availability()
operational_mode = get_operational_mode()

logger.info(f"Database connection: {'✓' if db_availability['database_connection'] else '✗'}")
logger.info(f"All required tables: {'✓' if db_availability['all_tables_available'] else '✗'}")
logger.info(f"Operational mode: {operational_mode.upper()}")

if db_availability['missing_tables']:
    logger.warning(f"Missing tables: {', '.join(db_availability['missing_tables'])}")
    logger.warning("Some database operations will be disabled")

# Initialize scheduler
scheduler = BackgroundScheduler()

# Content helper function for CSV generation
def scheduled_generate_content_csvs():
    """Background task to generate content-based CSV files."""
    try:
        logger.info("Starting scheduled content CSV generation...")
        import sys
        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        sys.argv = [
            "content_helper.py",
            os.path.join(DATA_DIR, "services.csv"),
            "-e", os.path.join(DATA_DIR, "service_master_enhanced.csv"),
            "-s", os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
        ]
        generate_content_csv_files()
        logger.info("Content CSV generation completed successfully")
    except Exception as e:
        logger.error(f"Error in scheduled content CSV generation: {str(e)}")

def scheduled_generate_demo_csvs():
    """Background task to generate demo CSV files."""
    try:
        logger.info("Starting scheduled demo CSV generation...")
        result = generate_demo_csv_files()
        logger.info(f"Demo CSV generation completed successfully. Files: {list(result.keys())}")
    except Exception as e:
        logger.error(f"Error in scheduled demo CSV generation: {str(e)}")

def scheduled_generate_district_csvs():
    """Background task to generate district CSV files."""
    try:
        logger.info("Starting scheduled district CSV generation...")
        result = generate_district_csv_files()
        logger.info(f"District CSV generation completed successfully. Files: {list(result.keys())}")
    except Exception as e:
        logger.error(f"Error in scheduled district CSV generation: {str(e)}")

def scheduled_csv_generation_task():
    """Combined task that runs all CSV generation functions."""
    logger.info("Starting nightly CSV generation tasks...")
    
    # Generate demo CSVs
    scheduled_generate_demo_csvs()
    
    # Generate district CSVs
    scheduled_generate_district_csvs()
    
    # Generate content CSVs
    scheduled_generate_content_csvs()
    
    logger.info("Completed nightly CSV generation tasks.")

# Schedule demo CSV generation to run every night at 11 PM
scheduler.add_job(
    func=scheduled_generate_demo_csvs,
    trigger=CronTrigger(hour=21, minute=0),  # 11:00 PM daily
    id='nightly_demo_csv_generation',
    name='Nightly Demo CSV Generation',
    replace_existing=True
)

# Schedule district CSV generation to run monthly on the 1st at 11:30 PM
scheduler.add_job(
    func=scheduled_generate_district_csvs,
    trigger=CronTrigger(day=1, hour=3, minute=0),  # 1st of every month at 3:00 AM
    id='monthly_district_csv_generation',
    name='Monthly District CSV Generation',
    replace_existing=True
)

# Start the scheduler
scheduler.start()

# Ensure scheduler shuts down when the application exits
atexit.register(lambda: scheduler.shutdown())

# Load data at startup
# grouped_df = pd.read_csv(os.path.join(DATA_DIR, "grouped_df.csv"), encoding="utf-8")
# service_df = pd.read_csv(os.path.join(DATA_DIR, "services.csv"), encoding="utf-8")
# final_df = pd.read_csv(os.path.join(DATA_DIR, "final_df.csv"), encoding="utf-8")
# with open(os.path.join(DATA_DIR, "cluster_service_map.pkl"), "rb") as f:
#     cluster_service_map = pickle.load(f)
# df_service_names = pd.read_csv(os.path.join(DATA_DIR, "service_id_with_name.csv"), encoding="utf-8")
# service_id_to_name = dict(zip(df_service_names['service_id'], df_service_names['service_name']))

# citizen_master = pd.read_csv(os.path.join(DATA_DIR, "ml_citizen_master.csv"), encoding="utf-8")
# provision_data = pd.read_csv(os.path.join(DATA_DIR, "ml_provision.csv"), encoding="utf-8")

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize scheduler when the app starts."""
    if not scheduler.running:
        scheduler.start()
    logger.info("FastAPI application started. Scheduler is running.")
    logger.info("Demo CSV generation scheduled for 11:00 PM daily.")
    logger.info("District CSV generation scheduled for 11:30 PM on the 1st of each month.")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown scheduler when the app stops."""
    if scheduler.running:
        scheduler.shutdown()
    logger.info("FastAPI application stopped. Scheduler shutdown.")

def block_service(service, caste=None):
    if not isinstance(service, str):
        return False
    s = service.lower()
    if "birth" in s or "death" in s:
        return False
    if caste is not None and str(caste).lower() == "general" and "caste" in s:
        return False
    return True

def load_under_18_services():
    """Load under-18 eligible services from database or CSV fallback."""
    try:
        # Try loading from database first
        from backend.config.database import DatabaseConfig
        db_config = DatabaseConfig()
        if db_config.test_connection():
            engine = db_config.get_engine()
            query = "SELECT service_id, service_name FROM under_18_services"
            df = pd.read_sql(query, engine)
            logger.info(f"Loaded {len(df)} under-18 services from database")
            return df
    except Exception as e:
        logger.warning(f"Could not load under_18_services from database: {e}")
    
    # Fallback to CSV if database fails
    try:
        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        csv_path = os.path.join(DATA_DIR, "under18_top_services.csv")
        df = pd.read_csv(csv_path, encoding='latin-1')
        # Handle CSVs with or without service_id column
        if 'service_id' in df.columns:
            df = df[['service_id', 'service_name']].drop_duplicates()
        else:
            # CSV only has service_name, create a simple DataFrame
            df = df[['service_name']].drop_duplicates()
        logger.info(f"Loaded {len(df)} under-18 services from CSV")
        return df
    except Exception as e:
        logger.error(f"Could not load under_18_services from CSV: {e}")
        return pd.DataFrame(columns=['service_name'])

def filter_recommendations_for_under_18(recommendations, under_18_services_df):
    """Filter recommendations to only include services eligible for under-18 users."""
    if under_18_services_df.empty:
        return recommendations
    
    # Normalize service names: lowercase, strip, remove extra spaces, remove special chars
    def normalize_service_name(name):
        if not isinstance(name, str):
            return ""
        # Convert to lowercase, strip whitespace
        normalized = name.lower().strip()
        # Replace multiple spaces with single space
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        # Remove common punctuation that might differ
        normalized = normalized.replace('-', ' ').replace('_', ' ')
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    # Get list of eligible service names (normalized)
    eligible_service_names = set(under_18_services_df['service_name'].apply(normalize_service_name))
    
    # Debug: Log eligible services
    logger.debug(f"Under-18 eligible services count: {len(eligible_service_names)}")
    
    # Filter recommendations
    if isinstance(recommendations, list):
        # For list of service names (district/demographic recommendations)
        filtered = []
        for rec in recommendations:
            if isinstance(rec, str):
                rec_normalized = normalize_service_name(rec)
                # Only exact match after normalization
                if rec_normalized in eligible_service_names:
                    filtered.append(rec)
                else:
                    logger.debug(f"Filtered out (not in under-18 list): {rec}")
        return filtered
    elif isinstance(recommendations, dict):
        # For dictionary of recommendations (content-based)
        filtered_dict = {}
        for key, values in recommendations.items():
            filtered_values = []
            for val in values:
                if isinstance(val, str):
                    val_normalized = normalize_service_name(val)
                    # Only exact match after normalization
                    if val_normalized in eligible_service_names:
                        filtered_values.append(val)
                    else:
                        logger.debug(f"Filtered out (not in under-18 list): {val}")
            if filtered_values:
                filtered_dict[key] = filtered_values
        return filtered_dict
    
    return recommendations

class RecommendRequest(BaseModel):
    mode: str  # "phone" or "manual"
    phone: Optional[str] = None
    district_id: Optional[int] = None
    gender: Optional[str] = None
    caste: Optional[str] = None
    age: Optional[int] = None
    religion: Optional[str] = None
    selected_service_id: Optional[int] = None

class RecommendResponse(BaseModel):
    district_recommendations: List[str]
    demographic_recommendations: List[str]
    item_recommendations: Dict[str, List[str]]  # service_name: [recommendations]

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    import pandas as pd
    import pickle
    import os

    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    # Load data only when this route is called
    grouped_df = pd.read_csv(os.path.join(DATA_DIR, "grouped_df.csv"), encoding="utf-8")
    service_df = pd.read_csv(os.path.join(DATA_DIR, "services.csv"), encoding="utf-8")
    final_df = pd.read_csv(os.path.join(DATA_DIR, "final_df.csv"), encoding="utf-8")
    with open(os.path.join(DATA_DIR, "cluster_service_map.pkl"), "rb") as f:
        cluster_service_map = pickle.load(f)
    df_service_names = pd.read_csv(os.path.join(DATA_DIR, "service_id_with_name.csv"), encoding="utf-8")
    service_id_to_name = dict(zip(df_service_names['service_id'], df_service_names['service_name']))

    citizen_master = pd.read_csv(os.path.join(DATA_DIR, "ml_citizen_master.csv"), encoding="utf-8")
    provision_data = pd.read_csv(os.path.join(DATA_DIR, "ml_provision.csv"), encoding="utf-8")

    # --- Phone Number Mode ---
    if req.mode.lower() == "phone":
        if not req.phone:
            raise HTTPException(status_code=400, detail="Phone number required for phone mode.")
        phone_columns = ['citizen_phone', 'phone', 'mobile']
        phone_col = None
        for col in phone_columns:
            if col in citizen_master.columns:
                phone_col = col
                break
        if not phone_col:
            raise HTTPException(status_code=500, detail="No phone column found in citizen master.")
        try:
            phone_int = int(req.phone)
            citizens_df = citizen_master[citizen_master[phone_col] == phone_int]
        except ValueError:
            citizens_df = citizen_master[citizen_master[phone_col].astype(str) == req.phone]
        if citizens_df.empty:
            raise HTTPException(status_code=404, detail="No citizen found for this phone number.")
        citizen_row = citizens_df.iloc[0]
        citizen_id = citizen_row['citizen_id']
        district_id = int(citizen_row['district_id'])
        gender = citizen_row.get('gender', None)
        caste = citizen_row.get('caste', None)
        age = citizen_row.get('age', None)
        religion = citizen_row.get('religion', None)
        used_services = provision_data[provision_data['customer_id'] == citizen_id]
        used_service_ids = used_services['service_id'].dropna().unique().tolist()
        item_service_ids = used_service_ids.copy()
        if req.selected_service_id and req.selected_service_id not in item_service_ids:
            item_service_ids.append(req.selected_service_id)
        
        # Get district recommendations
        top_services = get_top_services_for_district_from_csv(
            os.path.join(DATA_DIR, "district_top_services.csv"),
            district_id, top_n=MAX_RECOMMENDATIONS
        )
        top_services = [s for s in top_services if block_service(s, caste)][:MAX_RECOMMENDATIONS]
        
        # Get demographic recommendations
        citizen_master_data = citizens_df
        # Get searched service name if available
        searched_service_name = None
        if req.selected_service_id:
            searched_service_name = service_id_to_name.get(int(req.selected_service_id))
        
        try:
            demo_recs = recommend_services_2(
                citizen_id=citizen_id,
                df=final_df,
                grouped_df=grouped_df,
                cluster_service_map=cluster_service_map,
                service_id_to_name=service_id_to_name,
                service_df=service_df,
                top_n=MAX_RECOMMENDATIONS,
                citizen_master=citizen_master_data,
                searched_service_name=searched_service_name
            )
            demo_recs = [s for s in demo_recs if block_service(s, caste)]
        except Exception as e:
            demo_recs = [f"Error: {e}"]
        item_recs = {}
        data_file = os.path.join(DATA_DIR, "service_with_domains.csv")
        similarity_file = os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
        max_total_recs = MAX_RECOMMENDATIONS
        n_services = len(item_service_ids)
        recs_per_service = {}
        if n_services > 0:
            if req.selected_service_id and req.selected_service_id in item_service_ids:
                recs_per_service[req.selected_service_id] = min(3, max_total_recs)
                remaining_recs = max_total_recs - recs_per_service[req.selected_service_id]
                other_services = [sid for sid in item_service_ids if sid != req.selected_service_id]
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
        for sid in item_service_ids:
            try:
                num_similar_services = recs_per_service.get(sid, 0)
                if num_similar_services <= 0:
                    continue
                similar_services = find_similar_services_from_csv(
                    data_file, similarity_file, int(sid), num_similar_services
                )
                filtered_similar_services = [s for s in similar_services if block_service(s, caste)]
                service_name = service_id_to_name.get(int(sid), f"Service {sid}")
                if filtered_similar_services:  # Only add if there are recommendations
                    item_recs[service_name] = filtered_similar_services
            except Exception as e:
                item_recs[f"Service {sid}"] = [f"Error: {e}"]
        
        return RecommendResponse(
            district_recommendations=top_services,
            demographic_recommendations=demo_recs,
            item_recommendations=item_recs
        )

    # --- Manual Entry Mode ---
    elif req.mode.lower() == "manual":
        if not (req.district_id and req.gender and req.caste and req.age is not None and req.religion and req.selected_service_id):
            raise HTTPException(status_code=400, detail="All fields required for manual mode.")
        
        # Check if user is under 18
        is_under_18 = req.age < 18
        
        # Load under-18 services for age-based filtering (only for manual mode)
        under_18_services_df = load_under_18_services() if is_under_18 else pd.DataFrame(columns=['service_name'])
        
        # Get district recommendations
        top_services = get_top_services_for_district_from_csv(
            os.path.join(DATA_DIR, "district_top_services.csv"),
            req.district_id, top_n=MAX_RECOMMENDATIONS
        )
        top_services = [s for s in top_services if block_service(s, req.caste)][:MAX_RECOMMENDATIONS]
        manual_citizen_data = pd.DataFrame([{
            'citizen_id': 'manual_entry',
            'gender': req.gender,
            'caste': req.caste,
            'age': req.age,
            'religion': req.religion,
            'district_id': req.district_id
        }])
        # Get searched service name if available
        searched_service_name = None
        if req.selected_service_id:
            searched_service_name = service_id_to_name.get(int(req.selected_service_id))
        
        try:
            demo_recs = recommend_services_2(
                citizen_id='manual_entry',
                df=final_df,
                grouped_df=grouped_df,
                cluster_service_map=cluster_service_map,
                service_id_to_name=service_id_to_name,
                service_df=service_df,
                top_n=MAX_RECOMMENDATIONS,
                citizen_master=manual_citizen_data,
                searched_service_name=searched_service_name
            )
            demo_recs = [s for s in demo_recs if block_service(s, req.caste)]
        except Exception as e:
            demo_recs = [f"Error: {e}"]
        item_recs = {}
        data_file = os.path.join(DATA_DIR, "service_with_domains.csv")
        similarity_file = os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
        sid = req.selected_service_id
        try:
            similar_services = find_similar_services_from_csv(
                data_file, similarity_file, int(sid), MAX_RECOMMENDATIONS * 2 if is_under_18 else MAX_RECOMMENDATIONS
            )
            filtered_similar_services = [s for s in similar_services if block_service(s, req.caste)]
            # Filter for under-18 if applicable
            if is_under_18:
                logger.info(f"Before under-18 filter: {filtered_similar_services}")
                filtered_similar_services = filter_recommendations_for_under_18(filtered_similar_services, under_18_services_df)[:MAX_RECOMMENDATIONS]
                logger.info(f"After under-18 filter: {filtered_similar_services}")
            service_name = service_id_to_name.get(int(sid), f"Service {sid}")
            if filtered_similar_services:  # Only add if there are recommendations
                item_recs[service_name] = filtered_similar_services
        except Exception as e:
            item_recs[f"Service {sid}"] = [f"Error: {e}"]
        
        # Log recommendation counts for under-18 users
        if is_under_18:
            logger.info(f"Under-18 manual entry - District: {len(top_services)}, Demographic: {len(demo_recs)}, Content: {sum(len(v) for v in item_recs.values())}")
        
        return RecommendResponse(
            district_recommendations=top_services,
            demographic_recommendations=demo_recs,
            item_recommendations=item_recs
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'phone' or 'manual'.")

@app.post("/generate-district-csvs")
def generate_district_csvs():
    """
    Generate district-related CSV files using the helper.
    """
    try:
        result = generate_district_csv_files()
        return JSONResponse(content={"status": "success", "details": "District CSVs generated.", "files": list(result.keys())})
    except Exception as e:
        return JSONResponse(content={"status": "error", "details": str(e)}, status_code=500)

@app.post("/generate-demo-csvs")
def generate_demo_csvs():
    """
    Generate demographic-related CSV files using the helper.
    """
    try:
        result = generate_demo_csv_files()
        return JSONResponse(content={"status": "success", "details": "Demo CSVs generated.", "files": list(result.keys())})
    except Exception as e:
        return JSONResponse(content={"status": "error", "details": str(e)}, status_code=500)

@app.post("/convert-database-to-csv")
def convert_db_to_csv():
    """
    Convert all database tables to CSV files.
    This is the main endpoint for database-to-CSV conversion.
    """
    # Check database availability first
    if should_skip_database_operations():
        missing_tables = db_checker.get_missing_tables()
        return JSONResponse(content={
            "status": "skipped",
            "message": "Database operations disabled - missing required tables",
            "missing_tables": missing_tables,
            "operational_mode": get_operational_mode(),
            "recommendation": "Use CSV files directly to avoid errors"
        }, status_code=400)
    
    try:
        logger.info("Starting database to CSV conversion via API...")
        result = convert_database_to_csv()
        
        if result['status'] == 'completed':
            return JSONResponse(content={
                "status": "success",
                "message": f"Successfully converted {len(result['converted_files'])} tables to CSV",
                "converted_files": result['converted_files'],
                "errors": result['errors'] if result['errors'] else None
            })
        else:
            return JSONResponse(content={
                "status": "error",
                "message": "Database conversion failed",
                "error": result.get('error', 'Unknown error'),
                "errors": result['errors']
            }, status_code=500)
            
    except Exception as e:
        logger.error(f"Error in database conversion endpoint: {e}")
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/batch-convert-with-validation")
def batch_convert_endpoint():
    """
    Convert database to CSV with validation and detailed reporting.
    """
    # Check database availability first
    if should_skip_database_operations():
        missing_tables = db_checker.get_missing_tables()
        return JSONResponse(content={
            "status": "skipped",
            "message": "Database operations disabled - missing required tables",
            "missing_tables": missing_tables,
            "operational_mode": get_operational_mode(),
            "recommendation": "Use CSV files directly to avoid errors"
        }, status_code=400)
    
    try:
        logger.info("Starting batch conversion with validation via API...")
        result = batch_convert_with_validation()
        
        if result['status'] == 'completed':
            return JSONResponse(content={
                "status": "success",
                "message": f"Batch conversion completed. {len(result['converted_files'])} files converted.",
                "converted_files": result['converted_files'],
                "errors": result['errors'] if result['errors'] else None
            })
        else:
            return JSONResponse(content={
                "status": "error", 
                "message": "Batch conversion failed",
                "error": result.get('error', 'Unknown error'),
                "converted_files": result.get('converted_files', []),
                "errors": result['errors']
            }, status_code=500)
            
    except Exception as e:
        logger.error(f"Error in batch conversion endpoint: {e}")
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/data-status")
def get_data_status():
    """Check availability of all required data sources"""
    try:
        from datetime import datetime
        from backend.utils.data_utils import check_all_data_availability
        from backend.config.database import data_loader
        
        # Get database availability status
        db_status = check_database_availability()
        operational_mode = get_operational_mode()
        
        # Get CSV availability if data_loader exists
        csv_availability = None
        csv_status = None
        try:
            csv_availability = data_loader.check_data_availability()
            csv_status = data_loader.get_data_status()
        except Exception as e:
            logger.warning(f"Could not check CSV availability: {e}")
        
        return {
            "status": "success",
            "operational_mode": operational_mode,
            "database_status": db_status,
            "csv_availability": csv_availability,
            "csv_summary": csv_status.split("\n") if csv_status else None,
            "timestamp": datetime.now().isoformat(),
            "recommendations": {
                "use_database": db_status.get("all_tables_available", False),
                "use_csv_fallback": not db_status.get("all_tables_available", True),
                "can_convert_database": db_status.get("can_use_database_conversion", False)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/database-status")
def get_database_status():
    """
    Check database connection and table availability status.
    Use this endpoint to verify database setup before operations.
    """
    try:
        db_status = check_database_availability()
        operational_mode = get_operational_mode()
        
        return {
            "status": "success",
            "database_connection": db_status["database_connection"],
            "all_tables_available": db_status["all_tables_available"],
            "operational_mode": operational_mode,
            "table_status": db_status["table_status"],
            "missing_tables": db_status["missing_tables"],
            "available_tables": db_status["available_tables"],
            "recommendations": {
                "can_use_database_operations": db_status["all_tables_available"],
                "should_use_csv_fallback": not db_status["all_tables_available"],
                "can_convert_database": db_status["can_use_database_conversion"]
            },
            "message": f"Database mode: {operational_mode.upper()}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "operational_mode": "csv_only"
        }

@app.post("/generate-content-csvs")
def generate_content_csvs():
    """
    Generate content-related CSV files using the helper.
    """
    try:
        import sys
        DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        sys.argv = [
            "content_helper.py",
            os.path.join(DATA_DIR, "services.csv"),
            "-e", os.path.join(DATA_DIR, "service_master_enhanced.csv"),
            "-s", os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
        ]
        generate_content_csv_files()
        return JSONResponse(content={"status": "success", "details": "Content CSVs generated."})
    except Exception as e:
        return JSONResponse(content={"status": "error", "details": str(e)}, status_code=500)

# Temporarily disabled due to NumPy compatibility
# @app.post("/generate-content-csvs")
# def generate_content_csvs():
#     """
#     Generate content-related CSV files using the helper.
#     """
#     try:
#         import sys
#         DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
#         sys.argv = [
#             "content_helper.py",
#             os.path.join(DATA_DIR, "services.csv"),
#             "-e", os.path.join(DATA_DIR, "service_master_enhanced.csv"),
#             "-s", os.path.join(DATA_DIR, "openai_similarity_matrix.csv")
#         ]
#         generate_content_csv_files()
#         return JSONResponse(content={"status": "success", "details": "Content CSVs generated."})
#     except Exception as e:
#         return JSONResponse(content={"status": "error", "details": str(e)}, status_code=500)

@app.post("/trigger-nightly-csvs")
def trigger_nightly_csvs():
    """
    Manually trigger the nightly demo CSV generation task for testing.
    """
    try:
        scheduled_generate_demo_csvs()
        return JSONResponse(content={
            "status": "success", 
            "details": "Demo CSV generation task completed successfully."
        })
    except Exception as e:
        logger.error(f"Error in manual demo CSV generation: {str(e)}")
        return JSONResponse(content={
            "status": "error", 
            "details": str(e)
        }, status_code=500)

@app.post("/trigger-monthly-csvs")
def trigger_monthly_csvs():
    """
    Manually trigger the monthly district CSV generation task for testing.
    """
    try:
        scheduled_generate_district_csvs()
        return JSONResponse(content={
            "status": "success", 
            "details": "District CSV generation task completed successfully."
        })
    except Exception as e:
        logger.error(f"Error in manual district CSV generation: {str(e)}")
        return JSONResponse(content={
            "status": "error", 
            "details": str(e)
        }, status_code=500)

@app.post("/trigger-content-csvs")
def trigger_content_csvs():
    """
    Manually trigger the content CSV generation task for testing.
    """
    try:
        scheduled_generate_content_csvs()
        return JSONResponse(content={
            "status": "success", 
            "details": "Content CSV generation task completed successfully."
        })
    except Exception as e:
        logger.error(f"Error in manual content CSV generation: {str(e)}")
        return JSONResponse(content={
            "status": "error", 
            "details": str(e)
        }, status_code=500)

@app.post("/trigger-all-csvs")
def trigger_all_csvs():
    """
    Manually trigger both CSV generation tasks for testing.
    """
    try:
        scheduled_csv_generation_task()
        return JSONResponse(content={
            "status": "success", 
            "details": "All CSV generation tasks completed successfully."
        })
    except Exception as e:
        logger.error(f"Error in manual CSV generation: {str(e)}")
        return JSONResponse(content={
            "status": "error", 
            "details": str(e)
        }, status_code=500)

@app.get("/scheduler-status")
def get_scheduler_status():
    """
    Get the current status of the scheduler and scheduled jobs.
    """
    try:
        jobs = []
        for job in scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": str(job.next_run_time) if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        
        return JSONResponse(content={
            "scheduler_running": scheduler.running,
            "jobs": jobs
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "details": str(e)
        }, status_code=500)
