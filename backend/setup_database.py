"""
Database setup script for creating tables that match CSV structure.
Run this script to set up your PostgreSQL database with the correct schema.
"""

import os
import sys
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

# Add the parent directory to the path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_tables_from_csv_structure():
    """
    Create database tables based on CSV file structures.
    This reads your existing CSV files and creates matching database tables.
    """
    
    # Database connection
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://username:password@localhost:5432/recomm_db')
    
    try:
        engine = create_engine(DATABASE_URL)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connection successful")
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("Please check your DATABASE_URL in the .env file")
        return False
    
    # CSV files and their corresponding table names
    csv_mappings = {
        'ml_citizen_master.csv': 'ml_citizen_master',
        'ml_provision.csv': 'ml_provision',
        'ml_district.csv': 'ml_district', 
        'service_master.csv': 'ml_service_master'
    }
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    
    for csv_file, table_name in csv_mappings.items():
        csv_path = os.path.join(data_dir, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"⚠ CSV file not found: {csv_path}")
            continue
            
        try:
            # Read CSV to understand structure
            df = pd.read_csv(csv_path, nrows=0)  # Read only header
            print(f"Creating table {table_name} from {csv_file}...")
            
            # Generate CREATE TABLE statement
            columns = []
            for col in df.columns:
                # Use TEXT for all columns for simplicity - you can refine this
                columns.append(f'"{col}" TEXT')
            
            create_statement = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                {', '.join(columns)}
            );
            """
            
            # Execute CREATE TABLE
            with engine.connect() as conn:
                conn.execute(text(create_statement))
                conn.commit()
            
            print(f"✓ Table {table_name} created successfully")
            
        except Exception as e:
            print(f"✗ Error creating table {table_name}: {e}")
    
    print("\nDatabase setup completed!")
    print("You can now use the database conversion endpoints:")
    print("- POST /convert-database-to-csv")
    print("- POST /batch-convert-with-validation")
    
    return True

def load_csv_data_to_database():
    """
    Load existing CSV data into database tables.
    This populates your database with the CSV data you already have.
    """
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://username:password@localhost:5432/recomm_db')
    
    try:
        engine = create_engine(DATABASE_URL)
        print("Loading CSV data into database tables...")
        
        csv_mappings = {
            'ml_citizen_master.csv': 'ml_citizen_master',
            'ml_provision.csv': 'ml_provision',
            'ml_district.csv': 'ml_district',
            'service_master.csv': 'ml_service_master'
        }
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        
        for csv_file, table_name in csv_mappings.items():
            csv_path = os.path.join(data_dir, csv_file)
            
            if not os.path.exists(csv_path):
                print(f"⚠ CSV file not found: {csv_path}")
                continue
            
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
                
                # Load data to database
                df.to_sql(table_name, engine, if_exists='append', index=False)
                print(f"✓ Loaded {len(df)} rows into {table_name}")
                
            except Exception as e:
                print(f"✗ Error loading {csv_file} to {table_name}: {e}")
        
        print("Data loading completed!")
        
    except Exception as e:
        print(f"✗ Database loading failed: {e}")

if __name__ == "__main__":
    print("Database Setup Script")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    
    print("Choose an option:")
    print("1. Create database tables from CSV structure")
    print("2. Load CSV data into database")
    print("3. Do both (recommended for first setup)")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        create_tables_from_csv_structure()
    elif choice == "2":
        load_csv_data_to_database()
    elif choice == "3":
        if create_tables_from_csv_structure():
            load_csv_data_to_database()
    else:
        print("Invalid choice. Please run the script again.")
