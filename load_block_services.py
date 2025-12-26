"""
Load block_top_services.csv into PostgreSQL database.
Creates the block_top_services table and loads the data.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
import sys

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/recomm_db')

def load_block_services_to_db():
    """Load block_top_services.csv into PostgreSQL database."""
    
    # File path
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'block_top_services.csv')
    
    if not os.path.exists(data_file):
        print(f"❌ Error: File not found: {data_file}")
        return False
    
    try:
        # Create database engine
        print("Connecting to database...")
        engine = create_engine(DATABASE_URL)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connection successful")
        
        # Load CSV
        print(f"\nLoading CSV from: {data_file}")
        df = pd.read_csv(data_file, encoding='utf-8')
        print(f"✓ Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        
        # Display data info
        print(f"\nData summary:")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique blocks: {df['bsk_id'].nunique()}")
        print(f"  Unique services: {df['service_id'].nunique()}")
        print(f"  Unique municipalities: {df['block_mun_id'].nunique()}")
        
        # Create table schema
        print("\nCreating/replacing table in database...")
        
        create_table_sql = """
        DROP TABLE IF EXISTS block_top_services CASCADE;
        
        CREATE TABLE block_top_services (
            id SERIAL PRIMARY KEY,
            bsk_id INTEGER NOT NULL,
            bsk_name VARCHAR(255),
            block_mun_id INTEGER,
            service_id INTEGER NOT NULL,
            usage_count INTEGER NOT NULL,
            unique_customers INTEGER NOT NULL,
            service_name VARCHAR(500),
            rank_in_block INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT fk_bsk_id FOREIGN KEY (bsk_id) REFERENCES ml_bsk_master(bsk_id),
            CONSTRAINT fk_service_id FOREIGN KEY (service_id) REFERENCES ml_service_master(service_id)
        );
        
        CREATE INDEX idx_block_top_services_bsk_id ON block_top_services(bsk_id);
        CREATE INDEX idx_block_top_services_block_mun_id ON block_top_services(block_mun_id);
        CREATE INDEX idx_block_top_services_service_id ON block_top_services(service_id);
        CREATE INDEX idx_block_top_services_rank ON block_top_services(rank_in_block);
        """
        
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
        
        print("✓ Table created successfully")
        
        # Rename column to match database schema
        df_to_load = df.rename(columns={'bsk_name_x': 'bsk_name'})
        
        # Load data into database
        print("\nInserting data into database...")
        df_to_load.to_sql(
            'block_top_services',
            engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
        
        print(f"✓ Successfully inserted {len(df_to_load)} rows")
        
        # Verify the data
        print("\nVerifying data in database...")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM block_top_services"))
            count = result.scalar()
            print(f"✓ Total rows in database: {count}")
            
            # Show sample data
            result = conn.execute(text("""
                SELECT bsk_id, bsk_name, block_mun_id, service_name, usage_count, rank_in_block 
                FROM block_top_services 
                ORDER BY bsk_id, rank_in_block 
                LIMIT 10
            """))
            
            print("\nSample data from database:")
            print("-" * 100)
            for row in result:
                print(f"  Block {row[0]} ({row[1][:30]}...) - Rank {row[5]}: {row[3][:50]} - {row[4]} uses")
        
        print("\n✅ Successfully loaded block_top_services into database!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = load_block_services_to_db()
    sys.exit(0 if success else 1)
