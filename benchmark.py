import pandas as pd
import numpy as np
import time
import os
import psutil
from sqlalchemy import create_engine, text

#DB Connection
DB_URL = os.getenv("DB_URL", "postgresql://user:password@db:5432/sports_db")

def get_memory_usage():
    """Returns current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_pandas_benchmark(file_path):
    dataset_name = file_path.split('/')[-1]
    print(f"--- Testing File-Based (Pandas) on {dataset_name} ---")
    
    start_time = time.time()
    start_mem = get_memory_usage()

    #Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['athlete_id', 'date'])

    #Feature Calculation
    df['acute_load'] = df.groupby('athlete_id')['daily_load'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['chronic_load'] = df.groupby('athlete_id')['daily_load'].transform(lambda x: x.rolling(28, min_periods=1).mean())
    df['acwr'] = df['acute_load'] / (df['chronic_load'] + 0.01)

    df['load_std'] = df.groupby('athlete_id')['daily_load'].transform(lambda x: x.rolling(7, min_periods=1).std())
    df['monotony'] = df['daily_load'] / (df['load_std'] + 0.01)
    df['strain'] = df['daily_load'] * df['monotony']

    #Save & Measure 
    output = "temp_pandas_out.csv"
    df.to_csv(output, index=False)
    
    end_time = time.time()
    end_mem = get_memory_usage()
    
    #Calc Storage Size
    storage_mb = os.path.getsize(output) / (1024 * 1024)
    
    #Cleanup
    if os.path.exists(output): os.remove(output)

    return {
        "method": "File-Based",
        "dataset": dataset_name,
        "time_sec": round(end_time - start_time, 4),
        "memory_mb": round(end_mem - start_mem, 2),
        "storage_mb": round(storage_mb, 2)
    }

def run_sql_benchmark(file_path, table_name):
    dataset_name = file_path.split('/')[-1]
    print(f"Testing Database (SQL) on {dataset_name} ---")
    
    try:
        engine = create_engine(DB_URL)
        conn = engine.connect()
    except Exception as e:
        print(f"DB Error: {e}")
        return None

    # Pre-load data
    df_raw = pd.read_csv(file_path)
    df_raw.to_sql(table_name, engine, if_exists='replace', index=False)
    
    start_time = time.time()
    
    #SQL Query
    query = f"""
    CREATE TABLE {table_name}_features AS
    WITH rolling_stats AS (
        SELECT 
            athlete_id, date, daily_load,
            AVG(daily_load) OVER (PARTITION BY athlete_id ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as acute_load,
            AVG(daily_load) OVER (PARTITION BY athlete_id ORDER BY date ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) as chronic_load,
            STDDEV(daily_load) OVER (PARTITION BY athlete_id ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as load_std
        FROM {table_name}
    )
    SELECT *,
           (acute_load / NULLIF(chronic_load, 0)) as acwr,
           (daily_load / NULLIF(load_std, 0)) as monotony,
           (daily_load * (daily_load / NULLIF(load_std, 0))) as strain
    FROM rolling_stats;
    """
    
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}_features"))
        conn.execute(text(query))
        conn.commit()
        
        #Measure Storage Size (Postgres)
        size_query = text(f"SELECT pg_total_relation_size('{table_name}_features')")
        size_bytes = conn.execute(size_query).scalar()
        storage_mb = size_bytes / (1024 * 1024)

    end_time = time.time()
    
    return {
        "method": "Database",
        "dataset": dataset_name,
        "time_sec": round(end_time - start_time, 4),
        "memory_mb": "See Docker Stats", 
        "storage_mb": round(storage_mb, 2)
    }

#EXECUTION
if __name__ == "__main__":
    results = []
    datasets = [
        ('data_raw/dataset_small.csv', 'small_tbl'),
        ('data_raw/dataset_medium.csv', 'med_tbl'),
        ('data_raw/dataset_large.csv', 'large_tbl')
    ]

    print("Starting Infrastructure Benchmark...")
    for csv_path, tbl_name in datasets:
        if os.path.exists(csv_path):
            results.append(run_pandas_benchmark(csv_path))
            results.append(run_sql_benchmark(csv_path, tbl_name))

    print("\n" + "="*85)
    print(f"{'Method':<15} | {'Dataset':<20} | {'Time (s)':<10} | {'Mem (MB)':<18} | {'Storage (MB)':<10}")
    print("-" * 85)
    for r in results:
        if r:
            print(f"{r['method']:<15} | {r['dataset']:<20} | {r['time_sec']:<10} | {str(r['memory_mb']):<18} | {r['storage_mb']:<10}")
    print("="*85)