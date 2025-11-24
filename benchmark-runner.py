import pandas as pd
import numpy as np
import time
import os
import psutil
from sqlalchemy import create_engine, text

#Db Connection (Matched to docker-compose)
DB_URL = os.getenv("DB_URL", "postgresql://user:password@localhost:5432/sports_db")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def run_pandas_benchmark(file_path):
    print(f"--- Testing File-Based (Pandas) on {file_path} ---")
    start_time = time.time()
    start_mem = get_memory_usage()

    #Load Data
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['athlete_id', 'date'])

    #Feature Calc
    #ACWR
    df['acute_load'] = df.groupby('athlete_id')['daily_load'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['chronic_load'] = df.groupby('athlete_id')['daily_load'].transform(lambda x: x.rolling(28, min_periods=1).mean())
    df['acwr'] = df['acute_load'] / (df['chronic_load'] + 0.01)

    #Strain
    df['load_std'] = df.groupby('athlete_id')['daily_load'].transform(lambda x: x.rolling(7, min_periods=1).std())
    df['monotony'] = df['daily_load'] / (df['load_std'] + 0.01)
    df['strain'] = df['daily_load'] * df['monotony']

    #Save
    output = "temp_pandas_out.csv"
    df.to_csv(output, index=False)
    
    end_time = time.time()
    end_mem = get_memory_usage()
    
    #Cleanup
    if os.path.exists(output): os.remove(output)

    return {
        "method": "File-Based",
        "file": file_path,
        "time_sec": round(end_time - start_time, 4),
        "memory_mb": round(end_mem - start_mem, 2)
    }

def run_sql_benchmark(file_path, table_name):
    print(f"--- Testing Database (SQL) on {file_path} ---")
    engine = create_engine(DB_URL)
    
    #Pre-load data into DB (Not part of calc time, part of setup)
    df_raw = pd.read_csv(file_path)
    df_raw.to_sql(table_name, engine, if_exists='replace', index=False)
    
    start_time = time.time()
    
    #SQL Query replicating the Feature Engineering
    query = f"""
    CREATE TABLE {table_name}_features AS
    WITH rolling_stats AS (
        SELECT 
            athlete_id, date, daily_load, sleep_quality, injury_occurred,
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

    end_time = time.time()
    
    return {
        "method": "Database",
        "file": file_path,
        "time_sec": round(end_time - start_time, 4),
        "memory_mb": "N/A (Server-side)" 
    }

#EXECUTION
results = []
datasets = [
    ('data_raw/dataset_small.csv', 'small_tbl'),
    ('data_raw/dataset_medium.csv', 'med_tbl'),
    ('data_raw/dataset_large.csv', 'large_tbl')
]

for csv_path, tbl_name in datasets:
    if os.path.exists(csv_path):
        # Run Pandas
        results.append(run_pandas_benchmark(csv_path))
        # Run SQL
        results.append(run_sql_benchmark(csv_path, tbl_name))

#Print Results
print("\nFINAL BENCHMARK RESULTS")
print(f"{'Method':<15} | {'Dataset':<25} | {'Time (s)':<10} | {'Mem (MB)':<10}")
print("-" * 70)
for r in results:
    print(f"{r['method']:<15} | {r['file']:<25} | {r['time_sec']:<10} | {r['memory_mb']:<10}")