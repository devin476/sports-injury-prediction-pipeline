import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_athlete_data(num_athletes, num_days, output_filename):
    print(f"Generating {output_filename}...")
    
    #Params
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(num_days)]
    
    all_data = []

    for athlete_id in range(1, num_athletes + 1):
        #Base physiological
        base_load = np.random.randint(300, 600)
        base_hrv = np.random.randint(40, 80)
        
        #Gen daily records
        for d in dates:
            #Sim metrics with some variance
            daily_load = int(np.random.normal(base_load, 100))
            daily_load = max(0, daily_load) # Ensure no negative load
            
            sleep_quality = np.random.randint(1, 11) # 1-10 scale
            soreness = np.random.randint(1, 11) # 1-10 scale
            hrv = int(np.random.normal(base_hrv, 5))
            
            #Sim Injury (Event)
            #High load + Low sleep = slightly higher chance
            injury_prob = 0.001
            if daily_load > 700 and sleep_quality < 4:
                injury_prob = 0.02
            
            injury_occurred = 1 if np.random.random() < injury_prob else 0

            all_data.append({
                'athlete_id': athlete_id,
                'date': d,
                'daily_load': daily_load,
                'sleep_quality': sleep_quality,
                'soreness': soreness,
                'hrv': hrv, #Physiological metric
                'injury_occurred': injury_occurred
            })

    #Create DF
    df = pd.DataFrame(all_data)
    
    #Save to CSV
    df.to_csv(output_filename, index=False)
    print(f"Saved {output_filename}: {len(df)} rows.")

#EXECUTION
# Create directory for file-based
if not os.path.exists('data_raw'):
    os.makedirs('data_raw')

#SMALL Dataset 
# (1 team, 1 season -> ~20 athletes * 200 days = 4,000 rows)
generate_athlete_data(num_athletes=20, num_days=200, output_filename='data_raw/dataset_small.csv')

#MEDIUM Dataset 
# (10 teams, 2 seasons -> ~200 athletes * 400 days = 80,000 rows)
generate_athlete_data(num_athletes=200, num_days=400, output_filename='data_raw/dataset_medium.csv')

#LARGE Dataset 
# (League wide, multi-year -> ~1000 athletes * 1000 days = 1,000,000 rows)
generate_athlete_data(num_athletes=1000, num_days=1000, output_filename='data_raw/dataset_large.csv')