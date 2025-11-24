import pandas as pd
import numpy as np
import os

def engineer_features(input_path, output_path):
    print(f"Processing {input_path}...")
    
    #Load Data
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    #Sort by athlete and date
    df = df.sort_values(['athlete_id', 'date']).reset_index(drop=True)
    
    #Group by athlete
    g = df.groupby('athlete_id')

    #Acute Load (7-day MA)
    df['acute_load'] = g['daily_load'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    
    #Chronic Load (28-day MA)
    df['chronic_load'] = g['daily_load'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())
    
    # ACWR Calculation
    df['acwr'] = df['acute_load'] / (df['chronic_load'] + 0.01)
    
    #STRAIN & MONOTONY
    
    #Rolling Standard Deviation
    df['load_std'] = g['daily_load'].transform(lambda x: x.rolling(window=7, min_periods=1).std())
    
    #Monotony = Daily Load / StdDev
    #High monotony means little variation in training (bad)
    df['monotony'] = df['daily_load'] / (df['load_std'] + 0.01)
    
    #Strain = Daily Load * Monotony
    df['strain'] = df['daily_load'] * df['monotony']
    
 -
    
    #Rolling 7-day sleep quality average
    df['sleep_avg_7d'] = g['sleep_quality'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    
    #Rest Adequacy Index
    df['rest_adequacy'] = df['sleep_avg_7d'] / (df['acute_load'] + 0.01)
    
    #HISTORY INTERACTIONS
    
    # Cumulative previous injuries
    df['prior_injuries'] = g['injury_occurred'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    
    #Interaction Term
    df['load_x_history'] = df['daily_load'] * df['prior_injuries']
    
    #Fill NaNs
    df.fillna(0, inplace=True)
    
    #Save file
    df.to_csv(output_path, index=False)
    print(f"Success: Features saved to {output_path}")



if not os.path.exists('data_processed'):
    os.makedirs('data_processed')

#Test on the small dataset 
engineer_features('data_raw/dataset_small.csv', 'data_processed/features_small.csv')