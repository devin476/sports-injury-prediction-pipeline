import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import warnings

#Suppress warnings
warnings.filterwarnings('ignore')

def train_injury_model():
    print("PREDICTIVE MODELING")
    
    #Load the "Medium" dataset
    print("Loading Raw Data (Medium Dataset)...")
    try:
        df = pd.read_csv('data_raw/dataset_medium.csv')
    except FileNotFoundError:
        print("Error: dataset_medium.csv not found. Please run generate_data.py first.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['athlete_id', 'date'])
    
    # Feature Engineering
    print("Generating Automated Features...")
    
    #ACWR
    df['acute_load'] = df.groupby('athlete_id')['daily_load'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['chronic_load'] = df.groupby('athlete_id')['daily_load'].transform(lambda x: x.rolling(28, min_periods=1).mean())
    df['acwr'] = df['acute_load'] / (df['chronic_load'] + 0.01)
    
    #Strain
    df['load_std'] = df.groupby('athlete_id')['daily_load'].transform(lambda x: x.rolling(7, min_periods=1).std())
    df['monotony'] = df['daily_load'] / (df['load_std'] + 0.01)
    df['strain'] = df['daily_load'] * df['monotony']
    
    #Rest Adequacy
    df['sleep_avg'] = df.groupby('athlete_id')['sleep_quality'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['rest_adequacy'] = df['sleep_avg'] / (df['acute_load'] + 0.01)

    #Define Target Variable
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=7)
    df['injury_next_7d'] = df.groupby('athlete_id')['injury_occurred'].transform(
        lambda x: x.rolling(window=indexer, min_periods=1).max()
    )
    
    #Drop rows that break 7D window (end of the data)
    df = df.dropna(subset=['injury_next_7d'])
    
    #Train/Test Split
    features = ['daily_load', 'sleep_quality', 'soreness', 'acwr', 'strain', 'rest_adequacy']
    X = df[features]
    y = df['injury_next_7d']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    #Train Model (RF)
    print(f"Training Model on {len(X_train)} records...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    #Evaluate
    print("\n--- RESULTS ---")
    y_pred = rf.predict(X_test)
    
    #Print metrics
    print(classification_report(y_test, y_pred, target_names=['No Injury', 'Injury Risk']))
    
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score (Injury Class): {f1:.4f}")
    
    #Feature Importance
    print("\n--- FEATURE IMPORTANCE ---")
    importances = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)

if __name__ == "__main__":
    train_injury_model()