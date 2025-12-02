# Automated Feature Engineering Pipeline for Sports Injury Prediction

## Overview
This project benchmarks the scalability of File-Based (Pandas) vs. Database-Backed (PostgreSQL) architectures for processing athlete monitoring data.

## Project Structure
* `generate_data.py`: Generates synthetic datasets (Small, Medium, Large).
* `benchmark.py`: Runs the infrastructure speed test.
* `train_model.py`: Trains the Random Forest classifier.
* `docker-compose.yml`: Orchestrates the Python runner and PostgreSQL database.

## How to Run
This project runs entirely in Docker to ensure reproducibility.

1. **Build the Environment**
   ```docker compose up -d --build```
2.  **Generate Synthetic Data**
   ```docker compose exec benchmark-runner python generate_data.py```
3.  **Run Infrastructure Benchmark**
   ```docker compose exec benchmark-runner python benchmark.py```
4.  **Train Predictive Model**
   ```docker compose exec benchmark-runner python train_model.py```

## Authors
Titus Karuri, Devin Streeter
