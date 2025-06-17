import json
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def load_config(config_path='config.json'):
    """Loads the simulation configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        return None

def run_configurable_simulation(config):
    """
    Runs a discrete event simulation based on the provided configuration.
    This version correctly models parallel stations.
    """
    if not config:
        return pd.DataFrame()

    params = config['simulation_parameters']
    stages = config['production_stages']
    start_date = datetime.strptime(params['start_date'], '%Y-%m-%d')

    production_log = []
    # Initialize station availability times for each parallel station
    station_availability = {
        stage: [start_date] * details['station_count']
        for stage, details in stages.items()
    }

    # Keep track of when each ship finishes its previous stage
    ship_completion_times = {}

    for i in range(1, params['num_starships'] + 1):
        ship_id = f"SN{100 + i}"
        ship_completion_times[ship_id] = start_date

        for stage_name, stage_params in stages.items():
            while True:
                # Find the earliest available station for the current stage
                station_idx = np.argmin(station_availability[stage_name])
                available_time = station_availability[stage_name][station_idx]

                # A ship can only start a stage after it has finished the previous one
                # AND a station for the new stage is available.
                start_time = max(ship_completion_times[ship_id], available_time)

                processing_time_hours = max(1, np.random.normal(stage_params['mean_time_hours'], stage_params['std_dev_hours']))
                end_time = start_time + timedelta(hours=processing_time_hours)

                qc_passed = random.random() < stage_params['pass_rate']
                rework_penalty = 0

                production_log.append({
                    'ship_id': ship_id,
                    'stage': stage_name,
                    'station_id': f"{stage_name}_{station_idx + 1}",
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_hours': round(processing_time_hours, 2),
                    'qc_passed': qc_passed
                })

                if qc_passed:
                    # The station is free at the end_time
                    station_availability[stage_name][station_idx] = end_time
                    # The ship is ready for the next stage at end_time
                    ship_completion_times[ship_id] = end_time
                    break
                else:
                    # Rework happens, occupying the station for longer
                    rework_penalty = stage_params['mean_time_hours'] * 0.25
                    rework_end_time = end_time + timedelta(hours=rework_penalty)
                    # The station is free after rework is done
                    station_availability[stage_name][station_idx] = rework_end_time
                    # The ship is ready for its next attempt after rework
                    ship_completion_times[ship_id] = rework_end_time
                    print(f"REWORK: {ship_id} at {stage_name} will be ready for retry at {rework_end_time.strftime('%Y-%m-%d %H:%M')}")

    return pd.DataFrame(production_log)

if __name__ == '__main__':
    # Load configuration
    config = load_config()

    if config:
        # Run the simulation
        df_production_log = run_configurable_simulation(config)

        # Save the data
        output_filename = 'starship_production_log_configurable.csv'
        df_production_log.to_csv(output_filename, index=False)

        print(f"Successfully generated simulation data based on 'config.json'.")
        print(f"Data saved to '{output_filename}'")
        print("\nFirst 5 rows of the generated data:")
        print(df_production_log.head())
