import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import random
from datetime import datetime, timedelta
import json
import copy

# --- Configuration Loader ---
def load_config(config_path='config.json'):
    """Loads the simulation configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        return None

# --- Core Simulation Engine ---
def run_simulation(config):
    """Runs a discrete event simulation based on the provided configuration."""
    if not config:
        return pd.DataFrame()
    
    params = config['simulation_parameters']
    stages = config['production_stages']
    start_date = datetime.strptime(params['start_date'], '%Y-%m-%d')
    num_ships = params['num_starships']

    production_log = []
    station_availability = {stage: [start_date] * details['station_count'] for stage, details in stages.items()}
    ship_completion_times = {}

    for i in range(1, num_ships + 1):
        ship_id = f"SN{100 + i}"
        ship_completion_times[ship_id] = start_date

        for stage_name, stage_params in stages.items():
            while True:
                station_idx = np.argmin(station_availability[stage_name])
                available_time = station_availability[stage_name][station_idx]
                start_time = max(ship_completion_times[ship_id], available_time)

                processing_time_hours = max(1, np.random.normal(stage_params['mean_time_hours'], stage_params['std_dev_hours']))
                end_time = start_time + timedelta(hours=processing_time_hours)

                qc_passed = random.random() < stage_params['pass_rate']
                production_log.append({'ship_id': ship_id, 'stage': stage_name, 'start_time': start_time, 'end_time': end_time})

                if qc_passed:
                    station_availability[stage_name][station_idx] = end_time
                    ship_completion_times[ship_id] = end_time
                    break
                else:
                    rework_penalty = stage_params['mean_time_hours'] * 0.25
                    rework_end_time = end_time + timedelta(hours=rework_penalty)
                    station_availability[stage_name][station_idx] = rework_end_time
                    ship_completion_times[ship_id] = rework_end_time
    return pd.DataFrame(production_log)

# --- Load Baseline Data ---
try:
    df_baseline = pd.read_csv('starship_production_log_configurable.csv')
    df_baseline['start_time'] = pd.to_datetime(df_baseline['start_time'])
    df_baseline['end_time'] = pd.to_datetime(df_baseline['end_time'])
except FileNotFoundError:
    print("Error: 'starship_production_log_configurable.csv' not found.")
    print("Please run 'configurable_data_generator.py' first.")
    df_baseline = pd.DataFrame()

# --- KPI Calculation Function ---
def calculate_kpis(dataframe):
    if dataframe.empty:
        return 0, 0, 0
    df_completed = dataframe[dataframe['stage'] == 'Final_Checkout'].copy()
    if df_completed.empty:
        return 0, 0, 0
    total_ships = df_completed['ship_id'].nunique()
    total_time = (dataframe['end_time'].max() - dataframe['start_time'].min()).days
    throughput = total_ships / (total_time / 7) if total_time > 0 else 0
    return total_ships, throughput, total_time

# --- Initialize App and Baseline KPIs ---
app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Starship Production Dashboard"
baseline_config = load_config()
total_ships_base, throughput_base, total_time_base = calculate_kpis(df_baseline)

# --- Define Dashboard Layout ---
app.layout = html.Div(style={'fontFamily': 'sans-serif'}, children=[
    html.H1("Starship Production Operations Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # What-If Simulator Section
    html.Div(style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '20px', 'marginBottom': '20px'}, children=[
        html.H3("Process Improvement 'What-If' Simulator", style={'textAlign': 'center'}),
        html.P("Select a scenario to model its impact on key production metrics.", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='improvement-dropdown',
            options=[
                {'label': 'Current State (Baseline)', 'value': 'baseline'},
                {'label': 'Scenario 1: Add a Heat Shield Tiling Station', 'value': 'add_tiling_station'},
                {'label': 'Scenario 2: Improve Welding QC to 98% Pass Rate', 'value': 'improve_welding_qc'},
                {'label': 'Scenario 3: Reduce Plumbing/Wiring Time by 20% (Automation)', 'value': 'reduce_plumbing_time'}
            ],
            value='baseline',
            clearable=False,
            style={'marginBottom': '10px'}
        ),
        html.Button('Run Simulation', id='run-simulation-button', n_clicks=0, style={'display': 'block', 'margin': 'auto', 'backgroundColor': '#007bff', 'color': 'white'}),
        dcc.Loading(id="loading-1", type="default", children=html.Div(id='simulation-results-container', style={'marginTop': '20px', 'textAlign': 'center'}))
    ]),
    
    html.Hr(),
    html.H3("Baseline Performance Analysis", style={'textAlign': 'center'}),
    dcc.Graph(
        figure=px.box(
            df_baseline, x='stage', y=(df_baseline['end_time'] - df_baseline['start_time']).dt.total_seconds() / 3600,
            title='Baseline Cycle Time by Stage (Hours)',
            labels={'y': 'Duration (Hours)', 'x': 'Production Stage'}
        ).update_layout(title_x=0.5)
    )
])

# --- Callback for Interactive Simulation ---
@app.callback(
    Output('simulation-results-container', 'children'),
    Input('run-simulation-button', 'n_clicks'),
    State('improvement-dropdown', 'value')
)
def update_simulation_output(n_clicks, selected_improvement):
    if n_clicks == 0 or not baseline_config:
        return ""

    sim_config = copy.deepcopy(baseline_config)
    description = "Ran simulation with baseline parameters."

    if selected_improvement == 'add_tiling_station':
        sim_config['production_stages']['Heat_Shield_Tiling']['station_count'] += 1
        description = "Added a second Heat Shield Tiling station."
    elif selected_improvement == 'improve_welding_qc':
        sim_config['production_stages']['Welding']['pass_rate'] = 0.98
        description = "Improved Welding QC pass rate to 98%."
    elif selected_improvement == 'reduce_plumbing_time':
        sim_config['production_stages']['Plumbing_Wiring']['mean_time_hours'] *= 0.80
        description = "Reduced Plumbing/Wiring mean time by 20% via automation."

    sim_df = run_simulation(sim_config)
    total_ships_sim, throughput_sim, total_time_sim = calculate_kpis(sim_df)

    throughput_change = throughput_sim - throughput_base
    throughput_percent_change = (throughput_change / throughput_base) * 100 if throughput_base > 0 else 0

    return html.Div([
        html.H4("Simulation Results"),
        html.P(f"Scenario: {description}"),
        html.Table([
            html.Tr([html.Th("Metric"), html.Th("Baseline"), html.Th("Projected"), html.Th("Change")]),
            html.Tr([html.Td("Ships per Week"), html.Td(f"{throughput_base:.2f}"), html.Td(f"{throughput_sim:.2f}"), html.Td(f"{throughput_change:+.2f} ({throughput_percent_change:+.1f}%)")]),
            html.Tr([html.Td("Total Production Time (Days)"), html.Td(f"{total_time_base}"), html.Td(f"{total_time_sim}"), html.Td(f"{total_time_sim - total_time_base:+} days")])
        ], style={'margin': 'auto', 'border': '1px solid black', 'border-collapse': 'collapse', 'width': '80%'})
    ], style={'padding': '10px'})

if __name__ == '__main__':
    print("Dashboard script is ready.")
    print("To run the dashboard, execute the following commands in your terminal:")
    print("1. Ensure 'config.json' and 'starship_production_log_configurable.csv' are in the same directory.")
    print("2. pip install dash pandas plotly numpy")
    print("3. python your_dashboard_script_name.py")
    app.run(debug=True)
