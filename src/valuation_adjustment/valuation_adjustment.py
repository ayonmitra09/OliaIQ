import pandas as pd
import os


def apply_demographics_adjustment(state_code, county_code, base_value):
    # Load the pre-computed climate signals dataset
    BUCKET_ADJUSTMENTS = {
        "Climate Aligned Inflow": +0.02,
        "Mispriced Climate Risk": -0.05,
        "Risk-Aligned Climate Retreat": -0.02,
        "Long-Term Climate Opportunity": +0.05,
        "Could Not Analyze -- Missing Information": 0.00,
        "Statistically Neutral Climate/Migration Relationship": 0.00
    }
    
    signals_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'demographic_climate_signals.csv')
    
    if not os.path.exists(signals_path):
        raise FileNotFoundError(
            "Climate signals dataset not found. Please run demographic_signal.py first to generate it."
        )
    
    # Read CSV with proper dtypes to preserve leading zeros
    signal_df = pd.read_csv(signals_path, dtype={'state_code': str, 'county_code': str})
    
    # Ensure input codes are properly formatted
    state_code = str(state_code).zfill(2)
    county_code = str(county_code).zfill(3)
    
    row = signal_df.loc[(signal_df['county_code'] == county_code) & (signal_df['state_code'] == state_code)].squeeze()
    
    if row.empty:
        raise ValueError(f"No data found for state code {state_code} and county code {county_code}")
    
    bucket = row['climate_migration_signal']
    adjustment_pct = BUCKET_ADJUSTMENTS.get(bucket, 0.0)
    adjusted_value = base_value * (1 + adjustment_pct)

    return {
        "state": row['state_name'],
        "county": row['county_name'],
        "migration z score": float(row['z_migration']),
        "fema risk z score": float(row['z_risk']),
        "wildfire risk z score": float(row['z_wildfire']),
        "flood risk z score": float(row['z_flood']),
        "heatwave risk z score": float(row['z_heatwave']),
        "bucket": bucket,
        "adjustment_pct": adjustment_pct,
        "adjusted_value": adjusted_value,
    }