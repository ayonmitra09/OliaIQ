"""
Calculate structural climate risk signals based on absolute changes in climate metrics.
Uses min-max normalization with domain-relevant thresholds rather than z-scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_deltas(df):
    """Calculate changes in climate variables between 2020 and 2060."""
    # Heat stress deltas (positive = increased stress)
    df['delta_tx95f'] = df['TXge95F_2060'] - df['TXge95F_2020']  # Change in extreme heat days
    df['delta_tr'] = df['TR_2060'] - df['TR_2020']               # Change in tropical nights
    df['delta_tx95'] = df['TX95p_2060'] - df['TX95p_2020']       # Change in heat persistence
    df['delta_cdd'] = df['CDD_2060'] - df['CDD_2020']            # Change in consecutive dry days
    
    # Cold relief deltas (positive = decreased stress)
    df['delta_fd'] = df['FD_2020'] - df['FD_2060']               # Reduction in frost days
    df['delta_winter_min'] = df['WMin_2060'] - df['WMin_2020']   # Winter minimum warming
    
    return df

def minmax_normalize(series, min_threshold=None, max_cap=None):
    """
    Normalize values to 0-1 scale with optional thresholds and caps.
    
    Args:
        series: pandas Series of values to normalize
        min_threshold: Optional minimum threshold below which values are set to 0
        max_cap: Optional maximum cap above which values are capped
    """
    series = series.copy()
    
    # Apply threshold and cap
    if min_threshold is not None:
        series[series < min_threshold] = 0
    
    if max_cap is not None:
        series[series > max_cap] = max_cap
    
    # Get min and max after applying threshold/cap
    min_val = series.min()
    max_val = series.max()
    
    # Handle edge case where all values are the same
    if min_val == max_val:
        return pd.Series(0, index=series.index)
    
    # Normalize
    return (series - min_val) / (max_val - min_val)

def calculate_risk_components(df):
    """Calculate heat risk and cold relief components using absolute changes."""
    
    # Heat Risk Components (normalized without minimum thresholds)
    heat_components = {
        'extreme_heat': minmax_normalize(df['delta_tx95f']),  # Change in extreme heat days
        'heat_persistence': minmax_normalize(df['delta_tx95']),  # Change in heat persistence
        'tropical_nights': minmax_normalize(df['delta_tr']),  # Change in tropical nights
        'consecutive_dry': minmax_normalize(df['delta_cdd'])  # Change in consecutive dry days
    }
    
    # Calculate weighted heat risk
    df['heat_risk_change'] = (
        0.35 * heat_components['extreme_heat'] +
        0.25 * heat_components['heat_persistence'] +
        0.25 * heat_components['tropical_nights'] +
        0.15 * heat_components['consecutive_dry']
    )
    
    # Cold Relief Components (normalized with domain thresholds)
    cold_components = {
        'frost_reduction': minmax_normalize(df['delta_fd'], max_cap=30),  # Cap benefit at 30 days reduction
        'winter_min_change': minmax_normalize(df['delta_winter_min'], max_cap=5)  # Cap benefit at 5°C increase
    }
    
    # Calculate weighted cold relief (with 0.25 total weight)
    df['cold_risk_relief'] = (
        0.60 * cold_components['frost_reduction'] +
        0.40 * cold_components['winter_min_change']
    )
    
    # Store raw components for analysis
    for name, series in {**heat_components, **cold_components}.items():
        df[f'raw_{name}'] = series
    
    # Calculate net structural risk change
    df['structural_risk_change'] = df['heat_risk_change'] - 0.25 * df['cold_risk_relief']
    
    # Categorize risk changes
    df['risk_change_category'] = pd.cut(
        df['structural_risk_change'],
        bins=[-np.inf, -0.5, -0.25, 0.25, 0.5, np.inf],
        labels=[
            'High Risk Reduction',
            'Moderate Risk Reduction',
            'Stable Risk',
            'Moderate Risk Increase',
            'High Risk Increase'
        ]
    )
    
    return df

def main():
    """Process climate data and calculate risk signals."""
    data_dir = Path('data')
    
    # Read input data
    df = pd.read_csv(data_dir / 'climate_datasets' / 'climate_heat_precip_trends.csv')
    
    # Calculate deltas first
    df = calculate_deltas(df)
    
    # Calculate risk components
    df = calculate_risk_components(df)
    
    # Select columns for output
    output_cols = [
        'GEOID', 'state_name', 'county_name',
        'structural_risk_change',
        'heat_risk_change',
        'cold_risk_relief',
        'risk_change_category',
        'raw_extreme_heat',
        'raw_heat_persistence',
        'raw_tropical_nights',
        'raw_consecutive_dry',
        'raw_frost_reduction',
        'raw_winter_min_change',
        'delta_tx95f',
        'delta_tx95',
        'delta_tr',
        'delta_cdd',
        'delta_fd',
        'delta_winter_min'
    ]
    
    # Save results
    output_file = data_dir / 'hazardous_event_structural_signals' / 'structural_heat_index.csv'
    df[output_cols].to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print distribution summary
    print("\nRisk Change Distribution:")
    print(df['risk_change_category'].value_counts(normalize=True).sort_index())
    
    print("\nSummary Statistics:")
    print(f"Mean: {df['structural_risk_change'].mean():.3f}")
    print(f"Std Dev: {df['structural_risk_change'].std():.3f}")
    print(f"Range: {df['structural_risk_change'].min():.3f} to {df['structural_risk_change'].max():.3f}")

if __name__ == '__main__':
    main()
