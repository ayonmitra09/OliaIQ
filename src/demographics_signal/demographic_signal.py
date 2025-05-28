import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from pathlib import Path

# Get workspace root - more robust than counting parent directories
def get_workspace_root():
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find workspace root")

def load_county_data():
    workspace_root = get_workspace_root()
    path = workspace_root / 'data' / 'climate_datasets' / 'co-est2024-alldata.csv'
    
    df = pd.read_csv(path, encoding='latin1', dtype={'STATE': str, 'COUNTY': str})
    df['state_code'] = df['STATE'].str.zfill(2)
    df['county_code'] = df['COUNTY'].str.zfill(3)
    df = df[df['COUNTY'] != '000']

    # Calculate % population change from 2020 to 2024
    df['tot_population_change_pct'] = (
        (df['POPESTIMATE2024'] - df['POPESTIMATE2020']) / df['POPESTIMATE2020']
    ).round(4)

    # Calculate annualized population change percentage
    df['annual_population_change_pct'] = (
        (1 + df['tot_population_change_pct']) ** (1/4) - 1
    ).round(4)
    
    # Sum domestic migration from 2020-2024
    df['DOMESTICMIG'] = (
        df['DOMESTICMIG2021'] + 
        df['DOMESTICMIG2022'] + 
        df['DOMESTICMIG2023'] + 
        df['DOMESTICMIG2024']
    )

    df['NETMIG'] = (
        df['NETMIG2021'] + 
        df['NETMIG2022'] + 
        df['NETMIG2023'] + 
        df['NETMIG2024']
    )

    return df[[
        'state_code', 'county_code', 'STNAME', 'CTYNAME',
        'POPESTIMATE2020', 'POPESTIMATE2024',
        'DOMESTICMIG', 'NETMIG',
        'annual_population_change_pct'
    ]].rename(columns={
        'STNAME': 'state_name',
        'CTYNAME': 'county_name',
        'POPESTIMATE2020': 'population_2020',
        'POPESTIMATE2024': 'population_2024',
        'DOMESTICMIG': 'domestic_migration',
        'NETMIG': 'net_migration'
    })

def load_fema_nri_data():
    """
    Loads FEMA National Risk Index data and extracts basic fields:
    - state_code
    - county_code
    - state_name
    - county_name
    - risk_score
    """
    workspace_root = get_workspace_root()
    file_path = workspace_root / 'data' / 'climate_datasets' / 'NRI_Table_Counties.csv'

    df = pd.read_csv(file_path, dtype={'STATEFIPS': str, 'COUNTYFIPS': str})

    # Clean up and extract only what we need
    df['state_code'] = df['STATEFIPS'].str.zfill(2)
    df['county_code'] = df['COUNTYFIPS'].str.zfill(3)
    
    # Combine coastal and riverine flooding risks
    df['FLD_RISKS'] = df.apply(lambda row: 
        max(row['CFLD_RISKS'], row['RFLD_RISKS']) 
        if (row['CFLD_RISKS'] <= 25 or row['RFLD_RISKS'] <= 25)
        else (max(row['CFLD_RISKS'], row['RFLD_RISKS']) * 0.7 + 
              min(row['CFLD_RISKS'], row['RFLD_RISKS']) * 0.3),
        axis=1
    )
    
    # Replace NaN values with appropriate defaults for all column types
    numeric_columns = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
    non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64', 'float32', 'int32']).columns
    
    df[numeric_columns] = df[numeric_columns].fillna(0)
    df[non_numeric_columns] = df[non_numeric_columns].fillna('N/A')

    slim = df[[
        'state_code', 'county_code',
        'STATE', 'COUNTY',
        'RISK_SCORE', 'FLD_RISKS', 'WFIR_RISKS', 'HWAV_RISKS',
    ]].rename(columns={
        'STATE': 'state_name',
        'COUNTY': 'county_name',
        'RISK_SCORE': 'fema_risk_score',
        'FLD_RISKS': 'flood_risk',
        'WFIR_RISKS': 'wildfire_risk',
        'HWAV_RISKS': 'heat_wave_risk',
    })

    return slim

def compute_migration_risk_alignment(df):
    df = df.copy()
    
    # Z-score normalize migration and risk columns
    # Drop rows with NaN values before computing z-scores
    valid_rows = df.dropna(subset=['annual_population_change_pct', 'fema_risk_score', 
                                 'wildfire_risk', 'flood_risk', 'heat_wave_risk'])
    
    # Initialize all z-score columns with NaN
    df['z_migration'] = df['annual_population_change_pct'].map(lambda x: float('nan'))
    df['z_risk'] = df['fema_risk_score'].map(lambda x: float('nan'))
    df['z_wildfire'] = df['wildfire_risk'].map(lambda x: float('nan'))
    df['z_flood'] = df['flood_risk'].map(lambda x: float('nan'))
    df['z_heatwave'] = df['heat_wave_risk'].map(lambda x: float('nan'))
    
    # Calculate z-scores for valid rows
    df.loc[valid_rows.index, 'z_migration'] = zscore(valid_rows['annual_population_change_pct'])
    df.loc[valid_rows.index, 'z_risk'] = zscore(valid_rows['fema_risk_score'])
    df.loc[valid_rows.index, 'z_wildfire'] = zscore(valid_rows['wildfire_risk'])
    df.loc[valid_rows.index, 'z_flood'] = zscore(valid_rows['flood_risk'])
    df.loc[valid_rows.index, 'z_heatwave'] = zscore(valid_rows['heat_wave_risk'])

    neutral_zone = 0.5
    
    def classify_zscore_bucket(row):
        if pd.isna(row['z_migration']) or pd.isna(row['z_risk']):
            return "Could Not Analyze -- Missing Information"
        
        z_mig = row['z_migration']
        z_risk = row['z_risk']

        high_mig = z_mig > neutral_zone
        low_mig = z_mig < -neutral_zone
        high_risk = z_risk > neutral_zone
        low_risk = z_risk < -neutral_zone
        
        if high_mig and low_risk:
            return "Climate Aligned Inflow"
        elif high_mig and high_risk:
            return "Mispriced Climate Risk"
        elif low_mig and high_risk:
            return "Risk-Aligned Climate Retreat"
        elif low_mig and low_risk:
            return "Long-Term Climate Opportunity"
        else:
            return "Statistically Neutral Climate/Migration Relationship"

    df['climate_migration_signal'] = df.apply(classify_zscore_bucket, axis=1)
    
    return df

def get_migration_risk_df():
    df = load_county_data()
    fema_df = load_fema_nri_data()
    # Merge the two dataframes on the 'state_code' and 'county_code' columns
    merged_df = pd.merge(
        df, 
        fema_df.drop(['state_name', 'county_name'], axis=1), 
        on=['state_code', 'county_code'], 
        how='left'
    )
    migration_risk_df = compute_migration_risk_alignment(merged_df)
    return migration_risk_df

def generate_climate_signals_dataset():
    """
    Generate and save the demographic climate signals dataset.
    This should be run once to create the reference dataset.
    """
    print("Starting dataset generation...")
    migration_risk_df = get_migration_risk_df()
    print(f"Generated dataframe with shape: {migration_risk_df.shape}")
    
    # Create data directory if it doesn't exist
    workspace_root = get_workspace_root()
    #print(f"Workspace root: {workspace_root}")
    data_dir = workspace_root / 'data'
    #print(f"Data dir: {data_dir}")
    demographics_dir = data_dir / 'demographics_signal'
    print(f"Demographics dir: {demographics_dir}")
    #os.makedirs(demographics_dir, exist_ok=True)
    
    # Save to CSV
    output_path = demographics_dir / 'demographic_climate_signals.csv'
    print(f"Attempting to save to: {output_path}")
    #print(f"File exists before saving: {output_path.exists()}")
    migration_risk_df.to_csv(output_path, index=False)
    #print(f"File exists after saving: {output_path.exists()}")
    #print(f"Generated demographic climate signals dataset at: {output_path}")
    return output_path

def main():
    """Generate the demographic climate signals dataset."""
    try:
        output_path = generate_climate_signals_dataset()
        print(f"\nSuccessfully generated dataset at: {output_path}")
        
        # Print some basic statistics about the generated data
        df = pd.read_csv(output_path)
        print("\nDataset Summary:")
        print(f"Total counties analyzed: {len(df)}")
        print("\nClimate Migration Signal Distribution:")
        print(df['climate_migration_signal'].value_counts())
        
    except Exception as e:
        print(f"Error generating dataset: {str(e)}")
        raise

if __name__ == '__main__':
    main()