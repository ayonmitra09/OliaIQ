"""
Create a simplified county-level climate reference dataset.
One row per county, with columns for each variable-year combination.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

def load_county_mapping(census_file):
    """Load county mapping from Census estimates file."""
    print(f"Loading county mapping from {census_file}")
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            print(f"Trying {encoding} encoding...")
            census_df = pd.read_csv(census_file, encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"Failed with {encoding} encoding")
            continue
    else:
        raise ValueError("Could not read file with any of the attempted encodings")
    
    # Create GEOID by combining state and county FIPS
    census_df['GEOID'] = census_df['STATE'].astype(str).str.zfill(2) + census_df['COUNTY'].astype(str).str.zfill(3)
    
    # Keep only necessary columns
    mapping_df = census_df[['GEOID', 'STNAME', 'CTYNAME']].copy()
    mapping_df = mapping_df.rename(columns={'STNAME': 'state_name', 'CTYNAME': 'county_name'})
    
    print(f"Loaded {len(mapping_df)} county mappings")
    return mapping_df

def process_climate_data(nc_file, variables, years=range(2010, 2061, 5)):
    """Process climate data for specified variables and years."""
    print(f"Opening {nc_file}...")
    print(f"File exists: {nc_file.exists()}")
    
    ds = xr.open_dataset(nc_file)
    print("\nDataset dimensions:", ds.dims)
    print("Available variables:", list(ds.data_vars))
    
    # More robust GEOID type checking
    try:
        geoid_sample = ds.GEOID.values[0]
        print("GEOID type:", type(geoid_sample))
        print("Sample GEOID:", geoid_sample)
        if isinstance(geoid_sample, bytes):
            print("Sample GEOID decoded:", geoid_sample.decode('utf-8'))
    except Exception as e:
        print(f"Error checking GEOID: {str(e)}")
        print("Will attempt to proceed anyway...")
    
    # Initialize list to store data
    data_list = []
    
    # Define variable types and their handling methods
    integer_vars = ['CDD', 'FD', 'GSL', 'TR']  # Count-based variables
    float_vars = ['TX90p', 'TX95p', 'R95p', 'Rx5day', 'SDII', 'TXge95F']  # Measurement-based variables
    
    # Process each variable
    for var in variables:
        try:
            if var not in ds.data_vars:
                print(f"Warning: Variable {var} not found in dataset")
                print("Available variables:", list(ds.data_vars))
                continue
                
            print(f"\nProcessing {var}...")
            
            # Get the data
            var_data = ds[var]
            df = var_data.to_dataframe().reset_index()
            
            # Handle GEOID encoding
            if isinstance(df['GEOID'].iloc[0], bytes):
                df['GEOID'] = df['GEOID'].str.decode('utf-8')
            
            # Check for null GEOIDs
            if df['GEOID'].isnull().any():
                null_count = df['GEOID'].isnull().sum()
                print(f"Warning: Found {null_count} null GEOIDs")
                df = df.dropna(subset=['GEOID'])
            
            # Handle timedelta variables
            if df[var].dtype == 'timedelta64[ns]':
                print(f"Converting timedelta to days for {var}")
                df[var] = df[var].dt.total_seconds() / (24 * 3600)  # Convert to days
            
            # Convert time to year if needed
            if 'time' in df.columns:
                df['year'] = df['time'].dt.year
                df = df[df['year'].isin(years)]
            
            # Pivot to get years as columns with variable prefix
            print("Pivoting data...")
            df_pivot = df.pivot(index='GEOID', columns='year', values=var)
            df_pivot.columns = [f"{var}_{int(year)}" for year in df_pivot.columns]
            
            # Convert to appropriate data type
            print(f"Converting {var} to appropriate data type...")
            if var in integer_vars:
                df_pivot = df_pivot.round().astype('Int64')  # pandas nullable integer type
            elif var in float_vars:
                df_pivot = df_pivot.astype('float32')  # 32-bit float for efficiency
            
            # Verify column count
            print(f"Columns for {var}: {len(df_pivot.columns)}")
            print(f"Sample of values for {var}:")
            print(df_pivot.iloc[:2, :2])  # Show first 2 rows and 2 columns
            
            data_list.append(df_pivot)
            
        except Exception as e:
            print(f"\nError processing variable {var}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Skipping this variable and continuing...")
            continue
    
    if not data_list:
        raise ValueError("No data was successfully processed")
    
    # Combine all variables
    print("\nCombining all processed variables...")
    result_df = pd.concat(data_list, axis=1)
    print(f"\nFinal shape: {result_df.shape}")
    
    # Verify total column count
    print("\nColumn count verification:")
    print(f"Total columns: {len(result_df.columns)}")
    print("Columns by variable:")
    for var in variables:
        var_cols = [col for col in result_df.columns if col.startswith(var)]
        print(f"{var}: {len(var_cols)} columns")
    
    return result_df

def process_winter_tnn(nc_file, years=range(2010, 2061, 5)):
    """Process winter minimum temperature (TNn) from monthly data."""
    print("\nProcessing winter minimum temperature data...")
    print(f"Opening {nc_file}...")
    
    ds = xr.open_dataset(nc_file)
    var_data = ds['TNn']
    
    # Convert to DataFrame
    df = var_data.to_dataframe().reset_index()
    
    # Handle GEOID encoding
    if isinstance(df['GEOID'].iloc[0], bytes):
        df['GEOID'] = df['GEOID'].str.decode('utf-8')
    
    # Extract month and year
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    
    # Adjust year for December (should count towards next year's winter)
    df.loc[df['month'] == 12, 'year'] = df.loc[df['month'] == 12, 'year'] + 1
    
    # Filter for winter months and relevant years
    target_years = list(years) + [min(years) - 1]  # Include previous year for December
    df = df[df['year'].isin(target_years) & df['month'].isin([12, 1, 2])]
    
    # Group by GEOID and year to get winter minimum
    winter_min = df.groupby(['GEOID', 'year'])['TNn'].min().reset_index()
    
    # Filter for our target years
    winter_min = winter_min[winter_min['year'].isin(years)]
    
    # Pivot to get years as columns
    df_pivot = winter_min.pivot(index='GEOID', columns='year', values='TNn')
    df_pivot.columns = [f"WMin_{int(year)}" for year in df_pivot.columns]
    
    # Convert to float32 for consistency
    df_pivot = df_pivot.astype('float32')
    
    print(f"Processed winter minimum temperature data shape: {df_pivot.shape}")
    print("Sample of winter minimum values:")
    print(df_pivot.iloc[:2, :2])
    
    return df_pivot

def main():
    # File paths
    data_dir = Path("data")
    monthly_file = data_dir / "climate_datasets"/ "cmip6" / "CMIP6-LOCA2_Thresholds_WeightedMultiModelMean.ssp245_1950-2100_monthly_timeseries_by_county.nc"
    annual_file = data_dir / "climate_datasets"/ "cmip6" / "CMIP6-LOCA2_Thresholds_WeightedMultiModelMean.ssp245_1950-2100_annual_timeseries_by_county.nc"
    census_file = data_dir / "climate_datasets"/ "co-est2024-alldata.csv"
    
    # Load county mapping
    county_mapping = load_county_mapping(census_file)
    
    # Process winter TNn separately
    winter_tnn_df = process_winter_tnn(monthly_file)
    
    # Process annual data including percentile-based indices
    annual_vars = ['TX90p', 'TX95p', 'CDD', 'FD', 'GSL', 'R95p', 'Rx5day', 'SDII', 'TR', 'TXge95F']
    annual_df = process_climate_data(annual_file, annual_vars)
    
    # Combine annual data with winter TNn
    combined_df = pd.concat([annual_df, winter_tnn_df], axis=1)
    
    # Add state and county names
    final_df = combined_df.reset_index().merge(county_mapping, on='GEOID', how='left')
    
    # Reorder columns to put identifying information first
    cols = ['GEOID', 'state_name', 'county_name'] + [col for col in final_df.columns if col not in ['GEOID', 'state_name', 'county_name']]
    final_df = final_df[cols]
    
    # Save to CSV
    output_file = data_dir / "climate_datasets" / "climate_heat_precip_trends.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved reference dataset to {output_file}")
    print(f"Final shape: {final_df.shape}")
    print(f"Total columns: {len(final_df.columns)}")
    
    # Print sample of final dataset with dtypes
    print("\nSample of final dataset:")
    print(final_df.head())
    print("\nColumn data types:")
    print(final_df.dtypes)

if __name__ == "__main__":
    main() 