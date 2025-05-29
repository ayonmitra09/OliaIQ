"""
Create a simplified county-level flood and precipitation reference dataset.
One row per county, with columns for each variable-year combination.
Combines both CMIP6 threshold data and MWBM hydrological variables.

Variables included:
- CMIP6 Thresholds (annual):
  * R95pDAYS: Number of very wet days (> 95th percentile)
  * R20mm: Number of very heavy precipitation days (≥ 20mm)
  * R40mm: Number of extremely heavy precipitation days (≥ 40mm)
  * PRCPTOT: Annual total precipitation
  * Rx5day: Annual maximum 5-day precipitation
  * SDII: Simple daily precipitation intensity index
- MWBM (annual statistics from monthly data):
  * Runoff: Surface water runoff (mm/mo)
  * Deficit: Evapotranspiration deficit (mm/mo)
  * Storage: Soil moisture storage (mm)
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import dask.dataframe as dd
import dask

# Enable dask's multiprocessing scheduler
dask.config.set(scheduler='processes')

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

def process_threshold_data(nc_file, variables, years=range(2010, 2061, 5)):
    """Process CMIP6 threshold data for flood/precipitation variables."""
    print(f"Processing threshold data from {nc_file}...")
    
    ds = xr.open_dataset(nc_file)
    data_list = []
    
    # Define variable types for appropriate data conversion
    integer_vars = ['R95pDAYS', 'R20mm', 'R40mm']  # Count-based variables
    float_vars = ['PRCPTOT', 'Rx5day', 'SDII']  # Measurement-based variables
    
    for var in variables:
        try:
            if var not in ds.data_vars:
                print(f"Warning: Variable {var} not found in dataset")
                print("Available variables:", list(ds.data_vars))
                continue
                
            print(f"\nProcessing {var}...")
            var_data = ds[var]
            df = var_data.to_dataframe().reset_index()
            
            # Handle GEOID encoding
            if isinstance(df['GEOID'].iloc[0], bytes):
                df['GEOID'] = df['GEOID'].str.decode('utf-8')
            
            # Handle timedelta variables
            if df[var].dtype == 'timedelta64[ns]':
                print(f"Converting timedelta to days for {var}")
                df[var] = df[var].dt.total_seconds() / (24 * 3600)  # Convert to days
            
            # Convert time to year if needed
            if 'time' in df.columns:
                df['year'] = df['time'].dt.year
                df = df[df['year'].isin(years)]
            
            # Pivot to get years as columns
            df_pivot = df.pivot(index='GEOID', columns='year', values=var)
            df_pivot.columns = [f"{var}_{int(year)}" for year in df_pivot.columns]
            
            # Convert to appropriate type
            if var in integer_vars:
                df_pivot = df_pivot.round().astype('Int64')  # pandas nullable integer type
            elif var in float_vars:
                df_pivot = df_pivot.astype('float32')  # 32-bit float for efficiency
            
            data_list.append(df_pivot)
            
        except Exception as e:
            print(f"\nError processing variable {var}: {str(e)}")
            continue
    
    return pd.concat(data_list, axis=1) if data_list else None

def process_mwbm_data(nc_file, variables=['runoff', 'aet', 'pet', 'deficit', 'snow', 'stor'], years=range(2010, 2061, 5)):
    """Process monthly MWBM data into annual values with year-by-year processing."""
    print(f"Processing MWBM data from {nc_file}...")
    
    # Define aggregation method for each variable
    sum_vars = ['runoff', 'aet', 'pet', 'deficit', 'snow']
    mean_vars = ['stor']
    
    # Initialize dictionary to store results for each variable
    var_results = {var: pd.DataFrame() for var in variables}
    
    # Process one year at a time
    for year in years:
        print(f"\nProcessing year {year}...")
        
        # Open dataset for just this year
        with xr.open_dataset(
            nc_file,
            decode_timedelta=False,
        ) as ds:
            # Select data for this year only
            ds = ds.sel(time=str(year))
            
            for var in variables:
                try:
                    if var not in ds.data_vars:
                        print(f"Warning: Variable {var} not found in dataset")
                        continue
                    
                    print(f"Processing {var} for {year}...")
                    
                    # Convert to dataframe
                    df = ds[var].to_dataframe()
                    
                    # Handle GEOID encoding if needed
                    if isinstance(df.index.get_level_values('GEOID')[0], bytes):
                        df.index = df.index.set_levels([level.str.decode('utf-8') if isinstance(level[0], bytes) else level for level in df.index.levels])
                    
                    # Calculate annual value
                    if var in sum_vars:
                        annual_value = df.groupby('GEOID')[var].sum()
                    else:  # mean_vars
                        annual_value = df.groupby('GEOID')[var].mean()
                    
                    # Store result with year as column name
                    var_results[var][f"{var}_{year}"] = annual_value
                    
                except Exception as e:
                    print(f"\nError processing {var} for year {year}:")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {str(e)}")
                    continue
    
    # Combine all variables
    print("\nCombining all variables...")
    data_list = []
    for var in variables:
        if not var_results[var].empty:
            data_list.append(var_results[var])
    
    return pd.concat(data_list, axis=1) if data_list else None

def main():
    # File paths
    data_dir = Path("data")
    monthly_mwbm_file = data_dir / "climate_datasets" / "cmip6" / "CMIP6-LOCA2_MWBM_1950-2100_County2023_monthly.nc"
    annual_file = data_dir / "climate_datasets" / "cmip6" / "CMIP6-LOCA2_Thresholds_WeightedMultiModelMean.ssp245_1950-2100_annual_timeseries_by_county.nc"
    census_file = data_dir / "climate_datasets" / "co-est2024-alldata.csv"
    
    # Load county mapping
    county_mapping = load_county_mapping(census_file)
    
    # Process precipitation threshold data
    precip_vars = ['R95pDAYS', 'R20mm', 'R40mm', 'PRCPTOT', 'Rx5day', 'SDII']
    threshold_df = process_threshold_data(annual_file, precip_vars)
    
    # Process MWBM data
    mwbm_vars = ['runoff', 'aet', 'pet', 'deficit', 'snow', 'stor']
    mwbm_df = process_mwbm_data(monthly_mwbm_file, mwbm_vars)
    
    # Combine datasets
    combined_df = pd.concat([threshold_df, mwbm_df], axis=1)
    
    # Add state and county names
    final_df = combined_df.reset_index().merge(county_mapping, on='GEOID', how='left')
    
    # Reorder columns to put identifying information first
    cols = ['GEOID', 'state_name', 'county_name'] + [
        col for col in final_df.columns 
        if col not in ['GEOID', 'state_name', 'county_name']
    ]
    final_df = final_df[cols]
    
    # Save to CSV
    output_file = data_dir / "climate_datasets" / "climate_flood_precip_trends.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved reference dataset to {output_file}")
    print(f"Final shape: {final_df.shape}")
    
    # Print summary of included variables
    print("\nIncluded variables:")
    print("\nPrecipitation Threshold Variables:")
    for var in precip_vars:
        print(f"- {var}")
    print("\nMWBM Variables:")
    for var in mwbm_vars:
        print(f"- {var}")
        print(f"- {var} (annual mean)")

if __name__ == "__main__":
    main()
