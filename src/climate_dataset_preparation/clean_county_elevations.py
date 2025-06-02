import pandas as pd
import numpy as np

def load_census_counties():
    """
    Load and process Census county estimates data
    """
    # Try different encodings that Census commonly uses
    for encoding in ['latin1', 'windows-1252', 'cp1252']:
        try:
            census_df = pd.read_csv('data/climate_datasets/co-est2024-alldata.csv', encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    # Extract state and county names and FIPS codes
    census_df['state'] = census_df['STNAME']
    census_df['county'] = census_df['CTYNAME']
    census_df['fips'] = census_df['STATE'].astype(str).str.zfill(2) + census_df['COUNTY'].astype(str).str.zfill(3)
    
    # Remove state-level records (where COUNTY == 0)
    census_df = census_df[census_df['COUNTY'] != 0]
    
    # Remove Alaska and Hawaii (FIPS codes 02 and 15)
    census_df = census_df[~census_df['STATE'].isin([2, 15])]
    
    # Create a clean dataset with just state, county names and FIPS
    census_counties = census_df[['state', 'county', 'fips']].copy()
    
    return census_counties

def print_detailed_mismatches(elevation_df, census_df):
    """
    Print detailed information about mismatched counties using FIPS codes
    """
    elevation_fips = set(elevation_df['fips'])
    census_fips = set(census_df['fips'])
    
    missing = census_fips - elevation_fips
    extra = elevation_fips - census_fips
    
    if missing:
        print("\nMissing from elevation data (in Census but not in elevation data):")
        missing_list = []
        for fips in sorted(missing):
            census_row = census_df[census_df['fips'] == fips].iloc[0]
            missing_list.append({
                'state': census_row['state'],
                'county': census_row['county'],
                'fips': fips
            })
        
        # Print in a formatted table
        print("\n{:<20} {:<30} {:<10}".format("State", "County", "FIPS"))
        print("-" * 60)
        for item in sorted(missing_list, key=lambda x: (x['state'], x['county'])):
            print("{:<20} {:<30} {:<10}".format(
                item['state'], item['county'], item['fips']
            ))
    
    if extra:
        print("\nExtra in elevation data (in elevation data but not in Census):")
        extra_list = []
        for fips in sorted(extra):
            elevation_row = elevation_df[elevation_df['fips'] == fips].iloc[0]
            extra_list.append({
                'state': elevation_row['state'],
                'county': elevation_row['county'],
                'fips': fips
            })
        
        # Print in a formatted table
        print("\n{:<20} {:<30} {:<10}".format("State", "County", "FIPS"))
        print("-" * 60)
        for item in sorted(extra_list, key=lambda x: (x['state'], x['county'])):
            print("{:<20} {:<30} {:<10}".format(
                item['state'], item['county'], item['fips']
            ))

def clean_county_elevations():
    """
    Clean and validate county elevation data:
    1. Remove Alaska and Hawaii
    2. Check for and handle duplicates using FIPS codes
    3. Cross-reference with Census county data
    4. Print state-by-state record counts
    """
    print("Reading county elevation data...")
    df = pd.read_csv('data/climate_datasets/county_elevations.csv')
    print(f"Initial record count: {len(df)}")
    
    # Remove Alaska and Hawaii
    print("\nRemoving Alaska and Hawaii...")
    continental_df = df[~df['state'].isin(['Alaska', 'Hawaii'])]
    removed_count = len(df) - len(continental_df)
    print(f"Removed {removed_count} records from Alaska and Hawaii")
    
    # Check for duplicates using FIPS codes
    print("\nChecking for duplicates...")
    duplicates = continental_df.groupby(['fips']).size().reset_index(name='count')
    duplicates = duplicates[duplicates['count'] > 1]
    
    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} duplicate county entries:")
        print(duplicates)
        
        # For each duplicate, keep the record with the most extreme elevation range
        print("\nResolving duplicates by keeping records with largest elevation range...")
        continental_df['elevation_range'] = continental_df['max_elevation_m'] - continental_df['min_elevation_m']
        continental_df = continental_df.sort_values('elevation_range', ascending=False)
        continental_df = continental_df.drop_duplicates(['fips'], keep='first')
        continental_df = continental_df.drop('elevation_range', axis=1)
    
    # Cross-reference with Census county data
    print("\nCross-referencing with Census county data...")
    census_counties = load_census_counties()
    print(f"Census county count (excluding AK/HI): {len(census_counties)}")
    
    # Print detailed analysis of mismatches
    print("\nDetailed analysis of county mismatches:")
    print_detailed_mismatches(continental_df, census_counties)
    
    # Print state-by-state record counts
    print("\nRecord counts by state:")
    print("-" * 60)
    print("{:<25} {:<10} {:<10}".format("State", "Elevation", "Census"))
    print("-" * 60)
    
    elevation_counts = continental_df['state'].value_counts().sort_index()
    census_counts = census_counties['state'].value_counts().sort_index()
    
    all_states = sorted(set(elevation_counts.index) | set(census_counts.index))
    for state in all_states:
        elev_count = elevation_counts.get(state, 0)
        census_count = census_counts.get(state, 0)
        diff = elev_count - census_count
        status = ""
        if diff != 0:
            sign = "+" if diff > 0 else ""
            status = f"({sign}{diff})"
        print("{:<25} {:<10} {:<10} {}".format(
            state, elev_count, census_count, status
        ))
    
    print("\nSummary:")
    print(f"Total elevation records: {len(continental_df)}")
    print(f"Total Census records: {len(census_counties)}")
    print(f"Difference: {len(continental_df) - len(census_counties)}")
    
    # Save cleaned data with all FIPS codes preserved
    output_file = 'data/climate_datasets/county_elevations_cleaned.csv'
    continental_df.to_csv(output_file, index=False)
    print(f"\nCleaned data saved to {output_file}")
    
    return continental_df

if __name__ == "__main__":
    clean_county_elevations() 