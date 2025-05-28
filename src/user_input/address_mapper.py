import pandas as pd
import os
import requests
import time

def load_fips_mapping():
    """
    Loads FIPS mapping from the official 2020 Census TXT file.
    Returns a DataFrame with standardized state_code, county_code, and names.
    """

    # Build the path to the text file in the ../data folder
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'national_county2020.txt')
    
    # Load the pipe-delimited file
    fips_df = pd.read_csv(
        file_path,
        delimiter='|',
        dtype={'STATEFP': str, 'COUNTYFP': str},
        names=['state_name', 'state_code', 'county_code', 'count_code_2','county_name', 'class_code','func_code'],
        header=None
    )
    
    # Zero-pad state and county codes to make them merge-safe
    fips_df['state_code'] = fips_df['state_code'].str.zfill(2)
    fips_df['county_code'] = fips_df['county_code'].str.zfill(3)

    return fips_df[['state_code', 'county_code', 'state_name', 'county_name']]

def provide_fips_codes(address, max_retries=3):
    """
    Geocode an address using the Census Geocoder API and return FIPS codes directly.
    Includes retry logic for reliability.
    """
    base_url = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
    
    # Parameters for the Census Geocoder API
    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
        "layers": "all"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            data = response.json()
            
            # Check if we got any results
            result = data.get('result', {}).get('addressMatches', [])
            if not result:
                raise ValueError("Could not geocode the provided address")
            
            # Get the first match
            match = result[0]
            
            # Extract state and county FIPS codes
            geographies = match.get('geographies', {}).get('Counties', [])
            if not geographies:
                raise ValueError("Could not find county information for this address")
            
            county_info = geographies[0]
            state_code = county_info.get('STATE')
            county_code = county_info.get('COUNTY')
            
            # Load FIPS mapping to get state and county names
            fips_df = load_fips_mapping()
            location_info = fips_df[
                (fips_df['state_code'] == state_code) & 
                (fips_df['county_code'] == county_code)
            ].iloc[0]
            
            return location_info['state_code'], location_info['county_code']

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                raise ValueError(f"Census Geocoding service unavailable after {max_retries} attempts: {str(e)}")
            time.sleep(2)  # Wait 2 seconds before retrying
        except Exception as e:
            raise ValueError(f"Error geocoding address: {str(e)}")