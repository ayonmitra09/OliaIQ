"""
Process county area data from the US Census Bureau cartographic boundary files.
Saves a clean CSV file with GEOID and areas for all US counties.

Instructions:
1. Download the county shapefile from:
   https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip
2. Extract the zip file to landraiq_mvp/data/cb_2020_us_county_500k/
3. Run this script to process the data
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

def process_county_areas():
    """
    Process county area data from local shapefile.
    Returns DataFrame with county GEOIDs and their areas.
    """
    print("Processing county boundaries...")
    
    # Read the shapefile
    shp_path = Path('landraiq_mvp/data/cb_2020_us_county_500k/cb_2020_us_county_500k.shp')
    if not shp_path.exists():
        raise FileNotFoundError(
            f"Shapefile not found at {shp_path}. Please download and extract the county boundary file first."
        )
    
    counties = gpd.read_file(shp_path)
    
    # Project to USA Contiguous Albers Equal Area Conic projection
    counties = counties.to_crs('ESRI:102003')
    
    # Calculate areas in square kilometers using the geometry
    counties['total_area_km2'] = counties.geometry.area / 1e6  # Convert from m² to km²
    
    # Create clean DataFrame with just the needed columns
    result = counties[['GEOID', 'total_area_km2']].copy()
    
    return result

def main():
    """Process and save county area data."""
    try:
        county_areas = process_county_areas()
        
        # Save to CSV
        output_file = Path('landraiq_mvp/data/county_areas_2020.csv')
        county_areas.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully saved county area data to {output_file}")
        print(f"Total counties processed: {len(county_areas)}")
        print("\nSummary statistics (in square kilometers):")
        print(county_areas['total_area_km2'].describe())
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease follow these steps:")
        print("1. Download the county shapefile from:")
        print("   https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip")
        print("2. Extract the zip file to landraiq_mvp/data/cb_2020_us_county_500k/")
        print("3. Run this script again")

if __name__ == "__main__":
    main() 