import os
import requests
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from shapely.geometry import box, Polygon
import pandas as pd
from urllib.parse import urlencode
import tempfile
import shutil

# State FIPS to name mapping
STATE_FIPS = {
    '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas',
    '06': 'California', '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware',
    '11': 'District of Columbia', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii',
    '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa',
    '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine',
    '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota',
    '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska',
    '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico',
    '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio',
    '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island',
    '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas',
    '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington',
    '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming'
}

def get_elevation_tile(min_lon, min_lat, max_lon, max_lat, temp_dir):
    """
    Download elevation data from USGS 3DEP services for a specified bounding box
    
    Parameters:
    min_lon, min_lat (float): Southwest corner coordinates
    max_lon, max_lat (float): Northeast corner coordinates
    temp_dir (str): Directory to save the temporary downloaded data
    """
    if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90")
    if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180")
    
    # Create WKT polygon for the bounding box
    bbox = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat)
    ])
    
    # USGS 3DEP service parameters
    params = {
        'SERVICE': 'WCS',
        'VERSION': '1.0.0',
        'REQUEST': 'GetCoverage',
        'COVERAGE': '3DEPElevation',
        'CRS': 'EPSG:4326',
        'RESPONSE_CRS': 'EPSG:4326',
        'FORMAT': 'GeoTIFF',
        'BBOX': f"{min_lon},{min_lat},{max_lon},{max_lat}",
        'WIDTH': '1000',
        'HEIGHT': '1000'
    }
    
    # USGS 3DEP endpoint
    url = 'https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer'
    
    # Download the data
    print(f"Downloading DEM tile for bbox: {params['BBOX']}")
    response = requests.get(f"{url}?{urlencode(params)}", stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download DEM data: {response.status_code}")
    
    # Save to temporary file
    temp_file = os.path.join(temp_dir, 'temp_dem.tif')
    with open(temp_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    return temp_file

def process_elevation_by_county(dem_file, county_shp, results_df):
    """
    Calculate elevation statistics for each county that intersects with the DEM tile
    and append to the existing results DataFrame
    """
    print("Reading DEM file...")
    try:
        with rasterio.open(dem_file) as src:
            # Get the bounding box of the DEM tile
            bounds = src.bounds
            tile_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            print("Reading county shapefile...")
            counties = gpd.read_file(county_shp)
            print(f"Successfully read {len(counties)} counties")
            
            # Print CRS information
            print(f"DEM CRS: {src.crs}")
            print(f"Counties CRS: {counties.crs}")
            
            # Ensure both are in the same CRS
            if counties.crs != src.crs:
                counties = counties.to_crs(src.crs)
            
            # Filter counties that intersect with the tile
            counties['intersects'] = counties.geometry.intersects(tile_bbox)
            counties = counties[counties['intersects']]
            print(f"Processing {len(counties)} counties that intersect with this tile")
            
            new_results = []
            for idx, county in counties.iterrows():
                try:
                    # Get the geometry in the correct CRS
                    geom = [county.geometry.__geo_interface__]
                    
                    # Mask the raster to the county boundary
                    out_image, out_transform = mask(src, geom, crop=True)
                    
                    # Calculate statistics
                    valid_data = out_image[0][out_image[0] != src.nodata]
                    if len(valid_data) > 0:
                        min_elev = float(np.min(valid_data))
                        max_elev = float(np.max(valid_data))
                        mean_elev = float(np.mean(valid_data))
                        
                        # Get state name from FIPS code
                        state_fips = str(county['STATEFP'])
                        state_name = STATE_FIPS.get(state_fips, 'Unknown')
                        
                        # Create combined FIPS code
                        county_fips = str(county['COUNTYFP'])
                        fips = state_fips.zfill(2) + county_fips.zfill(3)
                        
                        new_results.append({
                            'state': state_name,
                            'county': county['NAME'],
                            'fips': fips,
                            'state_fips': state_fips,
                            'county_fips': county_fips,
                            'min_elevation_m': min_elev,
                            'max_elevation_m': max_elev,
                            'mean_elevation_m': mean_elev
                        })
                        print(f"Processed {state_name} - {county['NAME']} (FIPS: {fips})")
                except Exception as e:
                    print(f"Error processing county {county['NAME']}: {str(e)}")
            
            # Convert new results to DataFrame and append
            if new_results:
                new_df = pd.DataFrame(new_results)
                return pd.concat([results_df, new_df], ignore_index=True)
    except Exception as e:
        print(f"Error processing DEM file: {str(e)}")
    
    return results_df

def process_nationwide():
    """
    Process elevation data for the entire United States
    """
    # Create output directory if it doesn't exist
    output_dir = 'data/climate_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results DataFrame with FIPS columns
    results_df = pd.DataFrame(columns=[
        'state', 'county', 'fips', 'state_fips', 'county_fips',
        'min_elevation_m', 'max_elevation_m', 'mean_elevation_m'
    ])
    
    # Create temporary directory for DEM files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define grid of 5x5 degree tiles covering the continental US
        # Adjusted to include Alaska and Hawaii
        grids = []
        
        # Continental US
        for lat in range(25, 50, 5):  # 25°N to 50°N
            for lon in range(-125, -65, 5):  # 125°W to 65°W
                grids.append((lat, lon))
        
        # Alaska
        for lat in range(55, 75, 5):  # 55°N to 75°N
            for lon in range(-170, -130, 5):  # 170°W to 130°W
                grids.append((lat, lon))
        
        # Hawaii
        grids.extend([
            (19, -160),  # Cover Hawaiian islands
            (19, -155)
        ])
        
        # Process each grid tile
        for lat, lon in grids:
            try:
                print(f"\nProcessing tile at {lat}°N, {lon}°W")
                
                # Download DEM for this tile
                dem_file = get_elevation_tile(lon, lat, lon + 5, lat + 5, temp_dir)
                
                # Process counties in this tile
                results_df = process_elevation_by_county(
                    dem_file,
                    'data/climate_datasets/geographic_data/tl_2024_us_county.shp',
                    results_df
                )
                
                # Remove the temporary DEM file
                os.remove(dem_file)
                
                # Save progress after each tile
                output_file = os.path.join(output_dir, 'county_elevations.csv')
                results_df.to_csv(output_file, index=False)
                print(f"Progress saved to {output_file}")
                
            except Exception as e:
                print(f"Error processing tile at {lat}°N, {lon}°W: {str(e)}")
                continue

if __name__ == "__main__":
    process_nationwide()
