import os
import requests
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from shapely.geometry import box, Polygon
import pandas as pd
from urllib.parse import urlencode

# Add state FIPS to name mapping
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
    '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming', '72': 'Puerto Rico'
}

def get_elevation_tile(min_lon, min_lat, max_lon, max_lat, output_dir):
    """
    Download elevation data from USGS 3DEP services for a specified bounding box
    
    Parameters:
    min_lon, min_lat (float): Southwest corner coordinates
    max_lon, max_lat (float): Northeast corner coordinates
    output_dir (str): Directory to save the downloaded data
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
    
    # USGS 3DEP WMS service URL
    base_url = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer"
    
    # Parameters for the WMS request - reducing resolution to start
    params = {
        'SERVICE': 'WMS',
        'VERSION': '1.3.0',
        'REQUEST': 'GetMap',
        'FORMAT': 'image/tiff',
        'TRANSPARENT': 'true',
        'LAYERS': '3DEPElevation:None',
        'CRS': 'EPSG:4326',
        'STYLES': '',
        'WIDTH': '1000',  # Reduced from 3000 to test
        'HEIGHT': '1000',
        'BBOX': f"{min_lat},{min_lon},{max_lat},{max_lon}"  # WMS 1.3.0 uses lat,lon order
    }
    
    url = f"{base_url}?{urlencode(params)}"
    tile_name = f"dem_{min_lat}_{min_lon}_{max_lat}_{max_lon}"
    output_file = os.path.join(output_dir, f"{tile_name}.tif")
    
    print(f"Attempting to download from URL: {url}")
    print(f"Output will be saved to: {output_file}")
    
    if not os.path.exists(output_file):
        print(f"Downloading elevation data for area: {min_lat}°N to {max_lat}°N, {min_lon}°E to {max_lon}°E")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded to {output_file}")
            
            # Verify the file was created and has content
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
            else:
                print("Warning: File was not created or is empty")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error downloading elevation data: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
    else:
        print(f"File already exists: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    return output_file

def process_elevation_by_county(dem_file, county_shp):
    """
    Calculate elevation statistics for each county that intersects with the DEM tile
    """
    # Read the DEM data
    with rasterio.open(dem_file) as src:
        # Get the bounding box of the DEM tile
        bounds = src.bounds
        tile_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        
        # Read county data
        counties = gpd.read_file(county_shp)
        
        # Reproject counties to match DEM CRS if needed
        if counties.crs != src.crs:
            counties = counties.to_crs(src.crs)
        
        # Filter counties that intersect with the DEM tile
        counties['intersects'] = counties.geometry.intersects(tile_bbox)
        intersecting_counties = counties[counties['intersects']]
        
        results = []
        
        for idx, county in intersecting_counties.iterrows():
            try:
                # Mask the DEM data to the county boundary
                masked_data, masked_transform = mask(src, [county.geometry], crop=True)
                
                # Calculate statistics
                valid_data = masked_data[masked_data != src.nodata]
                if len(valid_data) > 0:
                    stats = {
                        'GEOID': county['GEOID'],
                        'NAME': county['NAME'],
                        'STATE': STATE_FIPS.get(county['STATEFP'], county['STATEFP']),  # Use state name instead of FIPS
                        'min_elevation': float(np.min(valid_data)),
                        'max_elevation': float(np.max(valid_data)),
                        'mean_elevation': float(np.mean(valid_data))
                    }
                    results.append(stats)
            except Exception as e:
                print(f"Error processing county {county['NAME']}: {str(e)}")
        
        return pd.DataFrame(results)

def main():
    # Create output directory if it doesn't exist
    output_dir = "data/climate_datasets/geographic_data/dem"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download a sample 5x5 degree tile (covering part of California)
    # Coordinates for a 5x5 degree tile centered around Northern California
    min_lat, max_lat = 37.5, 42.5  # 5 degree range
    min_lon, max_lon = -122.5, -117.5  # 5 degree range
    
    dem_file = get_elevation_tile(min_lon, min_lat, max_lon, max_lat, output_dir)
    
    if dem_file:
        # Process county elevations
        county_shp = "data/climate_datasets/geographic_data/tl_2024_us_county.shp"
        results_df = process_elevation_by_county(dem_file, county_shp)
        
        # Save results
        output_csv = "data/climate_datasets/geographic_data/county_elevations.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
