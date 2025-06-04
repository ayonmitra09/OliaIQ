"""
Process NOAA Sea Level Rise and Storm Surge data to create county-level flooding projections.
Combines SLR scenarios with hurricane storm surge data to estimate future inundation risks.
"""

import os
import tempfile
import zipfile
import logging
from pathlib import Path
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, box, shape
from shapely.ops import transform
import pyproj
from tqdm import tqdm
import warnings
import concurrent.futures
import json
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
NOAA_BASE_URL = "https://coast.noaa.gov/slrdata/Depth_Rasters"
SURGE_LAYER_URL = "https://tiles.arcgis.com/tiles/C8EMgrsFcRFL6LrL/arcgis/rest/services/Storm_Surge_HazardMaps_Category3_v3/MapServer/0"
CACHE_DIR = Path("data/cache")
MAX_WORKERS = 4  # Adjust based on available CPU cores and memory

# States with coastal SLR data
COASTAL_STATES = [
    'AK', 'AL', 'CA', 'CT', 'DC', 'DE', 'FL', 'GA', 'LA', 'MA', 
    'MD', 'ME', 'MS', 'NC', 'NH', 'NJ', 'NY', 'OR', 'PA', 'RI', 
    'SC', 'TX', 'VA', 'WA'
]

SLR_MAPPING = {
    2010: 0, 2015: 0.5, 2020: 0.5,
    2025: 0.5, 2030: 1, 2035: 1,
    2040: 1.5, 2045: 1.5, 2050: 2,
    2055: 2, 2060: 2.5
}
SURGE_MULTIPLIERS = {
    0.0: 1.00,
    0.5: 1.10,
    1.0: 1.25,
    1.5: 1.40,
    2.0: 1.60,
    2.5: 1.80,
    3.0: 2.00,
}

# NYC Metro Area Counties (NY, NJ, CT)
NYC_METRO_COUNTIES = {
    # New York City Boroughs only
    '36005': 'Bronx',    # Bronx County
    '36047': 'Brooklyn', # Kings County
    '36061': 'Manhattan',# New York County
    '36081': 'Queens',   # Queens County
    '36085': 'Staten Island', # Richmond County
}

# Get required SLR heights from mapping
REQUIRED_SLR_HEIGHTS = sorted(set(SLR_MAPPING.values()))

def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """Download a file from URL to specified path with error handling."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
                
        progress_bar.close()
        logger.info(f"Successfully downloaded {url} to {output_path}")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        raise

@lru_cache(maxsize=32)
def get_state_url_list(state_code: str) -> list:
    """Get list of SLR data URLs for a given state with caching."""
    cache_file = CACHE_DIR / f"urllist_{state_code}.json"
    
    # Check cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    
    try:
        url = f"{NOAA_BASE_URL}/{state_code}/URLlist_{state_code}.txt"
        response = requests.get(url)
        response.raise_for_status()
        urls = response.text.strip().split()
        
        # Cache the results
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(urls, f)
            
        logger.info(f"Found {len(urls)} SLR files for {state_code}")
        return urls
    except Exception as e:
        logger.error(f"Failed to get URL list for {state_code}: {str(e)}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_file_with_retry(url: str, output_path: Path) -> None:
    """Download a file with retry logic."""
    download_file(url, output_path)

def get_state_code_from_fips(fips: str) -> str:
    """Convert state FIPS code to two-letter state code."""
    # First two digits of FIPS are state code
    state_fips = fips[:2]
    # Map of FIPS codes to state codes for coastal states
    FIPS_TO_STATE = {
        '02': 'AK', '01': 'AL', '06': 'CA', '09': 'CT', '11': 'DC', 
        '10': 'DE', '12': 'FL', '13': 'GA', '22': 'LA', '25': 'MA',
        '24': 'MD', '23': 'ME', '28': 'MS', '37': 'NC', '33': 'NH', 
        '34': 'NJ', '36': 'NY', '41': 'OR', '42': 'PA', '44': 'RI',
        '45': 'SC', '48': 'TX', '51': 'VA', '53': 'WA'
    }
    return FIPS_TO_STATE.get(state_fips)

def process_slr_for_county(county: gpd.GeoSeries, raster_path: Path, counties_crs) -> tuple:
    """Process SLR raster for a single county with error handling."""
    try:
        with rasterio.open(raster_path) as src:
            # Validate raster data
            if src.count == 0:
                raise ValueError(f"Raster file {raster_path.name} has no bands")
            if src.width == 0 or src.height == 0:
                raise ValueError(f"Raster file {raster_path.name} has invalid dimensions")
            
            # Check for valid data range (SLR depths should be in feet)
            data = src.read(1)
            if not np.any(data > 0):
                raise ValueError(f"Raster file {raster_path.name} contains no positive depth values")
            if np.all(np.isnan(data)):
                raise ValueError(f"Raster file {raster_path.name} contains only NaN values")
            
            # Create a single-row GeoDataFrame for the county
            county_gdf = gpd.GeoDataFrame(geometry=[county.geometry], crs=counties_crs)
            
            # Reproject county geometry to raster CRS if needed
            if src.crs != county_gdf.crs:
                transformer = pyproj.Transformer.from_crs(
                    county_gdf.crs,
                    src.crs,
                    always_xy=True
                )
                county_geom = transform(transformer.transform, county.geometry)
            else:
                county_geom = county.geometry
                
            try:
                # Clip raster to county boundary
                out_image, out_transform = mask(src, [mapping(county_geom)], crop=True)
                
                # Validate clipped data
                if out_image.size == 0:
                    logger.warning(f"No data found in {raster_path.name} for county {county.GEOID}")
                    return 0.0, 0.0
                
                # Calculate pixel area in km²
                pixel_area_km2 = abs(out_transform[0] * out_transform[4]) / 1_000_000
                
                # Count inundated pixels (>0)
                inundated_pixels = np.count_nonzero(out_image > 0)
                inundated_area_km2 = inundated_pixels * pixel_area_km2
                
                # Calculate county area and percentage
                county_area_km2 = county_gdf.to_crs(src.crs).area.values[0] / 1_000_000
                pct_inundated = (inundated_area_km2 / county_area_km2) * 100
                
                # Validate results
                if inundated_area_km2 > county_area_km2:
                    logger.warning(f"Inundated area ({inundated_area_km2:.2f} km²) exceeds county area ({county_area_km2:.2f} km²) for {county.GEOID}")
                    pct_inundated = 100.0
                
                return inundated_area_km2, min(pct_inundated, 100.0)
                
            except (ValueError, rasterio.errors.RasterioError) as e:
                logger.warning(f"Failed to process county {county.GEOID}: {str(e)}")
                return 0.0, 0.0
                
    except Exception as e:
        logger.error(f"Error processing SLR for county {county.GEOID}: {str(e)}")
        return 0.0, 0.0

def process_state_data(state_code: str, counties: gpd.GeoDataFrame, temp_dir: Path) -> dict:
    """Process SLR data for a single state."""
    results = {}
    try:
        # Create temporary state directory
        state_dir = temp_dir / state_code
        state_dir.mkdir(exist_ok=True)
        
        # Get state's counties
        state_counties = counties[counties['STATEFP'] == get_fips_from_state_code(state_code)]
        if state_counties.empty:
            logger.error(f"No counties found for state {state_code}")
            return results
        
        logger.info(f"Processing {len(state_counties)} counties in state {state_code}")
        
        # Get URLs for this state
        urls = get_state_url_list(state_code)
        if not urls:
            logger.error(f"No SLR data URLs found for state {state_code}")
            return results
            
        # Track successful downloads and processing
        successful_heights = set()
        failed_heights = set()
        
        # Process each required height
        for height in REQUIRED_SLR_HEIGHTS:
            # Convert height to string format (e.g., "1_5" for 1.5)
            height_str = str(height).replace('.', '_')
            height_urls = [url for url in urls if f"slr_depth_{height_str}ft.tif" in url]
            
            if not height_urls:
                logger.warning(f"No SLR data files found for height {height}ft in state {state_code}")
                failed_heights.add(height)
                continue
                
            height_success = False
            # Process each region file for this height
            for url in height_urls:
                try:
                    filename = url.split('/')[-1]
                    output_path = state_dir / filename
                    
                    # Download file with retry
                    logger.info(f"Downloading {filename}...")
                    download_file_with_retry(url, output_path)
                    
                    # Validate file size
                    if not output_path.exists() or output_path.stat().st_size == 0:
                        logger.error(f"Downloaded file {filename} is empty or missing")
                        continue
                    
                    # Process counties in this state
                    for _, county in state_counties.iterrows():
                        if county.GEOID not in results:
                            results[county.GEOID] = {}
                        
                        area, pct = process_slr_for_county(county, output_path, state_counties.crs)
                        logger.debug(f"Processed {filename} for county {county.GEOID}: area={area:.2f}, pct={pct:.2f}")
                        
                        # Only count as success if we got non-zero results
                        if area > 0 or pct > 0:
                            height_success = True
                        
                        # Update or aggregate results
                        if height not in results[county.GEOID]:
                            results[county.GEOID][height] = {'area': area, 'pct': pct}
                        else:
                            # Take maximum values if multiple regions exist
                            results[county.GEOID][height]['area'] = max(
                                results[county.GEOID][height]['area'], area
                            )
                            results[county.GEOID][height]['pct'] = max(
                                results[county.GEOID][height]['pct'], pct
                            )
                    
                    # Clean up file after processing
                    output_path.unlink()
                    
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {str(e)}")
                    continue
            
            if height_success:
                successful_heights.add(height)
            else:
                failed_heights.add(height)
                logger.error(f"No valid data found for height {height}ft in state {state_code}")
        
        # Clean up state directory
        state_dir.rmdir()
        
        # Log summary for this state
        logger.info(f"State {state_code} processing summary:")
        logger.info(f"  Counties processed: {len(results)}")
        logger.info(f"  Successful heights: {sorted(successful_heights)}")
        if failed_heights:
            logger.error(f"  Failed heights: {sorted(failed_heights)}")
        
        # Validate we have enough data
        if not successful_heights:
            logger.error(f"No valid SLR data processed for state {state_code}")
            return {}
        
        # Remove any counties with no valid data
        results = {
            geoid: heights for geoid, heights in results.items() 
            if any(v['area'] > 0 or v['pct'] > 0 for v in heights.values())
        }
        
        logger.info(f"Final results for state {state_code}: {len(results)} counties with data")
        return results
        
    except Exception as e:
        logger.error(f"Failed to process state {state_code}: {str(e)}")
        return {}

def get_fips_from_state_code(state_code: str) -> str:
    """Convert state code to FIPS code."""
    # Map of state codes to FIPS codes
    STATE_TO_FIPS = {
        'NY': '36', 'NJ': '34', 'CT': '09',  # Add more as needed
        'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
        'DE': '10', 'DC': '11', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16',
        'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22',
        'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
        'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NM': '35',
        'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42',
        'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
        'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56'
    }
    return STATE_TO_FIPS.get(state_code, '')

def download_slr_data(temp_dir: Path, counties: gpd.GeoDataFrame) -> dict:
    """Download and process SLR GeoTIFFs state by state using parallel processing."""
    # Get unique states from counties
    state_fips = counties['STATEFP'].unique()
    state_codes = [get_state_code_from_fips(fips) for fips in state_fips]
    state_codes = [code for code in state_codes if code in COASTAL_STATES]
    
    logger.info(f"Processing SLR data for states: {', '.join(state_codes)}")
    logger.info(f"Found {len(counties)} counties in states: {', '.join(state_fips)}")
    all_results = {}
    
    # Process states in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_state = {
            executor.submit(process_state_data, state, counties, temp_dir): state
            for state in state_codes
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_state), 
                         total=len(state_codes),
                         desc="Processing states"):
            state = future_to_state[future]
            try:
                state_results = future.result()
                logger.info(f"Got results for state {state} with {len(state_results)} counties")
                all_results.update(state_results)
                logger.info(f"Completed processing SLR data for state: {state}")
            except Exception as e:
                logger.error(f"Failed to process state {state}: {str(e)}")
    
    logger.info(f"Final SLR results contain data for {len(all_results)} counties")
    if not all_results:
        raise ValueError("No SLR data was successfully processed for any counties")
    return all_results

def query_surge_features(bbox: tuple, spatial_ref: int = 4326) -> dict:
    """Query surge features from ArcGIS REST API with retry logic."""
    params = {
        'f': 'geojson',
        'geometry': json.dumps({
            'spatialReference': {'wkid': spatial_ref},
            'xmin': bbox[0],
            'ymin': bbox[1],
            'xmax': bbox[2],
            'ymax': bbox[3]
        }),
        'geometryType': 'esriGeometryEnvelope',
        'spatialRel': 'esriSpatialRelIntersects',
        'outFields': '*',
        'returnGeometry': True,
        'outSR': spatial_ref,
        'where': '1=1'  # Return all features
    }
    
    response = requests.get(
        f"{SURGE_LAYER_URL}/query",
        params=params,
        headers={'User-Agent': 'Mozilla/5.0'}  # Add user agent to avoid 403 errors
    )
    response.raise_for_status()
    return response.json()

def process_mom_surge(counties: gpd.GeoDataFrame) -> pd.DataFrame:
    """Process MOM storm surge data for all counties using ArcGIS REST API."""
    try:
        # Ensure counties are in WGS84 (EPSG:4326)
        if counties.crs != 'EPSG:4326':
            counties = counties.to_crs('EPSG:4326')
        
        results = []
        
        # Process counties in smaller chunks to avoid timeout
        chunk_size = 5  # Reduced from 10 to 5
        county_chunks = [counties.iloc[i:i+chunk_size] for i in range(0, len(counties), chunk_size)]
        
        for chunk in tqdm(county_chunks, desc="Processing storm surge by county group"):
            try:
                # Add buffer to bounding box to ensure we get all intersecting features
                bounds = chunk.total_bounds
                buffer_deg = 0.1  # ~11km at these latitudes
                bbox = (
                    bounds[0] - buffer_deg,  # xmin
                    bounds[1] - buffer_deg,  # ymin
                    bounds[2] + buffer_deg,  # xmax
                    bounds[3] + buffer_deg   # ymax
                )
                
                # Query surge features for this chunk
                surge_data = query_surge_features(bbox)
                
                if not surge_data.get('features'):
                    logger.warning(f"No surge features found for chunk with bounds: {bbox}")
                    # Add zero results for counties in this chunk
                    for _, county in chunk.iterrows():
                        results.append({
                            'GEOID': county.GEOID,
                            'baseline_mom_area_km2': 0.0,
                            'baseline_pct_cat3_surge_inundated': 0.0
                        })
                    continue
                
                # Convert to GeoDataFrame
                surge_gdf = gpd.GeoDataFrame.from_features(
                    surge_data['features'],
                    crs='EPSG:4326'
                )
                
                # Process each county in chunk
                for _, county in chunk.iterrows():
                    try:
                        if surge_gdf.intersects(county.geometry).any():
                            # Calculate intersection
                            intersection = gpd.overlay(
                                gpd.GeoDataFrame(geometry=[county.geometry], crs=counties.crs),
                                surge_gdf,
                                how='intersection'
                            )
                            
                            # Calculate areas using equal area projection
                            area_crs = pyproj.CRS.from_string('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs')
                            intersection_area = intersection.to_crs(area_crs).area.sum() / 1_000_000  # Convert to km²
                            county_area = gpd.GeoDataFrame(geometry=[county.geometry], crs=counties.crs).to_crs(area_crs).area.values[0] / 1_000_000
                            
                            baseline_pct = min((intersection_area / county_area) * 100, 100.0)
                        else:
                            intersection_area = 0.0
                            baseline_pct = 0.0
                            
                        results.append({
                            'GEOID': county.GEOID,
                            'baseline_mom_area_km2': intersection_area,
                            'baseline_pct_cat3_surge_inundated': baseline_pct
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to process surge for county {county.GEOID}: {str(e)}")
                        results.append({
                            'GEOID': county.GEOID,
                            'baseline_mom_area_km2': 0.0,
                            'baseline_pct_cat3_surge_inundated': 0.0
                        })
                        
            except Exception as e:
                logger.error(f"Failed to process chunk with bounds {bbox}: {str(e)}")
                # Add zero results for all counties in failed chunk
                for _, county in chunk.iterrows():
                    results.append({
                        'GEOID': county.GEOID,
                        'baseline_mom_area_km2': 0.0,
                        'baseline_pct_cat3_surge_inundated': 0.0
                    })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Failed to process MOM surge data: {str(e)}")
        raise

def create_temporal_projections(counties: gpd.GeoDataFrame, 
                              slr_results: dict,
                              surge_baseline: pd.DataFrame) -> pd.DataFrame:
    """Create temporal projections combining SLR and surge data with error handling."""
    try:
        results = []
        
        # Log the data we're working with
        logger.info(f"Creating projections for {len(counties)} counties")
        logger.info(f"Have SLR results for {len(slr_results)} counties")
        logger.info(f"Have surge results for {len(surge_baseline)} counties")
        
        for _, county in counties.iterrows():
            try:
                # Get baseline surge data (defaulting to 0 if not found)
                baseline_surge = surge_baseline[surge_baseline['GEOID'] == county.GEOID].iloc[0] if not surge_baseline.empty else pd.Series({
                    'baseline_mom_area_km2': 0.0,
                    'baseline_pct_cat3_surge_inundated': 0.0
                })
                
                # Check if we have SLR data for this county
                if county.GEOID not in slr_results:
                    logger.warning(f"No SLR data found for county {county.GEOID} ({county.NAME}, {county.STUSPS})")
                    continue
                
                county_slr = slr_results[county.GEOID]
                if not county_slr:
                    logger.warning(f"Empty SLR data for county {county.GEOID} ({county.NAME}, {county.STUSPS})")
                    continue
                
                for year in range(2010, 2065, 5):
                    slr_level = SLR_MAPPING[year]
                    surge_multiplier = SURGE_MULTIPLIERS[slr_level]
                    
                    # Get SLR values for this level
                    if slr_level == 0:
                        slr_area, slr_pct = 0.0, 0.0
                    else:
                        # Handle case where exact level might not be available
                        if slr_level not in county_slr:
                            # Find closest available level
                            available_levels = sorted(county_slr.keys())
                            if not available_levels:
                                logger.warning(f"No SLR levels found for county {county.GEOID}")
                                continue
                            closest_level = min(available_levels, key=lambda x: abs(x - slr_level))
                            logger.info(f"Using {closest_level}ft data for {slr_level}ft projection in {county.NAME}, {county.STUSPS}")
                            slr_area = county_slr[closest_level]['area']
                            slr_pct = county_slr[closest_level]['pct']
                        else:
                            slr_area = county_slr[slr_level]['area']
                            slr_pct = county_slr[slr_level]['pct']
                    
                    # Calculate adjusted surge with more granular multipliers
                    future_surge_pct = baseline_surge['baseline_pct_cat3_surge_inundated'] * surge_multiplier
                    
                    results.append({
                        'county_fips': county.GEOID,
                        'county_name': county.NAME,
                        'state_fips': county.STATEFP,
                        'state_name': county.STUSPS,
                        'year': year,
                        'slr_level': slr_level,
                        'slr_area_km2': slr_area,
                        'pct_slr_inundated': slr_pct,
                        'surge_multiplier': surge_multiplier,
                        'pct_surge_inundated': min(future_surge_pct, 100.0)
                    })
                    
            except Exception as e:
                logger.error(f"Failed to process projections for county {county.GEOID} ({county.NAME}, {county.STUSPS}): {str(e)}")
                continue
        
        if not results:
            raise ValueError("No results were generated. Check the logs for details.")
            
        results_df = pd.DataFrame(results)
        logger.info(f"Created projections for {len(results_df['county_fips'].unique())} counties")
        return results_df
        
    except Exception as e:
        logger.error(f"Failed to create temporal projections: {str(e)}")
        raise

def load_counties(test_mode: bool = False) -> gpd.GeoDataFrame:
    """Load county boundaries from TIGER/Line shapefile with error handling."""
    try:
        counties_path = Path("data/climate_datasets/geographic_data/tl_2024_us_county.shp")
        if not counties_path.exists():
            raise FileNotFoundError(f"County shapefile not found at {counties_path}")
            
        counties = gpd.read_file(counties_path)
        
        # Add state postal codes
        state_fips_to_postal = {
            '36': 'NY', '34': 'NJ', '09': 'CT',  # Add more as needed
            '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO',
            '10': 'DE', '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID',
            '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA',
            '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS',
            '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '35': 'NM',
            '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA',
            '44': 'RI', '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
            '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY'
        }
        counties['STUSPS'] = counties['STATEFP'].map(state_fips_to_postal)
        
        if test_mode:
            # Filter to NYC metro area counties
            counties = counties[counties['GEOID'].isin(NYC_METRO_COUNTIES.keys())]
            logger.info(f"Test mode: Processing {len(counties)} counties in NYC metro area")
        else:
            # Filter to contiguous US
            counties = counties[~counties['STATEFP'].isin(['02', '15', '72'])]
            logger.info(f"Processing all {len(counties)} counties in contiguous US")
            
        counties['GEOID'] = counties['STATEFP'] + counties['COUNTYFP']
        return counties
        
    except Exception as e:
        logger.error(f"Failed to load county data: {str(e)}")
        raise

def main(test_mode: bool = False):
    """Main execution function with error handling."""
    try:
        # Create temporary directory within project's cache
        cache_root = Path("data/cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        
        with tempfile.TemporaryDirectory(dir=cache_root) as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Created temporary directory: {temp_path.absolute()}")
            
            # Create subdirectory for SLR data
            slr_path = temp_path / "slr"
            slr_path.mkdir()
            
            logger.info("Loading county boundaries...")
            counties = load_counties(test_mode)
            
            logger.info("Downloading SLR data...")
            logger.info(f"SLR data will be temporarily stored in: {slr_path.absolute()}")
            slr_results = download_slr_data(slr_path, counties)
            
            logger.info("Processing storm surge data from ArcGIS REST API...")
            surge_baseline = process_mom_surge(counties)
            
            # Create temporal projections
            logger.info("Creating temporal projections...")
            results_df = create_temporal_projections(counties, slr_results, surge_baseline)
            
            # Save results
            output_dir = Path("data/climate_datasets")
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / ("nyc_coastal_flooding_trends.csv" if test_mode else "coastal_flooding_trends.csv")
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
            # Temporary directory and contents will be automatically cleaned up here
            
    except Exception as e:
        logger.error(f"Failed to complete processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Run in test mode for NYC metro area only
    main(test_mode=True) 