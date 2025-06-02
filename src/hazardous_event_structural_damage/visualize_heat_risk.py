"""
Visualize structural climate risk components across US counties.
Shows heat risk change, cold relief, and net structural risk from 2020 to 2060.
Uses absolute delta-based normalization rather than z-scores.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data():
    """Load risk data and county geometries."""
    data_dir = Path("data")
    
    # Load risk data
    risk_df = pd.read_csv(data_dir / "hazardous_event_structural_signals" / "structural_heat_index.csv")
    risk_df['GEOID'] = risk_df['GEOID'].astype(str).str.zfill(5)
    
    # Load county geometries from TIGER/Line
    counties = gpd.read_file(data_dir / "climate_datasets" / "geographic_data" / "tl_2024_us_county.shp")
    counties['GEOID'] = counties['STATEFP'].astype(str).str.zfill(2) + counties['COUNTYFP'].astype(str).str.zfill(3)
    
    # Filter out Alaska (02), Hawaii (15), and Puerto Rico (72)
    counties = counties[~counties['STATEFP'].isin(['02', '15', '72'])]
    
    # Create state boundaries by dissolving counties
    states = counties.copy()
    states['state_fips'] = states['STATEFP']
    states = states.dissolve(by='state_fips')
    
    # Merge data - using risk_df as the left side since it's our source of truth
    gdf = risk_df.merge(counties[['GEOID', 'geometry']], on='GEOID', how='left')
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    
    # Set the CRS to a good projection for the continental US
    target_crs = 'EPSG:5070'  # USA Contiguous Albers Equal Area Conic
    gdf = gdf.set_crs(counties.crs).to_crs(target_crs)
    states = states.to_crs(target_crs)
    
    # Print info about missing geometries
    missing = gdf[gdf['geometry'].isna()]['GEOID'].tolist()
    if missing:
        print("\nMissing geometries for:")
        print(f"Total missing: {len(missing)}")
        missing_by_state = pd.Series([x[:2] for x in missing]).value_counts()
        print("\nMissing by state:")
        for state_fips, count in missing_by_state.items():
            print(f"State FIPS {state_fips}: {count} counties")
    
    # Drop any rows with missing geometries
    gdf = gdf.dropna(subset=['geometry'])
    
    return gdf, states

def create_risk_maps(gdf, states):
    """Create maps for heat risk, cold relief, and structural risk."""
    fig, axes = plt.subplots(3, 1, figsize=(24, 36))
    
    # Calculate bounds for consistent map extent
    bounds = gdf.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    # Add padding
    padding = 0.05
    bounds = (
        bounds[0] - width * padding,
        bounds[1] - height * padding,
        bounds[2] + width * padding,
        bounds[3] + height * padding
    )
    
    # Common parameters for county layers
    county_kwargs = dict(
        missing_kwds={'color': 'lightgrey'},
        legend=True,
        legend_kwds={
            'label': 'Risk Score',
            'orientation': 'horizontal',
            'fraction': 0.046,
            'pad': 0.04,
            'aspect': 30
        }
    )
    
    # Common parameters for state overlay
    state_kwargs = dict(
        color='none',
        edgecolor='black',
        linewidth=1.0,
        alpha=0.7
    )
    
    # Heat Risk Change Map (0-1 scale from min-max normalization)
    gdf.plot(
        column='heat_risk_change',
        ax=axes[0],
        cmap='YlOrRd',  # Yellow to Red for heat
        vmin=0,
        vmax=1,
        **county_kwargs
    )
    states.plot(ax=axes[0], **state_kwargs)
    axes[0].set_title('Heat Risk Change (2020-2060)\nNormalized Absolute Change', fontsize=20, pad=20)
    axes[0].axis('off')
    axes[0].set_xlim(bounds[0], bounds[2])
    axes[0].set_ylim(bounds[1], bounds[3])
    
    # Cold Relief Map
    gdf.plot(
        column='cold_risk_relief',
        ax=axes[1],
        cmap='RdBu',  # Red-Blue for cold (blue is positive relief)
        vmin=0,
        vmax=1,
        **county_kwargs
    )
    states.plot(ax=axes[1], **state_kwargs)
    axes[1].set_title('Cold Risk Relief (2020-2060)\nWeighted and Normalized Absolute Change', fontsize=20, pad=20)
    axes[1].axis('off')
    axes[1].set_xlim(bounds[0], bounds[2])
    axes[1].set_ylim(bounds[1], bounds[3])
    
    # Structural Risk Map (difference between heat and cold)
    gdf.plot(
        column='structural_risk_change',
        ax=axes[2],
        cmap='RdYlBu_r',  # Red-Yellow-Blue (reversed)
        vmin=-0.25,
        vmax=0.75,
        **county_kwargs
    )
    states.plot(ax=axes[2], **state_kwargs)
    axes[2].set_title('Net Structural Risk Change (2020-2060)\nHeat Risk - Cold Relief', fontsize=20, pad=20)
    axes[2].axis('off')
    axes[2].set_xlim(bounds[0], bounds[2])
    axes[2].set_ylim(bounds[1], bounds[3])
    
    # Adjust layout and spacing
    plt.tight_layout(pad=4.0)
    
    # Save figure with high resolution
    output_dir = Path("data/reference_figures")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(
        output_dir / "risk_component_maps.png",
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.5
    )
    plt.close()

def main():
    """Main function to create risk visualization maps."""
    print("Loading data...")
    gdf, states = load_data()
    
    print("Creating risk component maps...")
    create_risk_maps(gdf, states)
    
    print("Maps saved to data/reference_figures/risk_component_maps.png")

if __name__ == "__main__":
    main() 