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
    
    # Load county geometries
    counties = gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
    counties['GEOID'] = counties['id'].astype(str).str.zfill(5)
    
    # Filter out Alaska (02), Hawaii (15), and Puerto Rico (72)
    counties = counties[~counties['GEOID'].str.startswith(('02', '15', '72'))]
    
    # Create state boundaries by dissolving counties
    states = counties.copy()
    states['state_fips'] = states['GEOID'].str[:2]
    states = states.dissolve(by='state_fips')
    
    # Merge data
    gdf = counties.merge(risk_df, on='GEOID', how='left')
    
    return gdf, states

def create_risk_maps(gdf, states):
    """Create maps for heat risk, cold relief, and structural risk."""
    fig, axes = plt.subplots(3, 1, figsize=(24, 36))
    
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
    
    # Cold Relief Map (0-0.25 scale due to weighting)
    gdf.plot(
        column='cold_risk_relief',
        ax=axes[1],
        cmap='RdBu',  # Red-Blue for cold (blue is positive relief)
        vmin=0,
        vmax=0.25,
        **county_kwargs
    )
    states.plot(ax=axes[1], **state_kwargs)
    axes[1].set_title('Cold Risk Relief (2020-2060)\nWeighted and Normalized Absolute Change', fontsize=20, pad=20)
    axes[1].axis('off')
    
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
    
    print("Maps saved to oliaiq_mvp/data/reference_figures/risk_component_maps.png")

if __name__ == "__main__":
    main() 