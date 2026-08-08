import pprint
import sys
from pathlib import Path

# Inject src/ into path so we can import from seroepi when running as a standalone script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from seroepi.constants import SpatialResolution

try:
    import geopandas as gpd
except ImportError:
    raise ImportError("geopandas is required for reverse geocoding. Install with seroepi[spatial]")


def main():
    # 1. Load the Natural Earth Shapefile
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    print("Downloading and parsing Natural Earth data...")
    gdf = gpd.read_file(url)

    # Setup the output directory
    out_dir = Path("../src/seroepi/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- ARTIFACT 1: The Raw Python Dictionary (Gazetteer) ---
    print("Generating gazetteer_data.py...")
    # 1. to_crs(epsg=3857): Projects the globe to a flat map (meters)
    # 2. .centroid: Calculates the true geometric center
    # 3. to_crs(epsg=4326): Converts the center point back to GPS degrees
    centroids = gdf.to_crs(epsg=3857).centroid.to_crs(epsg=4326)

    # Build the nested dictionary: {'Nigeria': {'iso3': 'NGA', 'lat': ..., 'lon': ...}}
    gazetteer_dict = {}
    for idx, row in gdf.iterrows():
        country_name = row["ADMIN"]
        # Wrap the rounded numpy values in the native Python float() function
        native_lon = float(round(centroids.x[idx], 4))
        native_lat = float(round(centroids.y[idx], 4))
        gazetteer_dict[country_name] = {
            "iso3": row["ADM0_A3"],
            "region": row["SUBREGION"],
            "spatial_resolution": SpatialResolution.COUNTRY.value,
            "centroid_lon": native_lon,
            "centroid_lat": native_lat,
        }

    # Write the dictionary to a valid Python file
    dict_path = out_dir / "gazetteer_data.py"
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write('"""\nAUTO-GENERATED GAZETTEER DATA\n')
        f.write('Do not edit manually. Generated from Natural Earth 110m.\n"""\n\n')
        # pformat turns the dictionary object into a beautifully formatted code string
        formatted_dict = pprint.pformat(gazetteer_dict, sort_dicts=True, indent=4)
        f.write(f"GAZETTEER_DICT = {formatted_dict}\n")

    # --- ARTIFACT 2: The Plotly Boundaries (GeoJSON) ---
    print("Generating world_boundaries.geojson...")
    # Simplify the geometry for the web
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.05)

    # Keep only what Plotly needs (Added ADMIN so choropleth can match on spatial names)
    minimal_gdf = gdf[["ADMIN", "ADM0_A3", "geometry"]]

    minimal_gdf.to_file(out_dir / "world_boundaries.geojson", driver="GeoJSON")
    print("Done! Assets safely bundled in seroepi/data/")


if __name__ == "__main__":
    main()
