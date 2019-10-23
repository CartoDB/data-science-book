import geopandas as gpd
from cartoframes.auth import set_default_credentials
from cartoframes.data import Dataset

set_default_credentials('ebook-sds')

def get_nyc_census_tracts():
    """Retrieve dataset on NYC Census Tracts
    
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame representation of table
    """
    base_query = 'SELECT * FROM census_tracts_cleaned ORDER BY cartodb_id ASC'
    data = gpd.GeoDataFrame(Dataset(base_query).download(decode_geom=True))
    data.crs = {'init': 'epsg:4326'}
    return data