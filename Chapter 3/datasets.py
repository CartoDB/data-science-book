import geopandas as gpd
from cartoframes.auth import set_default_credentials
from cartoframes import read_carto
from cartoframes import to_carto

set_default_credentials('ebook-sds')

def get_retail_store_minnesota():
    """Retrieve Retail Store Locations in Minnesota
    
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame representation of table
    """
    table_name = 'retail_store_minnesota'
    data = read_carto(table_name)
    data['store_id'] = data['store_id'].apply(lambda x: str(int(x)))
    data.crs = {'init': 'epsg:4326'}
    return data