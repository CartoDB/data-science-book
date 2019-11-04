import geopandas as gpd
from cartoframes.auth import set_default_credentials
from cartoframes.data import Dataset

set_default_credentials("ebook-sds")


def get_table(tablename):
    """Retrieve tablename as a GeoDataFrame ordered by database id

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame representation of table
    """
    base_query = ("SELECT * FROM {tablename} ORDER BY cartodb_id ASC").format(
        tablename=tablename
    )
    data = gpd.GeoDataFrame(Dataset(base_query).download(decode_geom=True))
    data.crs = {"init": "epsg:4326"}
    return data


def get_nyc_census_tracts():
    """Retrieve dataset on NYC Census Tracts

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame representation of table
    """
    return get_table("census_tracts_cleaned")


def get_safegraph_visits():
    """Retrieve Safegraph visit data for Panama City Beach in July 2019
    as a GeoDataFrame ordered by database id

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame representation of table
    """
    return get_table("safegraph_pcb_visits")
