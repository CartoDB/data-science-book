import geopandas as gpd
from cartoframes.auth import set_default_credentials
from cartoframes import read_carto
from cartoframes import to_carto

set_default_credentials("ebook-sds")


def get_table(tablename):
    """Retrieve tablename as a GeoDataFrame ordered by database id

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame representation of table
    """
    base_query = ("SELECT * FROM {tablename} ORDER BY cartodb_id ASC").format(
        tablename=tablename
    )
    data_carto = read_carto(base_query)
    ## Renaming the geometry column from 'the_geom' to 'geometry' 
    ## (pysal expect the geometry column to be called 'geometry')
    data = data_carto.copy()
    data['geometry'] = data.geometry
    data.drop(['the_geom'],axis = 1, inplace = True)
    data = gpd.GeoDataFrame(data, geometry = 'geometry')
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
