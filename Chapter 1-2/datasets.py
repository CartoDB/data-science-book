import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import shapely
from libpysal.weights import Queen
import pointpats
import pointpats.centrography

from cartoframes.auth import set_default_credentials
from cartoframes import read_carto
from cartoframes import to_carto

set_default_credentials('ebook-sds')

## The Meuse dataset from R gstat package
class GetMeuse():
    def __init__(self):
        self.data = read_carto('meuse')
        self.data['log_zinc'] = np.log(self.data['zinc'])
        self.data = self.data.to_crs({'init': 'epsg:28992'})
        self.data_lonlat = self.data.to_crs({'init': 'epsg:4326'})

        self.data_grid = read_carto('meuse_grid')
        self.data_grid = self.data_grid.to_crs({'init': 'epsg:28992'})
        self.data_grid_lonlat = self.data_grid.to_crs({'init': 'epsg:4326'})

    def loadpred_krg(self):

        self.data_krg = pd.read_csv('/tmp/meuse_krg.csv')
        self.data_krg  = gpd.GeoDataFrame(self.data_krg, geometry=gpd.points_from_xy(self.data_krg.x, self.data_krg.y))  
        self.data_krg.crs = {'init': 'epsg:28992'}
        self.data_krg_lonlat = self.data_krg.to_crs({'init': 'epsg:4326'})

        self.data_grid_krg = pd.read_csv('/tmp/meuse.grid_krg.csv')
        self.data_grid_krg  = gpd.GeoDataFrame(self.data_grid_krg, geometry=gpd.points_from_xy(self.data_grid_krg.x, self.data_grid_krg.y)) 
        self.data_grid_krg.crs = {'init': 'epsg:28992'}
        self.data_grid_krg_lonlat = self.data_grid_krg.to_crs({'init': 'epsg:4326'})

    def loadpred_INLAspde(self):
        
        self.data_INLAspde = pd.read_csv('/tmp/meuse_INLAspde.csv')
        self.data_INLAspde  = gpd.GeoDataFrame(self.data_INLAspde, geometry=gpd.points_from_xy(self.data_INLAspde.x, self.data_INLAspde.y))  
        self.data_INLAspde.crs = {'init': 'epsg:28992'}
        self.data_INLAspde_lonlat = self.data_INLAspde.to_crs({'init': 'epsg:4326'})

        self.data_grid_INLAspde = pd.read_csv('/tmp/meuse.grid_INLAspde.csv')
        self.data_grid_INLAspde  = gpd.GeoDataFrame(self.data_grid_INLAspde, geometry=gpd.points_from_xy(self.data_grid_INLAspde.x, self.data_grid_INLAspde.y)) 
        self.data_grid_INLAspde.crs = {'init': 'epsg:28992'}
        self.data_grid_INLAspde_lonlat = self.data_grid_INLAspde.to_crs({'init': 'epsg:4326'})
        
## The Boston dataset from R spData package
class GetBostonHousing():
    def __init__(self):
        self.data_carto = read_carto('boston_housing')
        ## Renaming the geometry column from 'the_geom' to 'geometry' 
        ## (pysal expect the geometry column to be called 'geometry')
        self.data = self.data_carto.copy()
        self.data['geometry'] = self.data.geometry
        self.data.drop(['the_geom'],axis = 1, inplace = True)
        self.data = gpd.GeoDataFrame(self.data, geometry = 'geometry')
        self.w = Queen.from_dataframe(self.data)

    def loadpred(self):
        self.data_preds = gpd.read_file('/tmp/boston_housing_predictions.shp')
        self.data_preds.crs = {'init': 'epsg:4326'}

## The Crime dataset from UK Police data
class GetCrimeLondon():
    def __init__(self, var, var_value):
        self.filename = '/tmp/UK_Police_street_crimes_2019_04.csv'
        self.data = read_carto('uk_police_street_crimes_2019_04')
        self.data = self.data[self.data[var] == var_value]
        self.data_lonlat = self.data
        self.data_lonlat = read_carto('''
            SELECT c.*
              FROM uk_police_street_crimes_2019_04 as c
              JOIN london_borough_excluding_mhw as g
              ON ST_Intersects(c.the_geom, g.the_geom)
        
        ''')
        self.data = self.data.to_crs({'init': 'epsg:32630'})

    def pp(self):
        self.pointpattern = pointpats.PointPattern(
            pd.concat([self.data.geometry.x,self.data.geometry.y], axis=1)
        )