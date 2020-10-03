# -*- coding: utf-8 -*-
"""Clustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ybFYuT_d60-LooT6UFFwFyBTkQb8bKYS
"""

!wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=4 "https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/FIRMS/c6/Global" --header "Authorization: Bearer 53D9FEBC-F293-11E9-AFC1-F32CA14C1506" -P ./data
!wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=4 "https://nrt3.modaps.eosdis.nasa.gov/api/v2/content/archives/FIRMS/viirs/Global" --header "Authorization: Bearer 53D9FEBC-F293-11E9-AFC1-F32CA14C1506" -P ./data

!pip install geopandas
!pip install rasterio
!apt-get -qq install python-cartopy python3-cartopy
!pip install contextily
import cartopy

import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import threading
import os
import sys

for day in [276]:
  print(day)
  modis_df = pd.read_csv("data/FIRMS/c6/Global/MODIS_C6_Global_MCD14DL_NRT_2020" + str(day) + ".txt")
  viirs_df = pd.read_csv("data/FIRMS/viirs/Global/VIIRS_I_Global_VNP14IMGTDL_NRT_2020" + str(day) + ".txt")

modis_df

#changes the confidence rating of the viirs data from text based low,nominal,high to int values
viirs_df['confidence'] = viirs_df['confidence'].apply(lambda x: {'low':30,'nominal':60,'high':90}[x])

viirs_df = viirs_df.drop(columns=['bright_ti4', 'scan', 'track', 'satellite', 'version', 'bright_ti5', 'daynight'])
modis_df = modis_df.drop(columns=['brightness', 'scan', 'track', 'satellite', 'version', 'bright_t31', 'daynight'])

date_to_day = lambda x: (datetime.strptime(x['acq_date'] + " " + x['acq_time'], '%Y-%m-%d %H:%M') -
                         datetime(1970,1,1)).total_seconds() / (3600 * 24)

df = pd.concat([viirs_df,modis_df])

df['day'] = df.apply(date_to_day, axis=1)

df

for point in [0, 1, 2, 3, 4, 5]:
    point_s = df.iloc[point]
    #print(point_s)
    near_points = df[(point_s.latitude - 0.01 <= df.latitude) &         # gets latitude and longitude recordings in the database close to the particular point
                     (df.latitude <= point_s.latitude + 0.01) &         # and within one day of recording the data
                     (point_s.longitude - 0.01 <= df.longitude) &       
                     (df.longitude <= point_s.longitude + 0.01) &
                     (point_s.day - 1 <= df.day) &
                     (df.day <= point_s.day + 1)]
    
    print(len(near_points))
    print(near_points)

def get_df(days):
    df = None
    for day in days:
        modis_df = pd.read_csv("data/FIRMS/c6/Global/MODIS_C6_Global_MCD14DL_NRT_2020" + str(day) + ".txt")
        viirs_df = pd.read_csv("data/FIRMS/viirs/Global/VIIRS_I_Global_VNP14IMGTDL_NRT_2020" + str(day) + ".txt")
        viirs_df['confidence'] = viirs_df['confidence'].apply(lambda x: {"low": 30, "nominal": 60, "high": 90}[x])
        viirs_df = viirs_df.drop(columns=['bright_ti4', 'scan', 'track', 'satellite', 'version', 'bright_ti5', 'daynight'])
        modis_df = modis_df.drop(columns=['brightness', 'scan', 'track', 'satellite', 'version', 'bright_t31', 'daynight'])
        if df is None:
            df = pd.concat([viirs_df, modis_df])
        else:
            df = pd.concat([df, viirs_df, modis_df])
    
    print(len(df))
    
    date_to_day = lambda x: (datetime.strptime(x['acq_date'] + " " + x['acq_time'], '%Y-%m-%d %H:%M') -
                         datetime(1970,1,1)).total_seconds() / (3600 * 24)
    df['day'] = df.apply(date_to_day, axis=1)
    
    return df

def get_points(spacial_delta, temporal_delta, df,index):
    point_s = df.iloc[index]
    near_points = df[(point_s.latitude - spacial_delta <= df.latitude) &
                     (df.latitude <= point_s.latitude + spacial_delta) &
                     (point_s.longitude - spacial_delta <= df.longitude) &
                     (df.longitude <= point_s.longitude + spacial_delta) &
                     (point_s.day - temporal_delta <= df.day) &
                     (df.day <= point_s.day + temporal_delta)]
    return near_points

def get_map(points, s3_scenes):
    points.sort_values(by=['day'])
    date_time = datetime.strptime(points.iloc[0]['acq_date'] + " " + points.iloc[0]['acq_time'], '%Y-%m-%d %H:%M')
    lats = points["latitude"].values
    lons = points["longitude"].values
    
    coords = []
    coords.append([np.min(lats) - 0.001, np.min(lons) - 0.001])
    coords.append([np.max(lats) + 0.001, np.max(lons) + 0.001])
    
    b5 = get_image(coords, date_time, 5, s3_scenes)
    b3 = get_image(coords, date_time, 3, s3_scenes)
    b4 = get_image(coords, date_time, 4, s3_scenes)
    
    print(np.min(b3), np.average(b3), np.max(b3))
    print(np.min(b4), np.average(b4), np.max(b4))
    print(np.min(b5), np.average(b5), np.max(b5))
    
    false_col = np.stack(((b3 - np.min(b3)) / (np.max(b3) - np.min(b3)), 
                          (b4 - np.min(b4)) / (np.max(b4) - np.min(b4)), 
                          (b5 - np.min(b5)) / (np.max(b5) - np.min(b5))), axis=2)
    return coords, false_col

del_pos = np.array([])
for a in df.index:
  y = (len(get_points(0.09, 10, df,a)))
  if(y<90):
    del_pos = np.append(del_pos,np.array(a))

new_df = df.drop(labels = del_pos)
new_df
'''
for a in df[:1000].index:
  x = np.append(x,np.array(a))
  y = np.append(y,np.array(len(get_points(0.09, 10, df,a))))
'''
#plt.plot(x,y)
#plt.show()
#print(y.mean())

new_df

world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
import contextily as ctx
from shapely.geometry import Point,Polygon

providers = {}

def get_providers(provider):
    if "url" in provider:
        providers[provider['name']] = provider
    else:
        for prov in provider.values():
            get_providers(prov)

get_providers(ctx.providers)
world_merc = world_map.to_crs(epsg=3857)
geo = [Point(xy) for xy in zip(new_df['longitude'],new_df['latitude'])]
geo[:3]
crs = {'init':'epsg:4326'}
#world_map.plot()
geo_df = gpd.GeoDataFrame(new_df,crs=crs,geometry = geo)
geo_df.head()
fig,ax = plt.subplots(figsize = (20,20))
#fig, ax = plt.subplots()
world_map.plot(ax = ax,color = 'grey')

ctx.add_basemap(ax,source = providers['NASAGIBS.ViirsEarthAtNight2012'])
geo_df.plot(ax = ax,markersize = .1,color = 'red')
plt.savefig("map.png")
plt.show();
#cv2.imwrite("map.jpg", (geo_df.plot(ax = ax,markersize = .1,color = 'red')))

from sklearn.cluster import DBSCAN

X = np.asarray(df.loc[:,('latitude','longitude')])
X
clustering = DBSCAN(eps=1,min_samples = 90).fit(X)
print(clustering.labels_)

labels = clustering.labels_.reshape((-1,1))
print(labels.shape)
dat = np.append(X,labels,axis = 1)

cluster = pd.DataFrame(dat,columns=['Latitude','Longitude','Cluster Label'])
mean_points = cluster.groupby('Cluster Label').mean()
mean_points
world_2 = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
g_points = [Point(xy) for xy in zip(mean_points['Longitude'],mean_points['Latitude'])]
geo_df = gpd.GeoDataFrame(mean_points,crs=crs,geometry = g_points)
geo_df.head()
fig,ax = plt.subplots(figsize = (20,20))
world_map.plot(ax = ax,color = 'grey')

ctx.add_basemap(ax,source = providers['NASAGIBS.ViirsEarthAtNight2012'])
geo_df.plot(ax = ax,markersize = 10,color = 'red')