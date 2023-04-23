# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!
import streamlit as st
from PIL import Image
import pandas as pd
import base64
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
import json
import time

from shapely.wkt import loads
import geopandas as gpd
import seaborn as sns
import fiona
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import branca.colormap as bcm
import folium
import random

from streamlit_folium import st_folium, folium_static



# Define a function to validate the input
def validate_input(input_text):
    try:
        input_value = int(input_text.strip())  # Convert the input to an integer
        if input_value <= 0 or input_value < 2023:
            raise ValueError("Please insert a year post 2023")
        return input_value
    except ValueError:
        raise ValueError("Please insert a year post 2023")
    
def produce_tourism_column(column):
    return column * 0.21 * (1 + (1.57 / 12))


def produce_loan_predictor(column):
    model = LinearRegression()
    X = np.arange(len(column)).reshape(-1, 1)
    model.fit(X, column)
    
    return model

def get_month_range(requested_year, quarter):
    requested_year = int(requested_year)
    quarter_start = (requested_year - 2019) * 12 + (quarter - 1) * 3
    
    return list(range(quarter_start, quarter_start + 4))

def get_quarter_avg_loans(quarter_range, predictor):
   return sum([predictor.predict([[month]]) for month in quarter_range])

districts = ['Eastern Macedonia and Thrace',
             'Central Macedonia',
             'Western Macedonia',
             'Epirus', 'Thessaly',
             'Ionian Islands',
             'Western Greece',
             'Central Greece',
             'Attica',
             'Peloponnesus',
             'Northern Aegean',
             'Southern Aegean',
             'Crete']

tourism_income_data = {
                      'Year': [2016, 2017, 2018,2021],
                      'Eastern Macedonia and Thrace': [288,282,322,134],
                      'Central Macedonia': [1688,1852,2275,1012],
                      'Western Macedonia': [68,45,61,38],
                      'Epirus': [218,216,222,127],
                      'Thessaly': [301,290,270,179],
                      'Northern Aegean': [131,167,164,68],
                      'Southern Aegean' : [1989+1147,2236+1417,2814+1600,1946+1175],
                      'Central Greece': [117,113,194,113],
                      'Western Greece': [146,159,212,128],
                      'Peloponnesus': [324,307,415,250],
                      'Ionian Islands': [1504,1775,1691,1297],
                      'Crete': [3095,3260,3134,2395],
                      'Attica': [1734,2083,2279,1466],
                    }

tourism_income_df = pd.DataFrame(tourism_income_data)


loans_distribution = {'Attica': 0.2,
                      'Southern Aegean' : 0.19,
                      'Crete': 0.15,
                      'Central Macedonia': 0.15,
                      'Ionian Islands': 0.1,
                      'Eastern Macedonia and Thrace': 0.07,
                      'Peloponnesus': 0.03,
                      'Epirus': 0.03,
                      'Thessaly': 0.02,
                      'Western Greece': 0.02,
                      'Central Greece': 0.02,
                      'Western Macedonia': 0.01,
                      'Northern Aegean': 0.01
                    }



#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide")

#---------------------------------#
# Title
image = Image.open('pythia.png')
st.image(image, width = 1000)
st.title('Loan Distributor ')
st.markdown("""
This app predicts:
""")
st.write("- **{}**".format("Loan Allocations for Tourism Enterprises"))
st.write("- **{}**".format("Gross income from tourism per area"))

st.markdown("""
Optimal Loan Allocation Display per **Area & Sector**:
""")

#---------------------------------#
# Page layout (continued)
## Divide page to 3 columns (col1 = sidebar, col2 and col3 = page contents)
col1 = st.sidebar
col2, col3 = st.columns((2,1))

#---------------------------------#
# Sidebar + Main panel
col1.header('Input Options')

## Read Year Input
requested_year = col1.slider('Year:', 2023, 2033, 2023) 

## Sidebar - Number of coins to display
quarter = col1.slider('Quarter:', 1, 4, 2)

## Sidebar - Percent change timeframe
district_selectbox = col1.selectbox('Select area of interest:',
                                    districts)

# calculate and save the tourism loans column
df = pd.read_excel('loans_data.xls')
loans = df['Credit-Stocks-Loans_to_private_sector']
df['Tourism Loans'] = produce_tourism_column(loans)

# train a model to predict future total loans distributed to tourism enterprises
predictor = produce_loan_predictor(df['Tourism Loans'])


quarter_range = get_month_range(requested_year, quarter)
quarter_total_loans = get_quarter_avg_loans(quarter_range, predictor)

# calculate average income for each area
tourism_income_df_means = tourism_income_df.mean()
# calculate weights for each area based on their income
area_weights = tourism_income_df_means / tourism_income_df_means.sum()
area_weights = area_weights[1:]

dfper = pd.read_csv("WGS84.csv")
dfper['the_geom'] = dfper['the_geom'].apply(loads)
dfper['value'] = area_weights.values

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(dfper, geometry='the_geom')

# Set the original CRS of the GeoDataFrame
# gdf.crs = "EPSG:3857"
gdf.crs = "EPSG:2100"

# Reproject the GeoDataFrame to the EPSG:4326 coordinate reference system
gdf = gdf.to_crs(epsg=2100)

# Calculate the map's center
center = gdf.geometry.centroid.unary_union.centroid

# Create a folium map
m = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="cartodb positron")

# Create a colormap
cmap = bcm.LinearColormap(colors=["white", "red"], vmin=dfper["value"].min(), vmax=dfper["value"].max())

# Function to style each polygon
def style_function(feature):
    value = feature["properties"]["value"]
    return {
        "fillOpacity": 0.5,
        "fillColor": cmap(value),
        "color": "black",
        "weight": 1,
    }

# Add polygons to the map
folium.GeoJson(
    gdf.to_json(),
    style_function=style_function,
    tooltip=folium.features.GeoJsonTooltip(fields=["PER"], labels=True, sticky=True),
).add_to(m)

# Add the colormap to the map
cmap.caption = "Tourism Revenue"
cmap.add_to(m)

# Display the map
folium_static(m, width=700)


data = {'Action': ['Hotels', 'Restaurants', 'Arts and culture', 'Outdoors', 'Night', 'Cruises', 'Luxury Services', 'Outdoor'],
        'Frequency': [32, 17, 10, 10, 10, 10, 10, 1]}

df = pd.DataFrame(data)

import plotly.express as px
import plotly.graph_objects as go

# Assuming you have already created your DataFrame and named it df

# Sort the DataFrame by 'Frequency' in descending order
df = df.sort_values('Frequency', ascending=False)

# Create the pie chart using Plotly Express
fig = px.pie(df, names='Action', values='Frequency', title='NLP Score of Actions')

# Show the plot
st.plotly_chart(fig)












