from flask import Flask, render_template_string, jsonify
import geopandas as gpd
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(open("index.html").read())

@app.route('/geojson')
def geojson():
    # Load your GeoDataFrame here (or replace this line with your data loading code)
    gdf = gpd.read_file("/Users/dimitriosfotinos/Downloads/BOG/nbs/tourism/gr_1km.shp")
    gdf = gdf.to_crs(epsg=4326)
    return jsonify(json.loads(gdf.to_json()))

if __name__ == '__main__':
    app.run(debug=True, port=5003)