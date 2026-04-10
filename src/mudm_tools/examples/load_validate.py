import mudm_tools.model as mj
import json

# load the mudm file
with open("tests/json/mudm/valid/fullexample.json") as f:
    data = json.load(f, strict=True)

# validate the mudm file
microjsonobj = mj.MuDM.model_validate(data)
print("done validating: {}".format(microjsonobj))

# load the geojson file
with open("tests/json/geojson/valid/featurecollection/basic.json") as f:
    data = json.load(f, strict=True)

# validate the geojson file
geojsonobj = mj.GeoJSON.model_validate(data)

print("done validating: {}".format(geojsonobj))
