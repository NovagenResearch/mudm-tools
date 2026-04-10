# Integration of TileJSON with muDM

## Purpose

This specification outlines how to use TileJSON to integrate tiled muDM data, both in json form as well as binary form. It provides examples of how TileJSON can be used to specify the tiling scheme and zoom levels for muDM data and its binary equivalent. It is based on the [TileJSON 3.0.0 specification](https://github.com/mapbox/tilejson-spec/blob/master/3.0.0/README.md), but extends it by recommending additional properties to enable integration of muDM data and fit purposes of microscopy imaging. The recommendations provided here are not intrinsic to the original TileJSON specification but have been tailored to suit the needs of microscopy metadata annotation and integration with muDM. However, all suggestions are designed to maintain compatibility with the original TileJSON specification.

## Background of TileJSON

[TileJSON](https://github.com/mapbox/tilejson-spec/) is a widely-used format in mapping applications for specifying tilesets. Developed to streamline the integration of different map layers, TileJSON is essential for ensuring consistency across mapping platforms. It describes tilesets through a JSON object, detailing properties like tile URLs, zoom levels, and spatial coverage.

## TileJSON for muDM Object Structure

- **`tilejson`:** Specifies the version of the TileJSON spec being used. Required for all TileJSON objects.
- **`name`:** The name of the tileset. Optional but recommended.
- **`description`:** Provide a brief description of the tileset. Optional but recommended.
- **`version`:** The version of the tileset. Optional but recommended.
- **`attribution`:** A link to the data source or other attribution information, e.g. organisational origin. Optional but recommended.
- **`tiles`:** Required. The URL pattern for accessing the vector tiles. The  `urlbase/{zlvl}/{t}/{c}/{z}/{x}/{y}` is the recommended default naming pattern for the tiles, in this order, where `urlbase` is the base URL (e.g. `http://example.com/tiles`), `{zlvl}` is the zoom level, `{t}` is the tileset timestamp, `{c}` is the channel, `{z}` is the z coordinate, and `{x}` and `{y}` are the x and y coordinates, respectively. If not using a timestamp, channel, or z coordinate, these can be omitted. The zoom level should always be first.
- **`minzoom` and `maxzoom`:** Defines the range of zoom levels for which the tiles are available.
- **`bounds`:** Optional. Specifies the geometrical bounds included in the tileset. Specified as an array of minimum four numbers in the order `[minX, minY, maxX, maxY]`, but may include up to a further six numbers for a total of ten, `[minT, minC, minZ, minX, minY, maxT, maxC, maxZ, maxX, maxY]`, where `minT` is the minimum tileset timestamp, `minC` is the minimum channel, `minZ` is the minimum z coordinate, `minX` and `minY` are the minimum x and y coordinates, `maxT` is the maximum tileset timestamp, `maxC` is the maximum channel, `maxZ` is the maximum z coordinate, and `maxX` and `maxY` are the maximum x and y coordinates.
- **`center`:** Optional. Indicates the center and suggested default view of the tileset. Minimum of three numbers in the order `[x, y, zoom]`, but may include up to a further three numbers for a total of six, `[t,c,z,x,y,zoom]`, where `t` is the tileset timestamp, `c` is the channel, `z` is the z coordinate, `x` and `y` are the x and y coordinates, and `zoom` is the zoom level. Zoom level should be last.
- **`vector_layers`:**  Required. Describes each layer within the vector tiles, and has the following structure:

  - **`id`:** Required. A unique identifier for the layer. Required for each layer.
  - **`fields`:** Required. A list of fields (attributes) and their data types. For muDM, this can either be an empty list, or a simple datatype indicator, that is either of `String`, `Number`, or `Bool`. Complex data types, such as arrays or objects are not allowed.  Required for each layer.
  - **`fieldranges`:** Optional. A dictionary of field names and their ranges. For example, `{"label": [0,100], "channel": [0,10]}`. Optional.
  - **`fieldenums`:** Optional. A dictionary of field names and their possible values. For example, `{"plate": ["A1", "A2", "B1", "B2"], "image": ["image1.tif", "image2.tif", "image3.tif"]}`. Optional.
  - **`fielddescriptions`:** Optional. A dictionary of field names and their descriptions. For example, `{"plate": "Well plate identifier", "image": "Image filename", "label": "Label identifier", "channel": "Channel identifier"}`. Optional.
  - **`description`:** Optional. A brief description of the layer.
  - **`minzoom` and `maxzoom`:** Optional. The range of zoom levels at which the layer is visible.
- **`fillzoom`:** Optional. An integer specifying the zoom level from which to generate overzoomed tiles.
- **`legend`:** Optional. Contains a legend to be displayed with the tileset.
- **`multiscale`:** Optional. A multiscale object as defined in the section [Multiscale Object](#multiscale-object). If this property is not present, the default coordinate system is assumed to be the same as the image coordinate system, using cartesian coordinates and pixels as units.
- **`scale_factor`:** Optional. A float specifying the scale factor for the tileset. If not present, the default value of 2 is assumed. Unless the use case requires a different scale factor, it is highly recommended to use the default value, as it is widely supported and assumed by many viewers.

The following fields of TileJSON may be used if the use case requires it, and are included here for completeness:

- **`scheme`:** The tiling scheme of the tileset.
- **`grids`:** The URL pattern for accessing grid data.
- **`data`:** Optional. The URL pattern for accessing data. Used for GeoJSON originally, which in this specification is replaced by muDM and used in the `tiles` field.
- **`template`:** Optional. Contains a mustache template to be used to format data from grids for interaction.

### Multiscale Object

A multiscale object represents the choice of axes (2-5D) and potentially their transformations that should be applied to the numerical data in order to arrive to the actual size of the object described. If the field is present, it MUST have the following properties:

- `"axes"`: Representing the choice of axes as an array of Axis objects.

It may contain either of, but NOT both of the following properties:

- `"coordinateTransformations"`: Representing the set of coordinate transformations that should be applied to the numerical data in order to arrive to the actual size of the object described. It MUST be an array of objects, each object representing a coordinate transformation. Each object MUST have properties as follows:
  - `"type"`: Representing the type of the coordinate transformation. Currently supported types are `"identity"`, `"scale"`, and `"translate"`. If the type is `"scale"`, the object MUST have the property `"scale"`, representing the scaling factor. It MUST be an array of numbers, with the number of elements equal to the number of axes in the coordinate system. If the type is `"translate"`, the object MUST have the property `"translate"`, representing the translation vector. It MUST be an array of numbers, with the number of elements equal to the number of axes in the coordinate system. If the type is `"identity"`, the object MUST NOT have any other properties.
- `"transformationMatrix"`: Representing the transformation matrix from the coordinate system of the image to the coordinate system of the muDM object. It MUST be an array of arrays of numbers, with the number of rows equal to the number of axes in the coordinate system, and the number of columns equal to the number of axes in the image coordinate system. The transformation matrix MUST be invertible.

### Axis Object

Together with the other axes in the axes array, an axis object represents the coordinate system of the muDM object (2D-5D)
It MUST have the following properties:

- `"name"`: Representing the name of the axis. It MUST be a string.
It may contain the following properties:
- `"unit"`: Representing the units of the corresponding axis of the geometries in the muDM object. It MUST be an array with the elements having any of the following values: `[“angstrom", "attometer", "centimeter", "decimeter", "exameter", "femtometer", "foot", "gigameter", "hectometer", "inch", "kilometer", "megameter", "meter", "micrometer", "mile", "millimeter", "nanometer", "parsec", "petameter", "picometer", "terameter", "yard", "yoctometer", "yottameter", "zeptometer", "zettameter“]`
- `"description"`: A string describing the axis.

## Pydantic Model for TileJSON for muDM

### TileJSON

::: mudm.tilemodel.TileJSON

### TileModel

::: mudm.tilemodel.TileModel

### TileLayer

::: mudm.tilemodel.TileLayer

## General tiling requirements

This specification is designed to be compatible with the [Vector Tile Specification](https://github.com/mapbox/vector-tile-spec/blob/master/2.1/README.md) and the [TileJSON 3.0.0 specification](https://github.com/mapbox/tilejson-spec/blob/master/3.0.0/README.md). The Vector Tile Specification specifically requires vector tiles to be agnostic of the global coordinate system, and thus followingly each tile has a relative coordinate system, which instead is defined in the TileJSON. Our ambitions in general are to follow the same principles.

One difference is that we here recommend that the file ending for binary tiles is `.pbf` instead of `.mvt` to avoid confusion with the Mapbox Vector Tile format. The binary tiles should be encoded in the [Protobuf format](https://developers.google.com/protocol-buffers) as defined in the Vector Tile Specification.

## Tiling Pipelines

muDM provides two tiling pipeline generations. The **Rust-accelerated pipeline** is the recommended approach for new projects, offering significantly better performance through parallel processing and native code. The **legacy Python pipeline** remains available for backwards compatibility.

### Rust-Accelerated 2D Pipeline

The Rust 2D pipeline uses `StreamingTileGenerator2D` for quadtree-based spatial indexing with parallel tile encoding. It supports two output formats:

- **PBF (MVT)**: Standard Mapbox Vector Tiles, compatible with Leaflet, MapLibre, and other web map viewers.
- **Tiled Parquet**: Columnar format with ZSTD compression, ideal for ML training pipelines with zero-copy reads.

```python
from mudm_tools._rs import StreamingTileGenerator2D
from mudm_tools.tiling2d import generate_parquet, generate_pbf

# Create the tile generator
gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64/4096)

# Ingest GeoJSON data
geojson_str = open("data.json").read()
bounds = (xmin, ymin, xmax, ymax)
gen.add_geojson(geojson_str, bounds)

# Output as PBF vector tiles
generate_pbf(gen, "tiles/", bounds, simplify=True)

# Or output as tiled Parquet for ML
generate_parquet(gen, "output.parquet", bounds, simplify=True)
```

Key features:

- **Buffer**: Controls tile overlap to prevent visual gaps. Use `buffer=64/4096` (default) for PBF output.
- **Simplification**: Douglas-Peucker simplification at coarse zoom levels (optional, enabled by default).
- **Partitioned output**: Hive-style partitioning by zoom level for Parquet.
- **PBF reader**: `read_pbf(path, bounds, zoom, tile_x, tile_y)` decodes the `{z}/{x}/{y}.pbf` directory tree.
- **Parquet reader**: `read_parquet(path, zoom, tile_x, tile_y)` reads tiled Parquet with optional filtering.

An example script is located at `src/mudm/examples/tiling_rust.py`.

### Rust-Accelerated 3D Pipeline

The 3D pipeline uses `StreamingTileGenerator` for octree-based spatial indexing of 3D mesh data (OBJ files). It supports 3D Tiles (GLB) with meshopt/Draco compression, tiled Parquet for ML, PBF3, and Neuroglancer precomputed format.

For the full API reference, geometry types, compression comparison, bucketed redistribution for large datasets, and viewer documentation, see the [3D Data and Tiling](3d.md) page.

### Legacy Python Pipeline

The original Python modules remain available for backwards compatibility:

#### MuDM2vt

The MuDM2vt module converts muDM objects to vector tiles in an intermediate JSON format, which can then be transformed into protobuf using `vt2pbf`.

```python
from mudm_tools import mudm2vt
```

::: mudm_tools.mudm2vt.mudm2vt.MuDMVt
    :docstring:

#### TileWriter

The TileWriter module generates binary tiles from a large muDM file, utilizing both mudm2vt and vt2pbf.

::: mudm_tools.tilewriter
    :docstring:

An example is located at `src/mudm/examples/tiling.py`.

#### TileReader

The TileReader module reads binary tiles and converts them back to muDM objects.

::: mudm_tools.tilereader
    :docstring:

An example is located at `src/mudm/examples/readtiles.py`.

## TileJSON for muDM example with Vector Layers

The below example illustrates a TileJSON for a muDM tileset multiple layers of detail. The tileset has a single vector layer, `image_layer` id of `vector_layers`, which contains a single vector layer describing images. The `fields` property of the this layer specifies the attributes of the layer, including the data types of the attributes. The `tiles` property specifies the URL pattern for accessing the vector tiles, which in this case is a 2D data set (no channels, time or z-axis) with zoom level.

This file is located in the `examples/tiles` directory of the repository, and is named `tiled_example.json`. It has a corresponding muDM file for each tile, located in the `examples/tiles/tiled_example` directory of the repository. The muDM files are organized according to the tiling scheme, with the directory structure `zlvl/x/y.json` where `zlvl` is the zoom level, `x` is the x coordinate, and `y` is the y coordinate. The muDM files contain the muDM objects for the corresponding tiles, and are named according to the tiling scheme. For example, the muDM object for the tile at zoom level 1, tile at (0,1) in the tiling scheme is located at `examples/tiles/tiled_example/1/0/1.json`. Examples for muDM objects at zoom levels 0, 1, and 2 are provided below.

```json
{
    "tilejson": "3.0.0",
    "name": "2D Data Example",
    "description": "A tileset showing 2D data with multiple layers of detail.",
    "version": "1.0.0",
    "attribution": "<a href='http://example.com'>Example</a>",
    "tiles": [
        "http://example.com/tiled_example/{zlvl}/{x}/{y}.json"
    ],
    "minzoom": 0,
    "maxzoom": 10,
    "bounds": [0, 0, 24000, 24000],
    "center": [12000, 12000, 0],
    "vector_layers": [
        {
            "id": "Tile_layer",
            "description": "Tile layer",
            "minzoom": 0,
            "maxzoom": 10,
            "fields": {
              "plate": "String",
              "image": "String",
              "label": "Number",
              "channel": "Number"
            }
        }
    ],
    "fillzoom": 3,
    "multiscale": {
        "axes": [
            {
                "name": "x",
                "unit": "micrometer",
                "type": "space",
                "description": "x-axis"
            },
            {
                "name": "y",
                "unit": "micrometer",
                "type": "space",
                "description": "y-axis"
            }
        ],
        "transformationMatrix": [
            [
                1.0,
                0.0,
                0.0
            ],
            [
                0.0,
                1.0,
                0.0
            ],
            [
                0.0,
                0.0,
                0.0
            ]
        ]
    }
}
```

## Tiled binary TileJSON

In addition to json format, tiles may be encoded in a binary protobuf format. Below follows a similar example to the one above, but with binary tiles. The `tiles` property specifies the URL pattern for accessing the binary tiles, which in this case is a 2D data set (no channels, time or z-axis) with zoom level. The `fillzoom` property specifies the zoom level from which to generate overzoomed tiles, which in this case starts at level 3, after the last specified layer.

```json
{
    "tilejson": "3.0.0",
    "name": "2D Data Example",
    "description": "A tileset showing 2D data with multiple layers of detail.",
    "version": "1.0.0",
    "attribution": "<a href='http://example.com'>Example</a>",
    "tiles": [
        "http://example.com/tiled_example/{zlvl}/{x}/{y}.pbf"
    ],
    "minzoom": 0,
    "maxzoom": 10,
    "bounds": [0, 0, 24000, 24000],
    "center": [12000, 12000, 0],
    "vector_layers": [
        {
            "id": "tile_layer",
            "description": "Tile layer",
            "minzoom": 0,
            "maxzoom": 10,
            "fields": {
              "plate": "String",
              "image": "String",
              "label": "Number",
              "channel": "Number"
            },
            "fieldranges": {
              "label": [0,100],
              "channel": [0,10]
            },
            "fieldenums": {
              "plate": ["A1", "A2", "B1", "B2"],
              "image": ["image1.tif", "image2.tif", "image3.tif"]
            },
            "fielddescriptions": {
              "plate": "Well plate identifier",
              "image": "Image filename",
              "label": "Label identifier",
              "channel": "Channel identifier"
            }
        }
    ],
    "fillzoom": 3
}
```

## Tiled binary example

The examples folder contains an example of how to generate binary tiles from one large muDM file. It uses a helper module that generates a large random polygon grid, as could be expected in an imaging setting, using typical imaging coordinates. It is also included below for reference.

### Example of creating binary tiles from a large muDM file

::: mudm_tools.examples.tiling.main
    :docstring:

## Tiled muDM Example

### Level 0

The following is an example of a muDM object at zoom level 0, tile at (0,0) in the tiling scheme. Example URL: `http://example.com/tiles/0/0/0.json`

```json
{
    "tilejson": "3.0.0",
    "name": "2D Data Example",
    "description": "A tileset showing 2D data with multiple layers of detail.",
    "version": "1.0.0",
    "attribution": "<a href='http://example.com'>Example</a>",
    "tiles": [
        "http://example.com/tiled_example/{zlvl}/{x}/{y}.json"
    ],
    "minzoom": 0,
    "maxzoom": 10,
    "bounds": [0, 0, 24000, 24000],
    "center": [12000, 12000, 0],
    "format": "json",
    "vector_layers": [
        {
            "id": "image_layer",
            "description": "Image layer",
            "minzoom": 0,
            "maxzoom": 10,
            "fields": {
              "plate": "String",
              "image": "String",
              "label": "Number",
              "channel": "Number"
            }
        }
    ],
    "fillzoom": 3
}
```

### Level 1

The following is an example of a muDM object at zoom level 1, tile at (0,1) in the tiling scheme. Example URL: `http://example.com/tiles/1/0/1.json`

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              0,
              10000
            ],
            [
              10000,
              10000
            ],
            [
              10000,
              20000
            ],
            [
              0,
              20000
            ],
            [
              0,
              10000
            ]
          ]
        ]
      },
      "properties": {
        "label": 3
      }
    }
  ],
  "multiscale": {
    "axes": [
      {
        "name": "x",
        "type": "space",
        "unit": "micrometer",
        "description": "x-axis"
      },
      {
        "name": "y",
        "type": "space",
        "unit": "micrometer",
        "description": "y-axis"
      }
    ]
  },
  "properties": {
    "plate": "label",
    "image": "x00_y01_p01_c1.ome.tif",
    "channel": 1.0
  }
}
```

### Level 2

The following is an example of a muDM object at zoom level 2, tile at (1,1) in the tiling scheme. Example URL: `http://example.com/tiles/2/1/3.json`

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              5000,
              15000
            ],
            [
              10000,
              15000
            ],
            [
              10000,
              20000
            ],
            [
              5000,
              20000
            ],
            [
              5000,
              15000
            ]
          ]
        ]
      },
      "properties": {
        "label": 13
      }
    }
  ],
  "multiscale": {
    "axes": [
      {
        "name": "x",
        "type": "space",
        "unit": "micrometer",
        "description": "x-axis"
      },
      {
        "name": "y",
        "type": "space",
        "unit": "micrometer",
        "description": "y-axis"
      }
    ]
  },
  "properties": {
    "plate": "label",
    "image": "x00_y01_p01_c1.ome.tif",
    "channel": 1.0
  }
}
```
