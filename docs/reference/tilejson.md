# TileJSON for muDM

This page is the **specification and model reference** for *TileJSON for muDM* — the JSON descriptor that ties a pyramid of tiled muDM data (vector or binary) together. It defines the object structure, the extension properties muDM adds on top of the upstream TileJSON 3.0.0 spec, and the Pydantic models that implement it.

!!! note "What this page is — and isn't"
    This page covers the **format** and the **data models** only. For step-by-step instructions on *generating* tiles with the original Python tooling (`TileWriter`, `TileReader`, `mudm2vt`), see the [Legacy Pipeline guide](../guides/legacy-pipeline.md). For the coordinate-system and multiscale concepts shared with OME-NGFF, see the [OME-NGFF guide](../guides/ome-ngff.md).

## Purpose

This specification outlines how to use TileJSON to integrate tiled muDM data, both in JSON form as well as binary form. It provides examples of how TileJSON can be used to specify the tiling scheme and zoom levels for muDM data and its binary equivalent. It is based on the [TileJSON 3.0.0 specification](https://github.com/mapbox/tilejson-spec/blob/master/3.0.0/README.md), but extends it by recommending additional properties to enable integration of muDM data and fit the purposes of microscopy imaging. The recommendations provided here are not intrinsic to the original TileJSON specification but have been tailored to suit the needs of microscopy metadata annotation and integration with muDM. However, all suggestions are designed to maintain compatibility with the original TileJSON specification.

## Background of TileJSON

[TileJSON](https://github.com/mapbox/tilejson-spec/) is a widely-used format in mapping applications for specifying tilesets. Developed to streamline the integration of different map layers, TileJSON is essential for ensuring consistency across mapping platforms. It describes tilesets through a JSON object, detailing properties like tile URLs, zoom levels, and spatial coverage.

## TileJSON for muDM object structure

The top-level object describes a complete tileset. Properties marked **Required** must be present for the object to be a valid TileJSON; the rest are optional extensions or upstream fields muDM carries through.

- **`tilejson`:** Specifies the version of the TileJSON spec being used. Required for all TileJSON objects.
- **`name`:** The name of the tileset. Optional but recommended.
- **`description`:** Provide a brief description of the tileset. Optional but recommended.
- **`version`:** The version of the tileset. Optional but recommended.
- **`attribution`:** A link to the data source or other attribution information, e.g. organisational origin. Optional but recommended.
- **`tiles`:** Required. The URL pattern for accessing the vector tiles. The `urlbase/{zlvl}/{t}/{c}/{z}/{x}/{y}` is the recommended default naming pattern for the tiles, in this order, where `urlbase` is the base URL (e.g. `http://example.com/tiles`), `{zlvl}` is the zoom level, `{t}` is the tileset timestamp, `{c}` is the channel, `{z}` is the z coordinate, and `{x}` and `{y}` are the x and y coordinates, respectively. If not using a timestamp, channel, or z coordinate, these can be omitted. The zoom level should always be first.
- **`minzoom` and `maxzoom`:** Defines the range of zoom levels for which the tiles are available.
- **`bounds`:** Optional. Specifies the geometrical bounds included in the tileset. Specified as an array of minimum four numbers in the order `[minX, minY, maxX, maxY]`, but may include up to a further six numbers for a total of ten, `[minT, minC, minZ, minX, minY, maxT, maxC, maxZ, maxX, maxY]`, where `minT` is the minimum tileset timestamp, `minC` is the minimum channel, `minZ` is the minimum z coordinate, `minX` and `minY` are the minimum x and y coordinates, `maxT` is the maximum tileset timestamp, `maxC` is the maximum channel, `maxZ` is the maximum z coordinate, and `maxX` and `maxY` are the maximum x and y coordinates.
- **`center`:** Optional. Indicates the center and suggested default view of the tileset. Minimum of three numbers in the order `[x, y, zoom]`, but may include up to a further three numbers for a total of six, `[t, c, z, x, y, zoom]`, where `t` is the tileset timestamp, `c` is the channel, `z` is the z coordinate, `x` and `y` are the x and y coordinates, and `zoom` is the zoom level. Zoom level should be last.
- **`vector_layers`:** Required. Describes each layer within the vector tiles, and has the structure documented in [Vector layer object](#vector-layer-object) below.
- **`fillzoom`:** Optional. An integer specifying the zoom level from which to generate overzoomed tiles.
- **`legend`:** Optional. Contains a legend to be displayed with the tileset.
- **`multiscale`:** Optional. A multiscale object as defined in [Multiscale object](#multiscale-object). If this property is not present, the default coordinate system is assumed to be the same as the image coordinate system, using cartesian coordinates and pixels as units.
- **`scale_factor`:** Optional. A float specifying the scale factor for the tileset. The Pydantic field default is `None` (the field is simply omitted); per this specification a consumer SHOULD assume a scale factor of `2` when the field is absent. Unless the use case requires a different scale factor, it is highly recommended to rely on `2`, as it is widely supported and assumed by many viewers.

The following fields of TileJSON may be used if the use case requires it, and are included here for completeness:

- **`scheme`:** The tiling scheme of the tileset.
- **`grids`:** The URL pattern for accessing grid data.
- **`data`:** Optional. The URL pattern for accessing data. Used for GeoJSON originally, which in this specification is replaced by muDM and used in the `tiles` field.
- **`template`:** Optional. Contains a mustache template to be used to format data from grids for interaction.

### Vector layer object

The `vector_layers` array is **required**; each entry describes one layer within the vector tiles.

| Property | Required | Description |
| --- | --- | --- |
| `id` | Yes | A unique identifier for the layer. |
| `fields` | Yes | A map of fields (attributes) to their data types. For muDM, this can either be an empty map, or a simple datatype indicator that is either `String`, `Number`, or `Bool`. Complex data types such as arrays or objects are **not** allowed. |
| `fieldranges` | No | A map of field names to their numeric ranges, e.g. `{"label": [0, 100], "channel": [0, 10]}`. |
| `fieldenums` | No | A map of field names to their possible values, e.g. `{"plate": ["A1", "A2", "B1", "B2"]}`. |
| `fielddescriptions` | No | A map of field names to human-readable descriptions, e.g. `{"plate": "Well plate identifier"}`. |
| `description` | No | A brief description of the layer. |
| `minzoom` / `maxzoom` | No | The range of zoom levels at which the layer is visible. |

!!! tip "Populating `fields`, `fieldranges`, and `fieldenums` automatically"
    You don't have to hand-write these. `mudm_tools.tilewriter.extract_fields_ranges_enums(microjson_file)` scans a muDM file's feature properties and returns `(field_names, field_ranges, field_enums)` that map directly onto these properties. See the [Legacy Pipeline guide](../guides/legacy-pipeline.md) for the end-to-end recipe.

### Multiscale object

A multiscale object represents the choice of axes (2–5D) and potentially their transformations that should be applied to the numerical data in order to arrive at the actual size of the object described. If the field is present, it MUST have the following property:

- `"axes"`: Representing the choice of axes as an array of [Axis objects](#axis-object).

It may contain **either** of, but NOT both of, the following properties:

- `"coordinateTransformations"`: Representing the set of coordinate transformations that should be applied to the numerical data in order to arrive at the actual size of the object described. It MUST be an array of objects, each object representing a coordinate transformation. Each object MUST have properties as follows:
    - `"type"`: Representing the type of the coordinate transformation. Currently supported types are `"identity"`, `"scale"`, and `"translation"`. If the type is `"scale"`, the object MUST have the property `"scale"`, representing the scaling factor. It MUST be an array of numbers, with the number of elements equal to the number of axes in the coordinate system. If the type is `"translation"`, the object MUST have the property `"translation"`, representing the translation vector. It MUST be an array of numbers, with the number of elements equal to the number of axes in the coordinate system. If the type is `"identity"`, the object MUST NOT have any other properties.
- `"transformationMatrix"`: Representing the transformation matrix from the coordinate system of the image to the coordinate system of the muDM object. It MUST be an array of arrays of numbers, with the number of rows equal to the number of axes in the coordinate system, and the number of columns equal to the number of axes in the image coordinate system. The transformation matrix MUST be invertible.

### Axis object

Together with the other axes in the `axes` array, an axis object represents the coordinate system of the muDM object (2D–5D). It MUST have the following property:

- `"name"`: Representing the name of the axis. It MUST be a string.

It may contain the following properties:

- `"type"`: The type of the axis. If present, it MUST be one of `"space"`, `"time"`, or `"channel"`.
- `"unit"`: Representing the units of the corresponding axis of the geometries in the muDM object. It MUST be one of the following values: `"angstrom"`, `"attometer"`, `"centimeter"`, `"decimeter"`, `"degree"`, `"exameter"`, `"femtometer"`, `"foot"`, `"gigameter"`, `"hectometer"`, `"inch"`, `"kilometer"`, `"megameter"`, `"meter"`, `"micrometer"`, `"mile"`, `"millimeter"`, `"nanometer"`, `"parsec"`, `"petameter"`, `"picometer"`, `"pixel"`, `"radian"`, `"terameter"`, `"yard"`, `"yoctometer"`, `"yottameter"`, `"zeptometer"`, `"zettameter"`.
- `"description"`: A string describing the axis.

## General tiling requirements

This specification is designed to be compatible with the [Vector Tile Specification](https://github.com/mapbox/vector-tile-spec/blob/master/2.1/README.md) and the [TileJSON 3.0.0 specification](https://github.com/mapbox/tilejson-spec/blob/master/3.0.0/README.md). The Vector Tile Specification specifically requires vector tiles to be **agnostic of the global coordinate system** — each tile carries a relative coordinate system, and the mapping back to global coordinates is defined in the TileJSON (via `bounds`). Our ambitions in general are to follow the same principles.

!!! warning "Use `.pbf`, not `.mvt`, for binary tiles"
    One difference from upstream: we recommend the file ending for binary tiles be **`.pbf`** instead of `.mvt`, to avoid confusion with the Mapbox Vector Tile format. The binary tiles should be encoded in the [Protobuf format](https://protobuf.dev/) as defined in the Vector Tile Specification.

## Pydantic models for TileJSON for muDM

The TileJSON-for-muDM models live in the **core `mudm` package** at `mudm.tilemodel` (not in `mudm_tools`). The legacy pipeline (`TileWriter` / `TileReader` / `TileHandler`) is constructed from a `mudm.tilemodel.TileModel` instance.

!!! example "Building a `TileModel` in code"
    ```python
    from mudm.tilemodel import TileLayer, TileModel
    from mudm_tools.tilewriter import extract_fields_ranges_enums, getbounds

    microjson_file = "data.json"

    # Derive layer schema + bounds straight from a muDM file
    field_names, field_ranges, field_enums = extract_fields_ranges_enums(microjson_file)
    bounds = getbounds(microjson_file, square=True)  # [minx, miny, maxx, maxy]

    layer = TileLayer(
        id="image_layer",
        fields=field_names,
        fieldranges=field_ranges,
        # NOTE: field_enums values are Python sets — convert to lists before
        # JSON serialization (e.g. {k: sorted(v) for k, v in field_enums.items()})
        fieldenums={k: sorted(v) for k, v in field_enums.items()},
        minzoom=0,
        maxzoom=10,
    )

    tile_model = TileModel(
        tilejson="3.0.0",
        name="2D Data Example",
        tiles=["http://example.com/tiled_example/{z}/{x}/{y}.pbf"],
        minzoom=0,
        maxzoom=10,
        bounds=bounds,
        vector_layers=[layer],
    )
    ```

### TileJSON

The version-field model from the core package.

::: mudm.tilemodel.TileJSON
    options:
      show_root_heading: true

### TileModel

The core TileJSON pyramid configuration model. Beyond the standard TileJSON fields (`tilejson`, `tiles`, `name`, `description`, `version`, `attribution`, `template`, `legend`, `scheme`, `grids`, `data`, `minzoom`, `maxzoom`, `bounds`, `center`, `fillzoom`, `vector_layers`) it adds the muDM extensions `multiscale` and `scale_factor`. This is the object passed to `TileWriter`, `TileReader`, and `TileHandler`.

::: mudm.tilemodel.TileModel
    options:
      show_root_heading: true

### TileLayer

The core vector-layer descriptor model. Beyond the standard TileJSON `vector_layers` fields (`id`, `fields`, `minzoom`, `maxzoom`, `description`) it adds the muDM extensions `fieldranges`, `fieldenums`, `fielddescriptions`, and `vocabularies`.

::: mudm.tilemodel.TileLayer
    options:
      show_root_heading: true

## TileJSON for muDM example with vector layers

The example below illustrates a TileJSON for a muDM tileset with multiple layers of detail. The tileset has a single vector layer that describes images. The `fields` property of the layer specifies the attributes of the layer, including the data types of the attributes. The `tiles` property specifies the URL pattern for accessing the vector tiles, which in this case is a 2D data set (no channels, time, or z-axis) with a zoom level.

The tile files are organized according to the tiling scheme, with the directory structure `zlvl/x/y.json` where `zlvl` is the zoom level, `x` is the x coordinate, and `y` is the y coordinate. The muDM files contain the muDM objects for the corresponding tiles. For example, the muDM object for the tile at zoom level 1, tile `(0, 1)` is located at `tiled_example/1/0/1.json`. Examples for muDM objects at zoom levels 0, 1, and 2 are provided further down.

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
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    }
}
```

## Tiled binary TileJSON

In addition to the JSON format, tiles may be encoded in a binary Protobuf format. Below is a similar example to the one above, but for binary tiles. The `tiles` property specifies the URL pattern for accessing the binary tiles (note the `.pbf` ending). The `fillzoom` property specifies the zoom level from which to generate overzoomed tiles, which in this case starts at level 3, after the last specified layer. This example also demonstrates the muDM `fieldranges`, `fieldenums`, and `fielddescriptions` extension properties.

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
              "label": [0, 100],
              "channel": [0, 10]
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

## Tiled muDM examples

The following are examples of the per-tile muDM objects that the TileJSON above points to. The detailed levels are each a `FeatureCollection` whose coordinates are expressed in the global coordinate space defined by the tileset `bounds`. The coarsest level may instead serve a TileJSON *descriptor* as an overview, as shown for Level 0 below.

### Level 0

At the coarsest zoom, this tile is a TileJSON descriptor (an overview of the tileset) rather than a `FeatureCollection`. Example URL: `http://example.com/tiles/0/0/0.json`

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

A muDM object at zoom level 1, tile `(0, 1)`. Example URL: `http://example.com/tiles/1/0/1.json`

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
            [0, 10000],
            [10000, 10000],
            [10000, 20000],
            [0, 20000],
            [0, 10000]
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

A muDM object at zoom level 2, tile `(1, 3)`. Example URL: `http://example.com/tiles/2/1/3.json`

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
            [5000, 15000],
            [10000, 15000],
            [10000, 20000],
            [5000, 20000],
            [5000, 15000]
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

## See also

- [Legacy Pipeline guide](../guides/legacy-pipeline.md) — generating and reading these tiles with `TileWriter`, `TileReader`, and `mudm2vt`.
- [OME-NGFF guide](../guides/ome-ngff.md) — the multiscale and coordinate-system concepts muDM shares with OME-NGFF.
- [Python API reference](python-api.md) — the full `mudm_tools` API surface.
