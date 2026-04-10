# Extending OME-NGFF with Tiled Data Models: Integrating TileJSON, muDM, and Vector Tiling

## Introduction

[OME Next-Generation File Format](https://ngff.openmicroscopy.org/latest/) is a format for storing bioimaging data in the cloud, for which there is a growing need to integrate with established vector tiling formats widely used in geospatial applications. Vector tiling formats like [Mapbox Vector Tiles (MVT)](https://github.com/mapbox/vector-tile-spec) and [muDM](https://polusai.github.io/microjson/)  and tiling descriptors like [TileJSON](https://github.com/mapbox/tilejson-spec/tree/master/3.0.0), which has been [adapted to muDM](https://polusai.github.io/microjson/tiling/) provide a standardized way to access and visualize large geospatial datasets. This document explores how these tiling models can be integrated with OME-NGFF.

The **TileJSON** serves as a endpoint mapping that can bridge between NGFF’s chunked multiscale data structures and other tiling models. Tiles may then be of different formats:

- **muDM**: muDM may be used for each tile, using its intrinsic coordinate system to annotate the features of the tile.
- **JSON Vector tiles**: JSON representations of vector data, like Mapbox Vector Tiles (MVT) in JSON format.
- **Binary tile formats** encoded either like Mapbox Vector Tiles (MVT, protobuf-encoded tiles), or GeoParquet (Apache Parquet-based vector tiles).

## Background: OME-NGFF

**OME-NGFF** provides a standardized way to store large image datasets (in for example Zarr format) with metadata describing scale transformations, coordinate systems, and multiple resolution levels. Each resolution level consists of a set of chunked arrays, allowing efficient partial retrieval and processing of large images by tiled raster data.

Key features of NGFF:

- **Multidimensional data** (2-5D):  time (t), channel (c), z-depth (z), and spatial dimensions (y, x).
- **Multiscale pyramids**, each level providing a different resolution.  
- **Flexible coordinate transformations** that can place data in a variety of coordinate reference systems (CRS).

## TileJSON as an endpoint mapping Layer

**TileJSON** is a well-established specification within the geospatial community that describes tiled data sources via a simple JSON schema. It is commonly used in web mapping to:

- Reference a set of tiled resources (raster or vector) defined by zoom levels and tile coordinates (z/x/y). The standard order differs from NGFF’s (z/y/x).
- Provide metadata such as bounding boxes, attribution, min/max zoom levels, and tile endpoints.

While TileJSON has traditionally been used in 2D applications, there is nothing that hinders it from being used with higher dimensions, including 5D as with NGFF. The muDM implementation of TileJSON outlines such usage. It thus can be used to map endpoints for a coordinate system that is not strictly geospatial, given the following:

1. NGFF’s multiscale pyramids could be mapped directly to TileJSON’s `zoom` levels, however, TileJSON assumes a zoom factor of 2, while NGFF may have arbitrary scale factors. This may require additional adaptions of the TileJSON schema, to allow for arbitrary scale factors. This could be done by defining a new `scale_factor` field in the TileJSON schema. Followingly, the `Multiscale` class in the muDM should be moved to the TileJSON schema.
2. Associate NGFF array chunks with tile indices (z/x/y) derived from spatial transformations, given the NGFF multiscale metadata and the corresponding TileJSON metadata.
3. A practical observation is that if the multiscale pyramids differs, transformations between the two systems are needed, in addition to what is described above, which could be avoided by using the same multiscale pyramid structure in both systems. This is also valid for the tile size (expressed in the global coordinates), which should be the same in both systems for a specific zoom level.
4. Layering of data is supported in TileJSON, as the array `vector_layers` in its schema. Layers are thus stored together for each tile, as a contrast to the NGFF, where the layers are stored in separate arrays as labeled images.
5. The NGFF raster data hierarchy could be expressed with a TileJSON which does not have a vector layer but instead just maps the raster data endpoints as formatted in the field `tiles` in the TileJSON schema.

## Incorporating muDM and Vector Tiles

**muDM** is a valid GeoJSON but with a few extra additions, including for example feature classes and parent-relations between features. It aligns well with TileJSON as individual tiles could be specified in muDM, but with the same agnostic coordinate system as for the binary vector tiles, and the intermediate vector tile JSON.

**Vector Tiles** is a binary format for encoding vector data in tiles, either using protobuf, or directly as vector tile JSON, or some other format, like GeoParquet. Protobuf is widely used in geospatial applications to express them in  compact form. The MVT format can be used to represent vector data in a tile-based system, with each tile containing a subset of the vector data. The MVT format is well-suited for representing vector data in a tile-based system, as it allows for efficient storage and retrieval of vector data.

## NGFF Labels and Vector Tiles

NGFF labels can be represented as vector tiles, where each tile contains a subset of the labels. This allows for efficient storage and retrieval of labels, as well as the ability to overlay them on raster data. Labels can be stored as both raster data and vector tiles, providing flexibility for different tools and applications. It is strongly suggested to use the same identifier for the labels in the NGFF and the vector tiles, to allow for easy integration between the two systems.

## Conclusion

TileJSON can serve as a bridge between OME-NGFF and individual vector tiles expressed in different formats, including muDM and binary vector tiles. This integration provides a standardized way to access and visualize large datasets while allowing efficient storage and retrieval of vector data. By aligning coordinate reference systems and handling higher-dimensional data through slicing or conventions, seamless integration between OME-NGFF and tiling models is achievable. Vector tiles may replace NGFF labels, or be stored in parallel for flexibility. For practical use, zoom levels and tile sizes should be consistent between the two systems.

The current version of muDM must be adapted to support the NGFF data model, and the TileJSON schema should be extended to support arbitrary scale factors.
