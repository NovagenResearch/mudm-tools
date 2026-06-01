pub(crate) mod attribute;
pub(crate) mod connectivity;
pub(crate) mod entropy;
pub(crate) mod header;
pub(crate) mod metadata;

use crate::core::bit_coder::ByteWriter;
use crate::core::mesh::Mesh;
use crate::core::shared::ConfigType;
use crate::encode::attribute::portabilization::PortabilizationType;
use crate::{debug_write, shared};
use thiserror::Error;

#[cfg(feature = "evaluation")]
use crate::eval;

pub trait EncoderConfig {
    type Encoder;
    fn get_encoder(&self) -> Self::Encoder;
}

#[derive(Debug, Clone)]
pub struct Config {
    #[allow(unused)]
    // This field is unused in the current implementation, as we only support edgebreaker.
    connectivity_encoder_cfg: connectivity::Config,
    #[allow(unused)]
    // This field is unused in the current implementation, as we only suport the default attribute encoder configuration.
    attribute_encoder_cfg: attribute::Config,
    geometry_type: header::EncodedGeometryType,
    encoder_method: shared::header::EncoderMethod,
    metdata: bool,
    /// mudm-tools fork patch (T2b): optional per-encode override of the
    /// portabilization used for the Position attribute. `None` keeps the
    /// upstream default (`PortabilizationType::default_for` => the embedded
    /// QuantizationCoordinateWise quantization transform). `Some(ToBits)` stores
    /// pre-quantized integer positions LOSSLESSLY, which the Neuroglancer
    /// multilod_draco format requires (it forbids Draco's built-in quantization).
    /// The attribute TYPE stays `Position` (spec-correct); only the
    /// portabilization id written to the stream + the actual portabilization
    /// switch. Defaults to `None` so the GLB/f32 path is byte-unchanged.
    pub position_portabilization: Option<PortabilizationType>,
    /// mudm-tools fork patch (A1): optional per-encode override of the Position
    /// quantization. `Some((qbits, grid_identity))` forces the Position
    /// QuantizationCoordinateWise transform to use `qbits` bits and, when
    /// `grid_identity` is true, to be the IDENTITY over the NG integer grid
    /// `[0, 2^qbits - 1]` (lossless v->v). `None` keeps the upstream data-bbox
    /// behavior (default quantization_bits=11). Only the u32 NG caller sets this;
    /// `None` keeps the GLB/f32 path byte-identical.
    pub position_quantization: Option<(u8, bool)>,
}

impl ConfigType for Config {
    fn default() -> Self {
        Self {
            connectivity_encoder_cfg: connectivity::Config::default(),
            attribute_encoder_cfg: attribute::Config::default(),
            geometry_type: header::EncodedGeometryType::TrianglarMesh,
            encoder_method: shared::header::EncoderMethod::Edgebreaker,
            metdata: false,
            position_portabilization: None,
            position_quantization: None,
        }
    }
}

impl Config {
    /// mudm-tools fork patch (T2b): builder for the lossless NG path. Returns a
    /// `Config` identical to `default()` except the Position attribute uses
    /// `ToBits` (lossless integer storage) instead of the quantizing default.
    pub fn with_position_to_bits() -> Self {
        let mut cfg = <Self as ConfigType>::default();
        cfg.position_portabilization = Some(PortabilizationType::ToBits);
        cfg
    }

    /// mudm-tools fork patch (A1): builder for the LOSSLESS, libdraco-conformant
    /// NG path. Keeps `QuantizationCoordinateWise` (portabilization id=2, which
    /// libdraco decodes today) but forces its quantization transform to be the
    /// IDENTITY over the NG integer grid `[0, 2^qbits - 1]` (min=0,
    /// range=2^qbits-1, bits=qbits). Then encode/dequant is v->v EXACT.
    ///
    /// `qbits` MUST equal the `vertex_quantization_bits` the NG caller
    /// pre-quantized the positions with (so the grid range matches the
    /// pre-quantization). The f32/GLB path keeps `Config::default()` untouched.
    pub fn with_ng_lossless(qbits: u8) -> Self {
        let mut cfg = <Self as ConfigType>::default();
        cfg.position_portabilization = Some(PortabilizationType::QuantizationCoordinateWise);
        cfg.position_quantization = Some((qbits, true));
        cfg
    }
}

#[remain::sorted]
#[derive(Error, Debug)]
pub enum Err {
    #[error("Attribute encoding error: {0}")]
    AttributeError(#[from] attribute::Err),
    #[error("Connectivity encoding error: {0}")]
    ConnectivityError(#[from] connectivity::Err),
    #[error("Header encoding error: {0}")]
    HeaderError(#[from] header::Err),
    #[error("Metadata encoding error: {0}")]
    MetadataError(#[from] metadata::Err),
}

/// Encodes the input mesh into a provided byte stream using the provided configuration.
pub fn encode<W>(mesh: Mesh, writer: &mut W, cfg: Config) -> Result<(), Err>
where
    W: ByteWriter,
{
    #[cfg(feature = "evaluation")]
    eval::scope_begin("compression info", writer);

    // Encode header
    header::encode_header(writer, &cfg)?;

    debug_write!("Header done, now starting metadata.", writer);

    // Encode metadata
    if cfg.metdata {
        #[cfg(feature = "evaluation")]
        eval::scope_begin("metadata", writer);
        metadata::encode_metadata(&mesh, writer)?;
        #[cfg(feature = "evaluation")]
        eval::scope_end(writer);
    }

    debug_write!("Metadata done, now starting connectivity.", writer);

    // Destruct the mesh so that attributes and faces have the different lifetime.
    let Mesh {
        mut attributes,
        faces,
        ..
    } = mesh;

    // Encode connectivity
    let conn_out = connectivity::encode_connectivity(&faces, &mut attributes, writer, &cfg)?;
    debug_write!("Connectivity done, now starting attributes.", writer);

    // Encode attributes
    attribute::encode_attributes(attributes, writer, conn_out, &cfg)?;

    debug_write!("All done", writer);

    #[cfg(feature = "evaluation")]
    eval::scope_end(writer);
    Ok(())
}
