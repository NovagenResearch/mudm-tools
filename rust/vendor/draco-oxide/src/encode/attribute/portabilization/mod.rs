pub mod octahedral_quantization;
pub mod quantization_coordinate_wise;
pub mod to_bits;

use crate::core::bit_coder::ByteWriter;
use crate::core::shared::{ConfigType, Vector};
use crate::debug_write;
use crate::prelude::{Attribute, AttributeType, NdVector};
use crate::shared::attribute::Portable;

pub enum Portabilization<Data, const N: usize>
where
    Data: Vector<N> + Portable,
{
    QuantizationCoordinateWise(quantization_coordinate_wise::QuantizationCoordinateWise<Data, N>),
    OctahedralQuantization(octahedral_quantization::OctahedralQuantization<Data, N>),
    ToBits(to_bits::ToBits<Data, N>),
}

impl<Data, const N: usize> Portabilization<Data, N>
where
    Data: Vector<N> + Portable,
    NdVector<N, i32>: Vector<N, Component = i32>,
    NdVector<N, f32>: Vector<N, Component = f32> + Portable,
{
    /// creates a new instance of the portabilization, computes the metadata, and
    /// writes the metadata to the stream.
    // enum_dispatch does not support associated functions, we explicitly write the
    // constructor.
    pub fn new<W>(att: Attribute, cfg: Config, writer: &mut W) -> Self
    where
        W: ByteWriter,
    {
        debug_write!("Start of Portabilization Metadata", writer);
        // cfg.type_.write_to(writer);
        let out = match cfg.type_ {
            PortabilizationType::QuantizationCoordinateWise => {
                Portabilization::QuantizationCoordinateWise(
                    quantization_coordinate_wise::QuantizationCoordinateWise::<_, N>::new(
                        att, cfg, writer,
                    ),
                )
            }
            PortabilizationType::OctahedralQuantization => Portabilization::OctahedralQuantization(
                octahedral_quantization::OctahedralQuantization::new(att, cfg, writer),
            ),
            PortabilizationType::ToBits => {
                Portabilization::ToBits(to_bits::ToBits::new(att, cfg, writer))
            }
            PortabilizationType::Integer => {
                unimplemented!("Integer portabilization is not implemented yet.")
            }
        };
        debug_write!("End of Portabilization Metadata", writer);
        out
    }

    pub fn portabilize(self) -> Attribute {
        match self {
            Portabilization::QuantizationCoordinateWise(qcw) => qcw.portabilize(),
            Portabilization::OctahedralQuantization(oct) => oct.portabilize(),
            Portabilization::ToBits(tb) => tb.portabilize(),
        }
    }
}

pub trait PortabilizationImpl<const N: usize>
where
    NdVector<N, i32>: Vector<N, Component = i32>,
{
    /// portabilizes the whole data.
    fn portabilize(self) -> Attribute;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortabilizationType {
    QuantizationCoordinateWise,
    OctahedralQuantization,
    #[allow(dead_code)]
    Integer,
    ToBits,
}

impl PortabilizationType {
    pub(crate) fn get_id(&self) -> u8 {
        match self {
            PortabilizationType::ToBits => 1,
            PortabilizationType::Integer => 1, // Integer is not used in the current implementation, but kept for compatibility.
            PortabilizationType::QuantizationCoordinateWise => 2,
            PortabilizationType::OctahedralQuantization => 3,
        }
    }

    pub(crate) fn write_to<W>(&self, writer: &mut W)
    where
        W: ByteWriter,
    {
        let id = self.get_id();
        writer.write_u8(id);
    }

    pub(crate) fn default_for(ty: AttributeType) -> Self {
        match ty {
            AttributeType::Normal => PortabilizationType::OctahedralQuantization,
            AttributeType::Custom => PortabilizationType::ToBits,
            _ => PortabilizationType::QuantizationCoordinateWise, // default
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub type_: PortabilizationType,
    pub quantization_bits: u8,
    /// mudm-tools fork patch (A1): grid-identity override for the lossless NG
    /// path. `None` = upstream data-bbox behavior (unchanged — every existing
    /// arm defaults to this, so the f32/GLB path is byte-identical BY
    /// CONSTRUCTION). `Some(())` = make `QuantizationCoordinateWise` the IDENTITY
    /// over the NG integer grid `[0, 2^quantization_bits - 1]` with `min = 0`, so
    /// encode/dequant is v->v EXACT. Only the u32 NG caller ever sets this (via
    /// `encode::Config::with_ng_lossless`).
    pub grid_identity: Option<()>,
}

impl ConfigType for Config {
    fn default() -> Self {
        Config {
            type_: PortabilizationType::QuantizationCoordinateWise,
            quantization_bits: 11,
            grid_identity: None,
        }
    }
}

impl Config {
    pub fn default_for(ty: AttributeType) -> Self {
        match ty {
            AttributeType::Normal => Config {
                type_: PortabilizationType::OctahedralQuantization,
                quantization_bits: 8,
                grid_identity: None,
            },
            AttributeType::TextureCoordinate => Config {
                type_: PortabilizationType::QuantizationCoordinateWise,
                quantization_bits: 10,
                grid_identity: None,
            },
            AttributeType::Custom => Config {
                type_: PortabilizationType::ToBits,
                quantization_bits: 11, // default quantization bits (not used for ToBits)
                grid_identity: None,
            },
            _ => Self::default(),
        }
    }
}
