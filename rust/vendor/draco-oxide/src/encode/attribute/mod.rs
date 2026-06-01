pub(crate) mod attribute_encoder;
pub(crate) mod portabilization;
pub(crate) mod prediction_transform;

use crate::encode::attribute::portabilization::PortabilizationType;
use crate::encode::connectivity::ConnectivityEncoderOutput;
#[cfg(feature = "evaluation")]
use crate::eval;

use crate::prelude::{Attribute, AttributeType, ByteWriter, ConfigType};
use crate::shared::connectivity::edgebreaker::TraversalType;

/// mudm-tools fork patch (T2b): the effective portabilization for an attribute,
/// applying the optional Position override carried on the top-level encode
/// `Config`. Non-Position attributes are unaffected; if no override is set the
/// upstream `PortabilizationType::default_for` is used (byte-unchanged).
fn effective_portabilization(ty: AttributeType, cfg: &super::Config) -> PortabilizationType {
    if ty == AttributeType::Position {
        if let Some(p) = cfg.position_portabilization {
            return p;
        }
    }
    PortabilizationType::default_for(ty)
}

pub fn encode_attributes<W>(
    atts: Vec<Attribute>,
    writer: &mut W,
    conn_out: ConnectivityEncoderOutput<'_>,
    cfg: &super::Config,
) -> Result<(), Err>
where
    W: ByteWriter,
{
    #[cfg(feature = "evaluation")]
    eval::scope_begin("attributes", writer);

    // Write the number of attribute encoders/decoders (In draco-oxide, this is the same as the number of attributes as
    // each attribute has its own encoder/decoder)
    writer.write_u8(atts.len() as u8);
    #[cfg(feature = "evaluation")]
    eval::write_json_pair("attributes count", atts.len().into(), writer);

    for (i, att) in atts.iter().enumerate() {
        if cfg.encoder_method == crate::shared::header::EncoderMethod::Edgebreaker {
            // encode decoder id
            writer.write_u8((i as u8).wrapping_sub(1));
            // encode attribute type
            att.get_domain().write_to(writer);
            // write traversal method for attribute encoding/decoding sequencer. We currently only support depth-first traversal.
            TraversalType::DepthFirst.write_to(writer);
        }
    }

    #[cfg(feature = "evaluation")]
    eval::array_scope_begin("attributes", writer);

    let mut port_atts: Vec<Attribute> = Vec::new();
    for att in &atts {
        // Write 1 to indicate that the encoder is for one attribute.
        writer.write_u8(1);

        att.get_attribute_type().write_to(writer);
        att.get_component_type().write_to(writer);
        writer.write_u8(att.get_num_components() as u8);
        writer.write_u8(0); // Normalized flag, currently not used.
        writer.write_u8(att.get_id().as_usize() as u8); // unique id

        // write the decoder type.
        // mudm-tools fork patch (T2b): honor the optional Position portabilization
        // override so the id written here matches the portabilization actually
        // applied in AttributeEncoder (otherwise the stream is invalid).
        effective_portabilization(att.get_attribute_type(), cfg).write_to(writer);
    }

    for (i, att) in atts.into_iter().enumerate() {
        #[cfg(feature = "evaluation")]
        eval::scope_begin("attribute", writer);

        let parents_ids = att.get_parents();
        let parents = parents_ids
            .iter()
            .map(|id| port_atts.iter().find(|att| att.get_id() == *id).unwrap())
            .collect::<Vec<_>>();

        let ty = att.get_attribute_type();
        let len = att.len();
        // mudm-tools fork patch (T2b): resolve the effective portabilization for
        // this attribute (override applies to Position only) and build the
        // per-attribute Config with it, so the encoder applies exactly what was
        // declared in the stream header above.
        let mut enc_cfg = attribute_encoder::Config::default_for(ty, len);
        enc_cfg.set_position_portabilization(effective_portabilization(ty, cfg));
        // mudm-tools fork patch (A1): thread the Position quantization override
        // (qbits + grid-identity flag) down to the portabilization config site.
        // Applies to Position only; non-Position attributes ignore it.
        if ty == AttributeType::Position {
            if let Some((qbits, grid_identity)) = cfg.position_quantization {
                enc_cfg.set_position_quantization(qbits, grid_identity);
            }
        }
        let encoder = attribute_encoder::AttributeEncoder::new(
            att,
            i,
            &parents,
            &conn_out,
            writer,
            enc_cfg,
        );

        let port_att = encoder.encode::<true, false>()?;
        port_atts.push(port_att);

        #[cfg(feature = "evaluation")]
        eval::scope_end(writer);
    }

    #[cfg(feature = "evaluation")]
    {
        eval::array_scope_end(writer);
        eval::scope_end(writer);
    }

    Ok(())
}

#[derive(Clone, Debug)]
pub struct Config {
    #[allow(unused)]
    // This field is unused in the current implementation, as we only support the default attribute encoder configuration.
    cfgs: Vec<attribute_encoder::Config>,
}

impl ConfigType for Config {
    fn default() -> Self {
        Self {
            cfgs: vec![attribute_encoder::Config::default()],
        }
    }
}

#[remain::sorted]
#[derive(thiserror::Error, Debug)]
pub enum Err {
    #[error("Attribute encoding error: {0}")]
    AttributeError(#[from] attribute_encoder::Err),
}
