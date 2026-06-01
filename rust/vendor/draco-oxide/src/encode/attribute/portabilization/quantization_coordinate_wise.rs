use crate::core::shared::{AttributeValueIdx, DataValue, Vector};
use crate::prelude::{Attribute, ByteWriter, NdVector};
use crate::shared::attribute::Portable;

use super::{Config, PortabilizationImpl};

pub(crate) struct QuantizationCoordinateWise<Data, const N: usize>
where
    Data: Vector<N>,
{
    att: Attribute,
    range_size: f32,
    min_values: NdVector<N, f32>,
    quantization_bits: u8,
    _phantom: std::marker::PhantomData<Data>,
}

impl<Data, const N: usize> QuantizationCoordinateWise<Data, N>
where
    NdVector<N, i32>: Vector<N, Component = i32>,
    NdVector<N, f32>: Vector<N, Component = f32> + Portable,
    Data: Vector<N> + Portable,
    Data::Component: DataValue,
{
    pub fn new<W>(att: Attribute, cfg: Config, writer: &mut W) -> Self
    where
        W: ByteWriter,
    {
        // mudm-tools fork patch (A1): when the grid-identity override is set,
        // bypass the data-bbox scan and force the transform to be the IDENTITY
        // over the NG integer grid: min = 0, range = (2^qbits - 1). Then
        // portabilize_value maps integer v -> v EXACTLY (the (1<<bits)-1 multiply
        // cancels the divisor), and libdraco dequant reverses it exactly. The
        // metadata LAYOUT is unchanged (only the values), so the stream stays
        // libdraco-conformant (portabilization id stays 2).
        let (min_values, delta_max) = if cfg.grid_identity.is_some() {
            (
                NdVector::<N, f32>::zero(),
                ((1u64 << cfg.quantization_bits) - 1) as f32,
            )
        } else {
            let mut min_values = NdVector::<N, f32>::zero();
            for val in att.unique_vals_as_slice::<Data>() {
                for i in 0..N {
                    let component = val.get(i).to_f64() as f32;
                    if component < *min_values.get(i) {
                        *min_values.get_mut(i) = component;
                    }
                }
            }

            let mut max_values = NdVector::<N, f32>::zero();
            for val in att.unique_vals_as_slice::<Data>() {
                for i in 0..N {
                    let component = val.get(i).to_f64() as f32;
                    if component > *max_values.get(i) {
                        *max_values.get_mut(i) = component;
                    }
                }
            }

            let mut delta_max = 0.0;
            for i in 0..N {
                let delta = *max_values.get(i) - *min_values.get(i);
                if delta > delta_max {
                    delta_max = delta;
                }
            }
            (min_values, delta_max)
        };

        // write metadata
        min_values.write_to(writer);
        delta_max.write_to(writer);
        writer.write_u8(cfg.quantization_bits);

        Self {
            att,
            range_size: delta_max,
            min_values,
            quantization_bits: cfg.quantization_bits,
            _phantom: std::marker::PhantomData,
        }
    }

    fn portabilize_value(&mut self, val: Data) -> NdVector<N, i32> {
        // convert value to float vector TODO: implement the vector conversion so that this will be one line
        let val: NdVector<N, f32> = {
            let mut out = NdVector::<N, f32>::zero();
            for i in 0..N {
                *out.get_mut(i) = val.get(i).to_f64() as f32;
            }
            out
        };
        let diff = val - self.min_values;
        let normalized = if self.range_size == 0.0 {
            diff
        } else {
            diff / self.range_size
        };
        let quantized = normalized * f32::from_u64((1 << self.quantization_bits) - 1);
        let mut out = NdVector::<N, i32>::zero();
        for i in 0..N {
            *out.get_mut(i) = (*quantized.get(i) + 0.5).to_i64() as i32;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::attribute::AttributeDomain;
    use crate::encode::attribute::portabilization::PortabilizationType;
    use crate::prelude::AttributeType;

    /// mudm-tools A1 (Step 1): with the grid-identity override, the QCW transform
    /// must be the IDENTITY over the NG integer grid [0, 2^qbits - 1]:
    /// portabilize_value(v) == v for EVERY integer v in the full range, at both
    /// qbits=10 and qbits=16. We sweep the entire range (not just endpoints) to
    /// catch any intermediate f32 rounding in the (v/range)*(2^bits-1) chain.
    fn assert_identity_over_grid(qbits: u8) {
        let qmax = (1u64 << qbits) - 1;

        // One vertex at the grid origin and one at the grid max so the attribute
        // carries values spanning the full range; the override must IGNORE this
        // data bbox and use [0, 2^qbits - 1] regardless.
        let data: Vec<NdVector<3, f32>> = vec![
            NdVector::from([0.0f32, 0.0, 0.0]),
            NdVector::from([qmax as f32, qmax as f32, qmax as f32]),
        ];
        let att = Attribute::new::<NdVector<3, f32>, 3>(
            data,
            AttributeType::Position,
            AttributeDomain::Position,
            Vec::new(),
        );

        let cfg = Config {
            type_: PortabilizationType::QuantizationCoordinateWise,
            quantization_bits: qbits,
            grid_identity: Some(()),
        };

        let mut sink: Vec<u8> = Vec::new();
        let mut qcw =
            QuantizationCoordinateWise::<NdVector<3, f32>, 3>::new(att, cfg, &mut sink);

        // Sweep EVERY integer in [0, 2^qbits). For qbits=16 that is 65536 values.
        for v in 0..=qmax {
            let vf = v as f32;
            let input = NdVector::<3, f32>::from([vf, vf, vf]);
            let out = qcw.portabilize_value(input);
            for i in 0..3 {
                assert_eq!(
                    *out.get(i),
                    v as i32,
                    "grid-identity QCW must map v->v exactly: qbits={}, v={}, got={}",
                    qbits,
                    v,
                    *out.get(i),
                );
            }
        }
    }

    #[test]
    fn grid_identity_is_exact_qbits_10() {
        assert_identity_over_grid(10);
    }

    #[test]
    fn grid_identity_is_exact_qbits_16() {
        assert_identity_over_grid(16);
    }
}

impl<Data, const N: usize> PortabilizationImpl<N> for QuantizationCoordinateWise<Data, N>
where
    NdVector<N, i32>: Vector<N, Component = i32>,
    NdVector<N, f32>: Vector<N, Component = f32> + Portable,
    Data: Vector<N> + Portable,
{
    fn portabilize(mut self) -> Attribute {
        let mut out = Vec::new();
        for i in 0..self.att.num_unique_values() {
            let i = AttributeValueIdx::from(i);
            out.push(self.portabilize_value(self.att.get_unique_val::<Data, N>(i)));
        }
        let mut port_att = Attribute::from_without_removing_duplicates(
            self.att.get_id(),
            out,
            self.att.get_attribute_type(),
            self.att.get_domain(),
            self.att.get_parents().clone(),
        );
        port_att.set_point_to_att_val_map(self.att.take_point_to_att_val_map());
        port_att
    }
}
