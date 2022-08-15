use crate::{CWSDefinition, NERDefinition, POSDefinition, Perceptron};
use anyhow::Result;
pub use apache_avro::{schema, Codec, Reader, Schema};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Copy, Clone, Debug)]
pub enum Format {
    JSON,
    AVRO(Codec),
}

pub trait ModelSerde<'de>: Serialize + Deserialize<'de> {
    fn load<R: std::io::Read>(reader: R, format: Format) -> Result<Self>;
    fn load_avro<R: std::io::Read>(reader: Reader<R>) -> Result<Self>;
    fn save<W: std::io::Write>(&self, writer: W, format: Format) -> Result<()>;
}

pub type SerdeModel<T, V> = Perceptron<T, HashMap<String, usize>, Vec<V>, V>;
pub type SerdeCWSModel = SerdeModel<CWSDefinition, f64>;
pub type SerdePOSModel = SerdeModel<POSDefinition, f64>;
pub type SerdeNERModel = SerdeModel<NERDefinition, f64>;

#[macro_export]
macro_rules! impl_model_serialization {
    ($name:tt, $raw_schema:ident) => {
        impl<'de> ModelSerde<'de> for $name {
            fn load<R: std::io::Read>(reader: R, format: Format) -> Result<Self> {
                let model = match format {
                    Format::JSON => serde_json::from_reader(reader)?,
                    Format::AVRO(_) => {
                        let schema = apache_avro::Schema::parse_str($raw_schema)?;
                        let reader = apache_avro::Reader::with_schema(&schema, reader)?;

                        let mut model = None;
                        for value in reader {
                            model = Some(apache_avro::from_value::<Self>(&value.unwrap())?);
                        }
                        model.unwrap()
                    }
                };
                Ok(model)
            }

            fn load_avro<R: std::io::Read>(reader: apache_avro::Reader<R>) -> Result<Self> {
                let model = {
                    let mut model = None;
                    for value in reader {
                        model = Some(apache_avro::from_value::<Self>(&value.unwrap())?);
                    }
                    model.unwrap()
                };
                Ok(model)
            }

            fn save<W: std::io::Write>(&self, writer: W, format: Format) -> Result<()> {
                match format {
                    Format::JSON => {
                        serde_json::to_writer(writer, self)?;
                    }
                    Format::AVRO(codec) => {
                        let schema = apache_avro::Schema::parse_str($raw_schema)?;
                        let mut writer = apache_avro::Writer::with_codec(&schema, writer, codec);
                        writer.append_ser(self)?;
                        writer.flush()?;
                    }
                }
                Ok(())
            }
        }
    };
    () => {};
}

static CWS_RAW_SCHEMA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/vendor/schema/cws.avsc"
));
static POS_RAW_SCHEMA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/vendor/schema/pos.avsc"
));
static NER_RAW_SCHEMA: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/vendor/schema/ner.avsc"
));

impl_model_serialization!(SerdeCWSModel, CWS_RAW_SCHEMA);
impl_model_serialization!(SerdePOSModel, POS_RAW_SCHEMA);
impl_model_serialization!(SerdeNERModel, NER_RAW_SCHEMA);
