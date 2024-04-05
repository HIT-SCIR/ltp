pub mod perceptron;
pub mod utils;

pub use perceptron::{
    Algorithm, CWSDefinition, NERDefinition, POSDefinition, PaMode, Perceptron, Trainer,
};
#[cfg(feature = "serialization")]
pub use perceptron::{Codec, Format, ModelSerde, Reader, SerdeModel, SerdeCWSModel, SerdePOSModel, SerdeNERModel};

#[cfg(feature = "serialization")]
pub type CWSModel = SerdeCWSModel;
#[cfg(feature = "serialization")]
pub type POSModel = SerdePOSModel;
#[cfg(feature = "serialization")]
pub type NERModel = SerdeNERModel;


