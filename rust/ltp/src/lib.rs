pub mod eisner;
pub mod entities;
pub mod hook;
pub mod perceptron;
pub mod stnsplit;
pub mod viterbi;

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

pub use eisner::eisner;
pub use entities::{drop_get_entities, get_entities};
pub use stnsplit::{stn_split, stn_split_with_options, SplitOptions};
pub use viterbi::viterbi_decode_postprocessing;
