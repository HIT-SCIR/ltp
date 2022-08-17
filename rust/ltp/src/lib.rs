pub mod eisner;
pub mod entities;
pub mod hook;
pub mod perceptron;
pub mod stnsplit;
pub mod viterbi;

pub use perceptron::{
    Algorithm, CWSDefinition, NERDefinition, POSDefinition, PaMode, Perceptron, Trainer, SerdeModel, SerdeCWSModel, SerdePOSModel, SerdeNERModel,
};
#[cfg(feature = "serialization")]
pub use perceptron::{Codec, Format, ModelSerde, Reader};

pub type CWSModel = SerdeCWSModel;
pub type POSModel = SerdePOSModel;
pub type NERModel = SerdeNERModel;

pub use eisner::eisner;
pub use entities::{drop_get_entities, get_entities};
pub use stnsplit::{stn_split, stn_split_with_options, SplitOptions};
pub use viterbi::viterbi_decode_postprocessing;
