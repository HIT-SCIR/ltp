mod definition;
mod feature;
mod model;
mod parameter;
#[cfg(feature = "serialization")]
mod serialization;
mod trainer;

pub use definition::{CWSDefinition, Definition, GenericItem, NERDefinition, POSDefinition};
pub use feature::{TraitFeature, TraitFeatureCompressUtils, TraitFeaturesTrainUtils};
pub use model::{PaMode, Perceptron};
pub use parameter::{
    TraitParameter, TraitParameterStorage, TraitParameterStorageCompressUtils,
    TraitParameterStorageTrainUtils, TraitParameterStorageUtils,
};
#[cfg(feature = "serialization")]
pub use serialization::{
    schema, Codec, Format, ModelSerde, Reader, Schema, SerdeCWSModel, SerdeModel, SerdeNERModel,
    SerdePOSModel,
};
pub use trainer::{Algorithm, Trainer};
pub type Sample = (Vec<Vec<String>>, Vec<usize>);
