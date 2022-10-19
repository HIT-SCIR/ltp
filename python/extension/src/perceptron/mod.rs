mod alg;
mod com;
mod model;
mod specialization;
mod trainer;

pub type Perceptron<T> = ltp::perceptron::SerdeModel<T, f64>;
pub use alg::PyAlgorithm;
pub use model::{EnumModel, ModelType, PyModel};
pub use specialization::{
    CharacterType, PyCWSModel, PyCWSTrainer, PyNERModel, PyNERTrainer, PyPOSModel, PyPOSTrainer,
};
pub use trainer::{EnumTrainer, PyTrainer};
