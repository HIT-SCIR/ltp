#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod algorithms;
mod hook;
mod perceptron;
mod stnsplit;

use crate::perceptron::{ModelType, PyModel, PyTrainer};
pub use algorithms::{py_eisner, py_get_entities, py_viterbi_decode_postprocess};
use hook::PyHook;
pub use perceptron::{
    CharacterType, PyAlgorithm, PyCWSModel, PyCWSTrainer, PyNERModel, PyNERTrainer, PyPOSModel,
    PyPOSTrainer,
};
use pyo3::prelude::*;
use stnsplit::StnSplit;

/// LTP Module
#[pymodule]
fn ltp_extension(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Algorithms Module
    let algorithms = PyModule::new(py, "algorithms")?;

    algorithms.add_class::<StnSplit>()?;
    algorithms.add_class::<PyHook>()?;
    algorithms.add_function(wrap_pyfunction!(py_eisner, m)?)?;
    algorithms.add_function(wrap_pyfunction!(py_get_entities, m)?)?;
    algorithms.add_function(wrap_pyfunction!(py_viterbi_decode_postprocess, m)?)?;

    // Perceptron Module
    let perceptron = PyModule::new(py, "perceptron")?;
    perceptron.add_class::<PyModel>()?;
    perceptron.add_class::<ModelType>()?;
    perceptron.add_class::<PyTrainer>()?;
    perceptron.add_class::<PyAlgorithm>()?;

    perceptron.add_class::<CharacterType>()?;
    perceptron.add_class::<PyCWSModel>()?;
    perceptron.add_class::<PyCWSTrainer>()?;

    perceptron.add_class::<PyPOSModel>()?;
    perceptron.add_class::<PyPOSTrainer>()?;

    perceptron.add_class::<PyNERModel>()?;
    perceptron.add_class::<PyNERTrainer>()?;

    m.add_submodule(algorithms)?;
    m.add_submodule(perceptron)?;
    Ok(())
}
