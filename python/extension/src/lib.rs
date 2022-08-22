#[cfg(not(target_env = "musl"))]
use mimalloc::MiMalloc;

#[cfg(not(target_env = "musl"))]
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
    PyAlgorithm, PyCWSModel, PyCWSTrainer, PyNERModel, PyNERTrainer, PyPOSModel, PyPOSTrainer,
};
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
use stnsplit::StnSplit;

/// Algorithms Module
#[pymodule]
fn algorithms(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<StnSplit>()?;
    m.add_class::<PyHook>()?;
    m.add_function(wrap_pyfunction!(py_eisner, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_entities, m)?)?;
    m.add_function(wrap_pyfunction!(py_viterbi_decode_postprocess, m)?)?;
    Ok(())
}

/// LTP Module
#[pymodule]
fn perceptron(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<ModelType>()?;
    m.add_class::<PyTrainer>()?;
    m.add_class::<PyAlgorithm>()?;

    m.add_class::<PyCWSModel>()?;
    m.add_class::<PyCWSTrainer>()?;

    m.add_class::<PyPOSModel>()?;
    m.add_class::<PyPOSTrainer>()?;

    m.add_class::<PyNERModel>()?;
    m.add_class::<PyNERTrainer>()?;
    Ok(())
}

/// LTP Module
#[pymodule]
fn ltp_extension(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_wrapped(wrap_pymodule!(algorithms))?;
    m.add_wrapped(wrap_pymodule!(perceptron))?;
    Ok(())
}
