#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod algorithms;
mod hook;
mod perceptron;
mod stnsplit;
mod utils;

use crate::perceptron::{ModelType, PyModel, PyTrainer};
pub use algorithms::{py_eisner, py_get_entities, py_viterbi_decode_postprocess};
use hook::PyHook;
pub use perceptron::{
    CharacterType, PyAlgorithm, PyCWSModel, PyCWSTrainer, PyNERModel, PyNERTrainer, PyPOSModel,
    PyPOSTrainer,
};
use pyo3::prelude::*;
use stnsplit::StnSplit;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// For users using multiprocessing in python, it is quite easy to fork the process running
// tokenizers, ending up with a deadlock because we internaly make use of multithreading. So
// we register a callback to be called in the event of a fork so that we can warn the user.
static mut REGISTERED_FORK_CALLBACK: bool = false;
extern "C" fn child_after_fork() {
    use utils::parallelism::*;
    if has_parallelism_been_used() && !is_parallelism_configured() {
        println!(
            "LTP: The current process just got forked, after parallelism has \
            already been used. Disabling parallelism to avoid deadlocks..."
        );
        println!("To disable this warning, you can either:");
        println!(
            "\t- Avoid using `LTP/legacy` model before the fork if possible\n\
            \t- Explicitly set the environment variable {}=(true | false)",
            ENV_VARIABLE
        );
        set_parallelism(false);
    }
}

/// LTP Module
#[pymodule]
fn ltp_extension(py: Python, m: &PyModule) -> PyResult<()> {
    // Register the fork callback
    #[cfg(target_family = "unix")]
    unsafe {
        if !REGISTERED_FORK_CALLBACK {
            libc::pthread_atfork(None, None, Some(child_after_fork));
            REGISTERED_FORK_CALLBACK = true;
        }
    }

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
