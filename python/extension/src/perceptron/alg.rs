use ltp::perceptron::{Algorithm, PaMode};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

/// The perceptron algorithm.
/// algorithm support "AP", "Pa", "PaI", "PaII"
/// AP: average perceptron, param is the threads
/// PA: parallel average perceptron, param is c(margin)
#[pyclass(module = "ltp_extension.perceptron", name = "Algorithm", subclass)]
#[pyo3(text_signature = "(self, algorithm, param = None)")]
#[derive(Clone, Serialize, Deserialize, Default, Debug, PartialEq)]
pub struct PyAlgorithm {
    pub(crate) algorithm: Algorithm<f64>,
}

impl Display for PyAlgorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.algorithm)
    }
}

#[pymethods]
impl PyAlgorithm {
    #[new]
    pub fn new(py: Python, algorithm: &str, param: Option<PyObject>) -> PyResult<Self> {
        let algorithm: Algorithm<f64> = match algorithm {
            "AP" => {
                if let Some(param) = param {
                    let param = param.extract::<usize>(py)?;
                    Ok(Algorithm::AP(param))
                } else {
                    Ok(Algorithm::AP(1usize))
                }
            }
            "Pa" => Ok(Algorithm::PA(PaMode::Pa)),
            "PaI" => {
                if let Some(c) = param {
                    let c = c.extract::<f64>(py)?;
                    Ok(Algorithm::PA(PaMode::PaI(c)))
                } else {
                    Err(PyValueError::new_err("param is needed"))
                }
            }
            "PaII" => {
                if let Some(c) = param {
                    let c = c.extract::<f64>(py)?;
                    Ok(Algorithm::PA(PaMode::PaII(c)))
                } else {
                    Err(PyValueError::new_err("param is needed"))
                }
            }
            _ => Err(PyValueError::new_err("algorithm is not supported"))?,
        }?;

        Ok(Self { algorithm })
    }

    fn __repr__(&self) -> String {
        format!("{}", self.algorithm)
    }
}
