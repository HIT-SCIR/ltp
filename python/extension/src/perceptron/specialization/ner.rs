use crate::impl_model;
use crate::perceptron::{Perceptron, PyAlgorithm};
use crate::utils::parallelism::MaybeParallelIterator;
use ltp::perceptron::{NERDefinition as Definition, Trainer};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use serde::{Deserialize, Serialize};

pub type Model = Perceptron<Definition>;

#[pyclass(module = "ltp_extension.perceptron", name = "NERModel", subclass)]
#[pyo3(text_signature = "(self, path)")]
#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct PyNERModel {
    pub model: Model,
}

impl_model!(PyNERModel);

#[pymethods]
impl PyNERModel {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        Ok(Self::inner_load(path)?)
    }

    #[args(args = "*", parallelism = true)]
    pub fn __call__(&self, py: Python, args: &PyTuple, parallelism: bool) -> PyResult<PyObject> {
        let first = args.get_item(0)?;
        let is_single = match first.get_type().name()? {
            "list" => match first.get_item(0)?.get_type().name()? {
                "str" => true,
                "list" => false,
                name => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "type list(\"{}\") has not been supported",
                        name
                    )));
                }
            },
            name => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "type \"{}\" has not been supported",
                    name
                )));
            }
        };

        match is_single {
            true => self.predict(
                py,
                args.get_item(0)?.extract()?,
                args.get_item(1)?.extract()?,
            ),
            false => self.batch_predict(
                py,
                args.get_item(0)?.extract()?,
                args.get_item(1)?.extract()?,
                parallelism,
            ),
        }
    }

    /// Predict a sentence
    #[pyo3(text_signature = "(self, words, pos)")]
    pub fn predict(&self, py: Python, words: Vec<&str>, pos: Vec<&str>) -> PyResult<PyObject> {
        Ok(PyList::new(
            py,
            self.model
                .predict((&words, &pos))?
                .into_iter()
                .map(|s| PyString::new(py, s)),
        )
        .into())
    }

    /// Predict batched sentences
    #[args(parallelism = true)]
    #[pyo3(text_signature = "(self, batch_words, batch_pos , parallelism=True)")]
    pub fn batch_predict(
        &self,
        py: Python,
        batch_words: Vec<Vec<&str>>,
        batch_pos: Vec<Vec<&str>>,
        parallelism: bool,
    ) -> PyResult<PyObject> {
        let result: Result<Vec<Vec<_>>, _> = batch_words
            .into_maybe_par_iter_cond(parallelism)
            .zip(batch_pos)
            .map(|(words, pos)| self.model.predict((&words, &pos)))
            .collect();
        let result = result?;
        let res = PyList::new(py, Vec::<&PyList>::with_capacity(result.len()));
        for snt in result {
            let snt_res = PyList::new(py, Vec::<&PyString>::with_capacity(snt.len()));
            for tag in snt {
                snt_res.append(PyString::new(py, tag))?;
            }
            res.append(snt_res)?;
        }
        Ok(res.into())
    }

    /// Load Model from a path
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    pub fn load(path: &str) -> PyResult<Self> {
        Ok(Self::inner_load(path)?)
    }

    /// Save Model to a path
    #[pyo3(text_signature = "(self, path)")]
    pub fn save(&self, path: &str) -> PyResult<()> {
        Ok(Self::inner_save(self, path)?)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.model)
    }
}

#[pyclass(module = "ltp_extension.perceptron", name = "NERTrainer", subclass)]
#[pyo3(text_signature = "(self, labels)")]
#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct PyNERTrainer {
    pub trainer: Trainer<Definition>,
}

#[pymethods]
impl PyNERTrainer {
    #[new]
    pub fn new(labels: Vec<String>) -> PyResult<Self> {
        Ok(Self {
            trainer: Trainer::new_with_define(Definition::new(labels)),
        })
    }

    /// Get the value of the epoch parameter.
    #[getter]
    pub fn get_epoch(&self) -> PyResult<usize> {
        Ok(self.trainer.epoch)
    }

    /// Set the value of the epoch parameter.
    #[setter]
    pub fn set_epoch(&mut self, value: usize) -> PyResult<()> {
        self.trainer.epoch = value;
        Ok(())
    }

    /// Get the value of the shuffle parameter.
    #[getter]
    pub fn get_shuffle(&self) -> PyResult<bool> {
        Ok(self.trainer.shuffle)
    }

    /// Set the value of the shuffle parameter.
    #[setter]
    pub fn set_shuffle(&mut self, value: bool) -> PyResult<()> {
        self.trainer.shuffle = value;
        Ok(())
    }

    /// Get the value of the verbose parameter.
    #[getter]
    pub fn get_verbose(&self) -> PyResult<bool> {
        Ok(self.trainer.verbose)
    }

    /// Set the value of the verbose parameter.
    #[setter]
    pub fn set_verbose(&mut self, value: bool) -> PyResult<()> {
        self.trainer.verbose = value;
        Ok(())
    }

    /// Get the value of the algorithm parameter.
    #[getter]
    pub fn get_algorithm(&self) -> PyResult<PyAlgorithm> {
        Ok(PyAlgorithm {
            algorithm: self.trainer.algorithm.clone(),
        })
    }

    /// Set the value of the algorithm parameter.
    #[setter]
    pub fn set_algorithm(&mut self, value: PyAlgorithm) -> PyResult<()> {
        self.trainer.algorithm = value.algorithm;
        Ok(())
    }

    /// Get the value of the eval_threads parameter.
    #[getter]
    pub fn get_eval_threads(&self) -> PyResult<usize> {
        Ok(self.trainer.eval_threads)
    }

    /// Set the value of the eval_threads parameter.
    #[setter]
    pub fn set_eval_threads(&mut self, value: usize) -> PyResult<()> {
        self.trainer.eval_threads = value;
        Ok(())
    }

    /// Get the value of the compress parameter.
    #[getter]
    pub fn get_compress(&self) -> PyResult<bool> {
        Ok(self.trainer.compress)
    }

    /// Set the value of the compress parameter.
    #[setter]
    pub fn set_compress(&mut self, value: bool) -> PyResult<()> {
        self.trainer.compress = value;
        Ok(())
    }

    /// Get the value of the ratio parameter.
    #[getter]
    pub fn get_ratio(&self) -> PyResult<f64> {
        Ok(self.trainer.ratio)
    }

    /// Set the value of the ratio parameter.
    #[setter]
    pub fn set_ratio(&mut self, value: f64) -> PyResult<()> {
        self.trainer.ratio = value;
        Ok(())
    }

    /// Get the value of the threshold parameter.
    #[getter]
    pub fn get_threshold(&self) -> PyResult<f64> {
        Ok(self.trainer.threshold)
    }

    /// Set the value of the threshold parameter.
    #[setter]
    pub fn set_threshold(&mut self, value: f64) -> PyResult<()> {
        self.trainer.threshold = value;
        Ok(())
    }

    /// Load Train Data from a path
    #[pyo3(text_signature = "(self, path)")]
    pub fn load_train_data(&mut self, data: &str) -> PyResult<()> {
        self.trainer.train_set = Some(self.trainer.load_dataset(data)?);
        Ok(())
    }

    /// Load Eval Data from a path
    #[pyo3(text_signature = "(self, path)")]
    pub fn load_eval_data(&mut self, data: &str) -> PyResult<()> {
        self.trainer.eval_set = Some(self.trainer.load_dataset(data)?);
        Ok(())
    }

    /// Train a Segmentor model
    #[pyo3(text_signature = "(self)")]
    pub fn train(&self) -> PyResult<PyNERModel> {
        let model = PyNERModel {
            model: self.trainer.build()?,
        };

        Ok(model)
    }

    /// Eval a Segmentor model
    #[pyo3(text_signature = "(self, model)")]
    pub fn eval(&self, model: &PyNERModel) -> PyResult<()> {
        self.trainer.evaluate(&model.model)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("{}", self.trainer)
    }
}
