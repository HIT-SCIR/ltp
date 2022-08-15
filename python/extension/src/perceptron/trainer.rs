use crate::perceptron::model::{EnumModel, ModelType, PyModel};
use crate::perceptron::PyAlgorithm;
use ltp::{CWSDefinition, NERDefinition, POSDefinition, Trainer};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EnumTrainer {
    CWS(Trainer<CWSDefinition>),
    POS(Trainer<POSDefinition>),
    NER(Trainer<NERDefinition>),
}

impl Display for EnumTrainer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EnumTrainer::CWS(ref trainer) => {
                write!(f, "CWSTrainer({})", trainer)
            }
            EnumTrainer::POS(trainer) => {
                write!(f, "POSTrainer({})", trainer)
            }
            EnumTrainer::NER(trainer) => {
                write!(f, "NERTrainer({})", trainer)
            }
        }
    }
}

#[pyclass(module = "ltp_extension.perceptron", name = "Trainer", subclass)]
#[pyo3(text_signature = "(self, model_type=ModelType.Auto, labels=None)")]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct PyTrainer {
    pub trainer: EnumTrainer,
}

#[pymethods]
impl PyTrainer {
    #[new]
    #[args(model_type = "ModelType::Auto", labels = "None")]
    pub fn new(model_type: ModelType, labels: Option<Vec<String>>) -> PyResult<Self> {
        let trainer = match (model_type, labels) {
            (ModelType::CWS, _) => EnumTrainer::CWS(Default::default()),
            (ModelType::POS, Some(labels)) => {
                EnumTrainer::POS(Trainer::new_with_define(POSDefinition::new(labels)))
            }
            (ModelType::NER, Some(labels)) => {
                EnumTrainer::NER(Trainer::new_with_define(NERDefinition::new(labels)))
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Not Supported Model Type",
                ));
            }
        };

        Ok(Self { trainer })
    }

    /// Get the value of the epoch parameter.
    #[getter]
    pub fn get_epoch(&self) -> PyResult<usize> {
        Ok(match &self.trainer {
            EnumTrainer::CWS(trainer) => trainer.epoch,
            EnumTrainer::POS(trainer) => trainer.epoch,
            EnumTrainer::NER(trainer) => trainer.epoch,
        })
    }

    /// Set the value of the epoch parameter.
    #[setter]
    pub fn set_epoch(&mut self, value: usize) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.epoch = value;
            }
            EnumTrainer::POS(trainer) => {
                trainer.epoch = value;
            }
            EnumTrainer::NER(trainer) => {
                trainer.epoch = value;
            }
        }
        Ok(())
    }

    /// Get the value of the epoch parameter.
    #[getter]
    pub fn get_shuffle(&self) -> PyResult<bool> {
        Ok(match &self.trainer {
            EnumTrainer::CWS(trainer) => trainer.shuffle,
            EnumTrainer::POS(trainer) => trainer.shuffle,
            EnumTrainer::NER(trainer) => trainer.shuffle,
        })
    }

    /// Set the value of the epoch parameter.
    #[setter]
    pub fn set_shuffle(&mut self, value: bool) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.shuffle = value;
            }
            EnumTrainer::POS(trainer) => {
                trainer.shuffle = value;
            }
            EnumTrainer::NER(trainer) => {
                trainer.shuffle = value;
            }
        }
        Ok(())
    }

    /// Get the value of the epoch parameter.
    #[getter]
    pub fn get_verbose(&self) -> PyResult<bool> {
        Ok(match &self.trainer {
            EnumTrainer::CWS(trainer) => trainer.verbose,
            EnumTrainer::POS(trainer) => trainer.verbose,
            EnumTrainer::NER(trainer) => trainer.verbose,
        })
    }

    /// Set the value of the epoch parameter.
    #[setter]
    pub fn set_verbose(&mut self, value: bool) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.verbose = value;
            }
            EnumTrainer::POS(trainer) => {
                trainer.verbose = value;
            }
            EnumTrainer::NER(trainer) => {
                trainer.verbose = value;
            }
        }
        Ok(())
    }

    /// Get the value of the epoch parameter.
    #[getter]
    pub fn get_algorithm(&self) -> PyResult<PyAlgorithm> {
        let algorithm = match &self.trainer {
            EnumTrainer::CWS(trainer) => trainer.algorithm.clone(),
            EnumTrainer::POS(trainer) => trainer.algorithm.clone(),
            EnumTrainer::NER(trainer) => trainer.algorithm.clone(),
        };
        Ok(PyAlgorithm { algorithm })
    }

    /// Set the value of the epoch parameter.
    #[setter]
    pub fn set_algorithm(&mut self, value: PyAlgorithm) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.algorithm = value.algorithm;
            }
            EnumTrainer::POS(trainer) => {
                trainer.algorithm = value.algorithm;
            }
            EnumTrainer::NER(trainer) => {
                trainer.algorithm = value.algorithm;
            }
        }
        Ok(())
    }

    /// Get the value of the epoch parameter.
    #[getter]
    pub fn get_eval_threads(&self) -> PyResult<usize> {
        Ok(match &self.trainer {
            EnumTrainer::CWS(trainer) => trainer.eval_threads,
            EnumTrainer::POS(trainer) => trainer.eval_threads,
            EnumTrainer::NER(trainer) => trainer.eval_threads,
        })
    }

    /// Set the value of the epoch parameter.
    #[setter]
    pub fn set_eval_threads(&mut self, value: usize) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.eval_threads = value;
            }
            EnumTrainer::POS(trainer) => {
                trainer.eval_threads = value;
            }
            EnumTrainer::NER(trainer) => {
                trainer.eval_threads = value;
            }
        }
        Ok(())
    }

    /// Get the value of the epoch parameter.
    #[getter]
    pub fn get_compress(&self) -> PyResult<bool> {
        Ok(match &self.trainer {
            EnumTrainer::CWS(trainer) => trainer.compress,
            EnumTrainer::POS(trainer) => trainer.compress,
            EnumTrainer::NER(trainer) => trainer.compress,
        })
    }

    /// Set the value of the epoch parameter.
    #[setter]
    pub fn set_compress(&mut self, value: bool) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.compress = value;
            }
            EnumTrainer::POS(trainer) => {
                trainer.compress = value;
            }
            EnumTrainer::NER(trainer) => {
                trainer.compress = value;
            }
        }
        Ok(())
    }

    /// Get the value of the epoch parameter.
    #[getter]
    pub fn get_ratio(&self) -> PyResult<f64> {
        Ok(match &self.trainer {
            EnumTrainer::CWS(trainer) => trainer.ratio,
            EnumTrainer::POS(trainer) => trainer.ratio,
            EnumTrainer::NER(trainer) => trainer.ratio,
        })
    }

    /// Set the value of the epoch parameter.
    #[setter]
    pub fn set_ratio(&mut self, value: f64) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.ratio = value;
            }
            EnumTrainer::POS(trainer) => {
                trainer.ratio = value;
            }
            EnumTrainer::NER(trainer) => {
                trainer.ratio = value;
            }
        }
        Ok(())
    }

    /// Get the value of the epoch parameter.
    #[getter]
    pub fn get_threshold(&self) -> PyResult<f64> {
        Ok(match &self.trainer {
            EnumTrainer::CWS(trainer) => trainer.threshold,
            EnumTrainer::POS(trainer) => trainer.threshold,
            EnumTrainer::NER(trainer) => trainer.threshold,
        })
    }

    /// Set the value of the epoch parameter.
    #[setter]
    pub fn set_threshold(&mut self, value: f64) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.threshold = value;
            }
            EnumTrainer::POS(trainer) => {
                trainer.threshold = value;
            }
            EnumTrainer::NER(trainer) => {
                trainer.threshold = value;
            }
        }
        Ok(())
    }

    /// Load Train Data from a path
    #[pyo3(text_signature = "(self, path)")]
    pub fn load_train_data(&mut self, data: &str) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.train_set = Some(trainer.load_dataset(data)?);
            }
            EnumTrainer::POS(trainer) => {
                trainer.train_set = Some(trainer.load_dataset(data)?);
            }
            EnumTrainer::NER(trainer) => {
                trainer.train_set = Some(trainer.load_dataset(data)?);
            }
        }
        Ok(())
    }

    /// Load Eval Data from a path
    #[pyo3(text_signature = "(self, path)")]
    pub fn load_eval_data(&mut self, data: &str) -> PyResult<()> {
        match &mut self.trainer {
            EnumTrainer::CWS(trainer) => {
                trainer.eval_set = Some(trainer.load_dataset(data)?);
            }
            EnumTrainer::POS(trainer) => {
                trainer.eval_set = Some(trainer.load_dataset(data)?);
            }
            EnumTrainer::NER(trainer) => {
                trainer.eval_set = Some(trainer.load_dataset(data)?);
            }
        }
        Ok(())
    }

    /// Train a model
    #[pyo3(text_signature = "(self)")]
    pub fn train(&self) -> PyResult<PyModel> {
        let model = match &self.trainer {
            EnumTrainer::CWS(trainer) => trainer.build().map(EnumModel::CWS)?,
            EnumTrainer::POS(trainer) => trainer.build().map(EnumModel::POS)?,
            EnumTrainer::NER(trainer) => trainer.build().map(EnumModel::NER)?,
        };
        Ok(PyModel { model })
    }

    /// Eval a Segmentor model
    #[pyo3(text_signature = "(self, model)")]
    pub fn eval(&self, model: &PyModel) -> PyResult<(f64, f64, f64)> {
        let res = match (&self.trainer, &model.model) {
            (EnumTrainer::CWS(trainer), EnumModel::CWS(model)) => trainer.evaluate(model)?,
            (EnumTrainer::POS(trainer), EnumModel::POS(model)) => trainer.evaluate(model)?,
            (EnumTrainer::NER(trainer), EnumModel::NER(model)) => trainer.evaluate(model)?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "The type of Trainer and Model not match!",
                ));
            }
        };
        Ok(res)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.trainer)
    }
}
