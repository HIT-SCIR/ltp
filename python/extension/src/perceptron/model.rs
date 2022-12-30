use crate::perceptron::Perceptron;
use crate::utils::parallelism::MaybeParallelIterator;
use ltp::{CWSDefinition, ModelSerde, NERDefinition, POSDefinition};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[pyclass(module = "ltp_extension.perceptron", name = "ModelType")]
#[pyo3(text_signature = "(self, model_type=None)")]
pub enum ModelType {
    Auto,
    CWS,
    POS,
    NER,
}

#[pymethods]
impl ModelType {
    #[new]
    #[args(model_type = "None")]
    pub fn new(model_type: Option<&str>) -> PyResult<Self> {
        Ok(match model_type {
            Some("cws") => ModelType::CWS,
            Some("pos") => ModelType::POS,
            Some("ner") => ModelType::NER,
            None => ModelType::Auto,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Not Supported Model Type",
                ));
            }
        })
    }
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::Auto
    }
}

pub type CWSModel = Perceptron<CWSDefinition>;
pub type POSModel = Perceptron<POSDefinition>;
pub type NERModel = Perceptron<NERDefinition>;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EnumModel {
    CWS(CWSModel),
    POS(POSModel),
    NER(NERModel),
}

impl Display for EnumModel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EnumModel::CWS(ref model) => {
                write!(f, "CWSModel({})", model)
            }
            EnumModel::POS(model) => {
                write!(f, "POSModel({})", model)
            }
            EnumModel::NER(model) => {
                write!(f, "NERModel({})", model)
            }
        }
    }
}

#[pyclass(module = "ltp_extension.perceptron", name = "Model", subclass)]
#[pyo3(text_signature = "(self, path, model_type=ModelType.Auto)")]
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct PyModel {
    pub model: EnumModel,
}

#[pymethods]
impl PyModel {
    #[new]
    #[args(model_type = "ModelType::Auto")]
    pub fn new(path: &str, model_type: ModelType) -> PyResult<Self> {
        Self::load(path, model_type)
    }

    /// Load Model from a path
    #[staticmethod]
    #[pyo3(text_signature = "(path, model_type=ModelType.Auto)")]
    #[args(model_type = "ModelType::Auto")]
    pub fn load(path: &str, model_type: ModelType) -> PyResult<Self> {
        let file = std::fs::File::open(path)?;
        let format = if path.ends_with(".json") {
            ltp::perceptron::Format::JSON
        } else {
            ltp::perceptron::Format::AVRO(ltp::perceptron::Codec::Deflate)
        };

        let model = match (model_type, format) {
            (ModelType::CWS, format) => ModelSerde::load(file, format).map(EnumModel::CWS)?,
            (ModelType::POS, format) => ModelSerde::load(file, format).map(EnumModel::POS)?,
            (ModelType::NER, format) => ModelSerde::load(file, format).map(EnumModel::NER)?,
            (ModelType::Auto, ltp::perceptron::Format::JSON) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Json Format Not Supported AutoDetect",
                ));
            }
            (ModelType::Auto, _) => {
                use ltp::perceptron::Schema;
                let reader = ltp::perceptron::Reader::new(file).map_err(anyhow::Error::from)?;
                match reader.writer_schema() {
                    Schema::Record { name, .. } => match name.name.as_str() {
                        "cws" => ModelSerde::load_avro(reader).map(EnumModel::CWS)?,
                        "pos" => ModelSerde::load_avro(reader).map(EnumModel::POS)?,
                        "ner" => ModelSerde::load_avro(reader).map(EnumModel::NER)?,
                        _ => {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "Not Supported Model Type",
                            ));
                        }
                    },
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Not Supported Model Type",
                        ));
                    }
                }
            }
        };

        Ok(Self { model })
    }

    /// Specialize the Model
    #[pyo3(text_signature = "(self)")]
    pub fn specialize(&self, py: Python) -> PyResult<PyObject> {
        match &self.model {
            EnumModel::CWS(model) => Ok(crate::perceptron::specialization::PyCWSModel {
                model: model.clone(),
            }
            .into_py(py)),
            EnumModel::POS(model) => Ok(crate::perceptron::specialization::PyPOSModel {
                model: model.clone(),
            }
            .into_py(py)),
            EnumModel::NER(model) => Ok(crate::perceptron::specialization::PyNERModel {
                model: model.clone(),
            }
            .into_py(py)),
        }
    }

    /// Save Model to a path
    #[pyo3(text_signature = "(self, path)")]
    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = std::fs::File::create(path)?;
        let format = if path.ends_with(".json") {
            ltp::perceptron::Format::JSON
        } else {
            ltp::perceptron::Format::AVRO(ltp::perceptron::Codec::Deflate)
        };
        match &self.model {
            EnumModel::CWS(model) => ModelSerde::save(model, file, format)?,
            EnumModel::POS(model) => ModelSerde::save(model, file, format)?,
            EnumModel::NER(model) => ModelSerde::save(model, file, format)?,
        }
        Ok(())
    }

    #[args(args = "*", parallelism = true)]
    pub fn __call__(&self, py: Python, args: &PyTuple, parallelism: bool) -> PyResult<PyObject> {
        let first = args.get_item(0)?;
        let is_single = match &self.model {
            EnumModel::CWS(_) => match first.get_type().name()? {
                "str" => true,
                "list" => false,
                name => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "type \"{}\" has not been supported",
                        name
                    )));
                }
            },
            EnumModel::POS(_) | EnumModel::NER(_) => match first.get_type().name()? {
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
            },
        };

        match is_single {
            true => self.predict(py, args),
            false => self.batch_predict(py, args, parallelism),
        }
    }

    /// Predict a sentence
    #[pyo3(text_signature = "(self, *args)")]
    #[args(args = "*")]
    pub fn predict(&self, py: Python, args: &PyTuple) -> PyResult<PyObject> {
        Ok(match &self.model {
            EnumModel::CWS(model) => {
                let text = args.get_item(0)?.extract()?;
                PyList::new(
                    py,
                    model
                        .predict(text)?
                        .into_iter()
                        .map(|s| PyString::new(py, s)),
                )
                .into()
            }
            EnumModel::POS(model) => {
                let words: Vec<&str> = args.get_item(0)?.extract()?;
                PyList::new(
                    py,
                    model
                        .predict(&words)?
                        .into_iter()
                        .map(|s| PyString::new(py, s)),
                )
                .into()
            }
            EnumModel::NER(model) => {
                let words: Vec<&str> = args.get_item(0)?.extract()?;
                let tags: Vec<&str> = args.get_item(1)?.extract()?;
                PyList::new(
                    py,
                    model
                        .predict((&words, &tags))?
                        .into_iter()
                        .map(|s| PyString::new(py, s)),
                )
                .into()
            }
        })
    }

    /// Predict batched sentences
    #[pyo3(text_signature = "(self, *args, parallelism = True)")]
    #[args(args = "*", parallelism = true)]
    pub fn batch_predict(
        &self,
        py: Python,
        args: &PyTuple,
        parallelism: bool,
    ) -> PyResult<PyObject> {
        let result = match &self.model {
            EnumModel::CWS(model) => {
                let batch_text: Vec<_> = args.get_item(0)?.extract()?;
                let result: Result<Vec<Vec<_>>, _> = batch_text
                    .into_maybe_par_iter_cond(parallelism)
                    .map(|text| model.predict(text))
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
                res
            }
            EnumModel::POS(model) => {
                let batch_words: Vec<Vec<&str>> = args.get_item(0)?.extract()?;
                let result: Result<Vec<Vec<_>>, _> = batch_words
                    .into_maybe_par_iter_cond(parallelism)
                    .map(|words| model.predict(&words))
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
                res
            }
            EnumModel::NER(model) => {
                let batch_words: Vec<Vec<&str>> = args.get_item(0)?.extract()?;
                let batch_pos: Vec<Vec<&str>> = args.get_item(1)?.extract()?;
                let result: Result<Vec<Vec<_>>, _> = batch_words
                    .into_maybe_par_iter_cond(parallelism)
                    .zip(batch_pos)
                    .map(|(words, tags)| model.predict((&words, &tags)))
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
                res
            }
        };

        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        format!("{}", self.model)
    }
}
