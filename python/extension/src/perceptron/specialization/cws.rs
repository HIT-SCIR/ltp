use crate::impl_model;
use crate::perceptron::{Perceptron, PyAlgorithm};
use crate::utils::parallelism::MaybeParallelIterator;
use ltp::perceptron::{CWSDefinition as Definition, Trainer};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use serde::{Deserialize, Serialize};

pub type Model = Perceptron<Definition>;

#[pyclass(module = "ltp_extension.perceptron", name = "CWSModel", subclass)]
#[pyo3(text_signature = "(self, path)")]
#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct PyCWSModel {
    pub model: Model,
}

impl_model!(PyCWSModel);

/// Digit: Digit character. (e.g. 0, 1, 2, ...)
/// Roman: Roman character. (e.g. A, B, C, ...)
/// Hiragana: Japanese Hiragana character. (e.g. あ, い, う, ...)
/// Katakana: Japanese Katakana character. (e.g. ア, イ, ウ, ...)
/// Kanji: Kanji (a.k.a. Hanzi or Hanja) character. (e.g. 漢, 字, ...)
/// Other: Other character.
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[pyclass(module = "ltp_extension.perceptron", name = "CharacterType")]
pub enum CharacterType {
    /// Digit character. (e.g. 0, 1, 2, ...)
    Digit = 1,

    /// Roman character. (e.g. A, B, C, ...)
    Roman = 2,

    /// Japanese Hiragana character. (e.g. あ, い, う, ...)
    Hiragana = 3,

    /// Japanese Katakana character. (e.g. ア, イ, ウ, ...)
    Katakana = 4,

    /// Kanji (a.k.a. Hanzi or Hanja) character. (e.g. 漢, 字, ...)
    Kanji = 5,

    /// Other character.
    Other = 6,
}

#[pymethods]
impl PyCWSModel {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        Ok(Self::inner_load(path)?)
    }

    /// 自定义新feature
    #[pyo3(text_signature = "(self, core, feature, s, b, m, e)")]
    pub fn add_feature_rule(&mut self, core: &str, s: f64, b: f64, m: f64, e: f64) -> PyResult<()> {
        self.model.add_core_rule(core, s, b, m, e);
        Ok(())
    }

    /// 启用自定义新 feature
    #[pyo3(text_signature = "(self, core, feature)")]
    pub fn enable_feature_rule(&mut self, core: &str, feature: &str) -> PyResult<()> {
        self.model.enable_feature_rule(core, feature);
        Ok(())
    }

    /// 移除自定义新 feature
    #[pyo3(text_signature = "(self, core, feature, s, b, m, e)")]
    pub fn disable_feature_rule(&mut self, feature: &str) -> PyResult<()> {
        self.model.disable_feature_rule(feature);
        Ok(())
    }

    /// 开启连续不同类型之间的强制切分
    #[pyo3(text_signature = "(self, a, b)")]
    pub fn enable_type_cut(&mut self, a: CharacterType, b: CharacterType) -> PyResult<()> {
        self.add_feature_rule("[FORCE_CUT]", 500.0, 500.0, -500.0, -500.0)?;
        self.enable_feature_rule("[FORCE_CUT]", &format!("d{}{}", a as u8, b as u8))?;
        Ok(())
    }

    /// 开启连续不同类型之间的强制切分(双向)
    #[pyo3(text_signature = "(self, a, b)")]
    pub fn enable_type_cut_d(&mut self, a: CharacterType, b: CharacterType) -> PyResult<()> {
        self.add_feature_rule("[FORCE_CUT]", 500.0, 500.0, -500.0, -500.0)?;
        self.enable_feature_rule("[FORCE_CUT]", &format!("d{}{}", a as u8, b as u8))?;
        self.enable_feature_rule("[FORCE_CUT]", &format!("d{}{}", b as u8, a as u8))?;
        Ok(())
    }

    /// 开启连续不同类型之间的强制连接
    #[pyo3(text_signature = "(self, a, b)")]
    pub fn enable_type_concat(&mut self, a: CharacterType, b: CharacterType) -> PyResult<()> {
        self.add_feature_rule("[FORCE_CONCAT]", -500.0, -500.0, 500.0, 500.0)?;
        self.enable_feature_rule("[FORCE_CONCAT]", &format!("d{}{}", a as u8, b as u8))?;
        Ok(())
    }

    /// 开启连续不同类型之间的强制连接(双向)
    #[pyo3(text_signature = "(self, a, b)")]
    pub fn enable_type_concat_d(&mut self, a: CharacterType, b: CharacterType) -> PyResult<()> {
        self.add_feature_rule("[FORCE_CONCAT]", -500.0, -500.0, 500.0, 500.0)?;
        self.enable_feature_rule("[FORCE_CONCAT]", &format!("d{}{}", a as u8, b as u8))?;
        self.enable_feature_rule("[FORCE_CONCAT]", &format!("d{}{}", b as u8, a as u8))?;
        Ok(())
    }

    /// 关闭连续不同类型之间的强制连接/切分
    #[pyo3(text_signature = "(self, a, b)")]
    pub fn disable_type_rule(&mut self, a: CharacterType, b: CharacterType) -> PyResult<()> {
        self.disable_feature_rule(&format!("d{}{}", a as u8, b as u8))?;
        Ok(())
    }

    /// 关闭连续不同类型之间的强制连接/切分(双向)
    #[pyo3(text_signature = "(self, a, b)")]
    pub fn disable_type_rule_d(&mut self, a: CharacterType, b: CharacterType) -> PyResult<()> {
        self.disable_feature_rule(&format!("d{}{}", a as u8, b as u8))?;
        self.disable_feature_rule(&format!("d{}{}", b as u8, a as u8))?;
        Ok(())
    }

    #[args(args = "*", parallelism = true)]
    pub fn __call__(&self, py: Python, args: &PyTuple, parallelism: bool) -> PyResult<PyObject> {
        let first = args.get_item(0)?;
        let is_single = match first.get_type().name()? {
            "str" => true,
            "list" => false,
            name => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "type \"{}\" has not been supported",
                    name
                )));
            }
        };

        match is_single {
            true => self.predict(py, args.get_item(0)?.extract()?),
            false => self.batch_predict(py, args.get_item(0)?.extract()?, parallelism),
        }
    }

    /// Predict a sentence
    #[pyo3(text_signature = "(self, text)")]
    pub fn predict(&self, py: Python, text: &str) -> PyResult<PyObject> {
        Ok(PyList::new(
            py,
            self.model
                .predict(text)?
                .into_iter()
                .map(|s| PyString::new(py, s)),
        )
        .into())
    }

    /// Predict batched sentences
    #[args(parallelism = true)]
    #[pyo3(text_signature = "(self, batch_text, parallelism=True)")]
    pub fn batch_predict(
        &self,
        py: Python,
        batch_text: Vec<&str>,
        parallelism: bool,
    ) -> PyResult<PyObject> {
        let result: Result<Vec<Vec<_>>, _> = batch_text
            .into_maybe_par_iter_cond(parallelism)
            .map(|text| self.model.predict(text))
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

#[pyclass(module = "ltp_extension.perceptron", name = "CWSTrainer", subclass)]
#[pyo3(text_signature = "(self)")]
#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct PyCWSTrainer {
    pub trainer: Trainer<Definition>,
}

#[pymethods]
impl PyCWSTrainer {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self {
            trainer: Trainer::new(),
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
    pub fn train(&self) -> PyResult<PyCWSModel> {
        let model = PyCWSModel {
            model: self.trainer.build()?,
        };

        Ok(model)
    }

    /// Eval a Segmentor model
    #[pyo3(text_signature = "(self, model)")]
    pub fn eval(&self, model: &PyCWSModel) -> PyResult<()> {
        self.trainer.evaluate(&model.model)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("{}", self.trainer)
    }
}
