use ltp::{stn_split_with_options, SplitOptions};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(module = "ltp_extension.algorithms", name = "StnSplit", subclass)]
#[pyo3(text_signature = "(self)")]
#[derive(Clone, Serialize, Deserialize, Default, Debug, PartialEq, Eq)]
pub struct StnSplit {
    pub options: SplitOptions,
}

#[pymethods]
impl StnSplit {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self {
            options: SplitOptions {
                use_zh: true,
                use_en: true,
                bracket_as_entity: true,
                zh_quote_as_entity: true,
                en_quote_as_entity: true,
            },
        })
    }

    /// split to sentences
    #[pyo3(text_signature = "(self, text)")]
    pub fn split(&self, py: Python, text: &str) -> PyResult<PyObject> {
        let res = PyList::new(
            py,
            stn_split_with_options(text, &self.options)
                .into_iter()
                .map(|s| PyString::new(py, s)),
        );
        Ok(res.into())
    }

    /// batch split to sentences
    #[args(threads = "8")]
    #[pyo3(text_signature = "(self, batch_text, threads=8)")]
    pub fn batch_split(
        &self,
        py: Python,
        batch_text: Vec<&str>,
        threads: usize,
    ) -> PyResult<PyObject> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let result = pool.install(|| {
            batch_text
                .into_par_iter()
                .map(|text| {
                    stn_split_with_options(text, &self.options)
                        .into_iter()
                        .collect()
                })
                .reduce(Vec::new, |mut acc, v| {
                    acc.extend(v);
                    acc
                })
        });

        let result = PyList::new(py, result.into_iter().map(|s| PyString::new(py, s)));

        Ok(result.into())
    }

    /// Get the value of the use_zh option.
    #[getter]
    pub fn get_use_zh(&self) -> PyResult<bool> {
        Ok(self.options.use_zh)
    }

    /// Set the value of the use_zh option.
    #[setter]
    pub fn set_use_zh(&mut self, value: bool) -> PyResult<()> {
        self.options.use_zh = value;
        Ok(())
    }

    /// Get the value of the use_en option.
    #[getter]
    pub fn get_use_en(&self) -> PyResult<bool> {
        Ok(self.options.use_en)
    }

    /// Set the value of the use_en option.
    #[setter]
    pub fn set_use_en(&mut self, value: bool) -> PyResult<()> {
        self.options.use_en = value;
        Ok(())
    }

    /// Get the value of the bracket_as_entity option.
    #[getter]
    pub fn get_bracket_as_entity(&self) -> PyResult<bool> {
        Ok(self.options.bracket_as_entity)
    }

    /// Set the value of the bracket_as_entity option.
    #[setter]
    pub fn set_bracket_as_entity(&mut self, value: bool) -> PyResult<()> {
        self.options.bracket_as_entity = value;
        Ok(())
    }

    /// Get the value of the zh_quote_as_entity option.
    #[getter]
    pub fn get_zh_quote_as_entity(&self) -> PyResult<bool> {
        Ok(self.options.zh_quote_as_entity)
    }

    /// Set the value of the zh_quote_as_entity option.
    #[setter]
    pub fn set_zh_quote_as_entity(&mut self, value: bool) -> PyResult<()> {
        self.options.zh_quote_as_entity = value;
        Ok(())
    }

    /// Get the value of the en_quote_as_entity option.
    #[getter]
    pub fn get_en_quote_as_entity(&self) -> PyResult<bool> {
        Ok(self.options.en_quote_as_entity)
    }

    /// Set the value of the en_quote_as_entity option.
    #[setter]
    pub fn set_en_quote_as_entity(&mut self, value: bool) -> PyResult<()> {
        self.options.en_quote_as_entity = value;
        Ok(())
    }
}
