use ltp::hook::Hook;
use pyo3::prelude::*;

#[pyclass(module = "ltp_extension.algorithms", name = "Hook", subclass)]
#[pyo3(text_signature = "(self)")]
#[derive(Clone, Debug)]
pub struct PyHook {
    pub hook: Hook,
}

#[pymethods]
impl PyHook {
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self { hook: Hook::new() })
    }

    pub fn __len__(&self) -> usize {
        self.hook.total()
    }

    /// add words to the hook, the freq can be zero
    #[args(freq = "None")]
    #[pyo3(text_signature = "(self, word, freq = None)")]
    pub fn add_word(&mut self, word: &str, freq: Option<usize>) -> usize {
        self.hook.add_word(word, freq)
    }

    /// hook to the new words
    #[pyo3(text_signature = "(self, sentence, words)")]
    pub fn hook<'a>(&self, sentence: &'a str, words: Vec<&str>) -> PyResult<Vec<&'a str>> {
        Ok(self.hook.hook(sentence, &words))
    }
}
