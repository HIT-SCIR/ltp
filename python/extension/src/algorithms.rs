use ltp::{drop_get_entities, eisner, viterbi_decode_postprocessing};
use pyo3::prelude::*;

/// Convert Tags to Entities
#[pyfunction]
#[pyo3(name = "get_entities", text_signature = "(tags)")]
pub fn py_get_entities(tags: Vec<&str>) -> PyResult<Vec<(&str, usize, usize)>> {
    Ok(drop_get_entities(tags))
}

/// Decode with Eisner's algorithm
#[pyfunction]
#[pyo3(
    name = "eisner",
    text_signature = "(scores, stn_length, remove_root=False)"
)]
pub fn py_eisner(
    scores: Vec<f32>,
    stn_length: Vec<usize>,
    remove_root: bool,
) -> PyResult<Vec<Vec<usize>>> {
    Ok(eisner(&scores, &stn_length, remove_root))
}

/// Viterbi Decode Postprocessing
#[pyfunction]
#[pyo3(
    name = "viterbi_decode_postprocess",
    text_signature = "(history, last_tags, stn_length, labels_num)"
)]
pub fn py_viterbi_decode_postprocess(
    history: Vec<i64>,
    last_tags: Vec<i64>,
    stn_lengths: Vec<usize>,
    labels_num: usize,
) -> PyResult<Vec<Vec<i64>>> {
    Ok(viterbi_decode_postprocessing(
        &history,
        &last_tags,
        &stn_lengths,
        labels_num,
    ))
}
