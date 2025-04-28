use crate::{BatchCallback, Callback};
use ltp::{CWSDefinition, NERDefinition, POSDefinition};
use rayon::prelude::*;
use std::{slice, str};
use ltp::utils::hook::Hook;
use ltp::utils::hook::Dag;

pub type Perceptron<T> = ltp::perceptron::SerdeModel<T, f64>;

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub enum EnumModel {
    CWS(Perceptron<CWSDefinition>),
    POS(Perceptron<POSDefinition>),
    NER(Perceptron<NERDefinition>),
}

pub struct Model {
    model: EnumModel,
}

fn rs_model_load(model_path: &str) -> *mut Model {
    use ltp::perceptron::{ModelSerde, Reader, Schema};
    let file = match std::fs::File::open(model_path) {
        Ok(file) => file,
        Err(err) => {
            println!("{}", err);
            return std::ptr::null_mut();
        }
    };

    let reader = match Reader::new(file) {
        Ok(reader) => reader,
        Err(_) => {
            println!("Not Correct Bin Format!");
            return std::ptr::null_mut();
        }
    };

    let model: Result<Box<Model>, _> = {
        match reader.writer_schema() {
            Schema::Record(record) => match record.name.name.as_str() {
                "cws" => ModelSerde::load_avro(reader)
                    .map(EnumModel::CWS)
                    .map(|model| Box::new(Model { model })),
                "pos" => ModelSerde::load_avro(reader)
                    .map(EnumModel::POS)
                    .map(|model| Box::new(Model { model })),
                "ner" => ModelSerde::load_avro(reader)
                    .map(EnumModel::NER)
                    .map(|model| Box::new(Model { model })),
                _ => {
                    return std::ptr::null_mut();
                }
            },
            _ => {
                return std::ptr::null_mut();
            }
        }
    };

    match model {
        Ok(model) => Box::into_raw(model),
        Err(err) => {
            println!("{}", err);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_load(model_path: *const u8) -> *mut Model {
    let model_path = unsafe { std::ffi::CStr::from_ptr(model_path as *const _) };
    rs_model_load(&model_path.to_string_lossy())
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_load_s(model_path: *const u8, model_path_len: usize) -> *mut Model {
    let model_path =
        unsafe { str::from_utf8_unchecked(slice::from_raw_parts(model_path, model_path_len)) };
    rs_model_load(model_path)
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_release(model: *mut *mut Model) {
    let _ = unsafe { Box::from_raw(*model) };
    unsafe { *model = std::ptr::null_mut() };
}

pub extern "C" fn rs_model_save(
    model: *const Model,
    model_path: &str,
) -> bool {
    use ltp::perceptron::ModelSerde;

    let file = match std::fs::File::open(model_path) {
        Ok(file) => file,
        Err(err) => {
            println!("{}", err);
            return false;
        }
    };

    let model_format = if model_path.ends_with(".json") {
        ltp::perceptron::Format::JSON
    } else {
        ltp::perceptron::Format::AVRO(ltp::perceptron::Codec::Deflate)
    };

    match {
        match unsafe { &(*model).model } {
            EnumModel::CWS(ref model) => ModelSerde::save(model, file, model_format),
            EnumModel::POS(ref model) => ModelSerde::save(model, file, model_format),
            EnumModel::NER(ref model) => ModelSerde::save(model, file, model_format),
        }
    } {
        Ok(_) => true,
        Err(err) => {
            println!("{}", err);
            false
        }
    }
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_save(
    model: *const Model,
    model_path: *const u8,
) -> bool {
    let model_path = unsafe { std::ffi::CStr::from_ptr(model_path as *const _) };
    rs_model_save(model, &model_path.to_string_lossy())
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_save_s(
    model: *const Model,
    model_path: *const u8,
    model_path_len: usize,
) -> bool {
    let model_path =
        unsafe { str::from_utf8_unchecked(slice::from_raw_parts(model_path, model_path_len)) };
    rs_model_save(model, model_path)
}


#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_cws_predict(
    model: *const Model,
    sentence: *const u8,
    sentence_len: usize,
    callback: Callback,
) -> usize {
    let sentence =
        unsafe { str::from_utf8_unchecked(slice::from_raw_parts(sentence, sentence_len)) };

    if let EnumModel::CWS(ref model) = unsafe { &(*model).model } {
        if let Ok(results) = model.predict(sentence) {
            for (idx, result) in results.iter().enumerate() {
                (callback.call)(
                    callback.state,
                    result.as_ptr(),
                    result.len(),
                    idx,
                    results.len(),
                );
            }
            results.len()
        } else {
            0
        }
    } else {
        0
    }
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_cws_batch_predict(
    model: *const Model,
    sentences: *const *const u8,
    sentences_len: *const usize,
    sentences_len_len: usize,
    callback: BatchCallback,
    threads: usize,
) -> usize {
    let sentences_len = unsafe { slice::from_raw_parts(sentences_len, sentences_len_len) };
    let sentences = unsafe { slice::from_raw_parts(sentences, sentences_len_len) };
    let sentences = sentences
        .iter()
        .zip(sentences_len)
        .map(|(&sentence, &len)| unsafe {
            str::from_utf8_unchecked(slice::from_raw_parts(sentence, len))
        })
        .collect::<Vec<_>>();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    if let EnumModel::CWS(ref model) = unsafe { &(*model).model } {
        let batch_words: Result<Vec<Vec<_>>, _> = pool.install(|| {
            sentences
                .into_par_iter()
                .map(|text| model.predict(text))
                .collect()
        });

        if let Ok(batch_words) = batch_words {
            for (batch_idx, words) in batch_words.iter().enumerate() {
                for (idx, word) in words.iter().enumerate() {
                    (callback.call)(
                        callback.state,
                        word.as_ptr(),
                        word.len(),
                        idx,
                        words.len(),
                        batch_idx,
                        batch_words.len(),
                    );
                }
            }
            batch_words.len()
        } else {
            0
        }
    } else {
        0
    }
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_pos_predict(
    model: *const Model,
    words: *const *const u8,
    words_len: *const usize,
    words_len_len: usize,
    callback: Callback,
) -> usize {
    let words_len = unsafe { slice::from_raw_parts(words_len, words_len_len) };
    let words = unsafe { slice::from_raw_parts(words, words_len_len) };
    let words = words
        .iter()
        .zip(words_len)
        .map(|(&word, &len)| unsafe { str::from_utf8_unchecked(slice::from_raw_parts(word, len)) })
        .collect::<Vec<_>>();

    if let EnumModel::POS(ref model) = unsafe { &(*model).model } {
        if let Ok(results) = model.predict(&words) {
            for (idx, result) in results.iter().enumerate() {
                (callback.call)(
                    callback.state,
                    result.as_ptr(),
                    result.len(),
                    idx,
                    results.len(),
                );
            }
            results.len()
        } else {
            0
        }
    } else {
        0
    }
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_pos_batch_predict(
    model: *const Model,
    batch_words: *const *const *const u8,
    batch_words_len: *const *const usize,
    batch_words_len_len: *const usize,
    batch_words_len_len_len: usize,
    callback: BatchCallback,
    threads: usize,
) -> usize {
    let batch_words_len_len =
        unsafe { slice::from_raw_parts(batch_words_len_len, batch_words_len_len_len) };
    let batch_words_len =
        unsafe { slice::from_raw_parts(batch_words_len, batch_words_len_len_len) };
    let batch_words = unsafe { slice::from_raw_parts(batch_words, batch_words_len_len_len) };
    let batch_words_len = batch_words_len
        .iter()
        .zip(batch_words_len_len)
        .map(|(&words_len, &len)| unsafe { slice::from_raw_parts(words_len, len) })
        .collect::<Vec<_>>();
    let batch_words = batch_words
        .iter()
        .zip(batch_words_len_len)
        .map(|(&words, &words_len)| unsafe { slice::from_raw_parts(words, words_len) })
        .collect::<Vec<_>>();
    let batch_words = batch_words
        .iter()
        .zip(batch_words_len)
        .map(|(words, words_len)| {
            words
                .iter()
                .zip(words_len)
                .map(|(&word, &len)| unsafe {
                    str::from_utf8_unchecked(slice::from_raw_parts(word, len))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    if let EnumModel::POS(ref model) = unsafe { &(*model).model } {
        let batch_pos: Result<Vec<Vec<_>>, _> = pool.install(|| {
            batch_words
                .into_par_iter()
                .map(|text| model.predict(&text))
                .collect()
        });

        if let Ok(batch_tags) = batch_pos {
            for (batch_idx, tags) in batch_tags.iter().enumerate() {
                for (idx, tag) in tags.iter().enumerate() {
                    (callback.call)(
                        callback.state,
                        tag.as_ptr(),
                        tag.len(),
                        idx,
                        tags.len(),
                        batch_idx,
                        batch_tags.len(),
                    );
                }
            }
            batch_tags.len()
        } else {
            0
        }
    } else {
        0
    }
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_ner_predict(
    model: *const Model,
    words: *const *const u8,
    words_len: *const usize,
    pos: *const *const u8,
    pos_len: *const usize,
    words_len_len: usize,
    callback: Callback,
) -> usize {
    let words_len = unsafe { slice::from_raw_parts(words_len, words_len_len) };
    let words = unsafe { slice::from_raw_parts(words, words_len_len) };
    let words = words
        .iter()
        .zip(words_len)
        .map(|(&word, &len)| unsafe { str::from_utf8_unchecked(slice::from_raw_parts(word, len)) })
        .collect::<Vec<_>>();

    let pos_len = unsafe { slice::from_raw_parts(pos_len, words_len_len) };
    let pos = unsafe { slice::from_raw_parts(pos, words_len_len) };
    let pos = pos
        .iter()
        .zip(pos_len)
        .map(|(&tag, &len)| unsafe { str::from_utf8_unchecked(slice::from_raw_parts(tag, len)) })
        .collect::<Vec<_>>();

    if let EnumModel::NER(ref model) = unsafe { &(*model).model } {
        if let Ok(results) = model.predict((&words, &pos)) {
            for (idx, result) in results.iter().enumerate() {
                (callback.call)(
                    callback.state,
                    result.as_ptr(),
                    result.len(),
                    idx,
                    results.len(),
                );
            }
            results.len()
        } else {
            0
        }
    } else {
        0
    }
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_ner_batch_predict(
    model: *const Model,
    batch_words: *const *const *const u8,
    batch_words_len: *const *const usize,
    batch_pos: *const *const *const u8,
    batch_pos_len: *const *const usize,
    batch_words_len_len: *const usize,
    batch_words_len_len_len: usize,
    callback: BatchCallback,
    threads: usize,
) -> usize {
    // common
    let batch_words_len_len =
        unsafe { slice::from_raw_parts(batch_words_len_len, batch_words_len_len_len) };
    let batch_words_len =
        unsafe { slice::from_raw_parts(batch_words_len, batch_words_len_len_len) };
    let batch_pos_len = unsafe { slice::from_raw_parts(batch_pos_len, batch_words_len_len_len) };

    // 第一层 Sentence words
    let batch_words = unsafe { slice::from_raw_parts(batch_words, batch_words_len_len_len) };
    let batch_words_len = batch_words_len
        .iter()
        .zip(batch_words_len_len)
        .map(|(&words_len, &len)| unsafe { slice::from_raw_parts(words_len, len) })
        .collect::<Vec<_>>();

    // 第一层 Sentence POS
    let batch_pos = unsafe { slice::from_raw_parts(batch_pos, batch_words_len_len_len) };
    let batch_pos_len = batch_pos_len
        .iter()
        .zip(batch_words_len_len)
        .map(|(&pos_len, &len)| unsafe { slice::from_raw_parts(pos_len, len) })
        .collect::<Vec<_>>();

    // 第二层 Sentence words
    let batch_words = batch_words
        .iter()
        .zip(batch_words_len_len)
        .map(|(&words, &words_len)| unsafe { slice::from_raw_parts(words, words_len) })
        .collect::<Vec<_>>();

    let batch_words = batch_words
        .iter()
        .zip(batch_words_len)
        .map(|(words, words_len)| {
            words
                .iter()
                .zip(words_len)
                .map(|(&word, &len)| unsafe {
                    str::from_utf8_unchecked(slice::from_raw_parts(word, len))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // 第二层 Sentence PO
    let batch_pos = batch_pos
        .iter()
        .zip(batch_words_len_len)
        .map(|(&pos, &pos_len)| unsafe { slice::from_raw_parts(pos, pos_len) })
        .collect::<Vec<_>>();

    let batch_pos = batch_pos
        .iter()
        .zip(batch_pos_len)
        .map(|(pos, pos_len)| {
            pos.iter()
                .zip(pos_len)
                .map(|(&pos, &len)| unsafe {
                    str::from_utf8_unchecked(slice::from_raw_parts(pos, len))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    if let EnumModel::NER(ref model) = unsafe { &(*model).model } {
        let batch_pos: Result<Vec<Vec<_>>, _> = pool.install(|| {
            batch_words
                .into_par_iter()
                .zip(batch_pos)
                .map(|(words, pos)| model.predict((&words, &pos)))
                .collect()
        });

        if let Ok(batch_tags) = batch_pos {
            for (batch_idx, tags) in batch_tags.iter().enumerate() {
                for (idx, tag) in tags.iter().enumerate() {
                    (callback.call)(
                        callback.state,
                        tag.as_ptr(),
                        tag.len(),
                        idx,
                        tags.len(),
                        batch_idx,
                        batch_tags.len(),
                    );
                }
            }
            batch_tags.len()
        } else {
            0
        }
    } else {
        0
    }
}

pub struct CHook {
    inner: Hook,
}

#[no_mangle]
pub extern "C" fn hook_new() -> *mut CHook {
    Box::into_raw(Box::new(CHook { inner: Hook::new() }))
}

#[no_mangle]
pub extern "C" fn hook_free(ptr: *mut CHook) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[no_mangle]
pub extern "C" fn hook_add_word(ptr: *mut CHook, word_ptr: *const u8, word_len: usize, pos: usize) {
    if ptr.is_null() { return; }
    let word = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(word_ptr, word_len)) };
    unsafe { (*ptr).inner.add_word(word, Some(pos)) };
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn run_hook_with_custom_word(
    hook_ptr: *mut CHook,               
    sentence_ptr: *const u8,
    sentence_len: usize,
    words_ptr: *const *const u8,
    word_lens_ptr: *const usize,
    word_count: usize,
    callback: Callback,
) -> usize {
    if hook_ptr.is_null() {
        return 0;
    }

    let sentence = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(sentence_ptr, sentence_len)) };
    let word_ptrs = unsafe { std::slice::from_raw_parts(words_ptr, word_count) };
    let word_lens = unsafe { std::slice::from_raw_parts(word_lens_ptr, word_count) };

    let cut_words = word_ptrs.iter()
        .zip(word_lens.iter())
        .map(|(&ptr, &len)| unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len))
        })
        .collect::<Vec<&str>>();

    let mut result_words = Vec::with_capacity(word_count);
    let mut route = Vec::with_capacity(word_count);
    let mut dag = Dag::with_size_hint(word_count);

    let hook = unsafe { &mut (*hook_ptr).inner };
    if hook.inner_hook(sentence, &cut_words, &mut result_words, &mut route, &mut dag).is_err() {
        return 0;
    }

    for (i, word) in result_words.iter().enumerate() {
        (callback.call)(
            callback.state,
            word.as_ptr(),
            word.len(),
            i,
            result_words.len(),
        );
    }

    result_words.len()
}

    

