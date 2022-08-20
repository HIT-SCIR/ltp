use crate::Callback;
use ltp::{CWSDefinition, NERDefinition, POSDefinition};
use std::slice;

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

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_load(model_path: *const u8, model_path_len: usize) -> *mut Model {
    use ltp::perceptron::{ModelSerde, Reader, Schema};
    let model_path = unsafe { slice::from_raw_parts(model_path, model_path_len) };
    let model_path = String::from_utf8_lossy(model_path);
    let model_path: &str = &model_path;

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
            Schema::Record { name, .. } => match name.name.as_str() {
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
pub extern "C" fn model_release(model: *mut *mut Model) {
    let _ = unsafe { Box::from_raw(*model) };
    unsafe { *model = std::ptr::null_mut() };
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn model_save(
    model: *const Model,
    model_path: *const u8,
    model_path_len: usize,
) -> bool {
    use ltp::perceptron::ModelSerde;
    let model_path = unsafe { slice::from_raw_parts(model_path, model_path_len) };
    let model_path = String::from_utf8_lossy(model_path);
    let model_path: &str = &model_path;

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
pub extern "C" fn model_cws_predict(
    model: *const Model,
    sentence: *const u8,
    sentence_len: usize,
    callback: Callback,
) -> usize {
    let sentence = unsafe { slice::from_raw_parts(sentence, sentence_len) };
    let sentence = String::from_utf8_lossy(sentence);

    if let EnumModel::CWS(ref model) = unsafe { &(*model).model } {
        if let Ok(results) = model.predict(&sentence) {
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
        .map(|(&word, &len)| unsafe {
            std::str::from_utf8_unchecked(slice::from_raw_parts(word, len))
        })
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
        .map(|(&word, &len)| unsafe {
            std::str::from_utf8_unchecked(slice::from_raw_parts(word, len))
        })
        .collect::<Vec<_>>();

    let pos_len = unsafe { slice::from_raw_parts(pos_len, words_len_len) };
    let pos = unsafe { slice::from_raw_parts(pos, words_len_len) };
    let pos = pos
        .iter()
        .zip(pos_len)
        .map(|(&tag, &len)| unsafe {
            std::str::from_utf8_unchecked(slice::from_raw_parts(tag, len))
        })
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
