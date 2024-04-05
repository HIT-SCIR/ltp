use crate::Callback;
use ltp::utils::stnsplit::{
    stn_split as r_stn_split, stn_split_with_options as r_stn_split_with_options, SplitOptions,
};
use std::slice;

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn stn_split(text: *const u8, text_len: usize, callback: Callback) -> usize {
    let text = unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(text, text_len)) };
    let sentences = r_stn_split(text);
    for (idx, sentence) in sentences.iter().enumerate() {
        (callback.call)(
            callback.state,
            text.as_ptr(),
            sentence.len(),
            idx,
            sentence.len(),
        );
    }
    sentences.len()
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn stn_split_with_options(
    text: *const u8,
    text_len: usize,
    callback: Callback,
    use_zh: bool,
    use_en: bool,
    bracket_as_entity: bool,
    zh_quote_as_entity: bool,
    en_quote_as_entity: bool,
) -> usize {
    let text = unsafe { std::str::from_utf8_unchecked(slice::from_raw_parts(text, text_len)) };
    let options = SplitOptions {
        use_zh,
        use_en,
        bracket_as_entity,
        zh_quote_as_entity,
        en_quote_as_entity,
    };

    let sentences = r_stn_split_with_options(text, &options);
    for (idx, sentence) in sentences.iter().enumerate() {
        (callback.call)(
            callback.state,
            text.as_ptr(),
            sentence.len(),
            idx,
            sentence.len(),
        );
    }
    sentences.len()
}
