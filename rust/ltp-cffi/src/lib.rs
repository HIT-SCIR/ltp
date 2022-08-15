use std::ffi::c_void;

pub mod model;
pub mod stnsplit;

/// The LTP CFFI API.
/// the call args:
///    state: your design
///    text: the part of predicts (word/tags/sentence and so on)
///    text_len: the length of part of predicts
///    index: the index of current predict
///    length: the length of current predict
#[repr(C)]
pub struct Callback {
    pub state: *mut c_void,
    // state, char*, char_len, current idx, max_num
    pub call: extern "C" fn(*mut c_void, *const u8, usize, usize, usize),
}
