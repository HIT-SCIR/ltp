#[cfg(not(target_env = "musl"))]
use mimalloc::MiMalloc;

#[cfg(not(target_env = "musl"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use std::ffi::c_void;

pub mod model;
pub mod stnsplit;

/// The LTP CFFI API.
/// the call args:
///    state: your design
///    tag: the predicted tag
///    tag_len: the length of tag
///    tag_index: the index of current predict
///    tag_total: the length of current predict
#[repr(C)]
pub struct Callback {
    pub state: *mut c_void,
    // state, char*, char_len, current idx, max_num
    pub call: extern "C" fn(*mut c_void, *const u8, usize, usize, usize),
}

/// The LTP CFFI API.
/// the call args:
///    state: your design
///    tag: the predicted tag
///    tag_len: the length of tag
///    tag_index: the index of current predict
///    tag_total: the length of current predict
///    batch_index: the predict index of current batch
///    batch_total: the batch size of current batch
#[repr(C)]
pub struct BatchCallback {
    pub state: *mut c_void,
    // state, char*, char_len, tag idx, tag num, batch index, batch num
    pub call: extern "C" fn(*mut c_void, *const u8, usize, usize, usize, usize, usize),
}
