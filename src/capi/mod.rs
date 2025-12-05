use crate::core::config::Config;
use crate::backend::Backend;
use crate::traits::dft::{DftSpec, DftDirection};
use crate::core::types::Complex32;
use libc::{c_void, c_int};
use std::ffi::CStr;
use std::os::raw::c_char;

// --- Opaque Types ---
// We use incomplete structs in C, but here we cast them to void or specific types.
// Best practice: Use unit structs or just raw pointers.

// --- Status Codes ---
#[repr(C)]
pub enum OmniStatus {
    Ok = 0,
    Error = -1,
    NullPointer = -2,
}

// --- Config ---

#[no_mangle]
pub extern "C" fn omni_config_create() -> *mut Config {
    let config = Config::new();
    Box::into_raw(Box::new(config))
}

#[no_mangle]
pub extern "C" fn omni_config_destroy(ptr: *mut Config) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

// --- Backend ---

#[no_mangle]
pub extern "C" fn omni_backend_create(config_ptr: *mut Config) -> *mut Backend {
    if config_ptr.is_null() {
        return std::ptr::null_mut();
    }
    // Take ownership of config? Or clone?
    // Usually "create(config)" implies the backend takes it or uses it.
    // Our Rust API: Backend::new(Config) -> takes ownership.
    // So C side gives up ownership.
    let config = unsafe { *Box::from_raw(config_ptr) };
    
    let backend = Backend::new(config);
    Box::into_raw(Box::new(backend))
}

#[no_mangle]
pub extern "C" fn omni_backend_destroy(ptr: *mut Backend) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

// --- DFT Plan (Opaque Wrapper) ---
// Since DftPlan is a Trait Object (Box<dyn DftPlan>), we can't just cast it to void* easily
// because fat pointers are 16 bytes. We need a thin wrapper struct.
pub struct DftPlanWrapper {
    inner: Box<dyn crate::traits::dft::DftPlan<Complex32>>,
}

#[no_mangle]
pub extern "C" fn omni_dft_create_plan_c32(
    backend_ptr: *mut Backend,
    length: usize,
    direction: i32, // 0 = Forward, 1 = Inverse
) -> *mut DftPlanWrapper {
    if backend_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let backend = unsafe { &*backend_ptr };

    let dir = match direction {
        0 => DftDirection::Forward,
        _ => DftDirection::Inverse,
    };

    let spec = DftSpec {
        length,
        direction: dir,
    };

    // Using the Dft trait on Backend
    use crate::traits::dft::Dft; // Trait must be in scope
    match backend.create_plan(spec) {
        Ok(plan) => {
            let wrapper = DftPlanWrapper { inner: plan };
            Box::into_raw(Box::new(wrapper))
        },
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn omni_dft_destroy_plan_c32(ptr: *mut DftPlanWrapper) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[no_mangle]
pub extern "C" fn omni_dft_execute_c32(
    plan_ptr: *mut DftPlanWrapper,
    input: *const Complex32,
    output: *mut Complex32,
) -> OmniStatus {
    if plan_ptr.is_null() || input.is_null() || output.is_null() {
        return OmniStatus::NullPointer;
    }

    let wrapper = unsafe { &*plan_ptr };
    let length = wrapper.inner.get_spec().length;
    
    // Create safe slices from raw pointers using the length from the spec
    let input_slice = unsafe { std::slice::from_raw_parts(input, length) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, length) };

    match wrapper.inner.process(input_slice, output_slice) {
        Ok(_) => OmniStatus::Ok,
        Err(_) => OmniStatus::Error,
    }
}
