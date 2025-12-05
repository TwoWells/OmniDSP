use crate::core::config::Config;
use crate::context::Context; // Was Backend
use crate::traits::dft::{DftSpec, DftDirection};
use crate::core::types::Complex32;
use libc::{c_void, c_int};
use std::ffi::CStr;
use std::os::raw::c_char;

// --- Status Codes ---
#[repr(C)]
pub enum OmniStatus {
    Ok = 0,
    Error = -1,
    NullPointer = -2,
    InitFailed = -3,
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

// --- Context (Manager) ---

#[no_mangle]
pub extern "C" fn omni_context_create(config_ptr: *mut Config) -> *mut Context {
    if config_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let config = unsafe { *Box::from_raw(config_ptr) };
    
    match Context::new(config) {
        Ok(ctx) => Box::into_raw(Box::new(ctx)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn omni_context_destroy(ptr: *mut Context) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

// --- DFT Plan (Opaque Wrapper) ---
pub struct DftPlanWrapper {
    inner: Box<dyn crate::traits::dft::DftPlan<Complex32>>,
}

#[no_mangle]
pub extern "C" fn omni_dft_create_plan_c32(
    context_ptr: *mut Context,
    length: usize,
    direction: i32, // 0 = Forward, 1 = Inverse
) -> *mut DftPlanWrapper {
    if context_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let context = unsafe { &*context_ptr };

    let dir = match direction {
        0 => DftDirection::Forward,
        _ => DftDirection::Inverse,
    };

    let spec = DftSpec {
        length,
        direction: dir,
    };

    // Using the Dft trait on Context
    use crate::traits::dft::Dft; 
    match context.create_plan(spec) {
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
    
    // Create safe slices from raw pointers
    let input_slice = unsafe { std::slice::from_raw_parts(input, length) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, length) };

    match wrapper.inner.process(input_slice, output_slice) {
        Ok(_) => OmniStatus::Ok,
        Err(_) => OmniStatus::Error,
    }
}