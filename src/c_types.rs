//! The C types and errno values the decoder shares with C, from `libc` where it has them; wasm32 without
//! an OS has no libc, so they are spelled out there at the widths `libc` gives the other targets.

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use libc::{
    intptr_t, off_t, ptrdiff_t, uintptr_t, EAGAIN, EINVAL, ENOENT, ENOMEM, ENOPROTOOPT, ERANGE,
};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[allow(non_camel_case_types)]
mod wasm {
    pub type intptr_t = isize;
    pub type off_t = i64;
    pub type ptrdiff_t = isize;
    pub type uintptr_t = usize;
    pub const ENOENT: i32 = 2;
    pub const EAGAIN: i32 = 11;
    pub const ENOMEM: i32 = 12;
    pub const EINVAL: i32 = 22;
    pub const ERANGE: i32 = 34;
    pub const ENOPROTOOPT: i32 = 92;
}
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use wasm::*;
