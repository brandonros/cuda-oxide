//! Bug-41 narrowed: `<[u8]>::reverse()` on a partial sub-slice of a
//! stack-resident `[u8; N]` array.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. On real hardware (vanity-miner-rs
//! slot 108, v1.53.0), the sub-slice's bytes don't get reversed
//! correctly — either they stay in original order, or the swap
//! pattern produces wrong content.
//!
//! `base58_encode_32` calls `output[..result_len].reverse()` after
//! the digit-extraction + alphabet-map loops. Slot 107 (the same
//! algorithm with a hand-rolled `while i < j { swap(i, j); ... }`)
//! PASSes; only the std `<[u8]>::reverse()` variant FAILs. So the
//! bug is specifically in how rustc/std lowers `[T]::reverse` on
//! GPU.
//!
//! `<[u8]>::reverse()` (std impl):
//! ```ignore
//! pub fn reverse(&mut self) {
//!     let half_len = self.len() / 2;
//!     let Range { start, end } = 0..half_len;
//!     // unsafe chunked-swap loop using get_unchecked_mut + ptr::swap
//! }
//! ```
//!
//! The interesting interactions in this shape:
//! * `arr[..result_len]` — slice indexing with a runtime upper bound
//!   produces `&mut [u8]` (a fat pointer).
//! * `<[u8]>::reverse()` — calls `get_unchecked_mut(i)` /
//!   `get_unchecked_mut(len - 1 - i)`, swaps via `ptr::swap`.
//!
//! If any of those steps miscompiles for slices, reverse() leaves
//! the array unchanged (or partially shuffled).
//!
//! ## What this test locks down
//!
//! Two shapes:
//!
//! 1. `check_reverse_partial` — slot 108 verbatim: 64-byte array,
//!    fill first 5 bytes, call `arr[..result_len].reverse()`,
//!    compare to expected post-reverse bytes.
//! 2. `check_reverse_full` — `arr.reverse()` on the FULL 8-byte
//!    array. Confirms whether the bug is specifically about
//!    sub-slicing or about reverse() in general.
//!
//! ## Build
//!
//!     cargo oxide build slice_reverse_partial

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[inline(never)]
fn check_reverse_partial() -> u32 {
    let mut arr = [0u8; 64];
    arr[0] = 0x11;
    arr[1] = 0x22;
    arr[2] = 0x33;
    arr[3] = 0x44;
    arr[4] = 0x55;

    // Runtime-known length — black_box prevents the optimizer from
    // proving result_len is the constant 5 and unrolling the reverse.
    let result_len = core::hint::black_box(5usize);

    arr[..result_len].reverse();

    // Expected: arr[0..5] reversed, rest unchanged.
    let expected: [u8; 5] = [0x55, 0x44, 0x33, 0x22, 0x11];
    for i in 0..5 {
        if arr[i] != expected[i] {
            return 0;
        }
    }
    for i in 5..64 {
        if arr[i] != 0 {
            return 0;
        }
    }
    1
}

#[inline(never)]
fn check_reverse_full() -> u32 {
    let mut arr: [u8; 8] = [
        core::hint::black_box(0xAAu8),
        core::hint::black_box(0xBBu8),
        core::hint::black_box(0xCCu8),
        core::hint::black_box(0xDDu8),
        core::hint::black_box(0xEEu8),
        core::hint::black_box(0xFFu8),
        core::hint::black_box(0x11u8),
        core::hint::black_box(0x22u8),
    ];

    arr.reverse();

    let expected: [u8; 8] = [0x22, 0x11, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA];
    for i in 0..8 {
        if arr[i] != expected[i] {
            return 0;
        }
    }
    1
}

#[inline(never)]
fn check_slice_reverse_partial() -> u32 {
    if check_reverse_partial() == 0 {
        return 0;
    }
    if check_reverse_full() == 0 {
        return 0;
    }
    1
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            *slot = check_slice_reverse_partial();
        }
    }
}

fn main() {
    println!("=== slice_reverse_partial ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let n_out = 4usize;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, n_out).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(n_out as u32), &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for (i, r) in result.iter().enumerate() {
        assert_eq!(*r, 1, "thread {} got {} (expected 1)", i, r);
    }
    println!("SUCCESS: `<[u8]>::reverse()` on partial + full slices");
}
