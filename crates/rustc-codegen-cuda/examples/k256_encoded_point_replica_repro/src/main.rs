//! PASSING-TWIN baseline for `k256_encoded_point_from_affine_coords_repro`
//! (K256-1 L0, slot 96). Hand-rolled re-implementation of sec1's
//! `EncodedPoint::from_affine_coordinates(GX, GY, compress=true)` body
//! using raw `[u8; 33]` instead of `GenericArray<u8, U33>` and a plain
//! `bool` instead of `subtle::Choice` for the parity bit. Same inputs,
//! same algorithm, same expected output (canonical SEC1-compressed
//! secp256k1 generator).
//!
//! Surfaced from vanity-miner-rs self_test slot 100, which PASSes on
//! device while slot 96 FAILs. This crate's `.ptx` is the diff target
//! for localizing the cross-crate monomorphization bug — the algorithm
//! is identical to slot 96, so any non-trivial diff in the emitted
//! `check` body is the bug surface.
//!
//! ## What this DOES NOT depend on
//!
//! * `k256` — no `EncodedPoint`, no `Tag`, no `sec1::point`.
//! * `elliptic-curve` / `generic-array` / `typenum` — no typed array sizes.
//! * `subtle` — no `Choice` / `ConditionallySelectable` / `CtOption`.
//!
//! The only crates touched at codegen time are `core` and the
//! `cuda-*` runtime. Any PTX shape this example does NOT contain that
//! the failing twin DOES contain is candidate-bug-surface.
//!
//! ## Build with
//!
//!     cargo run -p cargo-oxide -- build k256_encoded_point_replica_repro
//!
//! ## PTX diff workflow
//!
//! ```sh
//! diff -u \
//!   <(awk '/Begin function .*__check/,/End function .*__check/' \
//!       k256_encoded_point_from_affine_coords_repro/*.ptx) \
//!   <(awk '/Begin function .*__check/,/End function .*__check/' \
//!       k256_encoded_point_replica_repro/*.ptx)
//! ```

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

const SECP256K1_GX_BYTES: [u8; 32] = [
    0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
    0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
    0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
    0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
];
const SECP256K1_GY_BYTES: [u8; 32] = [
    0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65,
    0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
    0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19,
    0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8,
];
const SECP256K1_GENERATOR_COMPRESSED: [u8; 33] = [
    0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB,
    0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
    0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28,
    0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17,
    0x98,
];

/// Verbatim port of vanity-miner-rs `logic::check_from_affine_coords_replica`.
///
/// Algorithm (identical to sec1's `from_affine_coordinates` for
/// compress=true):
///   tag    = 0x02 if y[31] is even, 0x03 if odd
///   out[0] = tag
///   out[1..33] = x
#[inline(never)]
pub fn check() -> u32 {
    let x_bytes = &SECP256K1_GX_BYTES;
    let y_bytes = &SECP256K1_GY_BYTES;
    let last_y = core::hint::black_box(y_bytes[31]);
    let tag: u8 = if last_y & 1 == 1 { 0x03 } else { 0x02 };
    let mut bytes = [0u8; 33];
    bytes[0] = tag;
    bytes[1..33].copy_from_slice(x_bytes);
    (bytes == SECP256K1_GENERATOR_COMPRESSED) as u32
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn run(results: &mut [u32]) {
        results[0] = super::check();
    }
}

fn main() {
    println!("=== k256_encoded_point_replica_repro (passing twin for slot 96) ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let mut results = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    unsafe {
        module
            .run(&stream, LaunchConfig::for_num_elems(1), &mut results)
            .expect("kernel launch");
    }

    let host_result = results.to_host_vec(&stream).unwrap();
    let cpu_result = check();
    println!("device result[0] = {}", host_result[0]);
    println!("cpu   result     = {} (sanity)", cpu_result);

    assert_eq!(cpu_result, 1, "CPU path itself disagrees — repro is wrong");

    if host_result[0] != 1 {
        eprintln!();
        eprintln!("UNEXPECTED FAIL: passing twin failed on device.");
        eprintln!("This was supposed to be the working baseline.");
        eprintln!("If this fails, the bug has spread to the local-types path —");
        eprintln!("the diff strategy is no longer well-grounded; re-check assumptions.");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: hand-rolled [u8; 33] replica matches on device (as expected).");
    println!("PASS");
}
