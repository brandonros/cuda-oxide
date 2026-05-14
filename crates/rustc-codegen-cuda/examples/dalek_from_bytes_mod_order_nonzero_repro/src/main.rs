//! Runtime known-failure — `curve25519_dalek::Scalar::from_bytes_mod_order(
//! [1, 0, ..., 0]).to_bytes()` returns bytes that differ from the input
//! on device.
//!
//! Surfaced from vanity-miner-rs self_test slot 71. Companion to slot
//! 102 (`dalek_from_bytes_mod_order_zero_repro`) with a non-zero input.
//! `1 < ℓ`, so reduction is still a no-op and the round-trip must
//! preserve the input bytes.
//!
//! Diagnostic value: 71 + 102 both failing rules out a zero-specific
//! codegen path and points at a general from_bytes_mod_order failure.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. `cargo oxide run` produces `to_bytes()`
//! that disagrees with the input.
//!
//! ## Fix
//!
//! See `dalek_from_canonical_bytes_zero_repro` (DALEK-1 L0) for the
//! full root cause. Same `Deref→Field→Index` projection-lowering bug
//! inside `Scalar52::sub`'s `L.0[i]` access; nonzero input goes
//! through the same Montgomery-reduce path.
//!
//! ## Build with
//!
//!     cargo oxide build dalek_from_bytes_mod_order_nonzero_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

/// Verbatim port of vanity-miner-rs `logic::check_dalek_scalar_round_trip_one`.
#[inline(never)]
pub fn check() -> u32 {
    let mut input = [0u8; 32];
    input[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(input);
    let bytes = scalar.to_bytes();
    (bytes == input) as u32
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
    println!("=== dalek_from_bytes_mod_order_nonzero_repro ===");

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
        eprintln!("FAIL: device-side `Scalar::from_bytes_mod_order([1, 0, ..., 0]).to_bytes()`");
        eprintln!("      did not round-trip to the input.");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: Scalar::from_bytes_mod_order(one) round-trips on device.");
    println!("PASS");
}
