//! Runtime known-failure — `curve25519_dalek::Scalar::from_bytes_mod_order(
//! [0u8; 32]).to_bytes()` returns bytes that differ from `[0u8; 32]` on
//! device.
//!
//! Surfaced from vanity-miner-rs self_test slot 102. One altitude up
//! from slot 112: `from_bytes_mod_order` DOES call `reduce()` (unlike
//! `from_canonical_bytes`), exercising the full Montgomery-reduce
//! pipeline. Input is all-zero, so reduction is a no-op and the result
//! must round-trip to `[0; 32]`.
//!
//! Passing baseline: vanity-miner-rs slot 117 runs the same reduce
//! pipeline on the same input via a verbatim Scalar52 port inside the
//! test crate, and PASSES. So the algorithm is correct on device; the
//! bug is specifically in real curve25519_dalek's `from_bytes_mod_order`
//! entry point.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. `cargo oxide run` produces `to_bytes()`
//! that disagrees with the input `[0; 32]`.
//!
//! No fix yet — documents cross-crate monomorphization bug in
//! curve25519_dalek's Scalar entry points.
//!
//! ## Build with
//!
//!     cargo oxide build dalek_from_bytes_mod_order_zero_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

/// Verbatim port of vanity-miner-rs `logic::check_dalek_scalar_round_trip_zero`.
#[inline(never)]
pub fn check() -> u32 {
    let input = [0u8; 32];
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(core::hint::black_box(input));
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
    println!("=== dalek_from_bytes_mod_order_zero_repro ===");

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
        eprintln!("FAIL: device-side `Scalar::from_bytes_mod_order([0; 32]).to_bytes()`");
        eprintln!("      did not round-trip to [0; 32].");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: Scalar::from_bytes_mod_order(zero) round-trips on device.");
    println!("PASS");
}
