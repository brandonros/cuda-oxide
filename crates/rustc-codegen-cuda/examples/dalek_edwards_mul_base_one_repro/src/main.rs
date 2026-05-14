//! Runtime known-failure — `curve25519_dalek::EdwardsPoint::mul_base(
//! &Scalar::from_bytes_mod_order([1, 0, ..., 0])).compress().to_bytes()`
//! does not equal the well-known ed25519 basepoint encoding (RFC 8032)
//! on device.
//!
//! Surfaced from vanity-miner-rs self_test slot 72. Full fixed-base
//! scalar-mult path with the smallest non-trivial scalar.
//!
//! ## Diagnostic value
//!
//! Slot 70 (`clamp_integer`) PASSES. Slot 71 (`from_bytes_mod_order`
//! round-trip) FAILS. If 71 starts passing first and this still FAILS,
//! the bug is in mul_base / EdwardsPoint ops / compress() (field
//! inversion). If 71 and this both flip together when a fix lands,
//! it's the same upstream Scalar-construction bug.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. `cargo oxide run` produces compressed
//! bytes that disagree with the basepoint
//! `5866666666666666666666666666666666666666666666666666666666666666`.
//!
//! No fix yet — documents cross-crate monomorphization bug in
//! curve25519_dalek's Scalar / Edwards entry points.
//!
//! ## Build with
//!
//!     cargo oxide build dalek_edwards_mul_base_one_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

const ED25519_BASEPOINT_COMPRESSED: [u8; 32] = [
    0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
];

/// Verbatim port of vanity-miner-rs `logic::check_dalek_mul_base_scalar_one`.
#[inline(never)]
pub fn check() -> u32 {
    let mut scalar_bytes = [0u8; 32];
    scalar_bytes[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(scalar_bytes);
    let point = curve25519_dalek::EdwardsPoint::mul_base(&scalar);
    let compressed = point.compress().to_bytes();
    (compressed == ED25519_BASEPOINT_COMPRESSED) as u32
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
    println!("=== dalek_edwards_mul_base_one_repro ===");

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
        eprintln!("FAIL: device-side `EdwardsPoint::mul_base(1).compress().to_bytes()`");
        eprintln!("      did not produce the ed25519 basepoint encoding.");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: EdwardsPoint::mul_base(scalar=1) matches the basepoint on device.");
    println!("PASS");
}
