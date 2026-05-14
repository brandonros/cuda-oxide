//! Runtime known-failure — the full ed25519 public-key derivation
//! pipeline (clamp → `Scalar::from_bytes_mod_order` →
//! `EdwardsPoint::mul_base` → `compress().to_bytes()`) does not match a
//! known KAT on device.
//!
//! Surfaced from vanity-miner-rs self_test slot 2 (the upstream of
//! every solana-pipeline slot 11/12). Mirrors
//! `vanity-miner-rs/logic/src/ed25519.rs::ed25519_derive_public_key`
//! applied to a fixed hashed-private input.
//!
//! Test vector:
//!   hashed_priv (64 bytes) =
//!     152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb
//!     0f5b8817a8e43510f83333417178f2f59fdc3c723199303a5f9be71af2f7b664
//!   expected_pub (32 bytes) =
//!     0af764c1b6133a3a0abd7ef9c853791b687ce1e235f9dc8466d886da314dbea7
//!
//! ## Diagnostic value
//!
//! The "downstream integration test" for the DALEK-1 cluster. If
//! `dalek_from_bytes_mod_order_*_repro` and `dalek_edwards_mul_base_one_repro`
//! flip to pass, this one should also flip — it's their composition
//! through `clamp_integer` (slot 70, already PASSing).
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. `cargo oxide run` produces a derived
//! public key that does not match the KAT.
//!
//! ## Fix
//!
//! Flipped together with L3 (`dalek_edwards_mul_base_one_repro`) by the
//! `[Deref, Field, ConstantIndex]` projection-lowering fix in
//! `crates/mir-importer/src/translator/rvalue.rs`. See that example's
//! doc-block for the full root cause — same broken
//! `FieldElement51::conditional_assign` underneath, just exercised by the
//! full ed25519 derive instead of a single basepoint scalar multiply.
//!
//! ## Build with
//!
//!     cargo oxide build dalek_ed25519_derive_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

const HASHED_PRIV: [u8; 64] = [
    0x15, 0x2d, 0x53, 0x72, 0x3d, 0xa4, 0x20, 0x34,
    0x78, 0x57, 0x4b, 0x15, 0x31, 0x43, 0xa7, 0xea,
    0xa9, 0x21, 0xa8, 0xd8, 0x2c, 0x62, 0x95, 0x17,
    0xd6, 0xb1, 0x89, 0x49, 0xf0, 0x11, 0x1a, 0xbb,
    0x0f, 0x5b, 0x88, 0x17, 0xa8, 0xe4, 0x35, 0x10,
    0xf8, 0x33, 0x33, 0x41, 0x71, 0x78, 0xf2, 0xf5,
    0x9f, 0xdc, 0x3c, 0x72, 0x31, 0x99, 0x30, 0x3a,
    0x5f, 0x9b, 0xe7, 0x1a, 0xf2, 0xf7, 0xb6, 0x64,
];
const EXPECTED_PUB: [u8; 32] = [
    0x0a, 0xf7, 0x64, 0xc1, 0xb6, 0x13, 0x3a, 0x3a,
    0x0a, 0xbd, 0x7e, 0xf9, 0xc8, 0x53, 0x79, 0x1b,
    0x68, 0x7c, 0xe1, 0xe2, 0x35, 0xf9, 0xdc, 0x84,
    0x66, 0xd8, 0x86, 0xda, 0x31, 0x4d, 0xbe, 0xa7,
];

/// Mirrors `vanity-miner-rs/logic/src/ed25519.rs::ed25519_derive_public_key`.
#[inline(never)]
pub fn check() -> u32 {
    let mut input = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        input[i] = HASHED_PRIV[i];
        i += 1;
    }

    let clamped_input = curve25519_dalek::scalar::clamp_integer(input);
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(clamped_input);
    let point = curve25519_dalek::EdwardsPoint::mul_base(&scalar);
    let compressed_point = point.compress();
    let pub_key_bytes = compressed_point.to_bytes();

    (pub_key_bytes == EXPECTED_PUB) as u32
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
    println!("=== dalek_ed25519_derive_repro ===");

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

    assert_eq!(cpu_result, 1, "CPU path itself disagrees with KAT — repro is wrong");

    if host_result[0] != 1 {
        eprintln!();
        eprintln!("FAIL: device-side ed25519_derive_public_key disagrees with KAT.");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: full ed25519 derive matches KAT on device.");
    println!("PASS");
}
