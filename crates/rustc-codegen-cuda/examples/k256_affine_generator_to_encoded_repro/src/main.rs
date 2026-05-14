//! Runtime known-failure — `k256::AffinePoint::GENERATOR.to_encoded_point(true)`
//! produces wrong bytes on device.
//!
//! Surfaced from vanity-miner-rs self_test slot 93. One altitude up
//! from slot 96 (`EncodedPoint::from_affine_coordinates`): `AffinePoint::
//! to_encoded_point` internally does an `is_identity` check via
//! `subtle::Choice` (slot 94 PASS) and a `conditional_select` arm
//! (slot 95 PASS), then calls into `EncodedPoint::from_affine_
//! coordinates(x, y, compress)`. If this is the same bug as slot 96,
//! the fix that flips slot 96 should also flip this one.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. `cargo oxide run` returns wrong 33
//! bytes — differs from the SEC1-compressed secp256k1 generator.
//!
//! ## Fix
//!
//! See `k256_encoded_point_from_affine_coords_repro` (K256-1 L0) for the
//! full root cause. Same `MirConstructEnumOp` bug — `sec1::Tag`'s
//! `#[repr(u8)]` explicit discriminants (`CompressedEvenY = 2`, etc.) were
//! being overwritten with declaration-order indices.
//!
//! ## Build with
//!
//!     cargo oxide build k256_affine_generator_to_encoded_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

const SECP256K1_GENERATOR_COMPRESSED: [u8; 33] = [
    0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB,
    0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
    0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28,
    0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17,
    0x98,
];

/// Verbatim port of vanity-miner-rs `logic::check_k256_affine_generator_encode`.
#[inline(never)]
pub fn check() -> u32 {
    use k256::AffinePoint;
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    let g = AffinePoint::GENERATOR;
    let encoded = g.to_encoded_point(true);
    let bytes = encoded.as_bytes();
    if bytes.len() != 33 {
        return 0;
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(bytes);
    (out == SECP256K1_GENERATOR_COMPRESSED) as u32
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
    println!("=== k256_affine_generator_to_encoded_repro ===");

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

    assert_eq!(cpu_result, 1, "CPU path itself disagrees with constants — repro is wrong");

    if host_result[0] != 1 {
        eprintln!();
        eprintln!("FAIL: device-side `AffinePoint::GENERATOR.to_encoded_point(true)`");
        eprintln!("      did not produce the SEC1-compressed secp256k1 generator.");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: AffinePoint::GENERATOR.to_encoded_point(true) matches on device.");
    println!("PASS");
}
