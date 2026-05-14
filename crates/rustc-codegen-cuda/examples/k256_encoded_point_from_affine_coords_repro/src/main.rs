//! Runtime known-failure — `k256::EncodedPoint::from_affine_coordinates(
//! GX, GY, compress=true)` produces wrong bytes on device while the same
//! call on CPU returns the canonical 33-byte SEC1-compressed secp256k1
//! generator.
//!
//! Surfaced from vanity-miner-rs self_test slot 96. Passing baseline:
//! slot 100, which assembles `[u8; 33]` from the same `GX`/`GY` raw
//! bytes by hand (no `EncodedPoint`, no `FieldBytes::into`).
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. `cargo oxide run` returns wrong
//! 33 bytes — the device-side `EncodedPoint::as_bytes()` differs from
//! the well-known SEC1-compressed generator
//! `02 79BE667E F9DCBBAC 55A06295 CE870B07 029BFCDB 2DCE28D9 59F2815B 16F81798`.
//!
//! ## What's suspected
//!
//! Cross-crate monomorphization of `EncodedPoint::from_affine_coordinates`.
//! The hand-rolled `[u8; 33]` replica with the same coordinate bytes
//! passes (slot 100), so the algorithm is correct on device. The only
//! difference is the real-k256-crate codegen path: the function uses
//! `GenericArray<u8, U32>` (FieldBytes) and writes through
//! `GenericArray<u8, U33>` for the output. Suspicion: one of those
//! generic-typed reads/writes is being miscompiled when the function
//! is instantiated from a `#[kernel]`-rooted call graph.
//!
//! Slots 98/99/101/110 (GenericArray basic ops) PASS, and slot 100
//! (raw `[u8; 33]` replica) PASSES. So the bug is specifically in this
//! cross-crate function, not in the GenericArray ops themselves.
//!
//! No fix yet — documents cross-crate monomorphization bug.
//!
//! ## Build with
//!
//!     cargo oxide build k256_encoded_point_from_affine_coords_repro

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

/// Verbatim port of vanity-miner-rs `logic::check_k256_encoded_point_from_affine_coords`.
#[inline(never)]
pub fn check() -> u32 {
    use k256::EncodedPoint;
    use k256::elliptic_curve::FieldBytes;
    let x: &FieldBytes<k256::Secp256k1> = (&SECP256K1_GX_BYTES).into();
    let y: &FieldBytes<k256::Secp256k1> = (&SECP256K1_GY_BYTES).into();
    let encoded = EncodedPoint::from_affine_coordinates(x, y, true);
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
    println!("=== k256_encoded_point_from_affine_coords_repro ===");

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
    println!("device result[0] = {} (1 = pass, 0 = fail)", host_result[0]);
    println!("cpu   result     = {} (sanity: must be 1)", cpu_result);

    assert_eq!(cpu_result, 1, "CPU path itself disagrees with constants — repro is wrong");

    if host_result[0] != 1 {
        eprintln!();
        eprintln!("FAIL: device-side `EncodedPoint::from_affine_coordinates(GX, GY, true)`");
        eprintln!("      did not produce the SEC1-compressed secp256k1 generator.");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: EncodedPoint::from_affine_coordinates round-trips on device.");
    println!("PASS");
}
