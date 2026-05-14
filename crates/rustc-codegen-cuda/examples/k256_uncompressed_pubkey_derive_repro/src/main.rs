//! Runtime known-failure — full k256 uncompressed derive
//! `SecretKey::from_bytes(scalar=1).public_key().to_encoded_point(false)`
//! produces wrong 65 bytes on device.
//!
//! Surfaced from vanity-miner-rs self_test slot 5 (the uncompressed
//! primitive used by every ethereum-pipeline slot 13/14/15). Same code
//! shape as the compressed derive (slot 74), differing only in the
//! `to_encoded_point(false)` flag → SEC1 uncompressed: 0x04 || GX || GY.
//!
//! With scalar=1 the result must equal the well-known SEC1-uncompressed
//! secp256k1 generator G.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. `cargo oxide run` returns wrong 65
//! bytes — differs from the SEC1-uncompressed secp256k1 generator.
//!
//! ## What this stacks on
//!
//! If `k256_secret_key_derive_one_repro` (slot 74) flips to pass, this
//! example should also flip — they share `SecretKey::from_bytes` and
//! the `to_encoded_point` chain, differing only in the compress flag.
//!
//! ## Fix
//!
//! See `k256_encoded_point_from_affine_coords_repro` (K256-1 L0) for the
//! full root cause. The `MirConstructEnumOp` discriminant fix flipped this
//! prediction true — same chain, just with `compress = false`
//! (`Tag::Uncompressed`, explicit value `4`, declaration index `3`).
//!
//! ## Build with
//!
//!     cargo oxide build k256_uncompressed_pubkey_derive_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

const SECP256K1_GENERATOR_UNCOMPRESSED: [u8; 65] = [
    0x04,
    // GX
    0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
    0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
    0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
    0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
    // GY
    0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65,
    0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
    0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19,
    0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8,
];

/// Same shape as vanity-miner-rs `secp256k1_derive_public_key_uncompressed`
/// (`logic/src/secp256k1.rs`), called with scalar=1 so the result is the
/// well-known generator G.
#[inline(never)]
pub fn check() -> u32 {
    use core::mem::ManuallyDrop;
    use k256::SecretKey;
    use k256::elliptic_curve::sec1::ToEncodedPoint;

    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 1;

    let secret_key = ManuallyDrop::new(SecretKey::from_bytes((&priv_bytes).into()).unwrap());
    let public_key = secret_key.public_key();
    let encoded_point = public_key.to_encoded_point(false);
    let uncompressed_bytes = encoded_point.as_bytes();
    if uncompressed_bytes.len() != 65 {
        return 0;
    }
    let mut result = [0u8; 65];
    result.copy_from_slice(uncompressed_bytes);
    (result == SECP256K1_GENERATOR_UNCOMPRESSED) as u32
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
    println!("=== k256_uncompressed_pubkey_derive_repro ===");

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
        eprintln!("FAIL: device-side `SecretKey::from_bytes(1).public_key().to_encoded_point(false)`");
        eprintln!("      did not produce the SEC1-uncompressed secp256k1 generator.");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: full k256 uncompressed derive for scalar=1 matches on device.");
    println!("PASS");
}
