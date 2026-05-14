//! Runtime known-failure — full k256 derive
//! `SecretKey::from_bytes(scalar=1).public_key().to_encoded_point(true)`
//! produces wrong bytes on device.
//!
//! Surfaced from vanity-miner-rs self_test slot 74. This is the full
//! path vanity-miner-rs's `secp256k1_derive_public_key` takes (see
//! `vanity-miner-rs/logic/src/secp256k1.rs`). With scalar=1 the result
//! must equal the well-known SEC1-compressed secp256k1 generator G.
//!
//! `SecretKey` is wrapped in `ManuallyDrop` because `SecretKey`
//! zeroizes on Drop and cuda-oxide does not yet emit device-side
//! `drop_in_place` for the `Zeroize`-derived path.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. `cargo oxide run` returns wrong 33
//! bytes — differs from the SEC1-compressed secp256k1 generator.
//!
//! ## What this stacks on
//!
//! Slot 73 (`SecretKey::from_bytes` alone) PASSES, so the bug is
//! downstream of `from_bytes`. Slots 78 / 93 / 96 (the encoding
//! chain) FAIL — fixing them should flip this example too.
//!
//! ## Fix
//!
//! See `k256_encoded_point_from_affine_coords_repro` (K256-1 L0) for the
//! full root cause. The downstream `to_encoded_point` chain bottoms out in
//! the same broken `MirConstructEnumOp` lowering for `sec1::Tag`.
//!
//! ## Build with
//!
//!     cargo oxide build k256_secret_key_derive_one_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

const SECP256K1_GENERATOR_COMPRESSED: [u8; 33] = [
    0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB,
    0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
    0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28,
    0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17,
    0x98,
];

/// Verbatim port of vanity-miner-rs `logic::check_k256_derive_scalar_one`,
/// inlining `secp256k1_derive_public_key` (same shape as
/// `logic/src/secp256k1.rs`).
#[inline(never)]
pub fn check() -> u32 {
    use core::mem::ManuallyDrop;
    use k256::SecretKey;
    use k256::elliptic_curve::sec1::ToEncodedPoint;

    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 1;

    let secret_key = ManuallyDrop::new(SecretKey::from_bytes((&priv_bytes).into()).unwrap());
    let public_key = secret_key.public_key();
    let encoded_point = public_key.to_encoded_point(true);
    let compressed_bytes = encoded_point.as_bytes();
    if compressed_bytes.len() != 33 {
        return 0;
    }
    let mut result = [0u8; 33];
    result.copy_from_slice(compressed_bytes);
    (result == SECP256K1_GENERATOR_COMPRESSED) as u32
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
    println!("=== k256_secret_key_derive_one_repro ===");

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
        eprintln!("FAIL: device-side `SecretKey::from_bytes(1).public_key().to_encoded_point(true)`");
        eprintln!("      did not produce the SEC1-compressed secp256k1 generator.");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: full k256 compressed derive for scalar=1 matches on device.");
    println!("PASS");
}
