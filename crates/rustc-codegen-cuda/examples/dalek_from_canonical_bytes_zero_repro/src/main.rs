//! Runtime known-failure — `curve25519_dalek::Scalar::from_canonical_bytes(
//! [0u8; 32]).unwrap() == Scalar::ZERO` returns `false` on device.
//!
//! Surfaced from vanity-miner-rs self_test slot 112. Smallest leaf of
//! the DALEK-1 cluster: `from_canonical_bytes` does NOT call `reduce()`,
//! it only validates `bytes < ℓ` and wraps. For `[0; 32]`, `0 < ℓ`, so
//! it returns `CtOption::Some(Scalar { bytes: [0; 32] })`. Then
//! comparing against `Scalar::ZERO` via PartialEq must return `true`.
//!
//! Passing baseline: vanity-miner-rs slots 113–117 take verbatim ports
//! of dalek's Scalar52 + reduce pipeline (compiled inside the test
//! crate, not from the real dalek crate) and all PASS on the same
//! all-zero input. So the algorithm is correct on device; the bug is
//! specifically in the real `curve25519_dalek::Scalar` API surface
//! when monomorphized from a `#[kernel]`-rooted call graph.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. `cargo oxide run` returns 0 (`false`)
//! from the `check` body — either the CtOption unwrap returned None
//! unexpectedly, or the PartialEq comparison returned false despite
//! both operands being zero.
//!
//! ## Root cause
//!
//! `[crates/mir-importer/src/translator/rvalue.rs]` lowered the Place
//! projection chain `[Deref, Field, Index]` (taken from `&self.0[i]`
//! inside `Scalar52`'s `impl Index<usize>`) by walking Deref + Field
//! and then **silently dropping** the Index step — the resulting
//! pointer was `&array[0]` instead of `&array[i]`. Every iteration of
//! the `for i in 0..5 { ... L.0[i] ... }` loop inside `Scalar52::sub`
//! therefore re-loaded `L.0[0]`, so `sub(intermediate, L)` (the final
//! canonicalization step in `montgomery_reduce`) returned garbage,
//! `Scalar::reduce(zero) != zero`, `is_canonical(zero) == Choice(0)`,
//! and the validation branch of `from_canonical_bytes` returned None.
//!
//! The passing twin (`dalek_scalar52_reduce_pipeline_zero_repro`)
//! uses explicit `L.0[i]` (projection chain `[Field, Index]`, no Deref)
//! and went through a different lowering path that handled the Index,
//! which is why the byte-identical algorithm passed in-tree.
//!
//! ## Fix
//!
//! Added an `Index` arm to the projection-walking loop in rvalue.rs
//! (both the `Rvalue::Ref` and `Rvalue::RawPtr` sites): emit a
//! `MirArrayElementAddrOp` so the resulting pointer is offset to the
//! correct array element.
//!
//! ## Build with
//!
//!     cargo oxide build dalek_from_canonical_bytes_zero_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

/// Verbatim port of vanity-miner-rs `logic::check_dalek_from_canonical_zero`.
#[inline(never)]
pub fn check() -> u32 {
    use curve25519_dalek::Scalar;
    let opt = Scalar::from_canonical_bytes(core::hint::black_box([0u8; 32]));
    let s_opt: Option<Scalar> = opt.into();
    let s = match s_opt {
        Some(s) => s,
        None => return 0,
    };
    let zero = Scalar::ZERO;
    (s == zero) as u32
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
    println!("=== dalek_from_canonical_bytes_zero_repro ===");

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
        eprintln!("FAIL: device-side `Scalar::from_canonical_bytes([0; 32]).unwrap() == ZERO`");
        eprintln!("      returned false (or the CtOption unwrap returned None).");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS: Scalar::from_canonical_bytes(0) round-trips on device.");
    println!("PASS");
}
