//! Runtime known-failure — `(returned_struct.payload == [u8; 32])`
//! triggers `CUDA_ERROR_MISALIGNED_ADDRESS` (716) on device when
//! `returned_struct`'s `[u8; 32]` field is at struct offset 1.
//!
//! Surfaced from vanity-miner-rs self_test slot 13 (ethereum priv).
//! `EthereumVanityKeyResult` is a default-repr struct with a `bool` and
//! `[u8; 32]` private_key (among others); Rust's layout optimizer pulls
//! `bool` to offset 0 and lands `private_key: [u8; 32]` at offset 1,
//! producing a struct of size 117 align 1. `check_ethereum_priv` then
//! does `result.private_key == EXPECTED` which the NVPTX optimizer
//! vectorizes to two `ld.local.v2.b64` instructions per operand. The
//! `EXPECTED` local sits at the 16-aligned local depot start (safe),
//! but `result.private_key` at struct-base + 1 is only 1-aligned — and
//! `ld.local.v2.b64` requires natural 16-byte alignment of the
//! effective address. Result: misaligned load → hard PTX fault.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeded. `cargo oxide run` reached the kernel
//! launch, then the next `cuMemcpyDtoHAsync_v2` returned
//! `CUDA_ERROR_MISALIGNED_ADDRESS` (716) with sticky-error markers on
//! every subsequent CUDA call — classic symptom of a hard kernel fault
//! on the GPU.
//!
//! ## Root cause
//!
//! `convert_rust_raw_eq` in `crates/mir-lower/src/convert/ops/call.rs`
//! lowered `[T; N] == [T; N]` to a single `load iN` (no `align`
//! attribute) + `icmp eq`. With no explicit alignment, NVPTX `llc`
//! defaulted to the result-type's ABI alignment — for `i256` that's
//! large enough to trigger `ld.local.v2.b64` lowering, which requires
//! natural 16-byte alignment of the effective address.
//!
//! The pointee here is `[u8; 32]` (alignment 1), accessed at struct
//! offset +1, so the runtime address was 1-aligned and the wide load
//! hard-faulted.
//!
//! The historical companion bug `xoshiro_seed_misalign` was fixed by
//! bumping alloca alignment via `conservative_alloca_align`, which is
//! sufficient when the entire pointee value is at the alloca base. It
//! is *insufficient* when the access GEPs into a misaligned interior
//! field — the alloca is 16-aligned, but `alloca + 1` is not.
//!
//! ## Fix
//!
//! 1. Added an `llvm_load_alignment` attribute to `llvm.load` in
//!    `crates/dialect-llvm/src/ops/memory.rs`, plus an exporter hook in
//!    `crates/dialect-llvm/src/export.rs` that appends `, align N` when
//!    set.
//! 2. Added `get_type_alignment` in `crates/mir-lower/src/convert/types.rs`
//!    that computes a conservative natural alignment from an LLVM type
//!    (`[u8; N]` → 1, `[u64; N]` → 8, struct → max-of-fields, etc.).
//! 3. `convert_rust_raw_eq` now sets the load alignment to
//!    `get_type_alignment(pointee)`, so NVPTX honors the real pointer
//!    alignment instead of guessing from the result type.
//!
//! Slot 13 of vanity-miner-rs's self_test (`ethereum priv`), which
//! exercised this on the way to comparing
//! `EthereumVanityKeyResult::private_key == EXPECTED`, flipped FAIL →
//! PASS as a direct consequence; the rest of the self-test (slots
//! 14–117) flipped too, ending in `all 118 checks passed`.
//!
//! ## What this DOES NOT depend on
//!
//! * `curve25519-dalek` / `k256` / `subtle` — no crypto crates.
//! * `rand` / `xoshiro` — no RNG.
//!
//! Only `core` and the `cuda-*` runtime. The faulting `ld.local.v2.b64`
//! comes from the codegen of `[u8; 32] == [u8; 32]` against a 1-aligned
//! source.
//!
//! ## Build with
//!
//!     cargo oxide build misaligned_struct_field_array_eq_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

/// Default Rust repr — layout optimizer is free to reorder. With one
/// `bool` (align 1, size 1) and one `[u8; 32]` (align 1, size 32),
/// total size is 33 with alignment 1; rustc places the `bool` at
/// offset 0 and `payload` at offset 1, matching the EthereumVanityKey-
/// Result shape.
pub struct WithFlagFirst {
    pub payload: [u8; 32],
    pub flag: bool,
}

const EXPECTED: [u8; 32] = [
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
];

#[inline(never)]
fn make_struct() -> WithFlagFirst {
    WithFlagFirst {
        payload: EXPECTED,
        flag: true,
    }
}

#[inline(never)]
pub fn check() -> u32 {
    let s = core::hint::black_box(make_struct());
    (s.payload == EXPECTED) as u32
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
    println!("=== misaligned_struct_field_array_eq_repro ===");

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
    println!("cpu   result     = {} (sanity: must be 1)", cpu_result);

    assert_eq!(cpu_result, 1, "CPU path itself disagrees — repro is wrong");

    if host_result[0] != 1 {
        eprintln!();
        eprintln!("FAIL: device-side `struct.payload == EXPECTED` did not return 1.");
        eprintln!("      (Or the kernel faulted before reaching the comparison.)");
        std::process::exit(1);
    }

    println!();
    println!("SUCCESS");
    println!("PASS");
}
