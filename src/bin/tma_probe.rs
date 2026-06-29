//! Minimal TMA smoke test: does `TensorMap` + `Barrier::tma_load_2d` work for our
//! packed FP4 element (`e2m1x2`)? Mirrors the CubeCL `tensormap_load` runtime test
//! exactly, swapping the element type. Loads a [16,32] tile of e2m1x2 from a
//! [64,64] tensor at column offset 8 into 128-byte-aligned shared, copies it out,
//! and checks the staged bytes match the source slice. If this hangs, TMA is the
//! problem for our element type; if it passes, the deadlock in matmul_fp4_pipe is
//! in the loop / barrier reuse, not the TMA primitive.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;
use cubecl::{e2m1x2, ir::features::Tma};

const ROWS: usize = 16;
const COLS: usize = 32; // e2m1x2 elements (bytes)
const COL_OFF: i32 = 16; // 1-byte elements: innermost byte offset must be 16-aligned

#[cube(launch)]
fn tma_probe<N: Size>(input: &TensorMap<e2m1x2, Tiled>, output: &mut Array<Vector<e2m1x2, N>>) {
    let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut stage = SharedMemory::<Vector<e2m1x2, N>>::new_aligned(ROWS * COLS, 128usize);

    // e2m1x2 is 1 byte, so transaction byte count == element count
    let expected = select(UNIT_POS == 0, comptime!(ROWS * COLS) as u32, 0u32);
    if UNIT_POS == 0 {
        barrier.tma_load_2d(input, &mut stage.to_slice_mut(), 0, COL_OFF);
    }
    let token = barrier.arrive_and_expect_tx(1, expected);
    barrier.wait(token);

    let out_pos = UNIT_POS_Y * COLS as u32 + UNIT_POS_X;
    output[out_pos as usize] = stage[out_pos as usize];
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "cpu")]
    run::<cubecl::cpu::CpuRuntime>(&Default::default());
    #[cfg(not(any(feature = "cuda", feature = "cpu")))]
    panic!("build with --features cuda or --features cpu");
}

#[cfg(feature = "cuda")]
fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    if !client.features().tma.contains(Tma::Base) {
        println!("TMA not supported on this device; skipping");
        return;
    }

    let (h, w) = (64usize, 64usize); // source tensor in e2m1x2 elements
    // representable fp4 byte patterns so create/readback is lossless
    let src_f: Vec<f32> = (0..h * w * 2)
        .map(|i| [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][i % 8])
        .collect();
    let src = e2m1x2::from_f32_slice(&src_f); // h*w packed bytes
    let handle = client.create_from_slice(e2m1x2::as_bytes(&src));
    let input = unsafe { TensorArg::from_raw_parts(handle, [w, 1].into(), [h, w].into()) };
    let out = client.empty(ROWS * COLS * core::mem::size_of::<e2m1x2>());

    tma_probe::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(COLS as u32, ROWS as u32),
        1,
        TensorMapArg::new(
            TiledArgs { tile_size: [ROWS, COLS].into() },
            input,
            // e2m1x2's native type is an invalid CUtensorMap dtype (illegal-instruction
            // at runtime); describe the 1-byte packed data as u8, same byte layout.
            u8::as_type_native_unchecked(),
        ),
        unsafe { ArrayArg::from_raw_parts(out.clone(), ROWS * COLS) },
    );

    let bytes = client.read_one(out).unwrap();
    let got = e2m1x2::from_bytes(&bytes);
    let mut wrong = 0;
    for r in 0..ROWS {
        for c in 0..COLS {
            let exp = src[(r) * w + (COL_OFF as usize + c)];
            if got[r * COLS + c] != exp {
                wrong += 1;
            }
        }
    }
    println!("runtime: {:?}", R::name(&client));
    println!("tma_probe: {} ({wrong} wrong of {})", if wrong == 0 { "PASS" } else { "FAIL" }, ROWS * COLS);
}
