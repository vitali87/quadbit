//! Milestone 2 mechanics probe: validate a depth-2 TMA software pipeline (two
//! `full` mbarriers, prologue prefetch, main loop of wait/consume/re-arm with
//! buffer reuse) on a trivial tile-copy task before wiring it into the matmul.
//! Loads NTILES consecutive [ROWS,COLS] tiles of a [ROWS, NTILES*COLS] tensor
//! through the pipeline and writes each out; verifies every tile arrives.
//! If this PASSES, the barrier/token protocol is correct and portable to the
//! matmul mainloop; if it hangs, the parity/token handling is wrong.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;
use cubecl::e2m1x2;
use cubecl::ir::features::Tma;

const ROWS: usize = 16;
const COLS: usize = 32; // bytes; 16-aligned so 1-byte TMA offsets are legal
const NTILES: usize = 8;
const TILE: usize = ROWS * COLS;

#[cube(launch)]
fn pipe_probe<N: Size>(input: &TensorMap<e2m1x2, Tiled>, output: &mut Array<Vector<e2m1x2, N>>, #[comptime] ntiles: usize) {
    let full0 = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    let full1 = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
    sync_async_proxy_shared();
    let mut buf = SharedMemory::<Vector<e2m1x2, N>>::new_aligned(2 * TILE, 128usize);

    let bytes = comptime!(TILE as u32);
    let tid = UNIT_POS_Y * COLS as u32 + UNIT_POS_X;

    // prologue: prefetch tiles 0 and 1 into buffers 0 and 1
    let e0 = select(UNIT_POS == 0, bytes, 0u32);
    if UNIT_POS == 0 {
        let mut s = buf.slice_mut(0, TILE);
        full0.tma_load_2d(input, &mut s, 0, 0);
        let mut s1 = buf.slice_mut(TILE, 2 * TILE);
        full1.tma_load_2d(input, &mut s1, 0, COLS as i32);
    }
    let mut token0 = full0.arrive_and_expect_tx(1, e0);
    let mut token1 = full1.arrive_and_expect_tx(1, select(UNIT_POS == 0, bytes, 0u32));

    for t in 0..ntiles {
        let cur = t % 2;
        let base = cur * TILE;
        if cur == 0 {
            full0.wait(token0);
        } else {
            full1.wait(token1);
        }

        output[(t * TILE) + tid as usize] = buf[base + tid as usize];
        sync_cube(); // all threads done reading buf[base] before it is reused

        let next = t + 2;
        if next < ntiles {
            let col = (next * COLS) as i32;
            let e = select(UNIT_POS == 0, bytes, 0u32);
            if cur == 0 {
                if UNIT_POS == 0 {
                    let mut s = buf.slice_mut(0, TILE);
                    full0.tma_load_2d(input, &mut s, 0, col);
                }
                token0 = full0.arrive_and_expect_tx(1, e);
            } else {
                if UNIT_POS == 0 {
                    let mut s = buf.slice_mut(TILE, 2 * TILE);
                    full1.tma_load_2d(input, &mut s, 0, col);
                }
                token1 = full1.arrive_and_expect_tx(1, e);
            }
        }
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(not(feature = "cuda"))]
    panic!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    if !client.features().tma.contains(Tma::Base) {
        println!("TMA not supported; skipping");
        return;
    }
    let w = NTILES * COLS;
    let src_f: Vec<f32> = (0..ROWS * w * 2)
        .map(|i| [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][i % 8])
        .collect();
    let src = e2m1x2::from_f32_slice(&src_f); // ROWS*w packed bytes
    let handle = client.create_from_slice(e2m1x2::as_bytes(&src));
    let input = unsafe { TensorArg::from_raw_parts(handle, [w, 1].into(), [ROWS, w].into()) };
    let out = client.empty(NTILES * TILE * core::mem::size_of::<e2m1x2>());

    pipe_probe::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_2d(COLS as u32, ROWS as u32),
        1,
        TensorMapArg::new(TiledArgs { tile_size: [ROWS, COLS].into() }, input, u8::as_type_native_unchecked()),
        unsafe { ArrayArg::from_raw_parts(out.clone(), NTILES * TILE) },
        NTILES,
    );

    let bytes = client.read_one(out).unwrap();
    let got = e2m1x2::from_bytes(&bytes);
    let mut wrong = 0;
    for t in 0..NTILES {
        for r in 0..ROWS {
            for c in 0..COLS {
                let exp = src[r * w + (t * COLS + c)];
                if got[t * TILE + r * COLS + c] != exp {
                    wrong += 1;
                }
            }
        }
    }
    println!("runtime: {:?}", R::name(&client));
    println!("pipe_probe: {} ({wrong} wrong of {})", if wrong == 0 { "PASS" } else { "FAIL" }, NTILES * TILE);
}
