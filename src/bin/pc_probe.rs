//! Warp-specialization handshake probe: validate a dedicated TMA-producer warp
//! feeding a consumer warp through a 2-stage ring buffer with full/empty mbarriers
//! and phase-based `wait_parity` (cross-warp, no shared token). Producer TMAs tile
//! t into ring buf[t%2] and arrives `full`; consumer waits `full`, drains buf to
//! output[t], arrives `empty`; producer waits `empty` before reusing a buffer.
//! Verifies every tile arrives. If it hangs, the parity convention is flipped.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;
use cubecl::e2m1x2;
use cubecl::ir::features::Tma;

const ROWS: usize = 16;
const COLS: usize = 32; // bytes; 16-aligned
const TILE: usize = ROWS * COLS; // 512
const NTILES: usize = 8;

#[cube(launch)]
fn pc_probe(input: &TensorMap<e2m1x2, Tiled>, output: &mut Array<e2m1x2>, #[comptime] ntiles: usize) {
    let warp = UNIT_POS_X / 32;
    let lane = UNIT_POS_X % 32;
    let bytes = comptime!(TILE as u32);

    let full0 = Barrier::shared(1, UNIT_POS_X == 0);
    let full1 = Barrier::shared(1, UNIT_POS_X == 0);
    let empty0 = Barrier::shared(32, UNIT_POS_X == 0);
    let empty1 = Barrier::shared(32, UNIT_POS_X == 0);
    sync_async_proxy_shared();
    let mut buf = SharedMemory::<e2m1x2>::new_aligned(2 * TILE, 128usize);

    if warp == 1 {
        // producer warp: only the elected lane issues TMA + arrives
        if lane == 0 {
            // prologue: fill both ring stages
            let mut s0 = buf.slice_mut(0, TILE);
            full0.tma_load_2d(input, &mut s0, 0, 0);
            full0.arrive_and_expect_tx(1, bytes);
            let mut s1 = buf.slice_mut(TILE, 2 * TILE);
            full1.tma_load_2d(input, &mut s1, 0, COLS as i32);
            full1.arrive_and_expect_tx(1, bytes);
            // steady state
            for step in 2..ntiles {
                let col = (step * COLS) as i32;
                if step % 2 == 0 {
                    empty0.wait_parity(((step / 2 - 1) % 2) as u32);
                    let mut s = buf.slice_mut(0, TILE);
                    full0.tma_load_2d(input, &mut s, 0, col);
                    full0.arrive_and_expect_tx(1, bytes);
                } else {
                    empty1.wait_parity(((step / 2 - 1) % 2) as u32);
                    let mut s = buf.slice_mut(TILE, 2 * TILE);
                    full1.tma_load_2d(input, &mut s, 0, col);
                    full1.arrive_and_expect_tx(1, bytes);
                }
            }
        }
    } else {
        // consumer warp 0: all 32 lanes wait full, drain, arrive empty
        for step in 0..ntiles {
            let parity = ((step / 2) % 2) as u32;
            let base = (step % 2) * TILE;
            if step % 2 == 0 {
                full0.wait_parity(parity);
            } else {
                full1.wait_parity(parity);
            }
            #[unroll]
            for j in 0..comptime!(TILE / 32) {
                let idx = j * 32 + lane as usize;
                output[step * TILE + idx] = buf[base + idx];
            }
            if step % 2 == 0 {
                empty0.arrive();
            } else {
                empty1.arrive();
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
    let src: Vec<e2m1x2> = (0..ROWS * w).map(|i| e2m1x2::from_bits((i % 251 + 1) as u8)).collect();
    let handle = client.create_from_slice(e2m1x2::as_bytes(&src));
    let input = unsafe { TensorArg::from_raw_parts(handle, [w, 1].into(), [ROWS, w].into()) };
    let out = client.empty(NTILES * TILE);

    pc_probe::launch::<R>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(64),
        TensorMapArg::new(TiledArgs { tile_size: [ROWS, COLS].into() }, input, u8::as_type_native_unchecked()),
        unsafe { ArrayArg::from_raw_parts(out.clone(), NTILES * TILE) },
        NTILES,
    );

    let ob = client.read_one(out).unwrap();
    let got = e2m1x2::from_bytes(&ob);
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
    println!("pc_probe: {} ({wrong} wrong of {})", if wrong == 0 { "PASS" } else { "FAIL" }, NTILES * TILE);
}
