//! Shared-memory tiled CubeCL matmul on SM120. Each block computes one TILE x
//! TILE output square; the k dimension is walked in TILE-wide steps, staging one
//! tile of A and one of B into shared memory per step so every loaded element is
//! reused by TILE threads. Same all-ones inputs as the naive baseline, so each
//! output element must still equal n (exact correctness check from the size).

use std::time::Duration;

use cubecl::future;
use cubecl::prelude::*;

const TILE: usize = 16;

#[cube(launch_unchecked)]
fn tiled_matmul<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>, n: u32) {
    let mut a_tile = SharedMemory::<F>::new(TILE * TILE);
    let mut b_tile = SharedMemory::<F>::new(TILE * TILE);

    let tile = TILE;
    let nn = n as usize;
    let ty = UNIT_POS_Y as usize;
    let tx = UNIT_POS_X as usize;
    let row = CUBE_POS_Y as usize * tile + ty;
    let col = CUBE_POS_X as usize * tile + tx;

    let mut acc = F::new(0.0);

    let steps = nn / tile;
    for s in 0..steps {
        let k_base = s * tile;
        // load phase: each thread stages one element of each tile
        a_tile[ty * tile + tx] = a[row * nn + (k_base + tx)];
        b_tile[ty * tile + tx] = b[(k_base + ty) * nn + col];
        sync_cube();

        // compute phase: partial dot product out of shared memory
        #[unroll]
        for k in 0..tile {
            acc += a_tile[ty * tile + k] * b_tile[k * tile + tx];
        }
        sync_cube();
    }

    out[row * nn + col] = acc;
}

fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let n: u32 = 2048;
    let count = (n * n) as usize;
    let ones = vec![1.0f32; count];

    let a = client.create_from_slice(f32::as_bytes(&ones));
    let b = client.create_from_slice(f32::as_bytes(&ones));
    let out = client.empty(count * core::mem::size_of::<f32>());

    let tile = TILE as u32;
    let grid = n / tile;

    // warmup: pays the NVRTC JIT on the first launch and lets clocks boost
    for _ in 0..5 {
        unsafe {
            tiled_matmul::launch_unchecked::<f32, R>(
                &client,
                CubeCount::Static(grid, grid, 1),
                CubeDim::new_2d(tile, tile),
                ArrayArg::from_raw_parts(a.clone(), count),
                ArrayArg::from_raw_parts(b.clone(), count),
                ArrayArg::from_raw_parts(out.clone(), count),
                n,
            );
        }
    }
    let _ = future::block_on(client.sync());

    let mut best = Duration::MAX;
    for _ in 0..20 {
        let (_, profile) = client
            .profile(
                || unsafe {
                    tiled_matmul::launch_unchecked::<f32, R>(
                        &client,
                        CubeCount::Static(grid, grid, 1),
                        CubeDim::new_2d(tile, tile),
                        ArrayArg::from_raw_parts(a.clone(), count),
                        ArrayArg::from_raw_parts(b.clone(), count),
                        ArrayArg::from_raw_parts(out.clone(), count),
                        n,
                    );
                },
                "tiled_matmul",
            )
            .unwrap();
        let elapsed = future::block_on(profile.resolve()).duration();
        if elapsed < best {
            best = elapsed;
        }
    }

    let bytes = client.read_one(out).unwrap();
    let result = f32::from_bytes(&bytes);
    let mut wrong = 0;
    for v in result.iter() {
        if *v != n as f32 {
            wrong += 1;
        }
    }

    let secs = best.as_secs_f64();
    let gflops = 2.0 * (n as f64).powi(3) / secs / 1e9;
    println!("runtime: {:?}", R::name(&client));
    println!("shape: {n}x{n}x{n}");
    println!("best: {:.3} ms   {gflops:.1} GFLOP/s", secs * 1e3);
    println!(
        "correctness: {} ({wrong} wrong, expected each = {n})",
        if wrong == 0 { "PASS" } else { "FAIL" }
    );
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "cpu")]
    run::<cubecl::cpu::CpuRuntime>(&Default::default());
    #[cfg(not(any(feature = "cuda", feature = "cpu")))]
    panic!("build with --features cuda (on a GPU) or --features cpu");
}
