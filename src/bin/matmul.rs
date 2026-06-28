//! Naive CubeCL matmul: the Rust baseline GEMM on SM120. One thread per output
//! element, every read straight from global memory. Not a target kernel; it
//! establishes the honest baseline and the timing path. Inputs are all 1.0 so
//! every output element must equal n (exact correctness check from the size).

use std::time::Duration;

use cubecl::future;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn naive_matmul<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>, n: u32) {
    let row = ABSOLUTE_POS_Y;
    let col = ABSOLUTE_POS_X;
    if row < n && col < n {
        let nn = n as usize;
        let r = row as usize;
        let c = col as usize;
        let mut acc = F::new(0.0);
        for k in 0..nn {
            acc += a[r * nn + k] * b[k * nn + c];
        }
        out[r * nn + c] = acc;
    }
}

fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let n: u32 = 2048;
    let count = (n * n) as usize;
    let ones = vec![1.0f32; count];

    let a = client.create_from_slice(f32::as_bytes(&ones));
    let b = client.create_from_slice(f32::as_bytes(&ones));
    let out = client.empty(count * core::mem::size_of::<f32>());

    let tile = 16u32;
    let grid = n.div_ceil(tile);

    // warmup: pays the NVRTC JIT on the first launch and lets clocks boost
    for _ in 0..5 {
        unsafe {
            naive_matmul::launch_unchecked::<f32, R>(
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
                    naive_matmul::launch_unchecked::<f32, R>(
                        &client,
                        CubeCount::Static(grid, grid, 1),
                        CubeDim::new_2d(tile, tile),
                        ArrayArg::from_raw_parts(a.clone(), count),
                        ArrayArg::from_raw_parts(b.clone(), count),
                        ArrayArg::from_raw_parts(out.clone(), count),
                        n,
                    );
                },
                "naive_matmul",
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
