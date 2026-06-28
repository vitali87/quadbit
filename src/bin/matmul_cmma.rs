//! Tensor-core (cmma) CubeCL matmul on SM120. Each warp computes one 16x16
//! output tile: it loops over k in 16-wide steps, loading an A and B fragment
//! straight from global memory and issuing one `cmma::execute` (a warp-wide
//! 16x16x16 multiply-accumulate) per step into an f32 accumulator fragment.
//! Inputs are f16 all-ones, accumulator is f32, so each output must equal n
//! exactly (the f32 accumulator holds 2048 with no rounding).

use std::time::Duration;

use cubecl::cmma;
use cubecl::future;
use cubecl::prelude::*;
use half::f16;

const M: usize = 16; // tensor-core fragment dims (the one CUDA WMMA shape we use)

#[cube(launch_unchecked)]
fn cmma_matmul(a: &Array<f16>, b: &Array<f16>, out: &mut Array<f32>, n: u32) {
    let nn = n as usize;
    let tile_row = CUBE_POS_Y as usize * M;
    let tile_col = CUBE_POS_X as usize * M;

    // accumulator fragment, zeroed; layout is irrelevant until we store it
    let acc = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        M,
        M,
        M,
        cmma::MatrixLayout::Undefined,
        0.0f32,
    );

    let steps = nn / M;
    for s in 0..steps {
        let k = s * M;

        // A fragment: 16x16 window of row-major A at (tile_row, k), row stride n
        let a_start = tile_row * nn + k;
        let a_frag = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A,
            M,
            M,
            M,
            cmma::MatrixLayout::RowMajor,
            &a.slice(a_start, a_start + M * nn),
            n,
        );

        // B fragment: 16x16 window of row-major B at (k, tile_col), row stride n
        let b_start = k * nn + tile_col;
        let b_frag = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B,
            M,
            M,
            M,
            cmma::MatrixLayout::RowMajor,
            &b.slice(b_start, b_start + M * nn),
            n,
        );

        // acc += a_frag * b_frag  (D = A*B + C with acc as both C and D)
        cmma::execute::<f16, f16, f32, f32>(&a_frag, &b_frag, &acc, &acc);
    }

    let out_start = tile_row * nn + tile_col;
    cmma::store(
        &mut out.slice_mut(out_start, out_start + M * nn),
        &acc,
        n,
        cmma::MatrixLayout::RowMajor,
    );
}

fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let n: u32 = 2048;
    let count = (n * n) as usize;
    let ones: Vec<f16> = vec![f16::from_f32(1.0); count];

    let a = client.create_from_slice(f16::as_bytes(&ones));
    let b = client.create_from_slice(f16::as_bytes(&ones));
    let out = client.empty(count * core::mem::size_of::<f32>());

    let grid = n / (M as u32);

    // warmup: pays the NVRTC JIT on the first launch and lets clocks boost
    for _ in 0..5 {
        unsafe {
            cmma_matmul::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid, grid, 1),
                CubeDim::new_1d(32),
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
                    cmma_matmul::launch_unchecked::<R>(
                        &client,
                        CubeCount::Static(grid, grid, 1),
                        CubeDim::new_1d(32),
                        ArrayArg::from_raw_parts(a.clone(), count),
                        ArrayArg::from_raw_parts(b.clone(), count),
                        ArrayArg::from_raw_parts(out.clone(), count),
                        n,
                    );
                },
                "cmma_matmul",
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
