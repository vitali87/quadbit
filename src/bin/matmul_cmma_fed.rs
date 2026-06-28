//! Properly-fed tensor-core (cmma) CubeCL matmul on SM120. A block of 4 warps
//! (2x2) computes a 64x64 output tile; per k-step it stages a 64x16 tile of A
//! and a 16x64 tile of B into shared memory, then each warp loads 2 A-fragments
//! and 2 B-fragments from shared memory and issues 4 16x16x16 MMAs into its 2x2
//! grid of f32 accumulators. Shared-memory staging + multi-warp occupancy +
//! fragment-level reuse keep the tensor cores fed. f16 all-ones inputs, f32
//! accumulators, so each output must equal n exactly.

use std::time::Duration;

use cubecl::cmma;
use cubecl::future;
use cubecl::prelude::*;
use half::f16;

const MMA: usize = 16; // tensor-core fragment dim (m=n=k=16)
const BM: usize = 64; // block output rows
const BN: usize = 64; // block output cols
const BK: usize = 16; // k staged per step (one fragment deep)
const WARPS_N: usize = 2; // warps across the block in n
const WM: usize = 2; // fragments per warp in m (BM / MMA / WARPS_M)
const WN: usize = 2; // fragments per warp in n (BN / MMA / WARPS_N)
const N_THREADS: u32 = 128; // 4 warps

#[cube(launch_unchecked)]
fn cmma_fed(a: &Array<f16>, b: &Array<f16>, out: &mut Array<f32>, n: u32) {
    let nn = n as usize;
    let mut a_tile = SharedMemory::<f16>::new(BM * BK);
    let mut b_tile = SharedMemory::<f16>::new(BK * BN);

    let tid = UNIT_POS_X as usize;
    let plane = tid / 32; // warp id within block
    let wm = plane / WARPS_N; // warp row in the 2x2 warp grid
    let wn = plane % WARPS_N; // warp col

    let block_row = CUBE_POS_Y as usize * BM;
    let block_col = CUBE_POS_X as usize * BN;

    // this warp's 2x2 grid of accumulator fragments, zeroed, live across all k
    let acc00 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, MMA, MMA, MMA, cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc01 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, MMA, MMA, MMA, cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc10 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, MMA, MMA, MMA, cmma::MatrixLayout::Undefined, 0.0f32,
    );
    let acc11 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator, MMA, MMA, MMA, cmma::MatrixLayout::Undefined, 0.0f32,
    );

    let n_threads = N_THREADS as usize;
    let steps = nn / BK;
    for s in 0..steps {
        let k_base = s * BK;

        // cooperative load: 128 threads fill the 64x16 A-tile (1024 elems)
        for i in 0..(BM * BK) / n_threads {
            let idx = tid + i * n_threads;
            let r = idx / BK;
            let c = idx % BK;
            a_tile[idx] = a[(block_row + r) * nn + (k_base + c)];
        }
        // and the 16x64 B-tile (1024 elems)
        for i in 0..(BK * BN) / n_threads {
            let idx = tid + i * n_threads;
            let r = idx / BN;
            let c = idx % BN;
            b_tile[idx] = b[(k_base + r) * nn + (block_col + c)];
        }
        sync_cube();

        // load this warp's A-fragments (rows) from shared A-tile, stride BK
        let a0_start = (wm * WM) * MMA * BK;
        let a1_start = (wm * WM + 1) * MMA * BK;
        let a0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A, MMA, MMA, MMA, cmma::MatrixLayout::RowMajor,
            &a_tile.slice(a0_start, a0_start + MMA * BK), BK as u32,
        );
        let a1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::A, MMA, MMA, MMA, cmma::MatrixLayout::RowMajor,
            &a_tile.slice(a1_start, a1_start + MMA * BK), BK as u32,
        );

        // load this warp's B-fragments (cols) from shared B-tile, stride BN
        let b0_start = (wn * WN) * MMA;
        let b1_start = (wn * WN + 1) * MMA;
        let b0 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B, MMA, MMA, MMA, cmma::MatrixLayout::RowMajor,
            &b_tile.slice(b0_start, b0_start + MMA * BN), BN as u32,
        );
        let b1 = cmma::Matrix::<f16>::from_slice(
            cmma::MatrixIdent::B, MMA, MMA, MMA, cmma::MatrixLayout::RowMajor,
            &b_tile.slice(b1_start, b1_start + MMA * BN), BN as u32,
        );

        // 4 MMAs: each loaded fragment feeds two accumulates
        cmma::execute::<f16, f16, f32, f32>(&a0, &b0, &acc00, &acc00);
        cmma::execute::<f16, f16, f32, f32>(&a0, &b1, &acc01, &acc01);
        cmma::execute::<f16, f16, f32, f32>(&a1, &b0, &acc10, &acc10);
        cmma::execute::<f16, f16, f32, f32>(&a1, &b1, &acc11, &acc11);
        sync_cube();
    }

    // store the 2x2 fragment grid to global out, stride n
    let r0 = block_row + (wm * WM) * MMA;
    let r1 = block_row + (wm * WM + 1) * MMA;
    let c0 = block_col + (wn * WN) * MMA;
    let c1 = block_col + (wn * WN + 1) * MMA;
    let s00 = r0 * nn + c0;
    let s01 = r0 * nn + c1;
    let s10 = r1 * nn + c0;
    let s11 = r1 * nn + c1;
    cmma::store(&mut out.slice_mut(s00, s00 + MMA * nn), &acc00, n, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut out.slice_mut(s01, s01 + MMA * nn), &acc01, n, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut out.slice_mut(s10, s10 + MMA * nn), &acc10, n, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut out.slice_mut(s11, s11 + MMA * nn), &acc11, n, cmma::MatrixLayout::RowMajor);
}

fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let n: u32 = 2048;
    let count = (n * n) as usize;
    let ones: Vec<f16> = vec![f16::from_f32(1.0); count];

    let a = client.create_from_slice(f16::as_bytes(&ones));
    let b = client.create_from_slice(f16::as_bytes(&ones));
    let out = client.empty(count * core::mem::size_of::<f32>());

    let grid = n / (BM as u32);

    // warmup: pays the NVRTC JIT on the first launch and lets clocks boost
    for _ in 0..5 {
        unsafe {
            cmma_fed::launch_unchecked::<R>(
                &client,
                CubeCount::Static(grid, grid, 1),
                CubeDim::new_1d(N_THREADS),
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
                    cmma_fed::launch_unchecked::<R>(
                        &client,
                        CubeCount::Static(grid, grid, 1),
                        CubeDim::new_1d(N_THREADS),
                        ArrayArg::from_raw_parts(a.clone(), count),
                        ArrayArg::from_raw_parts(b.clone(), count),
                        ArrayArg::from_raw_parts(out.clone(), count),
                        n,
                    );
                },
                "cmma_fed",
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
