//! Register-blocked tiled CubeCL matmul on SM120. Each block stages a BM x BK
//! tile of A and a BK x BN tile of B into shared memory; each of the TX x TY
//! threads computes an RM x RN patch of outputs held in registers, so each
//! shared-memory load feeds many FMAs (outer-product accumulation). Same
//! all-ones inputs as the earlier baselines, so each output must still equal n.

use std::time::Duration;

use cubecl::future;
use cubecl::prelude::*;

const BM: usize = 64; // output tile rows per block
const BN: usize = 64; // output tile cols per block
const BK: usize = 16; // k-depth staged per step
const RM: usize = 4; // output rows per thread
const RN: usize = 4; // output cols per thread
const TX: u32 = (BN / RN) as u32; // 16 threads in x
const TY: u32 = (BM / RM) as u32; // 16 threads in y

#[cube(launch_unchecked)]
fn regblock_matmul<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>, n: u32) {
    let mut a_tile = SharedMemory::<F>::new(BM * BK);
    let mut b_tile = SharedMemory::<F>::new(BK * BN);

    let nn = n as usize;
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let n_threads = (TX * TY) as usize;
    let thread_id = ty * (TX as usize) + tx;

    // origin of this block's output tile in global coords
    let block_row = CUBE_POS_Y as usize * BM;
    let block_col = CUBE_POS_X as usize * BN;

    // this thread's patch origin within the block tile
    let patch_row = ty * RM;
    let patch_col = tx * RN;

    let mut acc = Array::<F>::new(RM * RN);
    for i in 0..RM * RN {
        acc[i] = F::new(0.0);
    }

    let steps = nn / BK;
    for s in 0..steps {
        let k_base = s * BK;

        // load phase: 256 threads cooperatively fill the two shared tiles.
        // A-tile is BM*BK elements; each thread loads BM*BK / n_threads of them.
        for i in 0..(BM * BK) / n_threads {
            let idx = thread_id + i * n_threads;
            let r = idx / BK;
            let c = idx % BK;
            a_tile[idx] = a[(block_row + r) * nn + (k_base + c)];
        }
        for i in 0..(BK * BN) / n_threads {
            let idx = thread_id + i * n_threads;
            let r = idx / BN;
            let c = idx % BN;
            b_tile[idx] = b[(k_base + r) * nn + (block_col + c)];
        }
        sync_cube();

        // compute phase: outer-product accumulation out of registers
        #[unroll]
        for k in 0..BK {
            let mut a_reg = Array::<F>::new(RM);
            let mut b_reg = Array::<F>::new(RN);
            #[unroll]
            for i in 0..RM {
                a_reg[i] = a_tile[(patch_row + i) * BK + k];
            }
            #[unroll]
            for j in 0..RN {
                b_reg[j] = b_tile[k * BN + (patch_col + j)];
            }
            #[unroll]
            for i in 0..RM {
                #[unroll]
                for j in 0..RN {
                    acc[i * RN + j] += a_reg[i] * b_reg[j];
                }
            }
        }
        sync_cube();
    }

    // write phase: each thread stores its RM x RN patch
    #[unroll]
    for i in 0..RM {
        #[unroll]
        for j in 0..RN {
            let r = block_row + patch_row + i;
            let c = block_col + patch_col + j;
            out[r * nn + c] = acc[i * RN + j];
        }
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

    let grid = n / (BM as u32);

    // warmup: pays the NVRTC JIT on the first launch and lets clocks boost
    for _ in 0..5 {
        unsafe {
            regblock_matmul::launch_unchecked::<f32, R>(
                &client,
                CubeCount::Static(grid, grid, 1),
                CubeDim::new_2d(TX, TY),
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
                    regblock_matmul::launch_unchecked::<f32, R>(
                        &client,
                        CubeCount::Static(grid, grid, 1),
                        CubeDim::new_2d(TX, TY),
                        ArrayArg::from_raw_parts(a.clone(), count),
                        ArrayArg::from_raw_parts(b.clone(), count),
                        ArrayArg::from_raw_parts(out.clone(), count),
                        n,
                    );
                },
                "regblock_matmul",
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
