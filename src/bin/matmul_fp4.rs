//! Full block-scaled FP4 (MXFP4) matmul on SM120, built up from the single-MMA
//! hello kernel. One warp owns one 16x8 output tile and walks the full K in
//! steps of 64 (one `mma.sync...kind::mxf4nvf4.block_scale` per step), feeding
//! the f32 accumulator registers back in as C so the products accumulate across
//! K. Register and per-block-scale plumbing is the same hand-managed
//! `position_of_nth`/`scales_index` path the hello kernel uses, since the
//! high-level WMMA `Matrix` API has no FP4. A/B fragments are read straight from
//! global FP4 memory (the naive-but-correct rung; shared staging comes next).
//! All-ones FP4 inputs with unit (2^0) UE8M0 scales, so each output equals K.

use std::time::Duration;

use cubecl::features::ScaledMmaConfig;
use cubecl::future;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::{e2m1x2, ue8m0};

const TM: usize = 16; // MMA tile rows (m)
const TN: usize = 8; // MMA tile cols (n)
const TK: usize = 64; // MMA tile depth (k), two scale blocks of 32
const SF: usize = 2; // scales_factor: scale blocks per TK along k

#[cube(launch_unchecked)]
fn matmul_fp4<A: Scalar, B: Scalar, CD: Numeric, S: Scalar, NA: Size, NB: Size, NC: Size>(
    a: &Tensor<Vector<A, NA>>,
    b: &Tensor<Vector<B, NB>>,
    c: &Tensor<Vector<CD, NC>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Vector<CD, NC>>,
    #[comptime] full_n: usize,
    #[comptime] full_k: usize,
) {
    let a_pack = A::packing_factor();
    let b_pack = B::packing_factor();

    let def = cmma::MmaDefinition::<A, B, CD>::new_scaled::<S>(TM, TN, TK, SF);
    let lane_id = UNIT_POS_PLANE;

    // this warp's output tile in the global grid
    let tile_row = CUBE_POS_Y as usize * TM;
    let tile_col = CUBE_POS_X as usize * TN;

    let k_steps = comptime!(full_k / TK);
    let scale_blocks_per_row = comptime!(full_k / (TK / SF)); // = full_k / 32

    let elem_count_a = def.elems_per_lane(MatrixIdent::A);
    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_count_a = comptime!(elem_count_a / vector_size_a);
    let mut registers_a = Array::<Vector<A, NA>>::new(vector_count_a);

    let elem_count_b = def.elems_per_lane(MatrixIdent::B);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let vector_count_b = comptime!(elem_count_b / vector_size_b);
    let mut registers_b = Array::<Vector<B, NB>>::new(vector_count_b);

    let elem_count_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vector_size_c = def.vector_size(MatrixIdent::Accumulator);
    let vector_count_c = comptime!(elem_count_c / vector_size_c);

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();
    let mut scales_register_a = Vector::<S, NS>::empty();
    let mut scales_register_b = Vector::<S, NS>::empty();

    // accumulator registers for this tile, zero-initialised from the C tensor,
    // kept live across every k-step
    let mut acc = Array::<Vector<CD, NC>>::new(vector_count_c);
    #[unroll]
    for i in 0..vector_count_c {
        let n_elem = i * vector_size_c;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
        let idx = (tile_row + row as usize) * full_n + (tile_col + col as usize);
        acc[i] = c[idx / c.vector_size()];
    }

    let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A);
    let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B);

    for step in 0..k_steps {
        let k_base = step * TK;
        let scale_base = step * SF;

        // Load A fragment (16x64) straight from global FP4
        #[unroll]
        for i in 0..vector_count_a {
            let n_elem = i * vector_size_a * a_pack;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
            let idx = (tile_row + row as usize) * full_k + (k_base + col as usize);
            registers_a[i] = a[idx / (a.vector_size() * a_pack)];
        }
        #[unroll]
        for i in 0..scales_count {
            let idx = (tile_row + scales_idx_a as usize) * scale_blocks_per_row + scale_base + i;
            scales_register_a[i] = scales_a[idx];
        }

        // Load B fragment (64x8); B is stored [n, k] so n is the slow axis
        #[unroll]
        for i in 0..vector_count_b {
            let n_elem = i * vector_size_b * b_pack;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
            let idx = (tile_col + col as usize) * full_k + (k_base + row as usize);
            registers_b[i] = b[idx / (b.vector_size() * b_pack)];
        }
        #[unroll]
        for i in 0..scales_count {
            let idx = (tile_col + scales_idx_b as usize) * scale_blocks_per_row + scale_base + i;
            scales_register_b[i] = scales_b[idx];
        }

        let registers_d = def.execute_scaled(
            &registers_a,
            &registers_b,
            &acc,
            scales_register_a,
            scales_register_b,
        );
        #[unroll]
        for i in 0..vector_count_c {
            acc[i] = registers_d[i];
        }
    }

    // Store the accumulated tile to global out
    #[unroll]
    for i in 0..vector_count_c {
        let n_elem = i * vector_size_c;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
        let idx = (tile_row + row as usize) * full_n + (tile_col + col as usize);
        out[idx / out.vector_size()] = acc[i];
    }
}

fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let (m, n, k) = (2048usize, 2048usize, 2048usize);

    type AB = e2m1x2;
    type S = ue8m0;
    let ab_elem = AB::cube_type();
    let ab_vector_size = 32 / ab_elem.size_bits();

    let supported = client.features().matmul.scaled_mma.contains(&ScaledMmaConfig {
        a_type: ab_elem,
        b_type: ab_elem,
        cd_type: f32::cube_type(),
        scales_type: S::cube_type(),
        m: TM as u32,
        n: TN as u32,
        k: TK as u32,
        scales_factor: SF as u32,
    });
    if !supported {
        println!("runtime: {:?}", R::name(&client));
        println!("scaled FP4 MMA NOT supported on this device; skipping");
        return;
    }

    let scale_cols = k / (TK / SF); // scale blocks per row = k / 32

    // all-ones FP4 inputs; unit (2^0) UE8M0 block scales
    let lhs = e2m1x2::from_f32_slice(&vec![1.0f32; m * k]);
    let rhs = e2m1x2::from_f32_slice(&vec![1.0f32; n * k]);
    let unit = ue8m0::from_bits(127);
    let lhs_scales: Vec<S> = vec![unit; m * scale_cols];
    let rhs_scales: Vec<S> = vec![unit; n * scale_cols];
    let zeros = vec![0.0f32; m * n];

    let lhs_h = client.create_from_slice(AB::as_bytes(&lhs));
    let rhs_h = client.create_from_slice(AB::as_bytes(&rhs));
    let lhs_scales_h = client.create_from_slice(S::as_bytes(&lhs_scales));
    let rhs_scales_h = client.create_from_slice(S::as_bytes(&rhs_scales));
    let c_h = client.create_from_slice(f32::as_bytes(&zeros));
    let out_h = client.empty(m * n * core::mem::size_of::<f32>());

    let grid_x = (n / TN) as u32;
    let grid_y = (m / TM) as u32;

    let launch = |out_buf: cubecl::server::Handle| unsafe {
        matmul_fp4::launch_unchecked::<AB, AB, f32, S, R>(
            &client,
            CubeCount::Static(grid_x, grid_y, 1),
            CubeDim::new_1d(32),
            ab_vector_size,
            ab_vector_size,
            2,
            TensorArg::from_raw_parts(lhs_h.clone(), [k / 2, 1].into(), [m, k / 2].into()),
            TensorArg::from_raw_parts(rhs_h.clone(), [k / 2, 1].into(), [n, k / 2].into()),
            TensorArg::from_raw_parts(c_h.clone(), [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(lhs_scales_h.clone(), [scale_cols, 1].into(), [m, scale_cols].into()),
            TensorArg::from_raw_parts(rhs_scales_h.clone(), [scale_cols, 1].into(), [n, scale_cols].into()),
            TensorArg::from_raw_parts(out_buf, [n, 1].into(), [m, n].into()),
            n,
            k,
        );
    };

    // warmup: pays the NVRTC JIT and lets clocks boost
    for _ in 0..5 {
        launch(out_h.clone());
    }
    let _ = future::block_on(client.sync());

    let mut best = Duration::MAX;
    for _ in 0..20 {
        let (_, profile) = client
            .profile(|| launch(out_h.clone()), "matmul_fp4")
            .unwrap();
        let elapsed = future::block_on(profile.resolve()).duration();
        if elapsed < best {
            best = elapsed;
        }
    }

    let bytes = client.read_one(out_h).unwrap();
    let result = f32::from_bytes(&bytes);
    let expected = k as f32;
    let mut wrong = 0;
    for v in result.iter() {
        if (*v - expected).abs() > 0.03 * expected {
            wrong += 1;
        }
    }

    let secs = best.as_secs_f64();
    let gflops = 2.0 * (m as f64) * (n as f64) * (k as f64) / secs / 1e9;
    println!("runtime: {:?}", R::name(&client));
    println!("shape: {m}x{n}x{k}");
    println!("out[0]: {}  (expected {expected})", result[0]);
    println!("best: {:.3} ms   {gflops:.1} GFLOP/s", secs * 1e3);
    println!(
        "correctness: {} ({wrong} wrong of {})",
        if wrong == 0 { "PASS" } else { "FAIL" },
        m * n
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
