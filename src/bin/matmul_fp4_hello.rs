//! Hello block-scaled FP4 (MXFP4) on SM120. One warp issues a single
//! `mma.sync...kind::mxf4nvf4.block_scale` (shape m16 n8 k64, scales_factor 2)
//! via CubeCL's low-level `cmma::MmaDefinition` + `execute_scaled` path: the
//! only route to FP4 tensor cores (the high-level `cmma::Matrix` WMMA path does
//! not support FP4). The kernel manages per-lane registers by hand using
//! `position_of_nth`/`scales_index`. Inputs are all-ones FP4 with unit (2^0)
//! UE8M0 block scales, so each of the 16x8 outputs must equal k = 64.
//! Kernel body adapted verbatim from cubecl v0.10.0 runtime test `kernel_scaled`.

use cubecl::features::ScaledMmaConfig;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::{e2m1x2, ue8m0};

#[cube(launch)]
fn kernel_scaled<A: Scalar, B: Scalar, CD: Numeric, S: Scalar, NA: Size, NB: Size, NC: Size>(
    a: &Tensor<Vector<A, NA>>,
    b: &Tensor<Vector<B, NB>>,
    c: &Tensor<Vector<CD, NC>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Vector<CD, NC>>,
    #[comptime] size_m: usize,
    #[comptime] size_n: usize,
    #[comptime] size_k: usize,
    #[comptime] scales_factor: usize,
) {
    let a_pack = A::packing_factor();
    let b_pack = B::packing_factor();

    let def =
        cmma::MmaDefinition::<A, B, CD>::new_scaled::<S>(size_m, size_n, size_k, scales_factor);
    let lane_id = UNIT_POS_PLANE;

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
    let mut registers_c = Array::<Vector<CD, NC>>::new(vector_count_c);

    let elem_count_d = def.elems_per_lane(MatrixIdent::Accumulator);
    let vector_size_d = def.vector_size(MatrixIdent::Accumulator);
    let vector_count_d = comptime!(elem_count_d / vector_size_d);

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();

    let mut scales_register_a = Vector::<S, NS>::empty();
    let mut scales_register_b = Vector::<S, NS>::empty();

    // Load A
    #[unroll]
    for i in 0..vector_count_a {
        let n_elem = i * vector_size_a * a_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
        let idx = row as usize * size_k + col as usize;
        let idx = idx / (a.vector_size() * a_pack);

        registers_a[i] = a[idx];
    }

    let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A);
    #[unroll]
    for i in 0..scales_count {
        scales_register_a[i] = scales_a[scales_idx_a as usize * scales_factor + i];
    }

    // Load B
    #[unroll]
    for i in 0..vector_count_b {
        let n_elem = i * vector_size_b * b_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
        let idx = col as usize * size_k + row as usize;
        let idx = idx / (b.vector_size() * b_pack);

        registers_b[i] = b[idx];
    }

    let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B);
    #[unroll]
    for i in 0..scales_count {
        scales_register_b[i] = scales_b[scales_idx_b as usize * scales_factor + i];
    }

    // Load C
    #[unroll]
    for i in 0..vector_count_c {
        let n_elem = i * vector_size_c;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
        let idx = row as usize * size_n + col as usize;
        let value = c[idx / c.vector_size()];
        registers_c[i] = value;
    }

    let registers_d = def.execute_scaled(
        &registers_a,
        &registers_b,
        &registers_c,
        scales_register_a,
        scales_register_b,
    );

    // Store D
    #[unroll]
    for i in 0..vector_count_d {
        let n_elem = i * vector_size_d;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
        let idx = row as usize * size_n + col as usize;
        out[idx / out.vector_size()] = registers_d[i];
    }
}

fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    let (m, n, k) = (16usize, 8usize, 64usize);
    let scales_factor = 2usize;

    type AB = e2m1x2;
    type S = ue8m0;
    let ab_elem = AB::cube_type();
    let ab_vector_size = 32 / ab_elem.size_bits();

    let supported = client.features().matmul.scaled_mma.contains(&ScaledMmaConfig {
        a_type: ab_elem,
        b_type: ab_elem,
        cd_type: f32::cube_type(),
        scales_type: S::cube_type(),
        m: m as u32,
        n: n as u32,
        k: k as u32,
        scales_factor: scales_factor as u32,
    });
    if !supported {
        println!("runtime: {:?}", R::name(&client));
        println!("scaled FP4 MMA NOT supported on this device; skipping");
        return;
    }

    // all-ones FP4 inputs; unit (2^0) UE8M0 block scales
    let lhs_data: Vec<f32> = vec![1.0; m * k];
    let rhs_data: Vec<f32> = vec![1.0; n * k];
    let lhs = e2m1x2::from_f32_slice(&lhs_data);
    let rhs = e2m1x2::from_f32_slice(&rhs_data);
    let unit = ue8m0::from_bits(127);
    let lhs_scales_data: Vec<S> = vec![unit; m * scales_factor];
    let rhs_scales_data: Vec<S> = vec![unit; n * scales_factor];
    let out_host = vec![0.0f32; m * n];

    let lhs_h = client.create_from_slice(AB::as_bytes(&lhs));
    let rhs_h = client.create_from_slice(AB::as_bytes(&rhs));
    let lhs_scales_h = client.create_from_slice(S::as_bytes(&lhs_scales_data));
    let rhs_scales_h = client.create_from_slice(S::as_bytes(&rhs_scales_data));
    let out_h = client.create_from_slice(f32::as_bytes(&out_host));

    unsafe {
        kernel_scaled::launch::<AB, AB, f32, S, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            ab_vector_size,
            ab_vector_size,
            2,
            TensorArg::from_raw_parts(lhs_h, [k / 2, 1].into(), [m, k / 2].into()),
            TensorArg::from_raw_parts(rhs_h, [k / 2, 1].into(), [n, k / 2].into()),
            TensorArg::from_raw_parts(out_h.clone(), [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(
                lhs_scales_h,
                [scales_factor, 1].into(),
                [m, scales_factor].into(),
            ),
            TensorArg::from_raw_parts(
                rhs_scales_h,
                [scales_factor, 1].into(),
                [n, scales_factor].into(),
            ),
            TensorArg::from_raw_parts(out_h.clone(), [n, 1].into(), [m, n].into()),
            m,
            n,
            k,
            scales_factor,
        );
    }

    let bytes = client.read_one(out_h).unwrap();
    let result = f32::from_bytes(&bytes);

    // reference: out[i][j] = sum_l lhs * lhs_scale * rhs * rhs_scale  (== 64 here)
    let mut wrong = 0;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                let blk = l / (k / scales_factor);
                let ls = lhs_scales_data[i * scales_factor + blk].to_f32();
                let rs = rhs_scales_data[j * scales_factor + blk].to_f32();
                sum += lhs_data[i * k + l] * ls * rhs_data[j * k + l] * rs;
            }
            let got = result[i * n + j];
            if (got - sum).abs() > 0.03 * sum.abs().max(1.0) {
                wrong += 1;
            }
        }
    }

    println!("runtime: {:?}", R::name(&client));
    println!("shape: m{m} n{n} k{k}  scales_factor={scales_factor}");
    println!("out[0]: {}  (expected 64)", result[0]);
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
