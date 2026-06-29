//! Single-tile FP4 (MXFP4) MMA that loads its A and B operands with the
//! Blackwell `ldmatrix` path instead of hand-managed `position_of_nth` gathers.
//! This is the correctness microtest for the patched CubeCL `cubecl-cpp` codegen
//! (see `vendor/cubecl-cpp-0.10.0`): for packed `e2m1` it now emits the
//! format-converting `ldmatrix.m8n16 .b8x16.b4x16_p64` (A, non-transposed) and
//! `ldmatrix.m16n16.trans .b8x16.b4x16_p64` (B, transposed) that ptxas accepts.
//!
//! RESULT (kept as evidence): the patched PTX assembles and RUNS, but the kernel
//! is wrong by exactly 0.5x (every output == 32, not 64). The fragment dump shows
//! 0x02020202 where packed e2m1x2 all-ones should be 0x22222222: the format
//! converting load `.b8x16.b4x16_p64` EXPANDS each 4-bit value into its own byte
//! (real value in the low nibble, high nibble zeroed). `execute_scaled`'s
//! mma.sync.mxf4nvf4 wants PACKED e2m1x2 (2 fp4/byte), so it reads the expanded
//! bytes as packed pairs => every other k-element is the inserted zero => half.
//! A lane's 16-byte fragment holds 32 packed fp4 but only 16 expanded, so half of
//! K is simply absent; loading all of A expanded needs 2x the registers, fatal at
//! the 255-reg ceiling. Conclusion: ldmatrix is NOT usable for the packed-mxf4
//! mma.sync path on SM120. The codegen fix itself works (PTX assembles + runs).
//!
//! A/B operands are staged into 16-byte-aligned shared memory (ldmatrix requires
//! a shared source), then each lane hands `load_matrix_inplace` the 16-byte row
//! it owns. The `dbg`/`fdbg` tensors dump the fragment layout and raw registers.

use cubecl::features::ScaledMmaConfig;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::{e2m1x2, ue8m0};

#[cube(launch)]
fn kernel_ldm<A: Scalar, B: Scalar, CD: Numeric, S: Scalar, NC: Size>(
    a: &Tensor<A>,
    b: &Tensor<B>,
    c: &Tensor<Vector<CD, NC>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Vector<CD, NC>>,
    dbg: &mut Tensor<f32>,
    fdbg: &mut Tensor<u32>,
    #[comptime] size_m: usize,
    #[comptime] size_n: usize,
    #[comptime] size_k: usize,
    #[comptime] scales_factor: usize,
) {
    let def =
        cmma::MmaDefinition::<A, B, CD>::new_scaled::<S>(size_m, size_n, size_k, scales_factor);
    let lane_id = UNIT_POS_PLANE;

    let a_pack = A::packing_factor();
    let b_pack = B::packing_factor();
    // packed columns: number of e2m1x2 storage elements across k
    let ka = comptime!(size_k / a_pack);
    let kb = comptime!(size_k / b_pack);

    // stage A (m x ka) and B (n x kb) into 16-byte-aligned shared memory.
    // single warp (32 lanes); both tile sizes are exact multiples of 32.
    let mut stage_a = SharedMemory::<A>::new_aligned(size_m * ka, 16usize);
    let mut stage_b = SharedMemory::<B>::new_aligned(size_n * kb, 16usize);
    #[unroll]
    for j in 0..comptime!(size_m * ka / 32) {
        let idx = j * 32 + lane_id as usize;
        stage_a[idx] = a[idx];
    }
    #[unroll]
    for j in 0..comptime!(size_n * kb / 32) {
        let idx = j * 32 + lane_id as usize;
        stage_b[idx] = b[idx];
    }
    sync_cube();

    // width = number of storage elements in a 16-byte ldmatrix row
    let elem_size = comptime!(A::type_size());
    let width = comptime!(16 / elem_size);

    // A: shape m8n16, 8-row matrices, factor = vectors_per_lane (expected 4 => 2x2 tiling)
    let factor_a = def.vectors_per_lane(MatrixIdent::A);
    let size!(NA) = def.vector_size(MatrixIdent::A);
    let mut registers_a = Array::<Vector<A, NA>>::new(factor_a);
    // lane -> matrix (lane/8), row within matrix (lane%8); 2x2 tile of A(16 x 2*width)
    let mat_a = (lane_id / 8) as usize;
    let rt_a = mat_a / 2; // row tile
    let ct_a = mat_a % 2; // col tile
    let grow_a = rt_a * 8 + (lane_id % 8) as usize;
    let start_a = grow_a * ka + ct_a * width;
    let slice_a = stage_a.slice(start_a, start_a + width);
    def.load_matrix_inplace::<A, NA>(&slice_a, &mut registers_a, MatrixIdent::A, factor_a, false);

    // B: shape m16n16 (transposed). Each m16n16 matrix carries 2 u32/lane, so the
    // hardware matrix count is vectors_per_lane/2, while the fragment array stays
    // sized to the full per-lane fragment (vectors_per_lane vectors).
    let frag_vecs_b = def.vectors_per_lane(MatrixIdent::B);
    let num_mat_b = comptime!(frag_vecs_b / 2);
    let size!(NB) = def.vector_size(MatrixIdent::B);
    let mut registers_b = Array::<Vector<B, NB>>::new(frag_vecs_b);
    let row_b = (lane_id % 16) as usize;
    let start_b = row_b * width;
    let slice_b = stage_b.slice(start_b, start_b + width);
    def.load_matrix_inplace::<B, NB>(&slice_b, &mut registers_b, MatrixIdent::B, num_mat_b, true);

    // dump raw fragment registers (all-ones e2m1x2 => each u32 should be 0x22222222)
    fdbg[lane_id as usize * 6 + 0] = u32::reinterpret(registers_a[0]);
    fdbg[lane_id as usize * 6 + 1] = u32::reinterpret(registers_a[1]);
    fdbg[lane_id as usize * 6 + 2] = u32::reinterpret(registers_a[2]);
    fdbg[lane_id as usize * 6 + 3] = u32::reinterpret(registers_a[3]);
    fdbg[lane_id as usize * 6 + 4] = u32::reinterpret(registers_b[0]);
    fdbg[lane_id as usize * 6 + 5] = u32::reinterpret(registers_b[1]);

    // diagnostics: dump CubeCL's actual fragment layout for A and B
    let epl_a = def.elems_per_lane(MatrixIdent::A);
    let epl_b = def.elems_per_lane(MatrixIdent::B);
    if lane_id == 0 {
        dbg[0] = factor_a as f32;
        dbg[1] = num_mat_b as f32;
        dbg[2] = epl_a as f32;
        dbg[3] = epl_b as f32;
        dbg[4] = width as f32;
        dbg[5] = ka as f32;
        dbg[6] = kb as f32;
        dbg[7] = elem_size as f32;
    }

    // scales (hello path)
    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();
    let mut scales_register_a = Vector::<S, NS>::empty();
    let mut scales_register_b = Vector::<S, NS>::empty();
    let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A);
    #[unroll]
    for i in 0..scales_count {
        scales_register_a[i] = scales_a[scales_idx_a as usize * scales_factor + i];
    }
    let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B);
    #[unroll]
    for i in 0..scales_count {
        scales_register_b[i] = scales_b[scales_idx_b as usize * scales_factor + i];
    }

    // C
    let elem_count_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vector_size_c = def.vector_size(MatrixIdent::Accumulator);
    let vector_count_c = comptime!(elem_count_c / vector_size_c);
    let mut registers_c = Array::<Vector<CD, NC>>::new(vector_count_c);
    #[unroll]
    for i in 0..vector_count_c {
        let n_elem = i * vector_size_c;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
        let idx = row as usize * size_n + col as usize;
        registers_c[i] = c[idx / c.vector_size()];
    }

    let registers_d = def.execute_scaled(
        &registers_a,
        &registers_b,
        &registers_c,
        scales_register_a,
        scales_register_b,
    );

    #[unroll]
    for i in 0..vector_count_c {
        let n_elem = i * vector_size_c;
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

    let lhs_data: Vec<f32> = vec![1.0; m * k];
    let rhs_data: Vec<f32> = vec![1.0; n * k];
    let lhs = e2m1x2::from_f32_slice(&lhs_data);
    let rhs = e2m1x2::from_f32_slice(&rhs_data);
    let unit = ue8m0::from_bits(127); // 2^0
    let lhs_scales_data: Vec<S> = vec![unit; m * scales_factor];
    let rhs_scales_data: Vec<S> = vec![unit; n * scales_factor];
    let out_host = vec![0.0f32; m * n];

    let lhs_h = client.create_from_slice(AB::as_bytes(&lhs));
    let rhs_h = client.create_from_slice(AB::as_bytes(&rhs));
    let lhs_scales_h = client.create_from_slice(S::as_bytes(&lhs_scales_data));
    let rhs_scales_h = client.create_from_slice(S::as_bytes(&rhs_scales_data));
    let out_h = client.create_from_slice(f32::as_bytes(&out_host));
    let dbg_host = vec![0.0f32; 8];
    let dbg_h = client.create_from_slice(f32::as_bytes(&dbg_host));
    let fdbg_host = vec![0u32; 32 * 6];
    let fdbg_h = client.create_from_slice(u32::as_bytes(&fdbg_host));

    unsafe {
        kernel_ldm::launch::<AB, AB, f32, S, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            // NC: f32 accumulator/output line size (a and b are scalar tensors)
            2,
            TensorArg::from_raw_parts(lhs_h, [k / 2, 1].into(), [m, k / 2].into()),
            TensorArg::from_raw_parts(rhs_h, [k / 2, 1].into(), [n, k / 2].into()),
            TensorArg::from_raw_parts(out_h.clone(), [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(lhs_scales_h, [scales_factor, 1].into(), [m, scales_factor].into()),
            TensorArg::from_raw_parts(rhs_scales_h, [scales_factor, 1].into(), [n, scales_factor].into()),
            TensorArg::from_raw_parts(out_h.clone(), [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(dbg_h.clone(), [1].into(), [8].into()),
            TensorArg::from_raw_parts(fdbg_h.clone(), [1].into(), [32 * 6].into()),
            m,
            n,
            k,
            scales_factor,
        );
    }

    let bytes = client.read_one(out_h).unwrap();
    let result = f32::from_bytes(&bytes);

    let dbg_bytes = client.read_one(dbg_h).unwrap();
    let d = f32::from_bytes(&dbg_bytes);
    println!(
        "layout: factor_a={} factor_b={} epl_a={} epl_b={} width={} ka={} kb={} elem_size={}",
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
    );
    let fdbg_bytes = client.read_one(fdbg_h).unwrap();
    let f = u32::from_bytes(&fdbg_bytes);
    println!("fragments (all-ones => expect 0x22222222 each):");
    for lane in [0usize, 1, 7, 8, 15, 16] {
        println!(
            "  lane {lane:2}: A=[{:08x} {:08x} {:08x} {:08x}] B=[{:08x} {:08x}]",
            f[lane * 6], f[lane * 6 + 1], f[lane * 6 + 2], f[lane * 6 + 3],
            f[lane * 6 + 4], f[lane * 6 + 5]
        );
    }
    println!("result matrix (m x n):");
    for i in 0..m {
        let row: Vec<String> = (0..n).map(|j| format!("{:>5.0}", result[i * n + j])).collect();
        println!("  {}", row.join(" "));
    }

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
