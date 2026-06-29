//! Milestone 3 swizzle derivation (A operand): does the plain `m8n8.x4.b16`
//! ldmatrix load the A fragment in the layout `mma.sync.mxf4` expects? Stages a
//! 16x64 fp4 A tile (16x32 e2m1x2 bytes) two ways from the same distinct-valued
//! data and dumps both per lane for host comparison:
//!   (1) ldmatrix m8n8.x4.b16, row-major, lane->matrix=lane/8 with the m16n8k16
//!       tile map (row-block = matrix%2, col-block = matrix/2)
//!   (2) the fed position_of_nth gather (known-correct mma layout)
//! MATCH => ldmatrix drops straight in with no swizzle; DIFFER => the per-lane diff
//! is the swizzle. Uses a vectorized stage for the gather (whole-vector reads, no
//! Vector::empty which CubeCL can't construct for e2m1x2) and a scalar stage for
//! ldmatrix.

use cubecl::features::ScaledMmaConfig;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::*;
use cubecl::{e2m1x2, ue8m0};

const MMA_M: usize = 16;
const MMA_N: usize = 8;
const MMA_K: usize = 64;
const SF: usize = 2;
const AB_BYTES: usize = MMA_M * MMA_K / 2; // 512 e2m1x2 bytes
const KB: usize = MMA_K / 2; // e2m1x2 bytes per A row = 32

const BB_BYTES: usize = MMA_N * MMA_K / 2; // 8*64 fp4 = 256 e2m1x2 bytes
const NKB: usize = MMA_K / 2; // e2m1x2 bytes per B row (n-major) = 32

#[cube(launch_unchecked)]
fn ldm_probe<A: Scalar, B: Scalar, CD: Numeric, S: Scalar, NA: Size>(
    a: &Tensor<Vector<A, NA>>,
    b: &Tensor<Vector<B, NA>>,
    a_ldm: &mut Tensor<u32>,
    a_gat: &mut Tensor<u32>,
    b_ldm: &mut Tensor<u32>,
    b_gat: &mut Tensor<u32>,
    #[comptime] na: usize,
) {
    let def = cmma::MmaDefinition::<A, B, CD>::new_scaled::<S>(MMA_M, MMA_N, MMA_K, SF);
    let lane_id = UNIT_POS_PLANE;
    let a_pack = A::packing_factor();
    let b_pack = B::packing_factor();
    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let factor_a = def.vectors_per_lane(MatrixIdent::A);
    let factor_b = def.vectors_per_lane(MatrixIdent::B);
    let na_vec = comptime!(AB_BYTES / na);
    let nb_vec = comptime!(BB_BYTES / na);

    let mut a_v = SharedMemory::<Vector<A, NA>>::new(na_vec);
    let mut a_s = SharedMemory::<A>::new_aligned(AB_BYTES, 128usize);
    let mut b_v = SharedMemory::<Vector<B, NA>>::new(nb_vec);
    let mut b_s = SharedMemory::<B>::new_aligned(BB_BYTES, 128usize);
    #[unroll]
    for j in 0..comptime!(na_vec / 32) {
        let idx = j * 32 + lane_id as usize;
        let vv = a[idx];
        a_v[idx] = vv;
        #[unroll]
        for k in 0..na {
            a_s[idx * na + k] = vv[k];
        }
    }
    #[unroll]
    for j in 0..comptime!(nb_vec / 32) {
        let idx = j * 32 + lane_id as usize;
        let vv = b[idx];
        b_v[idx] = vv;
        #[unroll]
        for k in 0..na {
            b_s[idx * na + k] = vv[k];
        }
    }
    sync_cube();

    // ---- A: plain m8n8.x4.b16, row-major, m16n8k16 tile map ----
    let mut a_reg = Array::<Vector<A, NA>>::new(factor_a);
    let mat = (lane_id / 8) as usize;
    let arow = (mat % 2) * 8 + (lane_id % 8) as usize;
    let acblk = mat / 2;
    let astart = arow * KB + acblk * 16;
    let aslice = a_s.slice(astart, astart + 16);
    def.load_matrix_inplace::<A, NA>(&aslice, &mut a_reg, MatrixIdent::A, factor_a, false);

    let mut a_gather = Array::<Vector<A, NA>>::new(factor_a);
    #[unroll]
    for i in 0..factor_a {
        let n_elem = i * vector_size_a * a_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
        let idx = (row as usize * MMA_K + col as usize) / (vector_size_a * a_pack);
        a_gather[i] = a_v[idx];
    }

    // ---- B: plain m8n8.x{factor}.b16 (non-transposed); B is [n8,k64]=[8 x 32 bytes].
    // mma B wants each lane to hold one n-row's k-data; lanes 0-3 -> n0, etc., so the
    // 8-row ldmatrix tile maps nrow = lane%8 with 2 k-blocks (16 bytes each).
    let mut b_reg = Array::<Vector<B, NA>>::new(factor_b);
    let nrow = (lane_id % 8) as usize; // n row 0..7
    let kblk = ((lane_id / 8) % 2) as usize; // k-half 0/1
    let bstart = nrow * NKB + kblk * 16;
    let bslice = b_s.slice(bstart, bstart + 16);
    def.load_matrix_inplace::<B, NA>(&bslice, &mut b_reg, MatrixIdent::B, factor_b, false);

    let mut b_gather = Array::<Vector<B, NA>>::new(factor_b);
    #[unroll]
    for i in 0..factor_b {
        let n_elem = i * vector_size_b * b_pack;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
        let idx = (col as usize * MMA_K + row as usize) / (vector_size_b * b_pack);
        b_gather[i] = b_v[idx];
    }

    let ab = lane_id as usize * 4;
    #[unroll]
    for i in 0..factor_a {
        a_ldm[ab + i] = u32::reinterpret(a_reg[i]);
        a_gat[ab + i] = u32::reinterpret(a_gather[i]);
    }
    let bb = lane_id as usize * 2;
    #[unroll]
    for i in 0..factor_b {
        b_ldm[bb + i] = u32::reinterpret(b_reg[i]);
        b_gat[bb + i] = u32::reinterpret(b_gather[i]);
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
    type AB = e2m1x2;
    type S = ue8m0;
    let ab_elem = AB::cube_type();
    let ab_vs = 32 / ab_elem.size_bits(); // 4

    let supported = client.features().matmul.scaled_mma.contains(&ScaledMmaConfig {
        a_type: ab_elem,
        b_type: ab_elem,
        cd_type: f32::cube_type(),
        scales_type: S::cube_type(),
        m: MMA_M as u32,
        n: MMA_N as u32,
        k: MMA_K as u32,
        scales_factor: SF as u32,
    });
    if !supported {
        println!("scaled FP4 MMA unsupported; skipping");
        return;
    }

    let mk = |n: usize| -> Vec<AB> { (0..n).map(|i| AB::from_bits((i % 251 + 1) as u8)).collect() };
    let a_h = client.create_from_slice(AB::as_bytes(&mk(AB_BYTES)));
    let b_h = client.create_from_slice(AB::as_bytes(&mk(BB_BYTES)));
    let a_ldm = client.empty(32 * 4 * 4);
    let a_gat = client.empty(32 * 4 * 4);
    let b_ldm = client.empty(32 * 2 * 4);
    let b_gat = client.empty(32 * 2 * 4);

    unsafe {
        ldm_probe::launch_unchecked::<AB, AB, f32, S, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(32),
            ab_vs,
            TensorArg::from_raw_parts(a_h, [1].into(), [AB_BYTES].into()),
            TensorArg::from_raw_parts(b_h, [1].into(), [BB_BYTES].into()),
            TensorArg::from_raw_parts(a_ldm.clone(), [1].into(), [32 * 4].into()),
            TensorArg::from_raw_parts(a_gat.clone(), [1].into(), [32 * 4].into()),
            TensorArg::from_raw_parts(b_ldm.clone(), [1].into(), [32 * 2].into()),
            TensorArg::from_raw_parts(b_gat.clone(), [1].into(), [32 * 2].into()),
            ab_vs,
        );
    }
    let cmp = |name: &str, lh: cubecl::server::Handle, gh: cubecl::server::Handle, regs: usize| -> usize {
        let lb = client.read_one(lh).unwrap();
        let gb = client.read_one(gh).unwrap();
        let l = u32::from_bytes(&lb);
        let g = u32::from_bytes(&gb);
        let mut mismatch = 0;
        for lane in 0..32 {
            let m = (0..regs).filter(|&i| l[lane * regs + i] != g[lane * regs + i]).count();
            mismatch += m;
            if lane < 4 || m > 0 {
                let ls: Vec<String> = (0..regs).map(|i| format!("{:08x}", l[lane * regs + i])).collect();
                let gs: Vec<String> = (0..regs).map(|i| format!("{:08x}", g[lane * regs + i])).collect();
                println!("{name} lane {lane:2}: ldm[{}] gather[{}] {}", ls.join(","), gs.join(","), if m == 0 { "ok" } else { "DIFF" });
            }
        }
        mismatch
    };
    println!("runtime: {:?}", R::name(&client));
    let ma = cmp("A", a_ldm, a_gat, 4);
    let mb = cmp("B", b_ldm, b_gat, 2);
    println!("ldm_probe: A {} ({ma} bad), B {} ({mb} bad)", if ma == 0 { "MATCH" } else { "DIFFER" }, if mb == 0 { "MATCH" } else { "DIFFER" });
}
