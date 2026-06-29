//! Milestone 2 toward closing the 3x gap to CUTLASS: a depth-2 (double-buffered)
//! TMA software pipeline on top of milestone 1's TMA staging. Two `full` mbarriers
//! and two shared buffers per operand: the prologue prefetches steps 0 and 1, and
//! each loop step computes stage `step%2` while the TMA for `step+2` is already in
//! flight into the buffer this stage just freed. So the per-step `wait` no longer
//! blocks on TMA latency (it completed under the previous stage's compute), which
//! milestone 1's single-stage path could not hide. Mechanics validated in
//! `src/bin/pipe_probe.rs`. Keeps fed's 2x8 MMA inner byte for byte; scales stay on
//! the manual per-step path. Verified against the same f32 reference as fed.

use std::time::Duration;

use cubecl::features::ScaledMmaConfig;
use cubecl::future;
use cubecl::ir::MatrixIdent;
use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;
use cubecl::{e2m1x2, ue8m0};

const MMA_M: usize = 16;
const MMA_N: usize = 8;
const MMA_K: usize = 64;
const SF: usize = 2;
const WARPS_M: usize = 2;
const WARPS_N: usize = 2;
const WM: usize = 2;
const WN: usize = 8;
const BM: usize = WARPS_M * WM * MMA_M; // 64
const BN: usize = WARPS_N * WN * MMA_N; // 128
const BK: usize = 128;
const BKH: usize = BK / 2; // packed e2m1x2 columns per k-step tile (bytes)
const KSUB: usize = BK / MMA_K;
const FV: usize = 8;
const VPR: usize = BK / FV;
const N_THREADS: u32 = (WARPS_M * WARPS_N * 32) as u32; // 128

#[cube(launch_unchecked)]
fn matmul_fp4_pipe2<A: Scalar, B: Scalar, CD: Numeric, S: Scalar, NA: Size, NB: Size, NC: Size>(
    a: &TensorMap<A, Tiled>,
    b: &TensorMap<B, Tiled>,
    c: &Tensor<Vector<CD, NC>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Vector<CD, NC>>,
    #[comptime] full_n: usize,
    #[comptime] full_k: usize,
) {
    let a_pack = A::packing_factor();
    let b_pack = B::packing_factor();
    let def = cmma::MmaDefinition::<A, B, CD>::new_scaled::<S>(MMA_M, MMA_N, MMA_K, SF);

    let tid = UNIT_POS_X as usize;
    let lane_id = UNIT_POS_PLANE;
    let warp = tid / 32;
    let warp_m = warp / WARPS_N;
    let warp_n = warp % WARPS_N;

    let block_row = CUBE_POS_Y as usize * BM;
    let block_col = CUBE_POS_X as usize * BN;

    let k_steps = comptime!(full_k / BK);
    let scale_blocks_per_row = comptime!(full_k / (MMA_K / SF));

    let elem_count_a = def.elems_per_lane(MatrixIdent::A);
    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_count_a = comptime!(elem_count_a / vector_size_a);
    let elem_count_b = def.elems_per_lane(MatrixIdent::B);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let vector_count_b = comptime!(elem_count_b / vector_size_b);
    let elem_count_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let vector_size_c = def.vector_size(MatrixIdent::Accumulator);
    let vector_count_c = comptime!(elem_count_c / vector_size_c);

    let scales_count = def.scales_count();
    let size!(NS) = def.scales_vector_size();

    // TMA needs 128-byte-aligned shared destinations; double-buffered (2 stages).
    let mut a_tile = SharedMemory::<Vector<A, NA>>::new_aligned(2 * BM * VPR, 128usize);
    let mut b_tile = SharedMemory::<Vector<B, NB>>::new_aligned(2 * BN * VPR, 128usize);
    let bps = comptime!(BK / (MMA_K / SF));
    let mut sa_tile = SharedMemory::<S>::new(BM * bps);
    let mut sb_tile = SharedMemory::<S>::new(BN * bps);

    let a_row0 = (warp_m * WM) * MMA_M;
    let a_row1 = (warp_m * WM + 1) * MMA_M;
    let b_col0 = (warp_n * WN) * MMA_N;
    let b_col1 = (warp_n * WN + 1) * MMA_N;
    let b_col2 = (warp_n * WN + 2) * MMA_N;
    let b_col3 = (warp_n * WN + 3) * MMA_N;
    let b_col4 = (warp_n * WN + 4) * MMA_N;
    let b_col5 = (warp_n * WN + 5) * MMA_N;
    let b_col6 = (warp_n * WN + 6) * MMA_N;
    let b_col7 = (warp_n * WN + 7) * MMA_N;

    let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A) as usize;
    let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B) as usize;

    let mut acc00 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc01 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc02 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc03 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc04 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc05 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc06 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc07 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc10 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc11 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc12 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc13 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc14 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc15 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc16 = Array::<Vector<CD, NC>>::new(vector_count_c);
    let mut acc17 = Array::<Vector<CD, NC>>::new(vector_count_c);
    #[unroll]
    for i in 0..vector_count_c {
        let n_elem = i * vector_size_c;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
        let r = row as usize;
        let cc = col as usize;
        let vs = c.vector_size();
        let row0 = (block_row + a_row0 + r) * full_n;
        let row1 = (block_row + a_row1 + r) * full_n;
        acc00[i] = c[(row0 + block_col + b_col0 + cc) / vs];
        acc01[i] = c[(row0 + block_col + b_col1 + cc) / vs];
        acc02[i] = c[(row0 + block_col + b_col2 + cc) / vs];
        acc03[i] = c[(row0 + block_col + b_col3 + cc) / vs];
        acc04[i] = c[(row0 + block_col + b_col4 + cc) / vs];
        acc05[i] = c[(row0 + block_col + b_col5 + cc) / vs];
        acc06[i] = c[(row0 + block_col + b_col6 + cc) / vs];
        acc07[i] = c[(row0 + block_col + b_col7 + cc) / vs];
        acc10[i] = c[(row1 + block_col + b_col0 + cc) / vs];
        acc11[i] = c[(row1 + block_col + b_col1 + cc) / vs];
        acc12[i] = c[(row1 + block_col + b_col2 + cc) / vs];
        acc13[i] = c[(row1 + block_col + b_col3 + cc) / vs];
        acc14[i] = c[(row1 + block_col + b_col4 + cc) / vs];
        acc15[i] = c[(row1 + block_col + b_col5 + cc) / vs];
        acc16[i] = c[(row1 + block_col + b_col6 + cc) / vs];
        acc17[i] = c[(row1 + block_col + b_col7 + cc) / vs];
    }

    let mut a0 = Array::<Vector<A, NA>>::new(vector_count_a);
    let mut a1 = Array::<Vector<A, NA>>::new(vector_count_a);
    let mut b0 = Array::<Vector<B, NB>>::new(vector_count_b);
    let mut b1 = Array::<Vector<B, NB>>::new(vector_count_b);
    let mut b2 = Array::<Vector<B, NB>>::new(vector_count_b);
    let mut b3 = Array::<Vector<B, NB>>::new(vector_count_b);
    let mut b4 = Array::<Vector<B, NB>>::new(vector_count_b);
    let mut b5 = Array::<Vector<B, NB>>::new(vector_count_b);
    let mut b6 = Array::<Vector<B, NB>>::new(vector_count_b);
    let mut b7 = Array::<Vector<B, NB>>::new(vector_count_b);
    let mut sa0 = Vector::<S, NS>::empty();
    let mut sa1 = Vector::<S, NS>::empty();
    let mut sb0 = Vector::<S, NS>::empty();
    let mut sb1 = Vector::<S, NS>::empty();
    let mut sb2 = Vector::<S, NS>::empty();
    let mut sb3 = Vector::<S, NS>::empty();
    let mut sb4 = Vector::<S, NS>::empty();
    let mut sb5 = Vector::<S, NS>::empty();
    let mut sb6 = Vector::<S, NS>::empty();
    let mut sb7 = Vector::<S, NS>::empty();

    let n_threads = N_THREADS as usize;
    let abuf = comptime!(BM * VPR);
    let bbuf = comptime!(BN * VPR);
    let ab_bytes = select(UNIT_POS_X == 0, comptime!((BM * BKH + BN * BKH) as u32), 0u32);
    let full0 = Barrier::shared(N_THREADS, UNIT_POS_X == 0);
    let full1 = Barrier::shared(N_THREADS, UNIT_POS_X == 0);
    sync_async_proxy_shared();

    // prologue: prefetch step 0 into buffer 0 and step 1 into buffer 1, so each
    // stage's TMA is in flight a step ahead of when it is consumed.
    if tid == 0 {
        let mut pa0 = a_tile.slice_mut(0, abuf);
        full0.tma_load_2d(a, &mut pa0, block_row as i32, 0);
        let mut pb0 = b_tile.slice_mut(0, bbuf);
        full0.tma_load_2d(b, &mut pb0, block_col as i32, 0);
        let mut pa1 = a_tile.slice_mut(abuf, 2 * abuf);
        full1.tma_load_2d(a, &mut pa1, block_row as i32, BKH as i32);
        let mut pb1 = b_tile.slice_mut(bbuf, 2 * bbuf);
        full1.tma_load_2d(b, &mut pb1, block_col as i32, BKH as i32);
    }
    let mut token0 = full0.arrive_and_expect_tx(1, ab_bytes);
    let mut token1 = full1.arrive_and_expect_tx(1, ab_bytes);

    for step in 0..k_steps {
        let cur = step % 2;
        let abase = cur * abuf;
        let bbase = cur * bbuf;

        // stage this step's scale blocks into shared (manual cooperative copy)
        #[unroll]
        for i in 0..(BM * bps) / n_threads {
            let s = tid + i * n_threads;
            sa_tile[s] = scales_a[(block_row + s / bps) * scale_blocks_per_row + step * bps + s % bps];
        }
        #[unroll]
        for i in 0..(BN * bps) / n_threads {
            let s = tid + i * n_threads;
            sb_tile[s] = scales_b[(block_col + s / bps) * scale_blocks_per_row + step * bps + s % bps];
        }

        // wait for this stage's A/B (prefetched a step earlier; should not block)
        if cur == 0 {
            full0.wait(token0);
        } else {
            full1.wait(token1);
        }
        sync_cube(); // scales visible to all, all threads aligned before compute

        #[unroll]
        for ks in 0..KSUB {
            let k_off = ks * MMA_K;
            let sk = ks * SF;

            #[unroll]
            for i in 0..vector_count_a {
                let n_elem = i * vector_size_a * a_pack;
                let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
                let div = a_pack; // shared vector already holds packed e2m1x2; div by pack
                a0[i] = a_tile[abase + ((a_row0 + row as usize) * BK + k_off + col as usize) / (div * vector_size_a)];
                a1[i] = a_tile[abase + ((a_row1 + row as usize) * BK + k_off + col as usize) / (div * vector_size_a)];
            }
            #[unroll]
            for i in 0..scales_count {
                sa0[i] = sa_tile[(a_row0 + scales_idx_a) * bps + sk + i];
                sa1[i] = sa_tile[(a_row1 + scales_idx_a) * bps + sk + i];
            }
            #[unroll]
            for i in 0..vector_count_b {
                let n_elem = i * vector_size_b * b_pack;
                let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
                let div = b_pack;
                b0[i] = b_tile[bbase + ((b_col0 + col as usize) * BK + k_off + row as usize) / (div * vector_size_b)];
                b1[i] = b_tile[bbase + ((b_col1 + col as usize) * BK + k_off + row as usize) / (div * vector_size_b)];
                b2[i] = b_tile[bbase + ((b_col2 + col as usize) * BK + k_off + row as usize) / (div * vector_size_b)];
                b3[i] = b_tile[bbase + ((b_col3 + col as usize) * BK + k_off + row as usize) / (div * vector_size_b)];
                b4[i] = b_tile[bbase + ((b_col4 + col as usize) * BK + k_off + row as usize) / (div * vector_size_b)];
                b5[i] = b_tile[bbase + ((b_col5 + col as usize) * BK + k_off + row as usize) / (div * vector_size_b)];
                b6[i] = b_tile[bbase + ((b_col6 + col as usize) * BK + k_off + row as usize) / (div * vector_size_b)];
                b7[i] = b_tile[bbase + ((b_col7 + col as usize) * BK + k_off + row as usize) / (div * vector_size_b)];
            }
            #[unroll]
            for i in 0..scales_count {
                sb0[i] = sb_tile[(b_col0 + scales_idx_b) * bps + sk + i];
                sb1[i] = sb_tile[(b_col1 + scales_idx_b) * bps + sk + i];
                sb2[i] = sb_tile[(b_col2 + scales_idx_b) * bps + sk + i];
                sb3[i] = sb_tile[(b_col3 + scales_idx_b) * bps + sk + i];
                sb4[i] = sb_tile[(b_col4 + scales_idx_b) * bps + sk + i];
                sb5[i] = sb_tile[(b_col5 + scales_idx_b) * bps + sk + i];
                sb6[i] = sb_tile[(b_col6 + scales_idx_b) * bps + sk + i];
                sb7[i] = sb_tile[(b_col7 + scales_idx_b) * bps + sk + i];
            }

            let d00 = def.execute_scaled(&a0, &b0, &acc00, sa0, sb0);
            let d01 = def.execute_scaled(&a0, &b1, &acc01, sa0, sb1);
            let d02 = def.execute_scaled(&a0, &b2, &acc02, sa0, sb2);
            let d03 = def.execute_scaled(&a0, &b3, &acc03, sa0, sb3);
            let d04 = def.execute_scaled(&a0, &b4, &acc04, sa0, sb4);
            let d05 = def.execute_scaled(&a0, &b5, &acc05, sa0, sb5);
            let d06 = def.execute_scaled(&a0, &b6, &acc06, sa0, sb6);
            let d07 = def.execute_scaled(&a0, &b7, &acc07, sa0, sb7);
            let d10 = def.execute_scaled(&a1, &b0, &acc10, sa1, sb0);
            let d11 = def.execute_scaled(&a1, &b1, &acc11, sa1, sb1);
            let d12 = def.execute_scaled(&a1, &b2, &acc12, sa1, sb2);
            let d13 = def.execute_scaled(&a1, &b3, &acc13, sa1, sb3);
            let d14 = def.execute_scaled(&a1, &b4, &acc14, sa1, sb4);
            let d15 = def.execute_scaled(&a1, &b5, &acc15, sa1, sb5);
            let d16 = def.execute_scaled(&a1, &b6, &acc16, sa1, sb6);
            let d17 = def.execute_scaled(&a1, &b7, &acc17, sa1, sb7);
            #[unroll]
            for i in 0..vector_count_c {
                acc00[i] = d00[i];
                acc01[i] = d01[i];
                acc02[i] = d02[i];
                acc03[i] = d03[i];
                acc04[i] = d04[i];
                acc05[i] = d05[i];
                acc06[i] = d06[i];
                acc07[i] = d07[i];
                acc10[i] = d10[i];
                acc11[i] = d11[i];
                acc12[i] = d12[i];
                acc13[i] = d13[i];
                acc14[i] = d14[i];
                acc15[i] = d15[i];
                acc16[i] = d16[i];
                acc17[i] = d17[i];
            }
        }
        sync_cube(); // all threads done reading this buffer before it is reused

        // prefetch step+2 into the buffer this stage just freed, and re-arm its barrier
        let next = step + 2;
        if next < k_steps {
            let kcol = (next * BKH) as i32;
            if cur == 0 {
                if tid == 0 {
                    let mut pa = a_tile.slice_mut(0, abuf);
                    full0.tma_load_2d(a, &mut pa, block_row as i32, kcol);
                    let mut pb = b_tile.slice_mut(0, bbuf);
                    full0.tma_load_2d(b, &mut pb, block_col as i32, kcol);
                }
                token0 = full0.arrive_and_expect_tx(1, ab_bytes);
            } else {
                if tid == 0 {
                    let mut pa = a_tile.slice_mut(abuf, 2 * abuf);
                    full1.tma_load_2d(a, &mut pa, block_row as i32, kcol);
                    let mut pb = b_tile.slice_mut(bbuf, 2 * bbuf);
                    full1.tma_load_2d(b, &mut pb, block_col as i32, kcol);
                }
                token1 = full1.arrive_and_expect_tx(1, ab_bytes);
            }
        }
    }

    #[unroll]
    for i in 0..vector_count_c {
        let n_elem = i * vector_size_c;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
        let r = row as usize;
        let cc = col as usize;
        let vs = out.vector_size();
        let row0 = (block_row + a_row0 + r) * full_n;
        let row1 = (block_row + a_row1 + r) * full_n;
        out[(row0 + block_col + b_col0 + cc) / vs] = acc00[i];
        out[(row0 + block_col + b_col1 + cc) / vs] = acc01[i];
        out[(row0 + block_col + b_col2 + cc) / vs] = acc02[i];
        out[(row0 + block_col + b_col3 + cc) / vs] = acc03[i];
        out[(row0 + block_col + b_col4 + cc) / vs] = acc04[i];
        out[(row0 + block_col + b_col5 + cc) / vs] = acc05[i];
        out[(row0 + block_col + b_col6 + cc) / vs] = acc06[i];
        out[(row0 + block_col + b_col7 + cc) / vs] = acc07[i];
        out[(row1 + block_col + b_col0 + cc) / vs] = acc10[i];
        out[(row1 + block_col + b_col1 + cc) / vs] = acc11[i];
        out[(row1 + block_col + b_col2 + cc) / vs] = acc12[i];
        out[(row1 + block_col + b_col3 + cc) / vs] = acc13[i];
        out[(row1 + block_col + b_col4 + cc) / vs] = acc14[i];
        out[(row1 + block_col + b_col5 + cc) / vs] = acc15[i];
        out[(row1 + block_col + b_col6 + cc) / vs] = acc16[i];
        out[(row1 + block_col + b_col7 + cc) / vs] = acc17[i];
    }
}

const FP4_MAGS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn xorshift(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

fn rand_fp4(state: &mut u32) -> f32 {
    let r = xorshift(state);
    let mag = FP4_MAGS[(r % 8) as usize];
    if (r >> 3) & 1 == 1 {
        -mag
    } else {
        mag
    }
}

fn a_map<R: Runtime>(handle: cubecl::server::Handle, m: usize, k: usize, tile_rows: usize) -> TensorMapArg<R, Tiled> {
    let arg = unsafe { TensorArg::from_raw_parts(handle, [k / 2, 1].into(), [m, k / 2].into()) };
    // describe the 1-byte packed e2m1x2 data as u8: e2m1x2's native type is an
    // invalid CUtensorMap dtype (illegal-instruction at runtime). Same byte layout.
    TensorMapArg::new(TiledArgs { tile_size: [tile_rows, BKH].into() }, arg, u8::as_type_native_unchecked())
}

fn verify<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    type AB = e2m1x2;
    type S = ue8m0;
    let ab_elem = AB::cube_type();
    let ab_vector_size = 32 / ab_elem.size_bits();

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
        println!("verify: scaled FP4 MMA unsupported; skipping");
        return;
    }

    let (m, n, k) = (512usize, 512usize, 512usize);
    let scale_cols = k / (MMA_K / SF);
    let blk = MMA_K / SF;

    let mut st = 0x9e3779b9u32;
    let lhs_f: Vec<f32> = (0..m * k).map(|_| rand_fp4(&mut st)).collect();
    let rhs_f: Vec<f32> = (0..n * k).map(|_| rand_fp4(&mut st)).collect();
    let lhs_sb: Vec<S> = (0..m * scale_cols).map(|_| ue8m0::from_bits(126 + (xorshift(&mut st) % 3) as u8)).collect();
    let rhs_sb: Vec<S> = (0..n * scale_cols).map(|_| ue8m0::from_bits(126 + (xorshift(&mut st) % 3) as u8)).collect();

    let lhs = e2m1x2::from_f32_slice(&lhs_f);
    let rhs = e2m1x2::from_f32_slice(&rhs_f);
    let zeros = vec![0.0f32; m * n];

    let lhs_h = client.create_from_slice(AB::as_bytes(&lhs));
    let rhs_h = client.create_from_slice(AB::as_bytes(&rhs));
    let lhs_sb_h = client.create_from_slice(S::as_bytes(&lhs_sb));
    let rhs_sb_h = client.create_from_slice(S::as_bytes(&rhs_sb));
    let c_h = client.create_from_slice(f32::as_bytes(&zeros));
    let out_h = client.empty(m * n * core::mem::size_of::<f32>());

    let grid_x = (n / BN) as u32;
    let grid_y = (m / BM) as u32;
    unsafe {
        matmul_fp4_pipe2::launch_unchecked::<AB, AB, f32, S, R>(
            &client,
            CubeCount::Static(grid_x, grid_y, 1),
            CubeDim::new_1d(N_THREADS),
            ab_vector_size,
            ab_vector_size,
            2,
            a_map::<R>(lhs_h, m, k, BM),
            a_map::<R>(rhs_h, n, k, BN),
            TensorArg::from_raw_parts(c_h, [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(lhs_sb_h, [scale_cols, 1].into(), [m, scale_cols].into()),
            TensorArg::from_raw_parts(rhs_sb_h, [scale_cols, 1].into(), [n, scale_cols].into()),
            TensorArg::from_raw_parts(out_h.clone(), [n, 1].into(), [m, n].into()),
            n,
            k,
        );
    }
    let bytes = client.read_one(out_h).unwrap();
    let result = f32::from_bytes(&bytes);

    let mut max_rel = 0.0f64;
    let mut wrong = 0usize;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                let sa = lhs_sb[i * scale_cols + l / blk].to_f32();
                let sb = rhs_sb[j * scale_cols + l / blk].to_f32();
                sum += lhs_f[i * k + l] * sa * rhs_f[j * k + l] * sb;
            }
            let rel = ((result[i * n + j] - sum).abs() / sum.abs().max(1.0)) as f64;
            if rel > max_rel {
                max_rel = rel;
            }
            if rel > 1e-3 {
                wrong += 1;
            }
        }
    }
    println!("runtime: {:?}", R::name(&client));
    println!("verify: shape {m}x{n}x{k}  random representable FP4 + random per-block scales");
    println!("verify: {} ({wrong} wrong of {}, max_rel={max_rel:.2e})", if wrong == 0 { "PASS" } else { "FAIL" }, m * n);
}

fn run<R: Runtime>(device: &R::Device) {
    for sz in [2048usize, 4096, 8192] {
        run_size::<R>(device, sz);
    }
}

fn run_size<R: Runtime>(device: &R::Device, sz: usize) {
    let client = R::client(device);
    let (m, n, k) = (sz, sz, sz);
    type AB = e2m1x2;
    type S = ue8m0;
    let ab_elem = AB::cube_type();
    let ab_vector_size = 32 / ab_elem.size_bits();

    let scale_cols = k / (MMA_K / SF);
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

    let grid_x = (n / BN) as u32;
    let grid_y = (m / BM) as u32;

    let launch = |out_buf: cubecl::server::Handle| unsafe {
        matmul_fp4_pipe2::launch_unchecked::<AB, AB, f32, S, R>(
            &client,
            CubeCount::Static(grid_x, grid_y, 1),
            CubeDim::new_1d(N_THREADS),
            ab_vector_size,
            ab_vector_size,
            2,
            a_map::<R>(lhs_h.clone(), m, k, BM),
            a_map::<R>(rhs_h.clone(), n, k, BN),
            TensorArg::from_raw_parts(c_h.clone(), [n, 1].into(), [m, n].into()),
            TensorArg::from_raw_parts(lhs_scales_h.clone(), [scale_cols, 1].into(), [m, scale_cols].into()),
            TensorArg::from_raw_parts(rhs_scales_h.clone(), [scale_cols, 1].into(), [n, scale_cols].into()),
            TensorArg::from_raw_parts(out_buf, [n, 1].into(), [m, n].into()),
            n,
            k,
        );
    };

    for _ in 0..5 {
        launch(out_h.clone());
    }
    let _ = future::block_on(client.sync());

    let mut best = Duration::MAX;
    for _ in 0..20 {
        let (_, profile) = client.profile(|| launch(out_h.clone()), "matmul_fp4_pipe2").unwrap();
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
    println!("shape: {m}x{n}x{k}   out[0]: {}  best: {:.3} ms   {gflops:.1} GFLOP/s   {}", result[0], secs * 1e3, if wrong == 0 { "PASS" } else { "FAIL" });
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        verify::<cubecl::cuda::CudaRuntime>(&Default::default());
        run::<cubecl::cuda::CudaRuntime>(&Default::default());
    }
    #[cfg(feature = "cpu")]
    {
        verify::<cubecl::cpu::CpuRuntime>(&Default::default());
        run::<cubecl::cpu::CpuRuntime>(&Default::default());
    }
    #[cfg(not(any(feature = "cuda", feature = "cpu")))]
    panic!("build with --features cuda (on a GPU) or --features cpu");
}
