//! Double-buffer (software-prefetch) experiment for the fed FP4 matmul. The
//! square-tile experiment showed the 2-blocks/SM config wins by BARRIER OVERLAP
//! (one block feeds the tensor cores while the other stalls at sync_cube), not by
//! staging intensity. Double buffering attacks that barrier directly: it stages
//! the NEXT k-tile into a second shared buffer while computing the current one,
//! so there is one barrier per step (after-stage) instead of two (after-stage +
//! after-compute), and the next tile's global loads are in flight during the
//! current compute (intra-block latency hiding). Unlike the cp.async pipeline
//! that regressed, these are plain loads (no mbarrier/async overhead); the only
//! cost is 2x shared memory, which still fits 2 blocks/SM at the 4-warp config.
//!
//! Fixed 4-warp 64x128 tile (the proven best). Variant selected by comptime
//! `double_buf`; timed interleaved (same clock) against the single-buffer
//! baseline. The staging block is duplicated at three sites (prologue, single
//! in-loop, double prefetch) rather than factored into a #[cube] helper.
//! ponytail: duplicated staging, throwaway A/B experiment, clarity over a helper.

use std::time::Duration;

use cubecl::features::ScaledMmaConfig;
use cubecl::future;
use cubecl::ir::MatrixIdent;
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
const KSUB: usize = BK / MMA_K;
const FV: usize = 8;
const VPR: usize = BK / FV;
const N_THREADS: u32 = (WARPS_M * WARPS_N * 32) as u32; // 128

#[cube(launch_unchecked)]
fn fed<A: Scalar, B: Scalar, CD: Numeric, S: Scalar, NA: Size, NB: Size, NC: Size>(
    a: &Tensor<Vector<A, NA>>,
    b: &Tensor<Vector<B, NB>>,
    c: &Tensor<Vector<CD, NC>>,
    scales_a: &Tensor<S>,
    scales_b: &Tensor<S>,
    out: &mut Tensor<Vector<CD, NC>>,
    #[comptime] full_n: usize,
    #[comptime] full_k: usize,
    #[comptime] double_buf: bool,
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
    let global_vpr = comptime!(full_k / FV);

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

    let nbuf = comptime!(if double_buf { 2 } else { 1 });
    let a_buf = comptime!(BM * VPR); // per-buffer vector counts
    let b_buf = comptime!(BN * VPR);
    let bps = comptime!(BK / (MMA_K / SF));
    let sa_buf = comptime!(BM * bps);
    let sb_buf = comptime!(BN * bps);
    let mut a_tile = SharedMemory::<Vector<A, NA>>::new(comptime!(a_buf * nbuf));
    let mut b_tile = SharedMemory::<Vector<B, NB>>::new(comptime!(b_buf * nbuf));
    let mut sa_tile = SharedMemory::<S>::new(comptime!(sa_buf * nbuf));
    let mut sb_tile = SharedMemory::<S>::new(comptime!(sb_buf * nbuf));

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

    // ---- prologue: for double buffering, prefetch step 0 into buffer 0 ----
    if double_buf {
        #[unroll]
        for i in 0..(BM * VPR) / n_threads {
            let s = tid + i * n_threads;
            a_tile[s] = a[(block_row + s / VPR) * global_vpr + s % VPR];
        }
        #[unroll]
        for i in 0..(BN * VPR) / n_threads {
            let s = tid + i * n_threads;
            b_tile[s] = b[(block_col + s / VPR) * global_vpr + s % VPR];
        }
        #[unroll]
        for i in 0..(BM * bps) / n_threads {
            let s = tid + i * n_threads;
            sa_tile[s] = scales_a[(block_row + s / bps) * scale_blocks_per_row + s % bps];
        }
        #[unroll]
        for i in 0..(BN * bps) / n_threads {
            let s = tid + i * n_threads;
            sb_tile[s] = scales_b[(block_col + s / bps) * scale_blocks_per_row + s % bps];
        }
        sync_cube();
    }

    for step in 0..k_steps {
        let cur = step % nbuf;
        let a_base = cur * a_buf;
        let b_base = cur * b_buf;
        let sa_base = cur * sa_buf;
        let sb_base = cur * sb_buf;

        // ---- single-buffer: stage current tile then barrier ----
        if !double_buf {
            let k_vec = step * VPR;
            #[unroll]
            for i in 0..(BM * VPR) / n_threads {
                let s = tid + i * n_threads;
                a_tile[a_base + s] = a[(block_row + s / VPR) * global_vpr + k_vec + s % VPR];
            }
            #[unroll]
            for i in 0..(BN * VPR) / n_threads {
                let s = tid + i * n_threads;
                b_tile[b_base + s] = b[(block_col + s / VPR) * global_vpr + k_vec + s % VPR];
            }
            #[unroll]
            for i in 0..(BM * bps) / n_threads {
                let s = tid + i * n_threads;
                sa_tile[sa_base + s] = scales_a[(block_row + s / bps) * scale_blocks_per_row + step * bps + s % bps];
            }
            #[unroll]
            for i in 0..(BN * bps) / n_threads {
                let s = tid + i * n_threads;
                sb_tile[sb_base + s] = scales_b[(block_col + s / bps) * scale_blocks_per_row + step * bps + s % bps];
            }
            sync_cube();
        }

        // ---- compute from the current buffer ----
        #[unroll]
        for ks in 0..KSUB {
            let k_off = ks * MMA_K;
            let sk = ks * SF;

            #[unroll]
            for i in 0..vector_count_a {
                let n_elem = i * vector_size_a * a_pack;
                let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
                let div = a.vector_size() * a_pack;
                a0[i] = a_tile[a_base + ((a_row0 + row as usize) * BK + k_off + col as usize) / div];
                a1[i] = a_tile[a_base + ((a_row1 + row as usize) * BK + k_off + col as usize) / div];
            }
            #[unroll]
            for i in 0..scales_count {
                sa0[i] = sa_tile[sa_base + (a_row0 + scales_idx_a) * bps + sk + i];
                sa1[i] = sa_tile[sa_base + (a_row1 + scales_idx_a) * bps + sk + i];
            }
            #[unroll]
            for i in 0..vector_count_b {
                let n_elem = i * vector_size_b * b_pack;
                let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
                let div = b.vector_size() * b_pack;
                b0[i] = b_tile[b_base + ((b_col0 + col as usize) * BK + k_off + row as usize) / div];
                b1[i] = b_tile[b_base + ((b_col1 + col as usize) * BK + k_off + row as usize) / div];
                b2[i] = b_tile[b_base + ((b_col2 + col as usize) * BK + k_off + row as usize) / div];
                b3[i] = b_tile[b_base + ((b_col3 + col as usize) * BK + k_off + row as usize) / div];
                b4[i] = b_tile[b_base + ((b_col4 + col as usize) * BK + k_off + row as usize) / div];
                b5[i] = b_tile[b_base + ((b_col5 + col as usize) * BK + k_off + row as usize) / div];
                b6[i] = b_tile[b_base + ((b_col6 + col as usize) * BK + k_off + row as usize) / div];
                b7[i] = b_tile[b_base + ((b_col7 + col as usize) * BK + k_off + row as usize) / div];
            }
            #[unroll]
            for i in 0..scales_count {
                sb0[i] = sb_tile[sb_base + (b_col0 + scales_idx_b) * bps + sk + i];
                sb1[i] = sb_tile[sb_base + (b_col1 + scales_idx_b) * bps + sk + i];
                sb2[i] = sb_tile[sb_base + (b_col2 + scales_idx_b) * bps + sk + i];
                sb3[i] = sb_tile[sb_base + (b_col3 + scales_idx_b) * bps + sk + i];
                sb4[i] = sb_tile[sb_base + (b_col4 + scales_idx_b) * bps + sk + i];
                sb5[i] = sb_tile[sb_base + (b_col5 + scales_idx_b) * bps + sk + i];
                sb6[i] = sb_tile[sb_base + (b_col6 + scales_idx_b) * bps + sk + i];
                sb7[i] = sb_tile[sb_base + (b_col7 + scales_idx_b) * bps + sk + i];
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

        // ---- double-buffer: prefetch the NEXT tile into the other buffer ----
        if double_buf {
            let nxt = (step + 1) % nbuf;
            let na_base = nxt * a_buf;
            let nb_base = nxt * b_buf;
            let nsa_base = nxt * sa_buf;
            let nsb_base = nxt * sb_buf;
            let k_vec = (step + 1) * VPR;
            if step + 1 < k_steps {
                #[unroll]
                for i in 0..(BM * VPR) / n_threads {
                    let s = tid + i * n_threads;
                    a_tile[na_base + s] = a[(block_row + s / VPR) * global_vpr + k_vec + s % VPR];
                }
                #[unroll]
                for i in 0..(BN * VPR) / n_threads {
                    let s = tid + i * n_threads;
                    b_tile[nb_base + s] = b[(block_col + s / VPR) * global_vpr + k_vec + s % VPR];
                }
                #[unroll]
                for i in 0..(BM * bps) / n_threads {
                    let s = tid + i * n_threads;
                    sa_tile[nsa_base + s] = scales_a[(block_row + s / bps) * scale_blocks_per_row + (step + 1) * bps + s % bps];
                }
                #[unroll]
                for i in 0..(BN * bps) / n_threads {
                    let s = tid + i * n_threads;
                    sb_tile[nsb_base + s] = scales_b[(block_col + s / bps) * scale_blocks_per_row + (step + 1) * bps + s % bps];
                }
            }
            sync_cube();
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

const VARIANTS: [bool; 2] = [false, true];

fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    type AB = e2m1x2;
    type S = ue8m0;
    let ab_elem = AB::cube_type();
    let ab_vs = 32 / ab_elem.size_bits();

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

    let make = |m: usize, n: usize, k: usize, seed: u32| {
        let scale_cols = k / (MMA_K / SF);
        let mut st = seed;
        let lhs_f: Vec<f32> = (0..m * k).map(|_| rand_fp4(&mut st)).collect();
        let rhs_f: Vec<f32> = (0..n * k).map(|_| rand_fp4(&mut st)).collect();
        let lhs_sb: Vec<S> = (0..m * scale_cols)
            .map(|_| ue8m0::from_bits(126 + (xorshift(&mut st) % 3) as u8))
            .collect();
        let rhs_sb: Vec<S> = (0..n * scale_cols)
            .map(|_| ue8m0::from_bits(126 + (xorshift(&mut st) % 3) as u8))
            .collect();
        let lhs = e2m1x2::from_f32_slice(&lhs_f);
        let rhs = e2m1x2::from_f32_slice(&rhs_f);
        let zeros = vec![0.0f32; m * n];
        let lhs_h = client.create_from_slice(AB::as_bytes(&lhs));
        let rhs_h = client.create_from_slice(AB::as_bytes(&rhs));
        let lsb_h = client.create_from_slice(S::as_bytes(&lhs_sb));
        let rsb_h = client.create_from_slice(S::as_bytes(&rhs_sb));
        let c_h = client.create_from_slice(f32::as_bytes(&zeros));
        (lhs_h, rhs_h, lsb_h, rsb_h, c_h, lhs_f, rhs_f, lhs_sb, rhs_sb)
    };

    let launch = |db: bool, m: usize, n: usize, k: usize, lhs_h: &cubecl::server::Handle, rhs_h: &cubecl::server::Handle, lsb_h: &cubecl::server::Handle, rsb_h: &cubecl::server::Handle, c_h: &cubecl::server::Handle, out_h: cubecl::server::Handle| {
        let scale_cols = k / (MMA_K / SF);
        unsafe {
            fed::launch_unchecked::<AB, AB, f32, S, R>(
                &client,
                CubeCount::Static((n / BN) as u32, (m / BM) as u32, 1),
                CubeDim::new_1d(N_THREADS),
                ab_vs,
                ab_vs,
                2,
                TensorArg::from_raw_parts(lhs_h.clone(), [k / 2, 1].into(), [m, k / 2].into()),
                TensorArg::from_raw_parts(rhs_h.clone(), [k / 2, 1].into(), [n, k / 2].into()),
                TensorArg::from_raw_parts(c_h.clone(), [n, 1].into(), [m, n].into()),
                TensorArg::from_raw_parts(lsb_h.clone(), [scale_cols, 1].into(), [m, scale_cols].into()),
                TensorArg::from_raw_parts(rsb_h.clone(), [scale_cols, 1].into(), [n, scale_cols].into()),
                TensorArg::from_raw_parts(out_h, [n, 1].into(), [m, n].into()),
                n,
                k,
                db,
            );
        }
    };

    let (vm, vn, vk) = (512usize, 512usize, 512usize);
    for db in VARIANTS {
        let (lhs_h, rhs_h, lsb_h, rsb_h, c_h, lhs_f, rhs_f, lhs_sb, rhs_sb) = make(vm, vn, vk, 0x1357);
        let out_h = client.empty(vm * vn * core::mem::size_of::<f32>());
        launch(db, vm, vn, vk, &lhs_h, &rhs_h, &lsb_h, &rsb_h, &c_h, out_h.clone());
        let bytes = client.read_one(out_h).unwrap();
        let result = f32::from_bytes(&bytes);
        let blk = MMA_K / SF;
        let scale_cols = vk / blk;
        let mut wrong = 0;
        let mut max_rel = 0.0f64;
        for i in 0..vm {
            for j in 0..vn {
                let mut sum = 0.0f32;
                for l in 0..vk {
                    let sa = lhs_sb[i * scale_cols + l / blk].to_f32();
                    let sb = rhs_sb[j * scale_cols + l / blk].to_f32();
                    sum += lhs_f[i * vk + l] * sa * rhs_f[j * vk + l] * sb;
                }
                let rel = ((result[i * vn + j] - sum).abs() / sum.abs().max(1.0)) as f64;
                if rel > max_rel {
                    max_rel = rel;
                }
                if rel > 1e-3 {
                    wrong += 1;
                }
            }
        }
        println!(
            "verify double_buf={db}: {} ({wrong} wrong, max_rel={max_rel:.2e})",
            if wrong == 0 { "PASS" } else { "FAIL" }
        );
    }

    let (m, n, k) = (4096usize, 4096usize, 4096usize);
    let (lhs_h, rhs_h, lsb_h, rsb_h, c_h, ..) = make(m, n, k, 0x2468);
    let out_h = client.empty(m * n * core::mem::size_of::<f32>());

    for db in VARIANTS {
        for _ in 0..5 {
            launch(db, m, n, k, &lhs_h, &rhs_h, &lsb_h, &rsb_h, &c_h, out_h.clone());
        }
    }
    let _ = future::block_on(client.sync());

    let mut best = [Duration::MAX; 2];
    for _ in 0..30 {
        for (idx, db) in VARIANTS.iter().enumerate() {
            let (_, profile) = client
                .profile(
                    || launch(*db, m, n, k, &lhs_h, &rhs_h, &lsb_h, &rsb_h, &c_h, out_h.clone()),
                    "fed",
                )
                .unwrap();
            let e = future::block_on(profile.resolve()).duration();
            if e < best[idx] {
                best[idx] = e;
            }
        }
    }
    let flop = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let g0 = flop / best[0].as_secs_f64() / 1e9;
    let g1 = flop / best[1].as_secs_f64() / 1e9;
    println!("shape: {m}x{n}x{k}  (interleaved, best of 30)");
    println!("single-buffer: {:.3} ms  {g0:.1} GFLOP/s", best[0].as_secs_f64() * 1e3);
    println!("double-buffer: {:.3} ms  {g1:.1} GFLOP/s", best[1].as_secs_f64() * 1e3);
    println!("speedup (double/single): {:.4}x", g1 / g0);
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "cpu")]
    run::<cubecl::cpu::CpuRuntime>(&Default::default());
    #[cfg(not(any(feature = "cuda", feature = "cpu")))]
    panic!("build with --features cuda (on a GPU) or --features cpu");
}
