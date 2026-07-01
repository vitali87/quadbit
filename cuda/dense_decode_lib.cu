// PyTorch-callable DECODE FP4 (split-N, direct bf16, no reduction) for small-M memory-bound
// shapes. Narrow TN=NWARP*8 column tile, each warp owns one 8-col n-tile over all 128 rows +
// full K -> grid = N/TN x M/128 blocks, direct bf16. Config is ADAPTIVE by N (from an occupancy
// sweep): large N wants more warps/block (NWARP=8), small N wants deeper pipelines + more blocks
// (NWARP=4, STAGES=8). More blocks than that backfires -- each N-block re-reads the full
// activation A from L2, so block count multiplies A traffic. Long-K decode (ffn down) should use
// split-K (dense_sk_lib) instead. Maps encode address+tiling not contents -> build weight map
// once, reuse every step; async launch avoids per-call sync (matches torch).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <cuda_bf16.h>

#define BM 128
#define BK 128
#define BKH 64

template <int NWARP, int STAGES>
__global__ void __launch_bounds__(NWARP * 32)
decode_mm(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          __nv_bfloat16 *C, int N, int Kfp4) {
    constexpr int TN = NWARP * 8;
    constexpr int ASZ = BM * BKH;
    constexpr int BSZ = TN * BKH;
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint8_t *a_s = smem;
    uint8_t *b_s = a_s + STAGES * ASZ;
    uint64_t *full = (uint64_t *)(b_s + STAGES * BSZ);
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * TN;
    int ksteps = Kfp4 / BK;
    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bkblk = (lane >> 3) & 1;
    int a_rowt[8];
#pragma unroll
    for (int mt = 0; mt < 8; mt++) a_rowt[mt] = mt * 16;
    int bcol = warp * 8;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    uint16_t zz = 0;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    asm volatile("bar.sync 0;");
    auto issue = [&](int s, int kstep) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(ASZ + BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&a_s[s * ASZ])), "l"(&mapA), "r"(kstep * BKH), "r"(block_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])), "l"(&mapB), "r"(kstep * BKH), "r"(block_col), "r"(bar));
    };
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < ksteps) issue(s, s);
    for (int step = 0; step < ksteps; step++) {
        int s = step % STAGES;
        uint32_t par = (step / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * ASZ, boff = s * BSZ;
#pragma unroll
        for (int ks = 0; ks < BK / 64; ks++) {
            int kb = ks * 32;
            uint32_t af[8][4], bf[2];
#pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                int ao = (a_rowt[mt] + arow) * BKH + kb + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
            }
            int bo = (bcol + nrow) * BKH + kb + bkblk * 16; bo ^= ((bo >> 7) & 3) << 4;
            uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];" : "=r"(bf[0]), "=r"(bf[1]) : "r"(bd));
#pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                float d0, d1, d2, d3;
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]), "r"(bf[0]), "r"(bf[1]),
                      "f"(acc[mt][0]), "f"(acc[mt][1]), "f"(acc[mt][2]), "f"(acc[mt][3]),
                      "r"(sa), "h"(zz), "h"(zz), "r"(sb), "h"(zz), "h"(zz));
                acc[mt][0] = d0; acc[mt][1] = d1; acc[mt][2] = d2; acc[mt][3] = d3;
            }
        }
        asm volatile("bar.sync 0;");
        int next = step + STAGES;
        if (tid == 0 && next < ksteps) issue(s, next);
    }
#pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        int gr = block_row + a_rowt[mt] + (lane >> 2), gc = block_col + bcol + (lane & 3) * 2;
        *reinterpret_cast<__nv_bfloat162 *>(&C[gr * N + gc]) = __floats2bfloat162_rn(acc[mt][0], acc[mt][1]);
        *reinterpret_cast<__nv_bfloat162 *>(&C[(gr + 8) * N + gc]) = __floats2bfloat162_rn(acc[mt][2], acc[mt][3]);
    }
}

// SPLIT-K variant of the efficient TN split-N structure: keeps the narrow TN column tile (so
// N-blocks stay 128 for N=4096, no A re-read blowup) and adds gridDim.z K-splits -> splits*128
// blocks fills the 188 SMs the plain decode kernel underfills at small N. Each z-CTA sums its
// K-subrange into an f32 workspace via atomicAdd (decode output M*N is tiny so the f32 pass is
// cheap); a convert kernel writes bf16. splits==1 writes bf16 directly (no workspace touch).
template <int NWARP, int STAGES>
__global__ void __launch_bounds__(NWARP * 32)
decode_mm_sk(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
             __nv_bfloat16 *C, float *Cf, int N, int Kfp4, int splits) {
    constexpr int TN = NWARP * 8;
    constexpr int ASZ = BM * BKH;
    constexpr int BSZ = TN * BKH;
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint8_t *a_s = smem;
    uint8_t *b_s = a_s + STAGES * ASZ;
    uint64_t *full = (uint64_t *)(b_s + STAGES * BSZ);
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * TN;
    int ksteps = Kfp4 / BK;
    int z = blockIdx.z;
    int kstart = (int)((long)z * ksteps / splits), kend = (int)((long)(z + 1) * ksteps / splits);
    int nsteps = kend - kstart;
    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bkblk = (lane >> 3) & 1;
    int a_rowt[8];
#pragma unroll
    for (int mt = 0; mt < 8; mt++) a_rowt[mt] = mt * 16;
    int bcol = warp * 8;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    uint16_t zz = 0;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    asm volatile("bar.sync 0;");
    auto issue = [&](int s, int kstep) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(ASZ + BSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&a_s[s * ASZ])), "l"(&mapA), "r"(kstep * BKH), "r"(block_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])), "l"(&mapB), "r"(kstep * BKH), "r"(block_col), "r"(bar));
    };
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < nsteps) issue(s, kstart + s);
    for (int step = 0; step < nsteps; step++) {
        int s = step % STAGES;
        uint32_t par = (step / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWK:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WK;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * ASZ, boff = s * BSZ;
#pragma unroll
        for (int ks = 0; ks < BK / 64; ks++) {
            int kb = ks * 32;
            uint32_t af[8][4], bf[2];
#pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                int ao = (a_rowt[mt] + arow) * BKH + kb + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
            }
            int bo = (bcol + nrow) * BKH + kb + bkblk * 16; bo ^= ((bo >> 7) & 3) << 4;
            uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];" : "=r"(bf[0]), "=r"(bf[1]) : "r"(bd));
#pragma unroll
            for (int mt = 0; mt < 8; mt++) {
                float d0, d1, d2, d3;
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]), "r"(bf[0]), "r"(bf[1]),
                      "f"(acc[mt][0]), "f"(acc[mt][1]), "f"(acc[mt][2]), "f"(acc[mt][3]),
                      "r"(sa), "h"(zz), "h"(zz), "r"(sb), "h"(zz), "h"(zz));
                acc[mt][0] = d0; acc[mt][1] = d1; acc[mt][2] = d2; acc[mt][3] = d3;
            }
        }
        asm volatile("bar.sync 0;");
        int next = step + STAGES;
        if (tid == 0 && next < nsteps) issue(s, kstart + next);
    }
    bool single = (splits == 1);
#pragma unroll
    for (int mt = 0; mt < 8; mt++) {
        int gr = block_row + a_rowt[mt] + (lane >> 2), gc = block_col + bcol + (lane & 3) * 2;
        if (single) {
            *reinterpret_cast<__nv_bfloat162 *>(&C[gr * N + gc]) = __floats2bfloat162_rn(acc[mt][0], acc[mt][1]);
            *reinterpret_cast<__nv_bfloat162 *>(&C[(gr + 8) * N + gc]) = __floats2bfloat162_rn(acc[mt][2], acc[mt][3]);
        } else {
            float *p0 = &Cf[gr * N + gc], *p1 = &Cf[(gr + 8) * N + gc];
            atomicAdd(p0, acc[mt][0]); atomicAdd(p0 + 1, acc[mt][1]);
            atomicAdd(p1, acc[mt][2]); atomicAdd(p1 + 1, acc[mt][3]);
        }
    }
}

__global__ void cvt_dec(const float *Cf, __nv_bfloat16 *C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = __float2bfloat16_rn(Cf[i]);
}

// adaptive config: large N -> 8 warps/block; small N -> 4 warps + deep pipeline.
#define BIG_N 8192
#define NW_BIG 8
#define ST_BIG 4
#define NW_SM 4
#define ST_SM 8

extern "C" int qb_decode_tn(int N) { return (N >= BIG_N ? NW_BIG : NW_SM) * 8; }

static void dmk(CUtensorMap *m, uint8_t *p, int Kb, int rows, int boxrows) {
    uint64_t gd[2] = {(uint64_t)Kb, (uint64_t)rows}; uint64_t gs[1] = {(uint64_t)Kb};
    uint32_t bd[2] = {(uint32_t)BKH, (uint32_t)boxrows}, es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("dmap fail %s\n", s); }
}

template <int NWARP, int STAGES>
static void launch(const CUtensorMap &mapA, const CUtensorMap &mapB, __nv_bfloat16 *C, int M, int N, int K) {
    constexpr int TN = NWARP * 8;
    int SMEM = STAGES * BM * BKH + STAGES * TN * BKH + STAGES * 8 + 128;
    static bool set[2] = {false, false};
    int idx = (NWARP == NW_BIG);
    if (!set[idx]) { cudaFuncSetAttribute(decode_mm<NWARP, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); set[idx] = true; }
    dim3 grid(N / TN, M / BM), block(NWARP * 32);
    decode_mm<NWARP, STAGES><<<grid, block, SMEM>>>(mapA, mapB, C, N, K);
}

static void dispatch(const CUtensorMap &mapA, const CUtensorMap &mapB, __nv_bfloat16 *C, int M, int N, int K) {
    if (N >= BIG_N) launch<NW_BIG, ST_BIG>(mapA, mapB, C, M, N, K);
    else launch<NW_SM, ST_SM>(mapA, mapB, C, M, N, K);
}

extern "C" int dense_fp4_decode(const void *A, const void *B, void *C, int M, int N, int K) {
    int Kb = K / 2;
    alignas(64) CUtensorMap mapA, mapB;
    dmk(&mapA, (uint8_t *)A, Kb, M, BM);
    dmk(&mapB, (uint8_t *)B, Kb, N, qb_decode_tn(N));
    dispatch(mapA, mapB, (__nv_bfloat16 *)C, M, N, K);
    return (int)cudaDeviceSynchronize();
}

// cached-map path. Build B map with boxrows = qb_decode_tn(N); A map with boxrows = 128 (BM).
extern "C" void *qb_encode_map(const void *ptr, int rows, int K, int boxrows) {
    CUtensorMap *m = (CUtensorMap *)aligned_alloc(64, sizeof(CUtensorMap));
    dmk(m, (uint8_t *)ptr, K / 2, rows, boxrows);
    return m;
}
extern "C" void qb_free_map(void *m) { free(m); }

extern "C" int dense_fp4_decode_cached(const void *mapA, const void *mapB, void *C, int M, int N, int K) {
    dispatch(*(const CUtensorMap *)mapA, *(const CUtensorMap *)mapB, (__nv_bfloat16 *)C, M, N, K);
    return (int)cudaDeviceSynchronize();
}

// fire-and-forget: no device sync (caller's stream handles it, like torch) -> decode steps pipeline.
extern "C" void dense_fp4_decode_cached_async(const void *mapA, const void *mapB, void *C, int M, int N, int K) {
    dispatch(*(const CUtensorMap *)mapA, *(const CUtensorMap *)mapB, (__nv_bfloat16 *)C, M, N, K);
}

template <int NWARP, int STAGES>
static void launch_sk(const CUtensorMap &mapA, const CUtensorMap &mapB, __nv_bfloat16 *C,
                      float *Cf, int M, int N, int K, int splits) {
    constexpr int TN = NWARP * 8;
    int SMEM = STAGES * BM * BKH + STAGES * TN * BKH + STAGES * 8 + 128;
    static bool set[2] = {false, false};
    int idx = (NWARP == NW_BIG);
    if (!set[idx]) { cudaFuncSetAttribute(decode_mm_sk<NWARP, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); set[idx] = true; }
    dim3 grid(N / TN, M / BM, splits), block(NWARP * 32);
    if (splits > 1) cudaMemsetAsync(Cf, 0, (size_t)M * N * sizeof(float));
    decode_mm_sk<NWARP, STAGES><<<grid, block, SMEM>>>(mapA, mapB, C, Cf, N, K, splits);
    if (splits > 1) {
        size_t ne = (size_t)M * N; int blk = 256;
        cvt_dec<<<(int)((ne + blk - 1) / blk), blk>>>(Cf, C, ne);
    }
}

// split-K over the narrow-TN split-N structure. Cf = M*N f32 workspace (ignored if splits==1).
// async (no internal sync); caller's stream syncs. Fills SMs the plain decode kernel underfills
// at small N (N=4096 -> 128 N-blocks * splits).
extern "C" void dense_fp4_decode_sk_async(const void *mapA, const void *mapB, void *C, void *Cf,
                                          int M, int N, int K, int splits) {
    if (N >= BIG_N) launch_sk<NW_BIG, ST_BIG>(*(const CUtensorMap *)mapA, *(const CUtensorMap *)mapB,
                                              (__nv_bfloat16 *)C, (float *)Cf, M, N, K, splits);
    else launch_sk<NW_SM, ST_SM>(*(const CUtensorMap *)mapA, *(const CUtensorMap *)mapB,
                                 (__nv_bfloat16 *)C, (float *)Cf, M, N, K, splits);
}
