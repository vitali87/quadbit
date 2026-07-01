// 2:4-sparse FP4 with SPLIT-K, PyTorch-callable, for weight-stationary DECODE. Orient
// C[out,tok] = W[out,in] @ X[tok,in]^T so the 2:4 weight is the compressed mma-A (M=out large)
// and tok is the thin N. The prefill sparse kernel then gives only M/(2*BM) blocks (56 for
// out=14336, 16 for 4096) -> SM underfill. Split-K (gridDim.z chunk-range CTAs, f32 atomic +
// convert) multiplies block count to fill the machine while the compressed weight streams once
// at HALF the DRAM bytes of dense FP4. Ported from sparse_fp4_lib matmul_sp; only the chunk
// range, epilogue (f32 atomic when splits>1), and a convert pass are added.
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#define BM 128
#define BN 128
#define WK 2
#define AROWB 32
#define BROWB 64
#define AW (AROWB * WK)
#define BW_ (BROWB * WK)
#define STAGES 2
#define ASZ (BM * AW)
#define BSZ (BN * BW_)
#define SCA 1024
#define SCB 512
#define MET 2048
#define SMEM (2*STAGES*ASZ + STAGES*BSZ + STAGES*WK*SCA + STAGES*WK*SCB + STAGES*WK*MET + 2*STAGES*8 + 128)

__global__ void __launch_bounds__(256)
matmul_sp_sk(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
             const uint8_t *scaleA, const uint8_t *scaleB, const uint32_t *meta,
             __nv_bfloat16 *C, float *Cf, int M, int N, int Klog, int splits) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;
    uint8_t *a_s = smem + wg * STAGES * ASZ;
    uint8_t *b_s = smem + 2 * STAGES * ASZ;
    uint8_t *scA_sm = b_s + STAGES * BSZ;
    uint8_t *scB_sm = scA_sm + STAGES * WK * SCA;
    uint8_t *met_sm = scB_sm + STAGES * WK * SCB;
    uint64_t *full = (uint64_t *)(met_sm + STAGES * WK * MET);
    uint64_t *empty = full + STAGES;

    int block_row = blockIdx.y * (2 * BM) + wg * BM;
    int a_load_row = blockIdx.y * (2 * BM), block_col = blockIdx.x * BN;
    int chunks_total = Klog / (128 * WK);
    int z = blockIdx.z;
    int c0 = (int)((long)z * chunks_total / splits);
    int c1 = (int)((long)(z + 1) * chunks_total / splits);
    int nchunk = c1 - c0;

    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bsub = (lane >> 3) & 3;
    int a_rowt[4], b_col[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_rowt[mt] = wm * 64 + mt * 16;
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = wn * 64 + j * 8;
    int ra_local = (lane & 3) * 8 + (lane >> 2), cb_local = lane >> 2;
    bool a_valid = ra_local < 16, b_valid = (lane & 3) == 0;
    int a_sidx[4], b_sidx[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_sidx[mt] = wg * 128 + a_rowt[mt] + ra_local;
#pragma unroll
    for (int n = 0; n < 8; n++) b_sidx[n] = b_col[n] + cb_local;
    int mma_row = (lane & 1) * 8 + (lane >> 2), Hh = (lane >> 1) & 1;
    int m_sidx[4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) m_sidx[mt] = (wg * 128 + a_rowt[mt] + mma_row) * 2 + Hh;

    float acc[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint16_t z16 = 0;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 256;" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        }
        asm volatile("fence.proxy.async.shared::cta;");
    }
    __syncthreads();

    auto issue = [&](int s, int chunk) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
                     "r"((uint32_t)(2 * ASZ + BSZ + WK * SCA + WK * SCB + WK * MET)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[s * ASZ])), "l"(&mapA), "r"(chunk * AW), "r"(a_load_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&smem[STAGES * ASZ + s * ASZ])), "l"(&mapA), "r"(chunk * AW), "r"(a_load_row + BM), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * BSZ])), "l"(&mapB), "r"(chunk * BW_), "r"(block_col), "r"(bar));
#pragma unroll
        for (int sub = 0; sub < WK; sub++) {
            int step = chunk * WK + sub;
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&scA_sm[(s * WK + sub) * SCA])), "l"(scaleA + (size_t)(step * M + a_load_row) * 4), "r"((uint32_t)SCA), "r"(bar));
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&scB_sm[(s * WK + sub) * SCB])), "l"(scaleB + (size_t)(step * N + block_col) * 4), "r"((uint32_t)SCB), "r"(bar));
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];" ::
                         "r"((uint32_t)__cvta_generic_to_shared(&met_sm[(s * WK + sub) * MET])), "l"((const uint8_t *)meta + (size_t)(step * M + a_load_row) * 8), "r"((uint32_t)MET), "r"(bar));
        }
    };
    if (tid == 0)
#pragma unroll
        for (int s = 0; s < STAGES; s++)
            if (s < nchunk) issue(s, c0 + s);

    for (int ci = 0; ci < nchunk; ci++) {
        int chunk = c0 + ci;
        int s = ci % STAGES; uint32_t par = (ci / STAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * ASZ, boff = s * BSZ;
#pragma unroll
        for (int sub = 0; sub < WK; sub++) {
            const uint32_t *scA = (const uint32_t *)(scA_sm + (s * WK + sub) * SCA);
            const uint32_t *scB = (const uint32_t *)(scB_sm + (s * WK + sub) * SCB);
            const uint32_t *mtA = (const uint32_t *)(met_sm + (s * WK + sub) * MET);
            uint32_t sav[4], sbv[8], ev[4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) { sav[mt] = a_valid ? scA[a_sidx[mt]] : 0x38383838u; ev[mt] = mtA[m_sidx[mt]]; }
#pragma unroll
            for (int n = 0; n < 8; n++) sbv[n] = b_valid ? scB[b_sidx[n]] : 0x38383838u;
            uint32_t af[4][4], bf[8][4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int ao = (a_rowt[mt] + arow) * AW + sub * AROWB + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * BW_ + sub * BROWB + bsub * 16; bo ^= ((bo >> 7) & 7) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(bf[n][0]), "=r"(bf[n][1]), "=r"(bf[n][2]), "=r"(bf[n][3]) : "r"(bd));
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = mt * 8 + n; float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]),
                          "r"(bf[n][0]), "r"(bf[n][1]), "r"(bf[n][2]), "r"(bf[n][3]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(ev[mt]), "r"(sav[mt]), "h"(z16), "h"(z16), "r"(sbv[n]), "h"(z16), "h"(z16));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        }
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])));
        int nci = ci + STAGES;
        if (tid == 0 && nci < nchunk) {
            asm volatile("{\n\t.reg .pred p;\nWE:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WE;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&empty[s])), "r"(par));
            issue(s, c0 + nci);
        }
    }
    bool single = (splits == 1);
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = mt * 8 + n;
            int gr = block_row + a_rowt[mt] + (lane >> 2), gc = block_col + b_col[n] + (lane & 3) * 2;
            if (single) {
                *reinterpret_cast<__nv_bfloat162 *>(&C[gr * N + gc]) = __floats2bfloat162_rn(acc[idx][0], acc[idx][1]);
                *reinterpret_cast<__nv_bfloat162 *>(&C[(gr + 8) * N + gc]) = __floats2bfloat162_rn(acc[idx][2], acc[idx][3]);
            } else {
                float *p0 = &Cf[gr * N + gc], *p1 = &Cf[(gr + 8) * N + gc];
                atomicAdd(p0, acc[idx][0]); atomicAdd(p0 + 1, acc[idx][1]);
                atomicAdd(p1, acc[idx][2]); atomicAdd(p1 + 1, acc[idx][3]);
            }
        }
}

__global__ void cvt_sp(const float *Cf, __nv_bfloat16 *C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = __float2bfloat16_rn(Cf[i]);
}

static void mk(CUtensorMap *m, uint8_t *p, int inner, int outer, int bi, int bo, CUtensorMapSwizzle sw) {
    uint64_t gd[2] = {(uint64_t)inner, (uint64_t)outer}; uint64_t gs[1] = {(uint64_t)inner};
    uint32_t bd[2] = {(uint32_t)bi, (uint32_t)bo}, es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("map fail %s\n", s); }
}

// Cf = caller f32 workspace of M*N (unused when splits==1).
extern "C" int sparse_fp4_mm_sk(const void *A, const void *B, const void *scaleA, const void *scaleB,
                                const void *meta, void *C, void *Cf, int M, int N, int Klog, int splits) {
    int KAb = Klog / 4, KBb = Klog / 2;
    alignas(64) CUtensorMap mapA, mapB;
    mk(&mapA, (uint8_t *)A, KAb, M, AW, BM, CU_TENSOR_MAP_SWIZZLE_64B);
    mk(&mapB, (uint8_t *)B, KBb, N, BW_, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    cudaFuncSetAttribute(matmul_sp_sk, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(N / BN, M / (2 * BM), splits), block(256);
    size_t ne = (size_t)M * N;
    if (splits > 1) cudaMemset(Cf, 0, ne * sizeof(float));
    matmul_sp_sk<<<grid, block, SMEM>>>(mapA, mapB, (const uint8_t *)scaleA, (const uint8_t *)scaleB,
                                        (const uint32_t *)meta, (__nv_bfloat16 *)C, (float *)Cf, M, N, Klog, splits);
    if (splits > 1) {
        int blk = 256, gr = (int)((ne + blk - 1) / blk);
        cvt_sp<<<gr, blk>>>((float *)Cf, (__nv_bfloat16 *)C, ne);
    }
    return (int)cudaDeviceSynchronize();
}
