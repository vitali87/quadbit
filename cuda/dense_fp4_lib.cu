// PyTorch-callable DENSE FP4 matmul (the pingpong kernel), unit ue8m0 scales, for the
// head-to-head benchmark vs cuBLAS bf16. C[M,N] = A[M,K] @ B[N,K]^T in FP4 (bf16 out).
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#define DBM 128
#define DBN 128
#define DBK 128
#define DBKH 64
#define DSTAGES 3
#define DASZ (DBM * DBKH)
#define DBSZ (DBN * DBKH)
#define DWG (DSTAGES * DASZ + DSTAGES * DBSZ)
#define DSMEM (2 * DWG + 2 * DSTAGES * 8 + 128)

__global__ void __launch_bounds__(256)
dmatmul_fp4(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
            __nv_bfloat16 *C, int N, int Kfp4) {
    extern __shared__ __align__(128) uint8_t smem[];
    int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127;
    int warp = wtid >> 5, lane = wtid & 31;
    int wm = warp >> 1, wn = warp & 1;
    uint8_t *a_s = smem + wg * DWG;
    uint8_t *b_s = a_s + DSTAGES * DASZ;
    uint64_t *full = (uint64_t *)(smem + 2 * DWG + wg * DSTAGES * 8);
    int sync_id = wg + 1;
    int block_row = blockIdx.y * DBM, block_col = blockIdx.x * (2 * DBN) + wg * DBN;
    int ksteps = Kfp4 / DBK;
    int arow = ((lane >> 3) & 1) * 8 + (lane & 7), acblk = (lane >> 3) >> 1;
    int nrow = lane & 7, bkblk = (lane >> 3) & 1;
    int a_rowt[4], b_col[8];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) a_rowt[mt] = wm * 64 + mt * 16;
#pragma unroll
    for (int j = 0; j < 8; j++) b_col[j] = wn * 64 + j * 8;
    float acc[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
    uint16_t z = 0;
    if (wtid == 0) {
#pragma unroll
        for (int s = 0; s < DSTAGES; s++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])));
        asm volatile("fence.proxy.async.shared::cta;");
    }
    asm volatile("bar.sync %0, 128;" ::"r"(sync_id));
    auto issue = [&](int s, int kstep) {
        uint32_t bar = (uint32_t)__cvta_generic_to_shared(&full[s]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((uint32_t)(DASZ + DBSZ)));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&a_s[s * DASZ])), "l"(&mapA), "r"(kstep * DBKH), "r"(block_row), "r"(bar));
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&b_s[s * DBSZ])), "l"(&mapB), "r"(kstep * DBKH), "r"(block_col), "r"(bar));
    };
    if (wtid == 0)
#pragma unroll
        for (int s = 0; s < DSTAGES; s++)
            if (s < ksteps) issue(s, s);
    for (int step = 0; step < ksteps; step++) {
        int s = step % DSTAGES;
        uint32_t par = (step / DSTAGES) & 1;
        asm volatile("{\n\t.reg .pred p;\nWA:\n\tmbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n\t@!p bra WA;\n\t}\n" ::"r"((uint32_t)__cvta_generic_to_shared(&full[s])), "r"(par));
        int aoff = s * DASZ, boff = s * DBSZ;
#pragma unroll
        for (int ks = 0; ks < DBK / 64; ks++) {
            int kb = ks * 32;
            uint32_t af[4][4], bf[8][2];
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                int ao = (a_rowt[mt] + arow) * DBKH + kb + acblk * 16; ao ^= ((ao >> 7) & 3) << 4;
                uint32_t ad = __cvta_generic_to_shared(&a_s[aoff + ao]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[mt][0]), "=r"(af[mt][1]), "=r"(af[mt][2]), "=r"(af[mt][3]) : "r"(ad));
            }
#pragma unroll
            for (int n = 0; n < 8; n++) {
                int bo = (b_col[n] + nrow) * DBKH + kb + bkblk * 16; bo ^= ((bo >> 7) & 3) << 4;
                uint32_t bd = __cvta_generic_to_shared(&b_s[boff + bo]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 {%0,%1}, [%2];"
                             : "=r"(bf[n][0]), "=r"(bf[n][1]) : "r"(bd));
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int n = 0; n < 8; n++) {
                    int idx = mt * 8 + n; float d0, d1, d2, d3;
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14},{%15,%16}, {%17},{%18,%19};"
                        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                        : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]), "r"(bf[n][0]), "r"(bf[n][1]),
                          "f"(acc[idx][0]), "f"(acc[idx][1]), "f"(acc[idx][2]), "f"(acc[idx][3]),
                          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
                    acc[idx][0] = d0; acc[idx][1] = d1; acc[idx][2] = d2; acc[idx][3] = d3;
                }
        }
        asm volatile("bar.sync %0, 128;" ::"r"(sync_id));
        int next = step + DSTAGES;
        if (wtid == 0 && next < ksteps) issue(s, next);
    }
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int idx = mt * 8 + n;
            int gr = block_row + a_rowt[mt] + (lane >> 2), gc = block_col + b_col[n] + (lane & 3) * 2;
            *reinterpret_cast<__nv_bfloat162 *>(&C[gr * N + gc]) = __floats2bfloat162_rn(acc[idx][0], acc[idx][1]);
            *reinterpret_cast<__nv_bfloat162 *>(&C[(gr + 8) * N + gc]) = __floats2bfloat162_rn(acc[idx][2], acc[idx][3]);
        }
}

static void dmk(CUtensorMap *m, uint8_t *p, int Kb, int rows, int boxrows) {
    uint64_t gd[2] = {(uint64_t)Kb, (uint64_t)rows}; uint64_t gs[1] = {(uint64_t)Kb};
    uint32_t bd[2] = {(uint32_t)DBKH, (uint32_t)boxrows}, es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("dmap fail %s\n", s); }
}

extern "C" int dense_fp4_mm(const void *A, const void *B, void *C, int M, int N, int K) {
    int Kb = K / 2;
    alignas(64) CUtensorMap mapA, mapB;
    dmk(&mapA, (uint8_t *)A, Kb, M, DBM);
    dmk(&mapB, (uint8_t *)B, Kb, N, DBN);
    cudaFuncSetAttribute(dmatmul_fp4, cudaFuncAttributeMaxDynamicSharedMemorySize, DSMEM);
    dim3 grid(N / (2 * DBN), M / DBM), block(256);
    dmatmul_fp4<<<grid, block, DSMEM>>>(mapA, mapB, (__nv_bfloat16 *)C, N, K);
    return (int)cudaDeviceSynchronize();
}
