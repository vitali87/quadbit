// Dense FP4 DECODE kernel: for small-M (few tokens) memory-bound shapes where the normal
// 128x256 tile underfills the SM array (M=128,N=4096 -> 16 blocks) and split-K's f32
// roundtrip + 3 launches are pure overhead. Here: a narrow 128 x TN(=32) tile, ONE
// warpgroup (128 threads / 4 warps), each warp owns one 8-col n-tile over ALL 128 rows and
// the FULL K. grid = (N/TN) x (M/128) -> 128 blocks at N=4096, filling the machine, and each
// block writes bf16 DIRECTLY (no K split, no reduction, no convert, one launch). B (weights)
// streams once from DRAM; A (128xK, tiny) is re-read from L2 by every N-block. Memory floor
// ~5us vs cuBLAS bf16 ~20us at 128/4096/4096.
//
// Run:  uv run modal run harness/run_cuda.py --cu matmul_fp4_decode

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>

#define BM 128
#define TN 32
#define BK 128
#define BKH 64
#define STAGES 4
#define ASZ (BM * BKH)   // 8192
#define BSZ (TN * BKH)   // 2048
#define SMEM (STAGES * ASZ + STAGES * BSZ + STAGES * 8 + 128)

__global__ void __launch_bounds__(128)
decode_mm(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
          __nv_bfloat16 *C, int N, int Kfp4) {
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
    int bcol = warp * 8;  // this warp's 8-col n-tile within the 32-col B tile

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

static void dmk(CUtensorMap *m, uint8_t *p, int Kb, int rows, int boxrows) {
    uint64_t gd[2] = {(uint64_t)Kb, (uint64_t)rows}; uint64_t gs[1] = {(uint64_t)Kb};
    uint32_t bd[2] = {(uint32_t)BKH, (uint32_t)boxrows}, es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, gd, gs, bd, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                                        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); printf("dmap fail %s\n", s); }
}

void run(int M, int N, int K) {
    int Kb = K / 2;
    uint8_t *dA, *dB; __nv_bfloat16 *dC;
    cudaMalloc(&dA, (size_t)M * Kb); cudaMalloc(&dB, (size_t)N * Kb);
    cudaMalloc(&dC, (size_t)M * N * sizeof(__nv_bfloat16));
    cudaMemset(dA, 0x22, (size_t)M * Kb); cudaMemset(dB, 0x22, (size_t)N * Kb);
    CUtensorMap mapA, mapB;
    dmk(&mapA, dA, Kb, M, BM); dmk(&mapB, dB, Kb, N, TN);
    cudaFuncSetAttribute(decode_mm, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    dim3 grid(N / TN, M / BM), block(128);
    for (int i = 0; i < 10; i++) decode_mm<<<grid, block, SMEM>>>(mapA, mapB, dC, N, K);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA err %s\n", cudaGetErrorString(cudaGetLastError())); return; }
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    int it = 100;
    cudaEventRecord(s);
    for (int i = 0; i < it; i++) decode_mm<<<grid, block, SMEM>>>(mapA, mapB, dC, N, K);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= it;
    size_t ne = (size_t)M * N;
    __nv_bfloat16 *hC = (__nv_bfloat16 *)malloc(ne * sizeof(__nv_bfloat16));
    cudaMemcpy(hC, dC, ne * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    int wrong = 0; for (size_t i = 0; i < ne; i++) if (__bfloat162float(hC[i]) != (float)K) wrong++;
    printf("decode %d/%d/%d: %6.1f us  %6.0f GFLOP/s  %s (%d wrong, grid %d)\n",
           M, N, K, ms * 1e3, 2.0 * M * N * K / (ms / 1e3) / 1e9, wrong == 0 ? "PASS" : "FAIL", wrong, (N / TN) * (M / BM));
    free(hC); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
    run(128, 4096, 4096);
    run(128, 14336, 4096);
    run(128, 4096, 14336);
    run(256, 4096, 4096);
    return 0;
}
